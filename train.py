import argparse
import json
import os
from typing import Dict, Optional, Tuple

import git
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

from config import Config
from datasets import TTSDataset
from datasets.ljspeech import LJSpeech
from glowtts import GlowTTS
from utils.diffwave import pretrained_diffwave


class Trainer:
    """Glow-TTS trainer.
    """
    def __init__(self, model: GlowTTS, ttsdata: TTSDataset, config: Config):
        """Initializer.
        Args:
            model: Glow-TTS model.
            ttsdata: dataset for tts models,
                which provides already batched and normalized speech dataset.
            config: unified configurations.
        """
        self.model = model
        self.ttsdata = ttsdata
        self.config = config

        trainset, testset = self.ttsdata.dataset(config.train.split)
        self.trainset = trainset.cache() \
            .shuffle(config.train.bufsiz) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        self.testset = testset.cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        self.trainsize = tf.data.experimental.cardinality(
            self.trainset).numpy().item()
        self.testsize = tf.data.experimental.cardinality(
            self.testset).numpy().item()

        self.optim = tf.keras.optimizers.Adam(
            config.train.lr(),
            config.train.beta1,
            config.train.beta2,
            config.train.eps)

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.cmap = tf.constant(
            plt.get_cmap('viridis').colors, dtype=tf.float32)

        self.diffwave = pretrained_diffwave()

    def train(self, epoch: int = 0):
        """Train GlowTTS.
        Args:
            epoch: starting epoch.
        """
        step = epoch * self.trainsize
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=self.trainsize, leave=False) as pbar:
                for iter, (text, mel, textlen, mellen) in enumerate(self.trainset):
                    with tf.GradientTape() as tape:
                        # tape.watch(self.model.trainable_variables)
                        loss, losses, attn = \
                            self.model.compute_loss(text, textlen, mel, mellen)

                    grad = tape.gradient(loss, self.model.trainable_variables)
                    self.optim.apply_gradients(
                        zip(grad, self.model.trainable_variables))

                    norm = tf.reduce_mean([
                        tf.norm(g) for g in grad if g is not None])
                    del grad

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {'loss': loss.numpy().item(),
                         'step': step,
                         'grad': norm.numpy().item()})

                    with self.train_log.as_default():

                        tf.summary.scalar('loss/loss', loss, step)
                        for key, value in losses.items():
                            tf.summary.scalar(f'loss/{key}', value, step)

                        lr = self.optim.lr
                        # check fixed learning rate.
                        if not isinstance(lr, tf.Variable):
                            lr = lr(step)
                        tf.summary.scalar('common/lr', lr, step)
                        tf.summary.scalar('common/grad norm', norm, step)

                        if (iter + 1) % (self.trainsize // 10) == 0:
                            tf.summary.image(
                                'align/mas', self.align_img(attn), step, max_outputs=1)

            self.model.write(
                '{}_{}.ckpt'.format(self.ckpt_path, epoch), self.optim)

            loss, (align, textlen, mellen) = self.eval_loss()
            with self.test_log.as_default():
                for key, value in loss.items():
                    tf.summary.scalar('loss/' + key, value, step)

                tf.summary.image(
                    'align/mas', self.align_img(align), step, max_outputs=1)

                mel, pmel, audio, amel, align = self.evaluate()
                tf.summary.image(
                    'mel/gt', self.mel_img(mel[None]), step)
                tf.summary.image(
                    'mel/eval', self.mel_img(pmel[None]), step)
                tf.summary.audio(
                    'audio/eval', audio[None, ..., None],
                    self.config.data.sr, step)
                tf.summary.image(
                    'mel/audio', self.mel_img(amel[None]), step)
                tf.summary.image(
                    'align/duration', self.align_img(align[None]), step)

                del text, mel, pmel, audio, align, textlen, mellen

    def eval_loss(self) -> Tuple[
            Dict[str, tf.Tensor],
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """Compute evaluation loss.
        Returns:
            loss: evaluation losses.
            align: [tf.float32; [B, T, S]], alignment.
            textlen: [tf.float32; [B]], text lengths for alignment plotting.
            mellen: [tf.float32; [B]], mel lengths for alignment plotting.
        """
        loss = {}
        align, textlen, mellen = None, None, None
        for text, mel, textlen, mellen in tqdm.tqdm(
                self.testset, total=self.testsize, leave='False'):
            loss_, losses, align = \
                self.model.compute_loss(text, textlen, mel, mellen)
            losses['loss'] = loss_
            for key, val in losses.items():
                if key not in loss:
                    loss[key] = []
                loss[key].append(val.numpy().item())

        for key, val in loss.items():
            loss[key] = sum(val) / len(val)
        return loss, (align, textlen, mellen)

    def evaluate(self, idx: Optional[int] = None) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate evaluation purpose audio.
        Args:
            idx: Optional[int], target index,
                if None is given, index will be randomized.
        Returns:
            mel: [tf.float32; [T, mel]]: gt mel-spectrogram.
            pmel: [tf.float32; [T', mel]]: predicted mel.
            audio: [tf.float32; [T' x hop]]: audio signal.
            amel: [tf.float32; [T', mel]]: mel from generated audio.
            align: [tf.float32; [T' / F, S]],
                align for text to mel.
        """
        if idx is None:
            idx = np.random.randint(self.config.data.batch)
        # [B, S], [B, T, mel], [B], [B]
        text, mel, textlen, mellen = next(iter(self.testset))
        # [1, S]
        text = text[idx:idx + 1, :textlen[idx]]
        # [1, T, mel]
        mel = mel[idx:idx + 1, :mellen[idx]]
        # [1, T', mel], [1], [1, T' / F, S]
        pmel, _, align = self.model(text, textlen[idx:idx + 1])
        # [1, T' x hop]
        audio, _ = self.diffwave(pmel)
        # [1, T', mel]
        amel = self.ttsdata.mel_fn(audio)
        # [T, mel], [T', mel], [T' x hop], [T', mel], [T' / F, S]
        return mel[0], pmel[0], audio[0], amel[0], align[0]

    def align_img(self, align: tf.Tensor) -> tf.Tensor:
        """Generate alignment images.
        Args:
            align: tf.Tensor, [tf.float32; [B, T, S]], attention alignment.
        Returns:
            [tf.int32, [B, S, T, 3]], align image.
        """
        # [B, T], otherwise, run min-max norm
        amin = tf.reduce_min(align, axis=-1, keepdims=True)
        amax = tf.reduce_max(align, axis=-1, keepdims=True)
        # [B, T, S]
        align = (align - amin) / (amax - amin)
        # in range(0, 255)
        align = tf.cast(align * 255, tf.int32)
        # [B, T, S, 3]
        align = tf.gather(self.cmap, align)
        # [B, S, T, 3]
        return tf.transpose(align, [0, 2, 1, 3])

    def mel_img(self, mel: tf.Tensor) -> tf.Tensor:
        """Generate mel-spectrogram images.
        Args:
            mel: [tf.float32; [B, T, mel]], mel-spectrogram.
        Returns:
            [tf.int32; [B, mel, T, 3]], mel-spectrogram in viridis color map.
        """
        # [B, mel, T]
        mel = tf.transpose(mel, [0, 2, 1])
        # minmax norm in range(0, 1)
        mel = (mel - tf.reduce_min(mel)) / (tf.reduce_max(mel) - tf.reduce_min(mel))
        # in range(0, 255)
        mel = tf.cast(mel * 255, tf.int32)
        # [B, mel, T, 3]
        mel = tf.gather(self.cmap, mel)
        # make origin lower
        mel = tf.image.flip_up_down(mel)
        return mel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--download', default=False, action='store_true')
    parser.add_argument('--from-raw', default=False, action='store_true')
    parser.add_argument('--ignore-warning', default=False, action='store_true')
    args = parser.parse_args()

    if args.ignore_warning:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    log_path = os.path.join(config.train.log, config.train.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    lj = LJSpeech(args.data_dir, args.download, not args.from_raw)
    ttsdata = TTSDataset(lj, config.data)

    glowtts = GlowTTS(config.model)
    trainer = Trainer(glowtts, ttsdata, config)

    if args.load_epoch > 0:
        super_path = os.path.join(config.train.ckpt, config.train.name)
        ckpt_path = '{}_{}.ckpt'.format(config.train.name, args.load_epoch)
        ckpt_path = next(
            name for name in os.listdir(super_path)
                 if name.startswith(ckpt_path) and name.endswith('.index'))
        ckpt_path = os.path.join(super_path, ckpt_path[:-6])
        
        print('[*] load checkpoint: ' + ckpt_path)
        trainer.model.restore(ckpt_path, trainer.optim)

    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    trainer.train(args.load_epoch)
