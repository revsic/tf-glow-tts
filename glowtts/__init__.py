from typing import Dict, Optional, Tuple

import tensorflow as tf
import numpy as np

from .config import Config
from .durator import DurationPredictor
from .flow import WaveNetFlow
from .transformer import Transformer


class GlowTTS(tf.keras.Model):
    """Glow-TTS: A Generative Flow for Text-to-Speech
        via Monotonic Alignment Search, Kim et al. In NeurIPS 2020. 
    """
    DURATION_MAX = 80

    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configuration object.
        """
        super().__init__()
        self.factor = config.factor
        self.temperature = config.temperature
        self.embedding = tf.keras.layers.Embedding(
            config.vocabs, config.channels)
        self.encoder = Transformer(config)
        self.decoder = WaveNetFlow(config)
        self.durator = DurationPredictor(
            config.dur_layers, config.channels,
            config.dur_kernel, config.dur_dropout)
        # mean-only training
        self.proj_mu = tf.keras.layers.Dense(config.neck)

    def call(self, inputs: tf.Tensor, lengths: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate mel-spectrogram from text inputs.
        Args:
            inputs: [tf.int32; [B, S]], input sequence.
            lengths: [tf.int32; [B]], input lengths.
        Returns:
            mel: [tf.float32; [B, T, M]], generated mel-spectrogram.
            mellen: [tf.int32; [B]], length of the mel-spectrogram.
            attn: [tf.float32; [B, T / F, S]], attention alignment.
        """
        # S
        _, seqlen = tf.shape(inputs)
        # [B, S]
        mask = self.mask(lengths, maxlen=seqlen)
        # [B, S, C]
        embedding = self.embedding(inputs) * mask[..., None]
        # [B, S, C]
        hidden, _ = self.encoder(embedding, mask)
        # [B, S, N]
        mean = self.proj_mu(hidden)

        # [B, S]
        duration = self.quantize(self.durator(hidden, mask), mask)
        # [B, T / F, S], [B, T / F]
        attn, mask = self.align(duration)

        # [B, T / F, N(=M x F)], []
        mean, std = tf.matmul(attn, mean), self.temperature
        # [B, T / F, M x F]
        sample = mean + tf.random.normal(tf.shape(mean)) * std

        # [B, T / F, M x F]
        mel = self.decoder.inverse(sample * mask[..., None], mask)
        # [B]
        mellen = tf.reduce_sum(mask, axis=-1)
        # [B, T, M], [B]
        mel, mellen = self.unfold(mel, mellen)
        return mel, mellen, attn

    def compute_loss(self, text: tf.Tensor, textlen: tf.Tensor,
                     mel: tf.Tensor, mellen: tf.Tensor) \
            -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
        """Compute loss for glow-tts.
        Args:
            text: [tf.int32; [B, S]], input text.
            textlen: [tf.int32; [B]], text lengths.
            mel: [tf.float32; [B, T, M]], mel-spectrogram.
            mellen: [tf.int32; [B]], mel lengths.
        Returns:
            loss: [tf.float32; []], loss tensor.
            ll: loss lists.
            attn: [tf.float32; [B, T, S]], attention alignment.
        """
        # [B, S]
        mask = self.mask(textlen, maxlen=tf.shape(text)[1])
        # [B, S, C]
        embedding = self.embedding(text) * mask[..., None]
        # [B, S, C]
        hidden, _ = self.encoder(embedding, mask)
        # [B, S, N], constant standard deviation
        mean = self.proj_mu(hidden)

        # [B, T', N], [B]
        mel, mellen = self.fold(mel, mellen)
        # [B, T']
        melmask = self.mask(mellen, maxlen=tf.shape(mel)[1])
        # [B, T', N], [B]
        latent, dlogdet = self.decoder(mel, melmask)

        # [B, T', S]
        attnmask = melmask[..., None] * mask[:, None]
        # [B, T', S], (mean - latent) ** 2
        dist = tf.reduce_sum(tf.square(latent), axis=-1, keepdims=True) - \
            2 * tf.matmul(latent, tf.transpose(mean, [0, 2, 1])) + \
            tf.reduce_sum(tf.square(mean), axis=-1)[:, None]
        # [B, T', S], assume constant standard deviation, 1.
        ll = -0.5 * (np.log(2 * np.pi) + dist) * attnmask
        # [B, T', S], attention alignment
        attn = tf.stop_gradient(self.monotonic_alignment_search(ll, attnmask))

        # [B, T', N]
        mean = tf.matmul(attn, mean)
        # [B]
        nll = 0.5 * (np.log(2 * np.pi) +
            tf.reduce_sum(tf.square(latent - mean), axis=[1, 2])) - dlogdet
        # []
        nll = tf.reduce_mean(
            nll / tf.cast(mellen * tf.shape(mean)[-1], tf.float32))

        # [B, S]
        logdur = self.durator(tf.stop_gradient(hidden), mask)
        # [B, S]
        gtdur = tf.math.log(tf.maximum(tf.reduce_sum(attn, axis=1), 1e-5)) * mask
        # [B]
        durloss = tf.reduce_sum(tf.square(logdur - gtdur), axis=-1) / \
            tf.cast(mellen, tf.float32)
        # []
        durloss = tf.reduce_mean(durloss)

        # []
        loss = nll + durloss
        return loss, {'nll': nll, 'durloss': durloss}, attn

    def quantize(self, logdur: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Convert log-duration to duration.
        Args:
            logdur: [tf.float32; [B, T]], log-duration.
            mask: [tf.float32; [B, T]], sequence mask.
        Returns:
            [tf.int32; [B, T]], duration.
        """
        # [B, T]
        dur = tf.round(tf.exp(logdur))
        # [B, T]
        dur = tf.clip_by_value(dur, 1., GlowTTS.DURATION_MAX) * mask
        return tf.cast(dur, tf.int32)

    def fold(self, inputs: tf.Tensor, lengths: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Fold inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            lengths: [tf.int32; [B]], input lengths.
        Returns:
            folded: [tf.float32; [B, T // F, C x F]], folded.
            lengths: [tf.int32; [B]], folded lengths.
        """
        # B, T, _
        bsize, timestep, channels = tf.shape(inputs)
        if timestep % self.factor != 0:
            rest = self.factor - timestep % self.factor
            # [B, T + R, C]
            inputs = tf.concat([inputs, tf.zeros([bsize, rest, channels])], axis=1)
            # T + R
            timestep = timestep + rest
        # [B, T // F, C x F]
        folded = tf.reshape(inputs, [bsize, timestep // self.factor, -1])
        # T / F
        lengths = tf.cast(tf.math.ceil(lengths / self.factor), tf.int32)
        return folded, lengths

    def unfold(self, inputs: tf.Tensor, lengths: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Recover folded inputs.
        Args:
            inputs: [tf.float32; [B, T // F, C x F]], folded tensor.
            lengths: [tf.int32; [B]], input lengths.
        Returns:
            recovered: [tf.float32; [B, T, C]], recovered.
            lengths: [tf.int32; [B]], recovered lengths.
        """
        # B, T // F, _
        bsize, timestep, _ = tf.shape(inputs)
        # [B, T, C]
        recovered = tf.reshape(inputs, [bsize, timestep * self.factor, -1])
        return recovered, lengths * self.factor

    def mask(self, lengths: tf.Tensor, maxlen: Optional[tf.Tensor] = None) \
            -> tf.Tensor:
        """Generate sequence mask from lengths.
        Args:
            lengths: [tf.int32; [B]], lengths.
            maxlen: tf.float32, maximum length.
        Returns:
            [tf.flaot32; [B, maxlen]], sequence mask.
        """
        if maxlen is None:
            maxlen = tf.reduce_max(lengths)
        # [B, S]
        return tf.cast(tf.range(maxlen)[None] < lengths[:, None], tf.float32)

    def align(self, duration: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate attention from duration.
        Args:
            duration: [tf.int32; [B, S]], duration vectors.
        Returns:
            attn: [tf.float32; [B, T, S]], attention map.
            mask: [tf.float32; [B, T]], sequence mask.
                where T = max(duration.sum(axis=-1)).
        """
        # S
        bsize, inplen = tf.shape(duration)
        # T
        maxlen = tf.reduce_max(tf.reduce_sum(duration, axis=-1))
        # [B x S]
        cumsum = tf.reshape(tf.math.cumsum(duration, axis=-1), [-1])
        # [B x S, T]
        cumattn = self.mask(cumsum, maxlen)
        # [B, S, T]
        cumattn = tf.reshape(cumattn, [bsize, inplen, maxlen])
        # [B, S, T]
        attn = cumattn - tf.pad(cumattn, [[0, 0], [1, 0], [0, 0]])[:, :-1]
        # [B, T]
        mask = cumattn[:, -1]
        # [B, T, S], [B, T]
        return tf.transpose(attn, [0, 2, 1]), mask

    def monotonic_alignment_search(self,
                                   ll: tf.Tensor,
                                   mask: tf.Tensor) -> tf.Tensor:
        """Monotonic aligment search, reference from jaywalnut310's glow-tts.
        https://github.com/jaywalnut310/glow-tts/blob/master/commons.py#L60

        Args:
            ll: [tf.float32; [B, T, S]], loglikelihood matrix.
            mask: [tf.float32; [B, T, S]], attention mask.
        Returns:
            [tf.float32; [B, T, S]], alignment.
        """
        # B, T, S
        bsize, timestep, seqlen = tf.shape(ll)
        # (expected) T x [B, S]
        direction = []
        # [B, S]
        prob = tf.zeros([bsize, seqlen], dtype=tf.float32)
        # [1, S]
        x_range = tf.range(seqlen)[None]
        for j in tf.range(timestep):
            # [B, S]
            prev = tf.pad(prob, [[0, 0], [1, 0]],
                          mode='CONSTANT',
                          constant_values=-np.inf)[:, :-1]
            cur = prob
            # larger value mask
            max_mask = cur >= prev
            # select larger value
            prob_max = tf.where(max_mask, cur, prev)
            # write direction
            direction.append(max_mask)
            # update prob
            prob = tf.where(x_range <= j, prob_max + ll[:, j], tf.float32.min)
        # [B, T, S]
        direction = tf.cast(tf.stack(direction, axis=1), tf.int32)
        # [B, T, S], mask
        direction = tf.where(tf.cast(mask, tf.bool), direction, 1)
        # (expected) T x [B, S]
        attn = [] # tf.zeros_like(ll, dtype=np.float32)
        # [B]
        index = tf.cast(tf.reduce_sum(mask[:, 0], axis=-1), tf.int32) - 1
        # [B], [B]
        index_range, values = tf.range(bsize), tf.ones(bsize)
        for j in tf.range(timestep)[::-1]:
            # [B, S]
            attn.append(tf.scatter_nd(
                tf.stack([index_range, index], axis=1),
                values, [bsize, seqlen]))
            # [B]
            dir = tf.gather_nd(
                direction,
                tf.stack([index_range, tf.cast(values, tf.int32) * j, index],
                         axis=1))
            # [B]
            index = index + dir - 1
        # [B, T, S]
        return tf.stack(attn, axis=1) * mask

    def write(self, path: str,
              optim: Optional[tf.keras.optimizers.Optimizer] = None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: path to write.
            optim: optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path: str,
                optim: Optional[tf.keras.optimizers.Optimizer] = None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: path to restore.
            optim: optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        return ckpt.restore(path)
