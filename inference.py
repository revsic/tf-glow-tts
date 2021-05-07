import argparse
import json

import tensorflow as tf

from config import Config
from datasets.normalizer import TextNormalizer
from glowtts import GlowTTS
from utils.diffwave import pretrained_diffwave

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--ckpt')
parser.add_argument('--text')

args = parser.parse_args()

with open(args.config) as f:
    config = Config.load(json.load(f))

tts = GlowTTS(config.model)
tts.restore(args.ckpt).expect_partial()

diffwave = pretrained_diffwave()

# [S]
label = tf.convert_to_tensor(TextNormalizer().labeling(args.text), dtype=tf.int32)
# S
textlen = tf.convert_to_tensor(len(label), dtype=tf.int32)

# [1, T, M]
mel, _, _ = tts(label[None], textlen[None])
# [1, T x H]
audio, _ = diffwave(mel)
# [T x H, 1], mono-channel
audio = tf.squeeze(audio, axis=0)[:, None]

tf.io.write_file('result.wav', tf.audio.encode_wav(audio, 22050))
