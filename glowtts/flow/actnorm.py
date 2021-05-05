from typing import Tuple

import tensorflow as tf


class ActNorm(tf.keras.Model):
    """Activation normalization.
    """
    def __init__(self):
        """Initializer.
        """
        super().__init__()
        self.init = tf.Variable(0., trainable=False)
    
    def ddi(self, inputs: tf.Tensor, mask: tf.Tensor):
        """Data-dependent initialization.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        """
        # []
        denom = tf.reduce_sum(mask)
        # [C]
        mean = tf.reduce_sum(inputs, axis=[0, 1]) / denom
        # [C]
        variance = tf.reduce_sum(tf.square(inputs), axis=[0, 1]) / denom - tf.square(mean)
        # [C]
        logstd = 0.5 * tf.math.log(tf.maximum(variance, 1e-5))
        # initialize
        self.mean = tf.Variable(mean, trainable=True)
        self.logstd = tf.Variable(logstd, trainable=True)
        self.init.assign(1.)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalize inputs with ddi.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], normalized.
            dlogdet: [tf.float32; [B]], likelihood contribution.
        """
        if self.init == 0.:
            self.ddi(inputs, mask)
        # [B, T, C]
        outputs = (inputs - self.mean[None, None]) \
            * tf.exp(-self.logstd[None, None]) \
            * mask[..., None]
        # [B]
        dlogdet = tf.reduce_sum(-self.logstd) * tf.reduce_sum(mask, axis=1)
        return outputs, dlogdet
    
    def inverse(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Denormalize inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], denormalized.
        """
        assert self.init == 1., "require ddi"
        # [B, T, C]
        x = inputs * tf.exp(self.logstd[None, None]) + self.mean[None, None]
        # [B, T, C]
        return x * mask[..., None]
