from typing import Tuple

import tensorflow as tf


class Inv1x1Conv(tf.keras.Model):
    """Invertible 1x1 grouped convolution.
    """
    def __init__(self, groups):
        """Initializer.
        Args:
            groups: int, size of the convolution groups.
        """
        super(Inv1x1Conv, self).__init__()
        self.groups = groups
        # [groups, groups]
        weight, _ = tf.linalg.qr(tf.random.normal([groups, groups]))
        self.weight = tf.Variable(weight)
    
    def transform(self, inputs: tf.Tensor, mask: tf.Tensor, weight: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Convolve inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32, [B, T]], sequence mask.
            weight: [tf.float32; [G, G]], convolutional weight.
        Returns:
            outputs: [tf.float32; [B, T, C]], convolved tensor.
            logdet: [tf.float32; [B]], log-determinant of conv2d derivation.
        """
        # [B, T, C // G, G]
        x = self.grouping(inputs)
        # [B, T, C // G, G]
        x = tf.nn.conv2d(x, weight[None, None], 1, padding='SAME')
        # []
        _, dlogdet = tf.linalg.slogdet(weight)
        # [B]
        dlogdet = dlogdet * tf.reduce_sum(mask, axis=-1) * \
            tf.cast(tf.shape(x)[2], tf.float32)
        # [B, T, C]
        outputs = self.recover(x)
        # [B, T, C], [B]
        return outputs, dlogdet

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward 1x1 convolution.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32, [B, T]], sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], convolved tensor.
            logdet: [tf.float32; [B]], log-determinant of conv2d derivation.
        """
        return self.transform(inputs, mask, self.weight)

    def inverse(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Inverse 1x1 convolution.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32, [B, T]], sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], convolved tensor.
        """
        outputs, _ = self.transform(inputs, mask, tf.linalg.inv(self.weight))
        return outputs

    def grouping(self, x: tf.Tensor) -> tf.Tensor:
        """Grouping tensor.
        Args:
            x: [tf.float32; [B, T, C]], input tensor.
        return:
            [tf.float32; [B, T, C // G, G]], grouped tensor.
        """
        # B, T, C
        bsize, timestep, channels = tf.shape(x)
        # [B, T, 2, C // G, G // 2]
        x = tf.reshape(x, [bsize, timestep, 2, channels // self.groups, self.groups // 2])
        # [B, T, C // G, 2, G // 2]
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        # [B, T, C // G, G]
        return tf.reshape(x, [bsize, timestep, channels // self.groups, self.groups])
    
    def recover(self, x: tf.Tensor) -> tf.Tensor:
        """Recover grouped tensor.
        Args:
            x: [tf.float32; [B, T, C // G, G]], grouped tensor.
        Returns:
            [tf.float32; [B, T, C]], recovered.
        """
        # B, T, C // G, G(=self.groups)
        bsize, timestep, splits, _ = tf.shape(x)
        # [B, T, C // G, 2, G // 2]
        x = tf.reshape(x, [bsize, timestep, splits, 2, self.groups // 2])
        # [B, T, 2, C // G, G // 2]
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        # [B, T, C]
        return tf.reshape(x, [bsize, timestep, splits * self.groups])
