import tensorflow as tf
import tensorflow_addons as tfa

class InterpolateSparse2d(tf.keras.layers.Layer):
    """ Efficiently interpolate tensor at given sparse 2D positions. """
    def __init__(self, mode='bicubic', align_corners=False, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1, 1]. """
        return 2. * (x / tf.constant([W-1, H-1], dtype=x.dtype)) - 1.

    def call(self, x, pos, H, W):
        """
        Input
            x: [B, H, W, C] feature tensor (channels last format)
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2D positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2D positions
        """
        grid = self.normgrid(pos, H, W)
        grid = tf.concat([grid, tf.zeros_like(grid[:, :, :1])], axis=-1)  # Add a zero channel for the grid
        grid = tf.image.convert_image_dtype(grid, dtype=x.dtype)  # Ensure the grid is in the same dtype as x

        x = tf.image.sample_distorted_bounding_box(
            tf.shape(x),
            bounding_boxes=tf.expand_dims(grid, 0),
            area_range=[0.0, 1.0],
            max_attempts=100
        )

        # Resample using tf.image
        x = tf.image.resize(x, size=(H, W), method=tf.image.ResizeMethod.BICUBIC if self.mode == 'bicubic' else tf.image.ResizeMethod.BILINEAR)

        x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B, C, H, W]
        x = tf.squeeze(x, axis=-2)  # Remove the last dimension

        return x
