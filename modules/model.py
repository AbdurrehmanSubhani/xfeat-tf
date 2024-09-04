import tensorflow as tf
from tensorflow.keras import layers, models

class BasicLayer(tf.keras.layers.Layer):
    """
    Basic Convolutional Layer: Conv2D -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=1, use_bias=False):
        super(BasicLayer, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, strides=stride, padding=padding, dilation_rate=dilation, use_bias=use_bias)
        self.bn = layers.BatchNormalization(scale=False)
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class XFeatModel(tf.keras.Model):
    """
    Implementation of architecture described in 
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super(XFeatModel, self).__init__()
        self.norm = layers.LayerNormalization(axis=[1, 2])

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.skip1 = models.Sequential([
            layers.AveragePooling2D(pool_size=4, strides=4),
            layers.Conv2D(24, 1, strides=1, padding='valid')
        ])

        self.block1 = models.Sequential([
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        ])

        self.block2 = models.Sequential([
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        ])

        self.block3 = models.Sequential([
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, kernel_size=1, padding='valid'),
        ])

        self.block4 = models.Sequential([
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        ])

        self.block5 = models.Sequential([
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, kernel_size=1, padding='valid'),
        ])

        self.block_fusion = models.Sequential([
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            layers.Conv2D(64, 1, padding='valid')
        ])

        self.heatmap_head = models.Sequential([
            BasicLayer(64, 64, kernel_size=1, padding='valid'),
            BasicLayer(64, 64, kernel_size=1, padding='valid'),
            layers.Conv2D(1, 1),
            layers.Activation('sigmoid')
        ])

        self.keypoint_head = models.Sequential([
            BasicLayer(64, 64, kernel_size=1, padding='valid'),
            BasicLayer(64, 64, kernel_size=1, padding='valid'),
            BasicLayer(64, 64, kernel_size=1, padding='valid'),
            layers.Conv2D(65, 1)
        ])

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher = models.Sequential([
            layers.Dense(512),
            layers.BatchNormalization(scale=False),
            layers.ReLU(),
            layers.Dense(512),
            layers.BatchNormalization(scale=False),
            layers.ReLU(),
            layers.Dense(512),
            layers.BatchNormalization(scale=False),
            layers.ReLU(),
            layers.Dense(512),
            layers.BatchNormalization(scale=False),
            layers.ReLU(),
            layers.Dense(64)
        ])

    def _unfold2d(self, x, ws=2):
        """
        Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        x = tf.image.extract_patches(images=x,
                                     sizes=[1, ws, ws, 1],
                                     strides=[1, ws, ws, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='VALID')
        B, H, W, C = x.shape
        return tf.reshape(x, [B, H, W, -1])

    def call(self, x):
        """
        input:
            x -> tf.Tensor(B, H, W, C) grayscale or rgb images
        return:
            feats     ->  tf.Tensor(B, H/8, W/8, 64) dense local features
            keypoints ->  tf.Tensor(B, H/8, W/8, 65) keypoint logit map
            heatmap   ->  tf.Tensor(B, H/8, W/8, 1) reliability map
        """
        x = tf.reduce_mean(x, axis=-1, keepdims=True)
        x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4 = tf.image.resize(x4, size=(x3.shape[1], x3.shape[2]), method='bilinear')
        x5 = tf.image.resize(x5, size=(x3.shape[1], x3.shape[2]), method='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))

        return feats, keypoints, heatmap
