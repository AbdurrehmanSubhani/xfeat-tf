import tensorflow as tf
import tensorflow_addons as tfa
import os
import requests

class LighterGlue(tf.keras.Model):
    """
        Lighter version of LightGlue :)
    """

    default_conf_xfeat = {
        "name": "xfeat",  # just for interfacing
        "input_dim": 64,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": 0.95,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    def __init__(self, weights_path=None, **kwargs):
        super().__init__(**kwargs)
        # Initialize LightGlue equivalent in TensorFlow (replace with actual model implementation)
        self.net = self._initialize_lightglue()  
        
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
        else:
            self._download_and_load_weights()

    def _initialize_lightglue(self):
        # Define the model architecture
        # Replace with actual implementation
        model = tf.keras.Sequential([
            # Add layers here to match LightGlue architecture
        ])
        return model

    def _load_weights(self, weights_path):
        # Load weights from file
        self.net.load_weights(weights_path)

    def _download_and_load_weights(self):
        url = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt"
        weights = requests.get(url).content
        with open('/tmp/xfeat-lighterglue.pt', 'wb') as f:
            f.write(weights)
        self._load_weights('/tmp/xfeat-lighterglue.pt')

    def call(self, data):
        result = self.net({
            'image0': {'keypoints': data['keypoints0'], 'descriptors': data['descriptors0'], 'image_size': data['image_size0']},
            'image1': {'keypoints': data['keypoints1'], 'descriptors': data['descriptors1'], 'image_size': data['image_size1']}
        })
        return result
