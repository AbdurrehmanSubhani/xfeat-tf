import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.image import resize

from modules.model import *
from modules.interpolator import InterpolateSparse2d

class XFeat(tf.keras.Model):
    """ 
    Implements the inference module for XFeat. 
    It supports inference for both sparse and semi-dense feature extraction & matching.
    """

    def __init__(self, weights=os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.h5', top_k=4096, detection_threshold=0.05):
        super().__init__()
        self.dev = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
        self.net = XFeatModel()  # Assuming XFeatModel is compatible with TensorFlow
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_weights(weights)
            else:
                self.net.set_weights(weights)

        self.interpolator = InterpolateSparse2d('bicubic')

        # Try to import LightGlue from Kornia
        self.kornia_available = False
        self.lighterglue = None
        try:
            import kornia
            self.kornia_available = True
        except ImportError:
            pass

    @tf.function
    def detectAndCompute(self, x, top_k=None, detection_threshold=None):
        """
        Compute sparse keypoints & descriptors. Supports batched mode.

        input:
            x -> tf.Tensor(B, C, H, W): grayscale or rgb image
            top_k -> int: keep best k features
        return:
            List[Dict]: 
                'keypoints'    ->   tf.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   tf.Tensor(N,): keypoint scores
                'descriptors'  ->   tf.Tensor(N, 64): local features
        """
        if top_k is None: top_k = self.top_k
        if detection_threshold is None: detection_threshold = self.detection_threshold
        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.net(x)
        M1 = tf.math.l2_normalize(M1, axis=1)

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        # Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores = tf.where(tf.reduce_all(tf.equal(mkpts, 0), axis=-1), -1, scores)

        # Select top-k features
        idxs = tf.argsort(-scores)
        mkpts_x = tf.gather(mkpts[..., 0], idxs, axis=-1)[:, :top_k]
        mkpts_y = tf.gather(mkpts[..., 1], idxs, axis=-1)[:, :top_k]
        mkpts = tf.concat([tf.expand_dims(mkpts_x, -1), tf.expand_dims(mkpts_y, -1)], axis=-1)
        scores = tf.gather(scores, idxs, axis=-1)[:, :top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        # L2-Normalize
        feats = tf.math.l2_normalize(feats, axis=-1)

        # Correct kpt scale
        mkpts = mkpts * tf.constant([rw1, rh1], dtype=mkpts.dtype)[None, None, :]

        valid = scores > 0
        return [{'keypoints': mkpts[b][valid[b]],
                 'scores': scores[b][valid[b]],
                 'descriptors': feats[b][valid[b]]} for b in range(B)]

    @tf.function
    def detectAndComputeDense(self, x, top_k=None, multiscale=True):
        """
        Compute dense *and coarse* descriptors. Supports batched mode.

        input:
            x -> tf.Tensor(B, C, H, W): grayscale or rgb image
            top_k -> int: keep best k features
        return: features sorted by their reliability score -- from most to least
            List[Dict]: 
                'keypoints'    ->   tf.Tensor(top_k, 2): coarse keypoints
                'scales'       ->   tf.Tensor(top_k,): extraction scale
                'descriptors'  ->   tf.Tensor(top_k, 64): coarse local features
        """
        if top_k is None: top_k = self.top_k
        if multiscale:
            mkpts, sc, feats = self.extract_dualscale(x, top_k)
        else:
            mkpts, feats = self.extractDense(x, top_k)
            sc = tf.ones_like(mkpts[:, :, 0])

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc}

    @tf.function
    def match_lighterglue(self, d0, d1):
        """
        Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched inference because of padding, but its possible to implement easily.
        input:
            d0, d1: Dict('keypoints', 'scores', 'descriptors', 'image_size (Width, Height)')
        output:
            mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        if not self.kornia_available:
            raise RuntimeError('We rely on kornia for LightGlue. Install with: pip install kornia')
        elif self.lighterglue is None:
            from modules.lighterglue import LighterGlue
            self.lighterglue = LighterGlue()

        data = {
            'keypoints0': tf.convert_to_tensor(d0['keypoints'])[None, ...],
            'keypoints1': tf.convert_to_tensor(d1['keypoints'])[None, ...],
            'descriptors0': tf.convert_to_tensor(d0['descriptors'])[None, ...],
            'descriptors1': tf.convert_to_tensor(d1['descriptors'])[None, ...],
            'image_size0': tf.convert_to_tensor(d0['image_size'])[None, ...],
            'image_size1': tf.convert_to_tensor(d1['image_size'])[None, ...]
        }

        # Dict -> log_assignment: [B x M+1 x N+1] matches0: [B x M] matching_scores0: [B x M] matches1: [B x N] matching_scores1: [B x N] matches: List[[Si x 2]], scores: List[[Si]]
        out = self.lighterglue(data)

        idxs = out['matches'][0]

        return d0['keypoints'][idxs[:, 0]].numpy(), d1['keypoints'][idxs[:, 1]].numpy()

    @tf.function
    def match_xfeat(self, img1, img2, top_k=None, min_cossim=-1):
        """
        Simple extractor and MNN matcher.
        For simplicity it does not support batched mode due to possibly different number of kpts.
        input:
            img1 -> tf.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
            img2 -> tf.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
            top_k -> int: keep best k features
        returns:
            mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        if top_k is None: top_k = self.top_k
        img1 = self.parse_input(img1)
        img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1, top_k=top_k)[0]
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]

        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)

        return out1['keypoints'][idxs0].numpy(), out2['keypoints'][idxs1].numpy()

    @tf.function
    def match_xfeat_star(self, im_set1, im_set2, top_k=None):
        """
        Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
        input:
            im_set1 -> tf.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
            im_set2 -> tf.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
            top_k -> int: keep best k features
        returns:
            matches -> np.ndarray (B, N, 2) xy coordinate matches from image1 to image2
        """
        if top_k is None: top_k = self.top_k
        im_set1 = self.parse_input(im_set1)
        im_set2 = self.parse_input(im_set2)

        B = im_set1.shape[0]

        # Extract coarse
        dset1 = self.detectAndComputeDense(im_set1, top_k)
        dset2 = self.detectAndComputeDense(im_set2, top_k)

        matches = []

        for i in range(B):
            idx0, idx1 = self.match(dset1[i]['descriptors'], dset2[i]['descriptors'])
            matches.append(np.stack([dset1[i]['keypoints'][idx0].numpy(),
                                     dset2[i]['keypoints'][idx1].numpy()], axis=1))

        return matches

    def preprocess_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        if x.ndim == 3:
            x = tf.expand_dims(x, 0)
        x = tf.cast(x, tf.float32) / 255.
        h, w = tf.shape(x)[2], tf.shape(x)[3]
        return x, h, w

    def get_kpts_heatmap(self, kpts):
        return tf.reduce_max(kpts, axis=1)

    def NMS(self, heatmap, threshold=0.05, kernel_size=5):
        kernel = tf.ones((kernel_size, kernel_size), dtype=heatmap.dtype)
        heatmap = tf.image.dilation2d(heatmap, kernel, padding='VALID')
        return tf.cast(heatmap > threshold, tf.float32)

    def extract_dualscale(self, x, top_k):
        x_small = resize(x, [x.shape[2] // 2, x.shape[3] // 2], method='bilinear')
        feats_small = self.detectAndCompute(x_small, top_k)
        feats_large = self.detectAndCompute(x, top_k)

        mkpts = tf.concat([feats_large['keypoints'], feats_small['keypoints']], axis=0)
        sc = tf.concat([tf.ones_like(feats_large['keypoints'][:, :, 0]), 
                        0.5 * tf.ones_like(feats_small['keypoints'][:, :, 0])], axis=0)
        feats = tf.concat([feats_large['descriptors'], feats_small['descriptors']], axis=0)

        return mkpts, sc, feats

    def extractDense(self, x, top_k):
        return self.detectAndCompute(x, top_k)

    def parse_input(self, x):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        if x.ndim == 3:
            x = tf.expand_dims(x, 0)
        return x
