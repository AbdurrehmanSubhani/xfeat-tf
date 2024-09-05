import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import tqdm

from tensorflow.keras import layers, models
from tensorflow.image import resize
from tensorflow.image import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
from tensorflow.image import rgb_to_grayscale
from tensorflow.image import crop_to_bounding_box

# Ensure reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def generate_random_tps(shape, grid=(8, 6), global_multiplier=0.3, prob=0.5):
    h, w = shape
    sh, sw = h / grid[0], w / grid[1]
    src = np.stack(np.meshgrid(np.arange(0, h + sh, sh), np.arange(0, w + sw, sw), indexing='ij'), -1).reshape(-1, 2)

    offsets = np.random.rand(grid[0] + 1, grid[1] + 1, 2) - 0.5
    offsets *= np.array([sh / 2, sw / 2]) * min(0.97, 2.0 * global_multiplier)
    dst = src + offsets if np.random.uniform() < prob else src

    src = (src / np.array([h, w])) * 2 - 1
    dst = (dst / np.array([h, w])) * 2 - 1

    weights, A = find_tps_transform(dst, src)
    return src, weights, A

def generate_random_homography(shape, global_multiplier=0.3):
    theta = np.radians(np.random.uniform(-30, 30))
    scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)
    tx, ty = -shape[1] / 2.0, -shape[0] / 2.0
    txn, tyn = np.random.normal(0, 120.0 * global_multiplier, 2)
    c, s = np.cos(theta), np.sin(theta)

    sx, sy = np.random.normal(0, 0.6 * global_multiplier, 2)
    p1, p2 = np.random.normal(0, 0.006 * global_multiplier, 2)

    H_t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    H_r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    H_a = np.array([[1, sy, 0], [sx, 1, 0], [0, 0, 1]])
    H_p = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])
    H_s = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    H_b = np.array([[1.0, 0, -tx + txn], [0, 1, -ty + tyn], [0, 0, 1]])

    H = H_b @ H_s @ H_p @ H_a @ H_r @ H_t
    return H

class AugmentationPipe(tf.keras.layers.Layer):
    def __init__(self, img_dir, warp_resolution=(1200, 900), out_resolution=(400, 300),
                 sides_crop=0.2, max_num_imgs=50, num_test_imgs=10, batch_size=1,
                 photometric=True, geometric=True, reload_step=1000):
        super(AugmentationPipe, self).__init__()
        self.device = '/device:CPU:0'
        self.half = 16

        self.dims = warp_resolution
        self.batch_size = batch_size
        self.out_resolution = out_resolution
        self.sides_crop = sides_crop
        self.max_num_imgs = max_num_imgs
        self.num_test_imgs = num_test_imgs
        self.reload_step = reload_step
        self.geometric = geometric
        self.cnt = 1

        self.dims_t = tf.convert_to_tensor([int(self.dims[0] * (1. - self.sides_crop)) - int(self.dims[0] * self.sides_crop) - 1,
                                            int(self.dims[1] * (1. - self.sides_crop)) - int(self.dims[1] * self.sides_crop) - 1], dtype=tf.float32)
        self.dims_s = tf.convert_to_tensor([self.dims_t[0] / out_resolution[0], self.dims_t[1] / out_resolution[1]], dtype=tf.float32)

        self.all_imgs = glob.glob(img_dir + '/*.jpg') + glob.glob(img_dir + '/*.png')

        if photometric:
            self.aug_list = tf.keras.Sequential([
                layers.Lambda(lambda x: adjust_brightness(x, delta=0.15)),
                layers.Lambda(lambda x: adjust_contrast(x, contrast_factor=0.15)),
                layers.Lambda(lambda x: adjust_saturation(x, saturation_factor=0.15)),
                layers.Lambda(lambda x: adjust_hue(x, delta=0.15)),
            ])
        else:
            self.aug_list = tf.keras.Sequential()

        if len(self.all_imgs) < 10:
            raise RuntimeError('Couldn\'t find enough images to train. Please check the path: ', img_dir)

        if len(self.all_imgs) - num_test_imgs < max_num_imgs:
            raise RuntimeError('Error: test set overlaps with training set! Decrease number of test imgs')

        self.load_imgs()

    def load_imgs(self):
        random.shuffle(self.all_imgs)
        train = []
        fast = cv2.FastFeatureDetector_create(30)
        for p in tqdm.tqdm(self.all_imgs[:self.max_num_imgs], desc='loading train'):
            im = cv2.imread(p)
            halfH, halfW = im.shape[0] // 2, im.shape[1] // 2
            if halfH > halfW:
                im = np.rot90(im)
                halfH, halfW = halfW, halfH

            if im.shape[0] != self.dims[1] or im.shape[1] != self.dims[0]:
                im = cv2.resize(im, self.dims)

            train.append(np.copy(im))

        self.train = np.array(train)
        self.test = np.array([
            cv2.resize(cv2.imread(p), self.dims)
            for p in tqdm.tqdm(self.all_imgs[-self.num_test_imgs:], desc='loading test')
        ])

    def norm_pts_grid(self, x):
        if len(x.shape) == 2:
            return (x.reshape(1, -1, 2) * self.dims_s / self.dims_t) * 2. - 1
        return (x * self.dims_s / self.dims_t) * 2. - 1

    def denorm_pts_grid(self, x):
        if len(x.shape) == 2:
            return ((x.reshape(1, -1, 2) + 1) / 2.) / self.dims_s * self.dims_t
        return ((x + 1) / 2.) / self.dims_s * self.dims_t

    def rnd_kps(self, shape, n=256):
        h, w = shape
        kps = np.random.rand(3, n).astype(np.float32)
        kps[0, :] *= w
        kps[1, :] *= h
        kps[2, :] = 1.0
        return kps

    def warp_points(self, H, pts):
        scale = self.dims_s.reshape(-1, 2)
        offset = np.array([int(self.dims[0] * self.sides_crop), int(self.dims[1] * self.sides_crop)], dtype=np.float32)
        pts = pts * scale + offset
        pts = np.vstack([pts.T, np.ones(pts.shape[0], dtype=np.float32)])
        warped = np.dot(H, pts)
        warped = warped / warped[2, ...]
        warped = warped[:2].T
        return (warped - offset) / scale

    def call(self, x, difficulty=0.3, TPS=False, prob_deformation=0.5, test=False):
        if self.cnt % self.reload_step == 0:
            self.load_imgs()

        if not self.geometric:
            difficulty = 0.

        x = x / 255.
        b, c, h, w = x.shape
        shape = (h, w)

        ######## Geometric Transformations
        H = np.array([generate_random_homography(shape, difficulty) for _ in range(self.batch_size)], dtype=np.float32)

        output = tf.image.warp_perspective(x, H, output_size=shape, padding_mode='constant')

        low_h = int(h * self.sides_crop)
        low_w = int(w * self.sides_crop)
        high_h = int(h * (1. - self.sides_crop))
        high_w = int(w * (1. - self.sides_crop))
        output = output[..., low_h:high_h, low_w:high_w]
        x = x[..., low_h:high_h, low_w:high_w]

        if TPS:
            s, w, A = zip(*[generate_random_tps(shape) for _ in range(self.batch_size)])
            s = tf.convert_to_tensor(s, dtype=tf.float32)
            w = tf.convert_to_tensor(w, dtype=tf.float32)
            A = tf.convert_to_tensor(A, dtype=tf.float32)

            c = tf.convert_to_tensor((np.random.rand(self.batch_size) < prob_deformation) * 2.0 - 1.0, dtype=tf.float32)
            s = s * c.reshape(-1, 1, 1)

            def tps_grid(x):
                g = self.norm_pts_grid(x.reshape(-1, 2))
                warped = self.warp_points(g, s)
                return warped.reshape(-1, h, w, 2)

            def tps_apply(x):
                x = tf.image.resize(x, (h, w))
                x = tf.image.warp_affine(x, tf.reshape(w, (self.batch_size, h, w, 2)), (h, w), mode='constant', padding_value=0)
                return x

            output = tps_apply(output)
            x = tps_apply(x)

        ######## Photometric Transformations
        if self.aug_list:
            x = self.aug_list(x)
            output = self.aug_list(output)

        x = tf.image.resize(x, self.out_resolution)
        output = tf.image.resize(output, self.out_resolution)

        return x, output
