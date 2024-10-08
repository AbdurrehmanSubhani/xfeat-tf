import os.path as osp
import numpy as np
import tensorflow as tf
import glob
from modules.dataset.megadepth.utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net
import numpy.random as rnd

class MegaDepthDataset(tf.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.3,
                 max_overlap_score=1.0,
                 load_depth=True,
                 img_resize=(800, 608),
                 df=32,
                 img_padding=False,
                 depth_padding=True,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]
        self.load_depth = load_depth
        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None #and img_padding and depth_padding

        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = kwargs.get('coarse_scale', 0.125)

        for idx in range(len(self.scene_info['image_paths'])):
            self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])

        for idx in range(len(self.scene_info['depth_paths'])):
            self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)

        if self.load_depth:
            # read depth. shape: (h, w)
            if self.mode in ['train', 'val']:
                depth0 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
                depth1 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
            else:
                depth0 = depth1 = tf.constant([])

            # read intrinsics of original size
            K_0 = tf.convert_to_tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=tf.float32).reshape((3, 3))
            K_1 = tf.convert_to_tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=tf.float32).reshape((3, 3))

            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = tf.convert_to_tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=tf.float32)[:4, :4]  # (4, 4)
            T_1to0 = tf.linalg.inv(T_0to1)

            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }

            # for LoFTR training
            if mask0 is not None:  # img_padding is True
                if self.coarse_scale:
                    ts_mask_0 = tf.image.resize(mask0[None, ..., tf.newaxis].astype(tf.float32),
                                                size=(int(mask0.shape[0] * self.coarse_scale),
                                                      int(mask0.shape[1] * self.coarse_scale)),
                                                method='nearest')[0, ..., 0].numpy()
                    ts_mask_1 = tf.image.resize(mask1[None, ..., tf.newaxis].astype(tf.float32),
                                                size=(int(mask1.shape[0] * self.coarse_scale),
                                                      int(mask1.shape[1] * self.coarse_scale)),
                                                method='nearest')[0, ..., 0].numpy()
                    data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        else:
            # read intrinsics of original size
            K_0 = tf.convert_to_tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=tf.float32).reshape((3, 3))
            K_1 = tf.convert_to_tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=tf.float32).reshape((3, 3))

            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = tf.convert_to_tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=tf.float32)[:4, :4]  # (4, 4)
            T_1to0 = tf.linalg.inv(T_0to1)

            data = {
                'image0': image0,  # (1, h, w)
                'image1': image1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }

        return data
