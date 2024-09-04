import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from modules.training.utils import plot_corrs

@tf.function
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (tf.Tensor): [N, L, 2] - <x, y>,
        depth0 (tf.Tensor): [N, H, W],
        depth1 (tf.Tensor): [N, H, W],
        T_0to1 (tf.Tensor): [N, 3, 4],
        K0 (tf.Tensor): [N, 3, 3],
        K1 (tf.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (tf.Tensor): [N, L]
        warped_keypoints0 (tf.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = tf.round(kpts0)
    kpts0_long = tf.clip_by_value(tf.cast(kpts0_long, tf.int32), 0, 2000 - 1)

    depth0 = tf.tensor_scatter_nd_update(depth0, [[i, 0, j] for i in range(depth0.shape[0]) for j in range(depth0.shape[2])], tf.zeros(depth0.shape[0] * depth0.shape[2]))
    depth1 = tf.tensor_scatter_nd_update(depth1, [[i, 0, j] for i in range(depth1.shape[0]) for j in range(depth1.shape[2])], tf.zeros(depth1.shape[0] * depth1.shape[2]))

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = tf.stack([tf.gather_nd(depth0[i], kpts0_long[i]) for i in range(kpts0.shape[0])], axis=0)  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Unproject
    kpts0_h = tf.concat([kpts0, tf.ones_like(kpts0[:, :, [0]])], axis=-1) * tf.expand_dims(kpts0_depth, axis=-1)  # (N, L, 3)
    kpts0_cam = tf.linalg.inv(K0) @ tf.transpose(kpts0_h, perm=[0, 2, 1])  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + tf.expand_dims(T_0to1[:, :3, 3], axis=-1)  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam)  # (N, 3, L)
    w_kpts0_h = tf.transpose(w_kpts0_h, perm=[0, 2, 1])  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth

    valid_mask = nonzero_mask #* consistent_mask * covisible_mask 

    return valid_mask, w_kpts0

@tf.function
def spvs_coarse(data, scale=8):
    """
    Supervise corresp with dense depth & camera poses
    """

    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = [x // scale for x in [H0, W0, H1, W1]]

    # 2. warp grids
    grid_pt1_c = tf.meshgrid(tf.range(h1), tf.range(w1), indexing='ij')
    grid_pt1_c = tf.stack(grid_pt1_c, axis=-1)  # [h1, w1, 2]
    grid_pt1_c = tf.reshape(grid_pt1_c, [1, h1 * w1, 2])
    grid_pt1_c = tf.tile(grid_pt1_c, [N, 1, 1])  # [N, hw, 2]
    grid_pt1_i = scale1 * grid_pt1_c

    # warp kpts bi-directionally and check reproj error
    nonzero_m1, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    nonzero_m2, w_pt1_og = warp_kpts(w_pt1_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])

    dist = tf.norm(grid_pt1_i - w_pt1_og, axis=-1)
    mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

    batched_corrs = [tf.concat([w_pt1_i[i, mask_mutual[i]] / data['scale0'][i],
                                grid_pt1_i[i, mask_mutual[i]] / data['scale1'][i]], axis=-1) for i in range(len(mask_mutual))]

    # Remove repeated correspondences - this is important for network convergence
    corrs = []
    for pts in batched_corrs:
        lut_mat12 = tf.fill([h1, w1, 4], -1.0)
        lut_mat21 = tf.identity(lut_mat12)
        src_pts = pts[:, :2] / scale
        tgt_pts = pts[:, 2:]
        try:
            lut_mat12 = tf.tensor_scatter_nd_update(lut_mat12, tf.cast(tf.stack([src_pts[:, 1], src_pts[:, 0]], axis=-1), tf.int32), tf.concat([src_pts, tgt_pts], axis=1))
            mask_valid12 = tf.reduce_all(lut_mat12 >= 0, axis=-1)
            points = tf.boolean_mask(lut_mat12, mask_valid12)

            # Target-src check
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21 = tf.tensor_scatter_nd_update(lut_mat21, tf.cast(tf.stack([tgt_pts[:, 1], tgt_pts[:, 0]], axis=-1), tf.int32), tf.concat([src_pts, tgt_pts], axis=1))
            mask_valid21 = tf.reduce_all(lut_mat21 >= 0, axis=-1)
            points = tf.boolean_mask(lut_mat21, mask_valid21)

            corrs.append(points)
        except:
            print('Error occurred during correspondence computation.')

    return corrs

@tf.function
def get_correspondences(pts2, data, idx):
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape

    pts2 = tf.expand_dims(pts2, axis=0)

    scale0 = data['scale0'][idx, None] if 'scale0' in data else 1
    scale1 = data['scale1'][idx, None] if 'scale1' in data else 1

    pts2 = scale1 * pts2 * 8

    # warp kpts bi-directionally and check reproj error
    nonzero_m1, pts1 = warp_kpts(pts2, data['depth1'][idx:idx+1], data['depth0'][idx:idx+1], data['T_1to0'][idx:idx+1],
                                 data['K1'][idx:idx+1], data['K0'][idx:idx+1])

    corrs = tf.concat([pts1[0, :] / data['scale0'][idx],
                       pts2[0, :] / data['scale1'][idx]], axis=-1)

    return corrs
