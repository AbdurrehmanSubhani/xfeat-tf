import tensorflow as tf
import numpy as np

def dual_softmax_loss(X, Y, temp=0.2):
    if X.shape != Y.shape or len(X.shape) != 2 or len(Y.shape) != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = tf.matmul(X, tf.transpose(Y)) * temp
    conf_matrix12 = tf.nn.log_softmax(dist_mat, axis=1)
    conf_matrix21 = tf.nn.log_softmax(tf.transpose(dist_mat), axis=1)

    conf12 = tf.exp(conf_matrix12)
    conf21 = tf.exp(conf_matrix21)
    conf = tf.reduce_max(conf12, axis=-1) * tf.reduce_max(conf21, axis=-1)

    target = tf.range(tf.shape(X)[0])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=conf_matrix12) + \
           tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=conf_matrix21)

    return tf.reduce_mean(loss), conf

def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = tf.abs(input - target)
    loss = tf.where(diff < beta, 0.5 * tf.square(diff) / beta, diff - 0.5 * beta)
    return tf.reduce_mean(loss) if size_average else tf.reduce_sum(loss)

def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    '''
        Compute Fine features and spatial loss
    '''
    C, H, W = tf.shape(f1)[0], tf.shape(f1)[1], tf.shape(f1)[2]
    N = tf.shape(pts1)[0]

    #Sort random offsets
    a, b = -(ws // 2), (ws // 2)
    offset_gt = (a - b) * tf.random.uniform([N, 2], dtype=f1.dtype) + b
    pts2_random = pts2 + offset_gt

    patches1 = utils.crop_patches(tf.expand_dims(f1, 0), tf.cast(pts1 + 0.5, tf.int32), size=ws)
    patches1 = tf.reshape(patches1, [C, N, ws * ws])
    patches1 = tf.transpose(patches1, perm=[1, 2, 0])

    patches2 = utils.crop_patches(tf.expand_dims(f2, 0), tf.cast(pts2_random + 0.5, tf.int32), size=ws)
    patches2 = tf.reshape(patches2, [C, N, ws * ws])
    patches2 = tf.transpose(patches2, perm=[1, 2, 0])

    patches1, patches2 = fine_module(patches1, patches2)

    features = tf.reshape(patches1, [N, ws, ws, C])[:, ws // 2, ws // 2, :]
    features = tf.reshape(features, [N, 1, 1, C])
    patches2 = tf.reshape(patches2, [N, ws, ws, C])

    heatmap_match = tf.reduce_sum(features * patches2, axis=-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    offset_gt = -offset_gt

    error = tf.reduce_mean(tf.reduce_sum(tf.square(offset_coords - offset_gt), axis=-1))

    return error

def alike_distill_loss(kpts, img):
    C, H, W = kpts.shape
    kpts = tf.transpose(kpts, perm=[1,2,0])
    img = tf.transpose(img, perm=[1,2,0])
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = img.numpy()

    alike_kpts = extract_alike_kpts(img)
    alike_kpts = tf.convert_to_tensor(alike_kpts, dtype=kpts.dtype)
    labels = tf.ones((H, W), dtype=tf.int64) * 64
    offsets = tf.cast(((alike_kpts / 8) - tf.floor(alike_kpts / 8)) * 8, tf.int64)
    offsets = offsets[:, 0] + 8 * offsets[:, 1]
    labels = tf.tensor_scatter_nd_update(labels, tf.cast(tf.floor(alike_kpts[:, 1] / 8), tf.int64), offsets)

    kpts = tf.reshape(kpts, [-1, C])
    labels = tf.reshape(labels, [-1])

    mask = labels < 64
    idxs_pos = tf.where(mask)[:, 0]
    idxs_neg = tf.where(~mask)[:, 0]
    perm = tf.random.shuffle(idxs_neg)[:len(idxs_pos) // 32]
    idxs_neg = tf.gather(idxs_neg, perm)
    idxs = tf.concat([idxs_pos, idxs_neg], axis=0)

    kpts = tf.gather(kpts, idxs)
    labels = tf.gather(labels, idxs)

    predicted = tf.argmax(kpts, axis=-1)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels, predicted), tf.float32))

    kpts = tf.nn.log_softmax(kpts)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=kpts)

    return tf.reduce_mean(loss), acc

def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp=1.0):
    C, H, W = kpts1.shape
    kpts1 = tf.transpose(kpts1, perm=[1,2,0]) * softmax_temp
    kpts2 = tf.transpose(kpts2, perm=[1,2,0]) * softmax_temp

    x, y = tf.meshgrid(tf.range(W), tf.range(H), indexing='xy')
    xy = tf.stack([x, y], axis=-1)
    xy *= 8

    hashmap = tf.ones((H * 8, W * 8, 2), dtype=tf.int64) * -1
    hashmap = tf.tensor_scatter_nd_update(hashmap, tf.cast(pts1, tf.int64), tf.cast(pts2, tf.int64))

    kpts1_offsets = tf.argmax(kpts1, axis=-1)
    kpts1_offsets_x = kpts1_offsets % 8
    kpts1_offsets_y = kpts1_offsets // 8
    kpts1_offsets_xy = tf.stack([kpts1_offsets_x, kpts1_offsets_y], axis=-1)

    kpts1_coords = xy + kpts1_offsets_xy
    kpts1_coords = tf.reshape(kpts1_coords, [-1, 2])
    gt_12 = tf.gather_nd(hashmap, kpts1_coords)
    mask_valid = tf.reduce_all(gt_12 >= 0, axis=-1)
    gt_12 = tf.boolean_mask(gt_12, mask_valid)

    labels2 = tf.cast(((gt_12 / 8) - tf.floor(gt_12 / 8)) * 8, tf.int64)
    labels2 = labels2[:, 0] + 8 * labels2[:, 1]

    kpts2_selected = tf.gather_nd(kpts2, tf.cast(gt_12 / 8, tf.int64))

    kpts1_selected = tf.nn.log_softmax(tf.boolean_mask(tf.reshape(kpts1, [-1, C]), mask_valid), axis=-1)
    kpts2_selected = tf.nn.log_softmax(kpts2_selected, axis=-1)

    labels1 = tf.argmax(kpts1_selected, axis=-1)
    predicted2 = tf.argmax(kpts2_selected, axis=-1)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels2, predicted2), tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels1, logits=kpts1_selected)) + \
           tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels2, logits=kpts2_selected))

    return loss, acc

def coordinate_classification_loss(coords1, pts1, pts2, conf):
    coords1_detached = pts1 * 8
    offsets1_detached = tf.cast(((coords1_detached / 8) - tf.floor(coords1_detached / 8)) * 8, tf.int64)
    labels1 = offsets1_detached[:, 0] + 8 * offsets1_detached[:, 1]

    coords1_log = tf.nn.log_softmax(coords1, axis=-1)
    predicted = tf.argmax(coords1, axis=-1)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels1, predicted), tf.float32))
    acc = tf.boolean_mask(acc, conf > 0.1)
    acc = tf.reduce_mean(acc)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels1, logits=coords1_log)
    conf = conf / tf.reduce_sum(conf)
    loss = tf.reduce_sum(loss * conf)

    return loss * 2., acc

def keypoint_loss(heatmap, target):
    return tf.reduce_mean(tf.abs(heatmap - target)) * 3.0

def hard_triplet_loss(X, Y, margin=0.5):
    if X.shape != Y.shape or len(X.shape) != 2 or len(Y.shape) != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = tf.reduce_sum(tf.square(X), axis=1, keepdims=True) + \
               tf.reduce_sum(tf.square(Y), axis=1) - 2.0 * tf.matmul(X, tf.transpose(Y))

    dist_pos = tf.linalg.diag_part(dist_mat)
    dist_neg = tf.reduce_min(dist_mat + tf.eye(tf.shape(X)[0]) * tf.reduce_max(dist_mat), axis=-1)

    return tf.reduce_mean(tf.maximum(dist_pos - dist_neg + margin, 0.0))
