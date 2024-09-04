import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_batch(augmentor, difficulty=0.3, train=True):
    Hs = []
    img_list = augmentor.train if train else augmentor.test
    batch_images = []

    for b in range(augmentor.batch_size):
        rdidx = np.random.randint(len(img_list))
        img = tf.convert_to_tensor(img_list[rdidx], dtype=tf.float32)
        img = tf.transpose(img, perm=[2, 0, 1])
        img = tf.expand_dims(img, axis=0)
        batch_images.append(img)

    batch_images = tf.concat(batch_images, axis=0)

    p1, H1 = augmentor(batch_images, difficulty)
    p2, H2 = augmentor(batch_images, difficulty, TPS=True, prob_deformation=0.7)

    return p1, p2, H1, H2



def plot_corrs(p1, p2, src_pts, tgt_pts):
    p1 = p1.numpy()
    p2 = p2.numpy()
    src_pts = src_pts.numpy()
    tgt_pts = tgt_pts.numpy()
    rnd_idx = np.random.randint(len(src_pts), size=200)
    src_pts = src_pts[rnd_idx, ...]
    tgt_pts = tgt_pts[rnd_idx, ...]

    fig, ax = plt.subplots(1, 2, figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts), 3))

    # Src image
    img = p1
    for i, p in enumerate(src_pts):
        ax[0].scatter(p[0], p[1], color=colors[i])
    ax[0].imshow(np.transpose(img, (1, 2, 0))[..., ::-1])

    # Target img
    img2 = p2
    for i, p in enumerate(tgt_pts):
        ax[1].scatter(p[0], p[1], color=colors[i])
    ax[1].imshow(np.transpose(img2, (1, 2, 0))[..., ::-1])
    plt.show()

def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop=None):
    negatives, positives = [], []
    rh, rw = tf.shape(p1)[-2], tf.shape(p1)[-1]
    ratio = tf.constant([rw / w, rh / h], dtype=tf.float32)

    H, mask1 = H
    H2, src, W, A, mask2 = H2

    x, y = tf.meshgrid(tf.range(w), tf.range(h), indexing='xy')
    mesh = tf.stack([x, y], axis=-1)
    target_pts = tf.reshape(mesh, (-1, 2)) * ratio

    for batch_idx in range(tf.shape(p1)[0]):
        T = (H[batch_idx], H2[batch_idx], 
             tf.expand_dims(src[batch_idx], 0), tf.expand_dims(W[batch_idx], 0), tf.expand_dims(A[batch_idx], 0))

        src_pts = augmentor.get_correspondences(target_pts, T)
        tgt_pts = target_pts

        mask_valid = tf.logical_and.reduce([
            src_pts[:, 0] >= 0, src_pts[:, 1] >= 0,
            src_pts[:, 0] < rw, src_pts[:, 1] < rh
        ])

        negatives.append(tf.boolean_mask(tgt_pts, tf.logical_not(mask_valid)))
        tgt_pts = tf.boolean_mask(tgt_pts, mask_valid)
        src_pts = tf.boolean_mask(src_pts, mask_valid)

        mask_valid = tf.logical_and(
            tf.gather_nd(mask1[batch_idx], tf.cast(src_pts, tf.int32)),
            tf.gather_nd(mask2[batch_idx], tf.cast(tgt_pts, tf.int32))
        )
        tgt_pts = tf.boolean_mask(tgt_pts, mask_valid)
        src_pts = tf.boolean_mask(src_pts, mask_valid)

        if crop is not None:
            rnd_idx = tf.random.shuffle(tf.range(tf.shape(src_pts)[0]))[:crop]
            src_pts = tf.gather(src_pts, rnd_idx)
            tgt_pts = tf.gather(tgt_pts, rnd_idx)

        src_pts /= ratio
        tgt_pts /= ratio

        padto = 10 if crop is not None else 2
        mask_valid1 = tf.logical_and.reduce([
            src_pts[:, 0] >= padto, src_pts[:, 1] >= padto,
            src_pts[:, 0] < (w - padto), src_pts[:, 1] < (h - padto)
        ])
        mask_valid2 = tf.logical_and.reduce([
            tgt_pts[:, 0] >= padto, tgt_pts[:, 1] >= padto,
            tgt_pts[:, 0] < (w - padto), tgt_pts[:, 1] < (h - padto)
        ])
        mask_valid = tf.logical_and(mask_valid1, mask_valid2)
        tgt_pts = tf.boolean_mask(tgt_pts, mask_valid)
        src_pts = tf.boolean_mask(src_pts, mask_valid)

        lut_mat = tf.ones([h, w, 4], dtype=tf.float32) * -1

        try:
            indices = tf.stack([tf.cast(src_pts[:, 1], tf.int32), tf.cast(src_pts[:, 0], tf.int32)], axis=1)
            lut_mat = tf.tensor_scatter_nd_update(lut_mat, indices, tf.concat([src_pts, tgt_pts], axis=1))
            mask_valid = tf.reduce_all(lut_mat >= 0, axis=-1)
            points = tf.boolean_mask(lut_mat, mask_valid)
            positives.append(points)
        except:
            print('Error occurred in processing.')

    return negatives, positives

def crop_patches(tensor, coords, size=7):
    B, C, H, W = tf.shape(tensor)
    x, y = coords[:, 0], coords[:, 1]
    y = tf.reshape(y, [-1, 1, 1])
    x = tf.reshape(x, [-1, 1, 1])
    halfsize = size // 2
    x_offset, y_offset = tf.meshgrid(tf.range(-halfsize, halfsize+1), tf.range(-halfsize, halfsize+1), indexing='xy')

    y_indices = y + tf.reshape(y_offset, [1, size, size]) + halfsize
    x_indices = x + tf.reshape(x_offset, [1, size, size]) + halfsize

    tensor_padded = tf.pad(tensor, [[0, 0], [0, 0], [halfsize, halfsize], [halfsize, halfsize]], mode='CONSTANT')

    patches = tf.gather_nd(tensor_padded, tf.stack([y_indices, x_indices], axis=-1), batch_dims=2)
    return patches

def subpix_softmax2d(heatmaps, temp=0.25):
    N, H, W = tf.shape(heatmaps)
    heatmaps = tf.nn.softmax(temp * tf.reshape(heatmaps, [-1, H*W]), axis=-1)
    heatmaps = tf.reshape(heatmaps, [-1, H, W])

    x, y = tf.meshgrid(tf.range(W), tf.range(H), indexing='xy')
    x = x - (W // 2)
    y = y - (H // 2)

    coords_x = tf.reduce_sum(x[None, ...] * heatmaps, axis=[1, 2])
    coords_y = tf.reduce_sum(y[None, ...] * heatmaps, axis=[1, 2])
    coords = tf.stack([coords_x, coords_y], axis=-1)

    return coords

def check_accuracy(X, Y, pts1=None, pts2=None, plot=False):
    dist_mat = tf.matmul(X, Y, transpose_b=True)
    nn = tf.argmax(dist_mat, axis=1)
    correct = tf.equal(nn, tf.range(tf.shape(X)[0]))

    if pts1 is not None and plot:
        import matplotlib.pyplot as plt
        canvas = tf.zeros((60, 80))
        pts1 = tf.boolean_mask(pts1, tf.logical_not(correct))
        indices = tf.cast(tf.stack([pts1[:, 1], pts1[:, 0]], axis=1), tf.int32)
        canvas = tf.tensor_scatter_nd_update(canvas, indices, tf.ones_like(pts1[:, 0]))
        plt.imshow(canvas.numpy()), plt.show()

    acc = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.shape(X)[0]
    return acc

def get_nb_trainable_params(model):
    trainable_params = np.sum([np.prod(v.shape.as_list()) for v in model.trainable_variables])
    print(f'Number of trainable parameters: {trainable_params / 1e6:.2f}M')
    return trainable_params