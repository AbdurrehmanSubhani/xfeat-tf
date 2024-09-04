import argparse
import os
import time
import sys

import tensorflow as tf
import numpy as np

from modules.model import XFeatModel  # Assuming XFeatModel is implemented in TensorFlow
from modules.dataset.augmentation import AugmentationPipe
from modules.training.utils import make_batch
from modules.training.losses import dual_softmax_loss, coordinate_classification_loss, alike_distill_loss, keypoint_loss, check_accuracy
from modules.dataset.megadepth.megadepth import MegaDepthDataset, megadepth_warper


def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

class Trainer:
    """
    Class for training XFeat with default params as described in the paper.
    We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
    The major bottleneck is to keep loading huge megadepth h5 files from disk, 
    the network training itself is quite fast.
    """

    def __init__(self, megadepth_root_path, 
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name='xfeat_default',
                       batch_size=10, n_steps=160_000, lr=3e-4, gamma_steplr=0.5, 
                       training_res=(800, 608), device_num="0", dry_run=False,
                       save_ckpt_every=500):

        self.dev = tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')
        with self.dev:
            self.net = XFeatModel()

        # Setup optimizer
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=30000,
            decay_rate=gamma_steplr,
            staircase=True)

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir=synthetic_root_path,
                                        device=self.dev, load_dataset=True,
                                        batch_size=int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution=training_res, 
                                        warp_resolution=training_res,
                                        sides_crop=0.1,
                                        max_num_imgs=3_000,
                                        num_test_imgs=5,
                                        photometric=True,
                                        geometric=True,
                                        reload_step=4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################

        ##################### MEGADEPTH INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = tf.data.Dataset.from_tensor_slices(npz_paths).map(
                lambda path: MegaDepthDataset(root_dir=TRAINVAL_DATA_SOURCE, npz_path=path)).cache().shuffle(buffer_size=100).repeat()
            self.data_loader = data.batch(int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size))
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = tf.summary.create_file_writer(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name


    def train_step(self, p1, p2, positives_c):

        with tf.GradientTape() as tape:
            feats1, kpts1, hmap1 = self.net(p1, training=True)
            feats2, kpts2, hmap2 = self.net(p2, training=True)

            loss_items = []
            for b in range(len(positives_c)):
                # Get positive correspondencies
                pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                # Grab features at corresponding idxs
                m1 = tf.gather_nd(feats1[b], tf.cast(pts1, tf.int32))
                m2 = tf.gather_nd(feats2[b], tf.cast(pts2, tf.int32))

                # grab heatmaps at corresponding idxs
                h1 = tf.gather_nd(hmap1[b, 0], tf.cast(pts1, tf.int32))
                h2 = tf.gather_nd(hmap2[b, 0], tf.cast(pts2, tf.int32))
                coords1 = self.net.fine_matcher(tf.concat([m1, m2], axis=-1))

                # Compute losses
                loss_ds, conf = dual_softmax_loss(m1, m2)
                loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                acc_pos = (acc_pos1 + acc_pos2) / 2

                loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                loss_items.append(tf.expand_dims(loss_ds, axis=0))
                loss_items.append(tf.expand_dims(loss_coords, axis=0))
                loss_items.append(tf.expand_dims(loss_kp, axis=0))
                loss_items.append(tf.expand_dims(loss_kp_pos, axis=0))

                if b == 0:
                    acc_coarse_0 = check_accuracy(m1, m2)

            loss = tf.reduce_mean(tf.concat(loss_items, axis=-1))

        grads = tape.gradient(loss, self.net.trainable_variables)
        tf.clip_by_global_norm(grads, 1.0)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))

        return loss, acc_coarse_0, acc_coords, acc_pos, loss_ds, loss_coords, loss_kp, loss_kp_pos, len(m1)

    def train(self):

        with self.dev:
            for i in range(self.steps):
                difficulty = 0.10

                p1s, p2s, H1, H2 = None, None, None, None
                d = None

                if self.augmentor is not None:
                    p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
                
                if self.data_iter is not None:
                    d = next(self.data_iter)

                img1, img2, positives_c = megadepth_warper(d, difficulty)

                if self.model_name == 'xfeat_default':
                    p1, p2, positives_c = tf.concat([img1, p1s], axis=0), tf.concat([img2, p2s], axis=0), positives_c
                elif self.model_name == 'xfeat_megadepth':
                    p1, p2 = img1, img2
                else:
                    p1, p2 = p1s, p2s

                if self.dry_run:
                    if i > 0: 
                        break
                    print(f"Dry run with batch_size = {self.batch_size}, model: {self.model_name}")

                loss, acc_coarse, acc_coords, acc_pos, loss_ds, loss_coords, loss_kp, loss_kp_pos, batch_size_used = self.train_step(p1, p2, positives_c)

                if i % 100 == 0:
                    print(f"step {i}, loss: {loss:.4f}, coarse acc: {acc_coarse:.4f}, keypoint acc: {acc_pos:.4f}, kp_loss: {loss_kp:.4f}, distill_loss: {loss_kp_pos:.4f}, coord_loss: {loss_coords:.4f}, batch_size: {batch_size_used}")

                if i % self.save_ckpt_every == 0:
                    self.net.save_weights(f"{self.ckpt_save_path}/ckpt_{self.model_name}_{i}.h5")

                    with self.writer.as_default():
                        tf.summary.scalar("loss", loss, step=i)
                        tf.summary.scalar("acc_coarse", acc_coarse, step=i)
                        tf.summary.scalar("acc_coords", acc_coords, step=i)
                        tf.summary.scalar("acc_pos", acc_pos, step=i)
                        tf.summary.scalar("loss_ds", loss_ds, step=i)
                        tf.summary.scalar("loss_coords", loss_coords, step=i)
                        tf.summary.scalar("loss_kp", loss_kp, step=i)
                        tf.summary.scalar("loss_kp_pos", loss_kp_pos, step=i)
                        self.writer.flush()

trainer = Trainer(
    megadepth_root_path=args.megadepth_root_path,
    synthetic_root_path=args.synthetic_root_path,
    ckpt_save_path=args.ckpt_save_path,
    model_name=args.training_type,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    lr=args.lr,
    gamma_steplr=args.gamma_steplr,
    training_res=args.training_res,
    device_num=args.device_num,
    dry_run=args.dry_run,
    save_ckpt_every=args.save_ckpt_every
)

trainer.train()
