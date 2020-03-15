import os
import sys
import time

from ssd.ssd_vgg16 import create_ssd_vgg16
from ssd.ssd_mobilenetv1 import create_ssd_mobilenetv1
from ssd.ssd_mobilenetv1_lite import create_ssd_mobilenetv1_lite
from ssd.ssd_mobilenetv2_lite import create_ssd_mobilenetv2_lite
from ssd.pre_ssd_mobilenetv1_lite import create_pre_ssd_mobilenetv1_lite
from ssd.test_pre_ssd_mobilenetv1_lite import create_test_pre_ssd_mobilenetv1_lite
from ssd.ssd import init_ssd

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from voc_data import create_batch_generator
from anchor import generate_default_boxes
from losses import create_losses

from settings import *

ARCH = 'test_pre_ssd300-mobilenetv1'
CHECKPOINT_DIR = 'checkpoint/test_pre_mobilenetv1_lite'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
default_boxes = generate_default_boxes(INFO[ARCH])

with tf.device('/device:GPU:1'):
    
    batch_generator, val_generator, info = create_batch_generator(
                DATA_DIR, DATA_YEAR, default_boxes,
                SIZE, BATCH_SIZE, NUM_BATCHES,
                mode='train', augmentation=['flip'])  # the patching algorithm is currently causing bottleneck sometimes


    dummy = tf.random.normal((1, 300, 300, 3))
    ssd = create_test_pre_ssd_mobilenetv1_lite()

    pretrained_type = 'new'
    net = init_ssd(ssd, pretrained_type)

    criterion = create_losses(NEG_RATIO, NUM_CLASSES)
    steps_per_epoch = info['length'] // BATCH_SIZE

    lr_fn = PiecewiseConstantDecay(
            boundaries=[int(steps_per_epoch * NUM_EPOCHS * 2 / 3),
                        int(steps_per_epoch * NUM_EPOCHS * 5 / 6)],
            values=[INITIAL_LR, INITIAL_LR * 0.1, INITIAL_LR * 0.01])

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=INITIAL_LR,
        momentum=MOMENTUM)

    train_log_dir = 'logs/train'
    val_log_dir = 'logs/val'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @tf.function
    def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
        with tf.GradientTape() as tape:
            confs, locs = ssd(imgs)

            conf_loss, loc_loss = criterion(
                confs, locs, gt_confs, gt_locs)

            loss = conf_loss + loc_loss

            #l2 regularization
            l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
            l2_loss = WEIGHT_DECAY * tf.math.reduce_sum(l2_loss)
            loss += l2_loss

        gradients = tape.gradient(loss, ssd.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

        return loss, conf_loss, loc_loss, None

    for epoch in range(NUM_EPOCHS):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()

        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):

            loss, conf_loss, loc_loss, l2_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)

            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)

            if (i + 1) % 50 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0

        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = ssd(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs)

            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

        if (epoch + 1) % 100 == 0:
            ssd.save_weights(
                os.path.join(CHECKPOINT_DIR, 'ssd_epoch_{}.h5'.format(epoch + 1)))