import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import math
import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2 as cv

# noinspection PyPep8Naming
import tensorflow_utils as tf_utils
import utils as utils


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


class DCGAN(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.batch_size = self.flags.batch_size

        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [1024, 512, 256, 128]  # 4, 8, 16, 32
        self.dis_c = [64, 128, 256, 512]  # 32, 16, 8, 4

        self._build_net()
        self._tensorboard()
        print("Initialized DCGAN SUCCESS!")

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='output')
        self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim], name='latent_vector')

        self.g_samples = self.generator(self.z)
        d_real, d_logit_real = self.discriminator(self.Y)
        d_fake, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

        # discriminator loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
        self.d_loss = d_loss_real + d_loss_fake

        # generator loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
        self.g_loss_without_mean = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake))

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        # Optimizer
        dis_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1) \
            .minimize(self.d_loss, var_list=d_vars)
        dis_ops = [dis_op] + self._dis_train_ops
        self.dis_optim = tf.group(*dis_ops)

        gen_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1) \
            .minimize(self.g_loss, var_list=g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

    def _tensorboard(self):
        # tf.summary.scalar('loss/d_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, data, y=None, name='g_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            data_flatten = flatten(data)

            if not self.flags.y_dim:
                # 4 x 4
                h0_linear = tf_utils.linear(data_flatten, 4 * 4 * self.gen_c[0], name='h0_linear')
                h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 4, self.gen_c[0]])
                h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
                h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')

                # 8 x 8
                h1_deconv = tf_utils.deconv2d(h0_relu, self.gen_c[1], name='h1_deconv2d')
                h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
                h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')

                # 16 x 16
                h2_deconv = tf_utils.deconv2d(h1_relu, self.gen_c[2], name='h2_deconv2d')
                h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
                h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')

                # 32 x 32
                h3_deconv = tf_utils.deconv2d(h2_relu, self.gen_c[3], name='h3_deconv2d')
                h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
                h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')

                # 64 x 64
                output = tf_utils.deconv2d(h3_relu, self.image_size[2], name='h4_deconv2d')
                return tf.nn.tanh(output)

            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.flags.y_dim])
                z = concat([data_flatten, y], 1)

                # 4 x 4
                h0_linear = tf_utils.linear(data_flatten, 4 * 4 * self.gen_c[0], name='h0_linear')
                h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 4, self.gen_c[0]])
                h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
                h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')
                h0 = conv_cond_concat(h0_relu, yb)

                # 8 x 8
                h1_deconv = tf_utils.deconv2d(h0, self.gen_c[1], name='h1_deconv2d')
                h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
                h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')
                h1 = conv_cond_concat(h1_relu, yb)

                # 16 x 16
                h2_deconv = tf_utils.deconv2d(h1, self.gen_c[2], name='h2_deconv2d')
                h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
                h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')
                h2 = conv_cond_concat(h2_relu, yb)

                # 32 x 32
                h3_deconv = tf_utils.deconv2d(h2, self.gen_c[3], name='h3_deconv2d')
                h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
                h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')
                h3 = conv_cond_concat(h3_relu, yb)

                # 64 x 64
                output = tf_utils.deconv2d(h3, self.image_size[2], name='h4_deconv2d')
                return tf.nn.tanh(output)

    def discriminator(self, data, y=None, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            if not self.flags.y_dim:
                # 64 -> 32
                h0_conv = tf_utils.conv2d(data, self.dis_c[0], name='h0_conv2d')
                h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')

                # 32 -> 16
                h1_conv = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
                h1_batchnorm = tf_utils.batch_norm(h1_conv, name='h1_batchnorm', _ops=self._dis_train_ops)
                h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

                # 16 -> 8
                h2_conv = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
                h2_batchnorm = tf_utils.batch_norm(h2_conv, name='h2_batchnorm', _ops=self._dis_train_ops)
                h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

                # 8 -> 4
                h3_conv = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
                h3_batchnorm = tf_utils.batch_norm(h3_conv, name='h3_batchnorm', _ops=self._dis_train_ops)
                h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

                h3_flatten = flatten(h3_lrelu)
                h4_linear = tf_utils.linear(h3_flatten, 1, name='h4_linear')

                return tf.nn.sigmoid(h4_linear), h4_linear

            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.flags.y_dim])
                x = conv_cond_concat(data, yb)

                h0_conv = tf_utils.conv2d(x, self.dis_c[0] + self.flags.y_dim, name='h0_conv2d')
                h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')
                h0 = conv_cond_concat(h0_lrelu, yb)

                # 32 -> 16
                h1_conv = tf_utils.conv2d(h0, self.dis_c[1] + self.flags.y_dim, name='h1_conv2d')
                h1_batchnorm = tf_utils.batch_norm(h1_conv, name='h1_batchnorm', _ops=self._dis_train_ops)
                h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')
                h1 = conv_cond_concat(h1_lrelu, yb)

                # 16 -> 8
                h2_conv = tf_utils.conv2d(h1, self.dis_c[2], name='h2_conv2d')
                h2_batchnorm = tf_utils.batch_norm(h2_conv, name='h2_batchnorm', _ops=self._dis_train_ops)
                h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')
                h2 = conv_cond_concat(h2_lrelu, yb)

                # 8 -> 4
                h3_conv = tf_utils.conv2d(h2, self.dis_c[3], name='h3_conv2d')
                h3_batchnorm = tf_utils.batch_norm(h3_conv, name='h3_batchnorm', _ops=self._dis_train_ops)
                h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

                h3_flatten = flatten(h3_lrelu)
                h3 = concat([h3_flatten, y], 1)

                h4_linear = tf_utils.linear(h3, 1, name='h4_linear')

                return tf.nn.sigmoid(h4_linear), h4_linear

    def train_step(self, imgs):
        feed = {self.z: self.sample_z(num=self.batch_size), self.Y: imgs}

        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed)
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=feed)

        return [d_loss, g_loss], summary

    def sample_imgs(self):
        g_feed = {self.z: self.sample_z(num=self.flags.sample_batch)}
        y_fakes = self.sess.run(self.g_samples, feed_dict=g_feed)

        return [y_fakes]

    def sample_z(self, num=64):
        return np.random.uniform(-1., 1., size=[num, self.flags.z_dim])

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.batch_size),
                                                  ('d_loss', loss[0]), ('g_loss', loss[1]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def plots(self, imgs_, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake = np.reshape(imgs_[0], (self.flags.sample_batch, 64, 64, 3))

        imgs = []
        for img in imgs_fake:
            imgs.append(img)

        # parameters for plot size
        scale, margin = 0.04, 0.01
        n_cols, n_rows = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        output = (imgs[0]).reshape(self.image_size[0], self.image_size[1], self.image_size[2])
        for row_index in range(n_rows - 1):
            output = cv.vconcat([output, (imgs[(row_index + 1) * n_cols]).reshape(self.image_size[0],
                                                                                  self.image_size[1],
                                                                                  self.image_size[2])])

        for col_index in range(n_cols - 1):
            out = (imgs[col_index + 1]).reshape(self.image_size[0], self.image_size[1], self.image_size[2])
            for row_index in range(n_rows - 1):
                out = cv.vconcat([out, (imgs[(row_index + 1) * n_cols + col_index + 1]).reshape(self.image_size[0],
                                                                                                self.image_size[1],
                                                                                                self.image_size[2])])
            output = cv.hconcat([output, out])

        output = np.uint8(output*255.0)
        print(np.min(output), np.max(output))

        cv.imwrite(save_file + '/sample_{}.png'.format(str(iter_time)), output)
        # fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        # gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        # gs.update(wspace=margin, hspace=margin)
        #
        # imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]
        #
        # # save more bigger image
        # for col_index in range(n_cols):
        #     for row_index in range(n_rows):
        #         ax = plt.subplot(gs[row_index * n_cols + col_index])
        #         plt.axis('off')
        #         ax.set_xticklabels([])
        #         ax.set_yticklabels([])
        #         ax.set_aspect('equal')
        #         """
        #         if self.image_size[2] == 3:
        #             plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
        #                 self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
        #         elif self.image_size[2] == 1:
        #             plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
        #                 self.image_size[0], self.image_size[1]), cmap='Greys_r')
        #         else:
        #             raise NotImplementedError
        #             """
        #
        # plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        # plt.close(fig)
