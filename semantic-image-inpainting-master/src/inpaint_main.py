# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf

from inpaint_solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '1', 'gpu index, default: 1')
tf.flags.DEFINE_string('dataset', 'celebA', 'dataset name for choice [celebA|svhn|vub], default: celebA')

tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate to update latent vector z, default: 0.01')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum term of the NAG optimizer for latent vector, default: 0.9')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')
tf.flags.DEFINE_integer('y_dim', 0, 'dimension of y label, default: 0')
tf.flags.DEFINE_float('lamb', 3, 'hyper-parameter for prior loss, default: 3')  # lambda is 0.003 in the paper
tf.flags.DEFINE_bool('is_blend', True, 'blend predicted image to original image, default: true')
tf.flags.DEFINE_string('mask_type', 'center', 'mask type choice in [center|random|half|pattern|eye], default: center')
tf.flags.DEFINE_integer('img_size', 64, 'image height or width, default: 64')
tf.flags.DEFINE_bool('glasses', False, 'generates images with glasses when True, default: False')

tf.flags.DEFINE_integer('iters', 1500, 'number of iterations to optimize latent vector, default: 1500')
tf.flags.DEFINE_integer('num_try', 20, 'number of random samples, default: 20')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('sample_batch', 2, 'number of sampling images, default: 2')
tf.flags.DEFINE_string('load_model', None,
                       'saved DCGAN model that you wish to test, (e.g. 20180704-1736), default: None')
tf.flags.DEFINE_string('load_label', 'list_attr_celeba.txt', 'folder of label text files, default: list_attr_celeba.txt')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    solver.test()


if __name__ == '__main__':
    tf.app.run()
