# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf

from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size for one feed forward, default: 256')
tf.flags.DEFINE_string('dataset', 'celebA', 'dataset name for choice [celebA|svhn|vub], default: vub')

tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: False')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 500, 'save frequency for model, default: 100')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 50')
tf.flags.DEFINE_integer('sample_batch', 64, 'number of sampling images for check generator quality, default: 64')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')

tf.flags.DEFINE_integer('y_dim', 0, 'dimension of y label, default: 0')
tf.flags.DEFINE_string('load_label', 'list_attr_celeba.txt', 'folder of label text files, default: list_attr_celeba.txt')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:  
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
