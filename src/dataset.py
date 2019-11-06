# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
import numpy as np
import scipy.io
import scipy.misc
import cv2 as cv
import utils as utils

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '../../Data')


class CelebA(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (64, 64, 3)
        self.input_height = self.input_width = 108
        self.num_trains, self.num_vals = 0, 0

        self.celeba_dataset_path = os.path.join('../Data', self.dataset_name, 'img_align_celeba')
        self.celeba_train_path = os.path.join('../Data', self.dataset_name, 'train')
        self.celeba_val_path = os.path.join('../Data', self.dataset_name, 'val')

        #
        # if not os.path.isfile(os.path.join('../../Data', self.dataset_name, 'train')):
        #     os.mkdir(self.celeba_train_path)
        #     os.mkdir(self.celeba_val_path)
        #     for i in range(1, 141820):
        #         shutil.move(os.path.join(self.celeba_dataset_path, '{:0>6}.jpg'.format(i)), self.celeba_train_path)
        #     for i in range(141820, 202600):
        #         shutil.move(os.path.join(self.celeba_dataset_path, '{:0>6}.jpg'.format(i)), self.celeba_val_path)
        self._load_celeba()

        np.random.seed(seed=int(time.time()))  # set random seed according to the current time

    def _load_celeba(self):
        print('Load {} dataset...'.format(self.dataset_name))

        self.train_data = utils.all_files_under(self.celeba_train_path)
        self.num_trains = len(self.train_data)

        self.val_data = utils.all_files_under(self.celeba_val_path)
        self.num_vals = len(self.val_data)

        celeba_attr_f = open("../Data/celebA/" + self.flags.load_label, "r").readlines()
        if self.flags.y_dim:
            self.y_label = [[0 if i == '-1' else 1 for i in x.split()[1:]] for x in celeba_attr_f[2:]]
            train_temp = [[x] for x in self.train_data]
            train_label = [x for x in self.y_label[:self.num_trains]]
            self.train_data_with_label = np.concatenate([train_temp, train_label], axis=1)
            val_temp = [[x] for x in self.val_data]
            val_label = [x for x in self.y_label[self.num_trains:]]
            self.val_data_with_label = np.concatenate([val_temp, val_label], axis=1)
        else:
            self.y_label = None

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        if self.flags.y_dim:
            index = [x for x in range(len(self.train_data_with_label))]
            batch_datas = np.random.choice(index, batch_size, replace=True)
            batch_labels = [self.train_data_with_label[x][1:] for x in batch_datas]
            batch_paths = [self.train_data_with_label[x][0] for x in batch_datas]
        else:
            batch_paths = np.random.choice(self.train_data, batch_size, replace=True)
            batch_labels = None
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs), batch_labels

    def val_next_batch(self, batch_size):
        if self.flags.y_dim:
            index = [x for x in range(len(self.val_data_with_label))]
            batch_datas = np.random.choice(index, batch_size, replace=True)
            batch_labels = [self.val_data_with_label[x][1:] for x in batch_datas]
            batch_paths = [self.val_data_with_label[x][0] for x in batch_datas]
        else:
            batch_paths = np.random.choice(self.val_data, batch_size, replace=True)
            batch_labels = None
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs), batch_labels


class SVHN(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (64, 64, 3)
        self.num_trains, self.num_vals = 0, 0

        self.svhn_train_path = os.path.join('../Data', self.dataset_name, 'train_32x32.mat')
        self.svhn_val_path = os.path.join('../Data', self.dataset_name, 'test_32x32.mat')
        self._load_svhn()

        np.random.seed(seed=int(time.time()))  # set random seed according to the current time

    def _load_svhn(self):
        print('Load {} dataset...'.format(self.dataset_name))

        # Convert mat data [W, H, C, N] to [N, H, W, C] and normalized to [-1. 1.]
        self.train_data = np.transpose(scipy.io.loadmat(self.svhn_train_path)['X'], (3, 0, 1, 2)).astype(np.float32)
        self.val_data = np.transpose(scipy.io.loadmat(self.svhn_val_path)['X'], (3, 0, 1, 2)).astype(np.float32)
        self.num_trains, self.num_vals = len(self.train_data), len(self.val_data)

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_indexs = np.random.choice(range(self.train_data.shape[0]), batch_size, replace=False)
        batch_imgs = self.train_data[batch_indexs]

        # resize (32, 32, 3) to (64, 64, 3) and random flip
        batch_imgs_ = [utils.random_flip(
            utils.transform(scipy.misc.imresize(batch_imgs[idx], (self.image_size[0], self.image_size[1]))))
            for idx in range(batch_imgs.shape[0])]

        return np.asarray(batch_imgs_)

    def val_next_batch(self, batch_size):
        batch_indexs = np.random.choice(range(self.val_data.shape[0]), batch_size, replace=False)
        batch_imgs = self.val_data[batch_indexs]

        # resize (32, 32, 3) to (64, 64, 3)
        batch_imgs_ = [utils.transform(scipy.misc.imresize(batch_imgs[idx], (self.image_size[0], self.image_size[1])))
                       for idx in range(batch_imgs.shape[0])]

        return np.asarray(batch_imgs_)


class VUB(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (64, 64, 3)
        self.input_height = self.input_width = 108
        self.num_trains, self.num_vals = 0, 0
        self.vub_train_path = os.path.join(DATA_DIR, self.dataset_name,'train')
        self.vub_val_path = os.path.join(DATA_DIR, self.dataset_name,'val')
        self.vub_raw_data_path = os.path.join(DATA_DIR, self.dataset_name)
        self._edit_vub()
        self._load_vub()

        np.random.seed(seed=int(time.time()))  # set random seed according to the current time

    def _edit_vub(self):
        train_file_size = os.path.getsize(self.vub_train_path)
        val_file_size = os.path.getsize(self.vub_val_path)
        exists = train_file_size & val_file_size
        if exists:
            return
        else:
            files = utils.all_files_under(os.path.join(DATA_DIR, '{}/{}'.format(self.dataset_name, 'urban')))
            count = 0
            totfiles = len(files) * 16
            print(os.path.abspath(DATA_DIR))
            for file in files:
                image = cv.imread(file)
                height, width, channels = image.shape
                if height > width:
                    image = image[0:width, :]
                else:
                    image = image[:, 0:height]
                image = cv.resize(image, (256, 256))
                for i in range(4):
                    for j in range(4):
                        temp = image.copy()[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
                        # crop the image to [64, 64, 3] format
                        if count < totfiles / 5 * 4:
                            cv.imwrite(os.path.join(DATA_DIR, '{}/{}/{:04d}.bmp'.format(
                                self.dataset_name, 'train', count)), temp)
                        else:
                            cv.imwrite(os.path.join(DATA_DIR, '{}/{}/{:04d}.bmp'.format(
                                self.dataset_name,'val', count)), temp)
                        count += 1

    def _load_vub(self):
        print('Load {} dataset...'.format(self.dataset_name))
        self.train_data = utils.all_files_under(self.vub_train_path)
        self.num_trains = len(self.train_data)

        self.val_data = utils.all_files_under(self.vub_val_path)
        self.num_vals = len(self.val_data)
        print('dataset size = {}'.format(self.num_trains))
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.train_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs)

    def val_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.val_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs)


# noinspection PyPep8Naming
def Dataset(flags, dataset_name):
    if dataset_name == 'celebA':
        return CelebA(flags, dataset_name)
    elif dataset_name == 'svhn':
        return SVHN(flags, dataset_name)
    elif dataset_name == 'vub':
        return VUB(flags, dataset_name)
    else:
        raise NotImplementedError


