import re
import os
import numpy as np
import config
from random import shuffle
from utils import sparse_tuple_from, resize_image, label_to_array

from scipy.misc import imsave

class DataManager(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio, max_char_count,shuffle=True):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        print(train_test_ratio)
        self.shuffle=shuffle
        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path

        self.examples_path = examples_path
        self.max_char_count = max_char_count
        data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.train_data=data[0:self.test_offset]
        self.test_data=data[self.test_offset:-1]
        self.init_op_test_batches()  #测试 or 验证数据 只执行一次init_op_test_batches




    def get_label(self,f):
        return  f.split('.')[-2].split('_')[1]

    def __load_data(self):
        """
            Load all the images in the folder
        """

        print('Loading data from {}'.format(self.examples_path))

        examples = []

        count = 0
        skipped = 0
        for f in os.listdir(self.examples_path):
            if "(" in  f:
                os.remove(os.path.join(self.examples_path, f))
                continue
            if len(self.get_label(f)) > self.max_char_count:
                continue
            arr, initial_len = resize_image(
                os.path.join(self.examples_path, f),
                self.max_image_width
            )
            examples.append(
                (
                    arr,
                    f,
                    label_to_array(self.get_label(f))
                )
            )
            imsave('blah.png', arr)
            count += 1
        shuffle(examples)
        return examples, len(examples)

    def init_op_train_batches(self):
        if self.shuffle :
            shuffle(self.train_data)
        train_batches = []
        num_batch=int(np.floor(len(self.train_data)/self.batch_size))
        for index in range(num_batch):
            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.train_data[index*self.batch_size:(1+index)*self.batch_size])
            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )
            batch_dt = sparse_tuple_from(
                    np.array(raw_batch_la)

            )
            # batch_dt = sparse_tuple_from(
            #     np.reshape(
            #         np.array(raw_batch_la),
            #         (-1)
            #     )
            # )
            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (len(raw_batch_x), self.max_image_width, 32, 1)
            )
            train_batches.append((batch_y, batch_dt, batch_x))
        self.train_batches= train_batches

    def init_op_test_batches(self):
        test_batches = []
        num_batch=int(np.floor(len(self.test_data)/self.batch_size))
        for index in range(num_batch):
            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.test_data[index*self.batch_size:(1+index)*self.batch_size])
            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )
            batch_dt = sparse_tuple_from(

                    np.array(raw_batch_la)

            )
            # batch_dt = sparse_tuple_from(
            #     np.reshape(
            #         np.array(raw_batch_la),
            #         (-1)
            #     )
            # )
            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
            self.test_batches= test_batches
