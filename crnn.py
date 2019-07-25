import os
import time
import numpy as np
import tensorflow as tf
import config
import  time
import shutil
from scipy.misc import imread, imresize, imsave
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util
from data_manager import DataManager
from utils import sparse_tuple_from, resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRNN(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_valid_ratio, restore):
        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')
        self.train_valid_ratio=train_valid_ratio
        self.__restore = restore

        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()

        # Building graph
        with self.__session.as_default():
            (
                self.__inputs,
                self.__targets,
                self.__seq_len,
                self.__logits,
                self.__decoded,
                self.__optimizer,
                self.__acc,
                self.__cost,
                self.__max_char_count,
                self.__init
            ) = self.crnn(max_image_width, batch_size)
            self.__init.run()

        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Loading last save if needed
            if self.__restore:

                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)
                    print('Restoring from epoch:{}'.format( self.step))

        # Creating data_manager
        self.__data_manager = DataManager(batch_size, model_path, examples_path, max_image_width, train_valid_ratio, self.__max_char_count)


    def save_PbModel(self):
        output_name=self.__decoded.op.name
        #output_name = self.__decoded.name.split(":")[0]
        input1_name=self.__inputs.name.split(":")[0]
        input2_name = self.__seq_len.name.split(":")[0]
        print("模型保存为pb格式，输入节点name：{}，{},输出节点name: {}".format(input1_name,input2_name,output_name))
        #constant_graph = graph_util.convert_variables_to_constants(self.__session, self.__session.graph_def, [output_name])
        constant_graph=graph_util.convert_variables_to_constants(self.__session,self.__session.graph_def,["SparseToDense"])
        with tf.gfile.GFile(self.__model_path+'Model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    def crnn(self, max_width, batch_size):
        def BidirectionnalRNN(inputs, seq_len):
            """
                Bidirectionnal LSTM Recurrent Neural Network part
            """
            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                # Forward
                lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_1 = rnn.BasicLSTMCell(256)
                #lstm_bw_cell_1 = rnn.BasicLSTMCell(256)
                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)

                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                # Forward
                lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)

                outputs = tf.concat(outputs, 2)


            return outputs

        def CNN(inputs):
            """
                Convolutionnal Neural Network part
            """

            # 64 / 3 x 3 / 1 / 1
            conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # 128 / 3 x 3 / 1 / 1
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # 256 / 3 x 3 / 1 / 1
            conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm1 = tf.layers.batch_normalization(conv3)

            # 256 / 3 x 3 / 1 / 1
            conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 1
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 3 x 3 / 1 / 1
            conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm2 = tf.layers.batch_normalization(conv5)

            # 512 / 3 x 3 / 1 / 1
            conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 2
            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 2 x 2 / 1 / 0
            conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)

            return conv7

        inputs = tf.placeholder(tf.float32, [batch_size, max_width, 32, 1])

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        cnn_output = CNN(inputs)

        reshaped_cnn_output = tf.reshape(cnn_output, [batch_size, -1, 512])

        max_char_count = reshaped_cnn_output.get_shape().as_list()[1]
        print("max_width：{}，max_char_count:{}".format(max_width,max_char_count))
        crnn_model = BidirectionnalRNN(reshaped_cnn_output, seq_len)

        logits = tf.reshape(crnn_model, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")

        logits = tf.matmul(logits, W) + b

        #logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES])
        logits = tf.reshape(logits, [batch_size, 24, config.NUM_CLASSES])

        print(config.NUM_CLASSES)
        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))
        print("logits的NAME :{}".format(logits.name))
        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        # Training step
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        #print("decoded的NAME :{}".format(decoded[0].name))
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        #print("dense_decoded的shape{}".format(dense_decoded.shape))
        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        init = tf.global_variables_initializer()

        return inputs, targets, seq_len, logits, dense_decoded, optimizer, acc, cost, max_char_count, init

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            for i in range(self.step, iteration_count + self.step):
                self.__data_manager.init_op_train_batches()
                iter_loss = 0
                for _, batch_dt, batch_x in self.__data_manager.train_batches:
                    op, decoded, loss_value = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size,
                            self.__targets: batch_dt } )
                    iter_loss += loss_value

                if i % 2 == 0 and self.train_valid_ratio<1:
                    self.valid()
                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=self.step
                )

                print('epoch:[{}]  loss: {}'.format(self.step, iter_loss))

                self.step += 1
        return None

    def test(self,b_savePb=False,b_test=True,b_saveNG=True):
        if b_saveNG:
            # 创建空txt文档保存错误样本的路径
            NG_path =  self.__data_manager.examples_path + '_NG'
            if  not os.path.exists(NG_path):
                os.mkdir(NG_path)

        with self.__session.as_default():
            if b_savePb:
                self.save_PbModel()
            if b_test:
                count=0
                r=0
                for batch_y, _, batch_x in self.__data_manager.test_batches:
                    t1 = time.time()
                    decoded = self.__session.run(
                        self.__decoded,
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size
                        }
                    )
                    for i, y in enumerate(batch_y):
                        count += 1
                        label = self.__data_manager.get_label(batch_y[i])
                        predict = ground_truth_to_word(decoded[i])

                        if label == predict:
                            r += 1
                        else:
                            print(label)
                            print(predict)
                            if b_saveNG:
                                wrong_example=os.path.join(self.__data_manager.examples_path, batch_y[i])
                                object_path=os.path.join(NG_path, batch_y[i])
                                shutil.copyfile(wrong_example,object_path)

                acc=r/count
                print("Testing ,count:{},acc:{}".format(count,acc))
                if b_saveNG:
                    print("错误样本保存到文件：{}".format(NG_path))

        return None
    def valid(self):
        count = 0
        r = 0
        for batch_y, _, batch_x in self.__data_manager.test_batches:
            t1 = time.time()
            decoded = self.__session.run(
                self.__decoded,
                feed_dict={
                    self.__inputs: batch_x,
                    self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size
                }
            )
            for i, y in enumerate(batch_y):
                count += 1
                label=self.__data_manager.get_label(batch_y[i])
                predict=ground_truth_to_word(decoded[i])
                if  label==predict :
                    r += 1
        acc = r / count
        print("validation, count:{}, acc:{}".format(count, acc))