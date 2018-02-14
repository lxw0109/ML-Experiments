#!/usr/bin/env python3
# coding: utf-8
# File: 2_mnist_keras.py
# Author: lxw
# Date: 2/14/18 11:34 PM

"""
一个手写数字识别的CNN
Trains a simple convnet on the MNIST dataset. Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning). 16 seconds per epoch on a GRID K520 GPU.
Reference: 
[keras 手把手入门#1-MNIST手写数字识别 深度学习实战闪电入门](http://nooverfit.com/wp/keras-手把手入门1-手写数字识别-深度学习实战/)
"""

import keras
import numpy as np
import time

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def run():
    # batch_size 太小会导致训练慢，过拟合等问题;太大会导致欠拟合。所以要适当选择
    batch_size = 128
    # 0-9手写数字, 共10个类别
    num_classes = 10
    epochs = 2
    # input image dimensions
    img_rows, img_cols = 28, 28

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()    # 会下载数据，比较慢
    """
    # 没有尝试成功，很慢
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets("../data/mnist.npz", one_hot=True)

    x_train, y_train = mnist_data.train.images, mnist_data.train.labels
    x_test, y_test = mnist_data.test.images, mnist_data.train.labels
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
    """
    # [【keras】解决example案例中MNIST数据集下载不了的问题](http://blog.csdn.net/houchaoqun_xmu/article/details/78492718)
    f = np.load("../data/input/mnist.npz")
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]
    f.close()

    # the data, shuffled and split between train and test sets
    # keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，
    # 其实就是格式差别而已
    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:    # tensorflow: "channels_last"
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 把数据变成float32更精确
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    model = Sequential()
    # 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
    # 卷积核的窗口选用3*3像素窗口
    model.add(Conv2D(32, activation="relu", input_shape=input_shape, nb_row=3, nb_col=3))
    # 64个通道的卷积层
    model.add(Conv2D(64, activation="relu", nb_row=3, nb_col=3))    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # 池化层是2*2像素的
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 对于池化层的输出，采用0.35概率的Dropout
    model.add(Dropout(0.35))
    # 展平所有像素，比如[28*28] -> [784]
    model.add(Flatten())
    # 对所有像素使用全连接层，输出为128，激活函数选用relu
    model.add(Dense(128, activation="relu"))
    # 对输入采用0.5概率的Dropout
    model.add(Dropout(0.5))
    # 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
    model.add(Dense(num_classes, activation="softmax"))
    # 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    model.save("../data/output/mnist_cnn_keras.h5")

    # 测试集上评估准确率
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])    # Test loss: 0.04647096006094944
    print("Test accuracy:", score[1])    # Test accuracy: 0.9846


if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = time.time()
    print("Time Cost:", end_time - start_time)    # Time Cost: 554.4480702877045
