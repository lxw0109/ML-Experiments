#!/usr/bin/env python3
# coding: utf-8
# File: 1_demo.py
# Author: lxw
# Date: 2/12/18 9:54 AM
# 最好不用这个，感觉不太对

'''
1. TensorFlow安装：
采用清华大学开源软件镜像站安装:
```bash
$ pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
  https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/cpu/tensorflow-1.2.1-cp36-cp36m-linux_x86_64.whl
```
References:
[TensorFlow学习笔记1——安装](https://www.cnblogs.com/CongYuUestcer/p/7345634.html)
[TensorFlow下载与安装](http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/os_setup.html)
2. MNIST
[MNIST机器学习入门](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html?plg_nld=1&plg_uin=1&plg_auth=1&plg_nld=1&plg_usr=1&plg_vkey=1&plg_dev=1)

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)    # ConnectionResetError: [Errno 104] Connection reset by peer
# print(type(mnist))  # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型
y_ = tf.placeholder(tf.float32, [None, 10])  # y_ = tf.placeholder("float", [None, 10])  # OK
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 采用梯度下降法(Gradient Descent)以0.01的学习速率最小化交叉熵

# 在一个Session里面启动模型，并且初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# print (sess.run(b))  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 开始训练模型，让模型循环训练1000次
EPOCH_NUM = 1000
BATCH_SIZE = 64  # 1: 0.83, 2: 0.87, 4: 0.87, 8: 0.898, 16: 893, 32: 0.909, 64: 0.91, 128: 0.91, 256: 0.098
max_i = 0
for i in range(EPOCH_NUM):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    # print type(batch_xs), type(batch_ys)  # <type 'numpy.ndarray'> <type 'numpy.ndarray'>
    # print batch_xs.shape, batch_ys.shape  # (BATCH_SIZE, 784) (BATCH_SIZE, 10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    max_i = i

print "max_i: {}".format(max_i)  # 999

# 评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images}))

# 计算模型在测试集上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
