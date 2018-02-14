#!/usr/bin/env python3
# coding: utf-8
# File: 1_demo.py
# Author: lxw
# Date: 2/12/18 9:54 AM

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
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)    # ConnectionResetError: [Errno 104] Connection reset by peer

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)    # TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵

# 在一个Session里面启动模型，并且初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 开始训练模型，让模型循环训练1000次
for i in range(1000):
    # 该循环的每个步骤都会随机抓取训练数据中的100个批处理数据点，然后用这些数据点作为参数替换之前的占位符来运行train_step
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
