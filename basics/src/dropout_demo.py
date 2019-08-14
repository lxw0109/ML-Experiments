#!/usr/bin/python
# -*- coding: utf-8 -*-
# file: dropout_demo.py
# author: liu.xiaowei
# date: 2019/2/1
"""
References:
[使用tensorflow来验证dropout的实际效果](https://blog.csdn.net/hwtkao150/article/details/79291049)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

N_SAMPLE = 20
N_HIDDEN = 300
LR = 0.01

x = np.linspace(-1, 1, N_SAMPLE)[:, np.newaxis]
y = x + 0.3 * np.random.randn(N_SAMPLE)[:, np.newaxis]

test_x = x.copy()
test_y = test_x + 0.3 * np.random.randn(N_SAMPLE)[:, np.newaxis]

plt.scatter(x,y,c='magenta',s=50,alpha=0.5,label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)

# overfitting net
o1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
o2 = tf.layers.dense(o1, N_HIDDEN, tf.nn.relu)
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(tf_y, o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

# dropout net
d1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)
d2 = tf.layers.dense(d1, N_HIDDEN, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

for t in range(500):
    sess.run([o_train, d_train], feed_dict={tf_x: x, tf_y: y, tf_is_training: True})
    if t % 10 == 0:
        plt.cla()
        [o_loss_, d_loss_, o_out_, d_out_] = sess.run([o_loss, d_loss, o_out, d_out],
                                                      feed_dict={tf_x: test_x, tf_y: test_y, tf_is_training: False})
        # print "o_loss_: {}, d_loss_: {}, o_out_: {}, d_out_: {}".format(o_loss_, d_loss_, o_out_, d_out_)
        print "o_loss_: {}, d_loss_: {}".format(o_loss_, d_loss_)
        plt.scatter(x, y, c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x, o_out_, 'r-', lw=3, label='overfitting')
        plt.plot(test_x, d_out_, 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss = %.4f' % o_loss_, fontdict={'size': 10, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 10, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

plt.ioff()
