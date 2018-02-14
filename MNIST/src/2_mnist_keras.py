#!/usr/bin/env python3
# coding: utf-8
# File: 2_mnist_keras.py
# Author: lxw
# Date: 2/14/18 11:34 PM

"""
一个手写数字识别的CNN
Reference: 
[keras 手把手入门#1-MNIST手写数字识别 深度学习实战闪电入门](http://nooverfit.com/wp/keras-手把手入门1-手写数字识别-深度学习实战/)
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K