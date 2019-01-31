#!/usr/bin/python
# -*- coding: utf-8 -*-
# file: log_loss_diy.py
# author: liu.xiaowei
# date: 2019/1/31
# 自定义logloss函数 vs. scikit-learn库中`sklearn.metrics.log_loss`函数
"""
References:
[对数损失函数(Logarithmic Loss Function)的原理和 Python 实现](https://www.cnblogs.com/klchang/p/9217551.html)
"""

import numpy as np

from sklearn.metrics import log_loss

def log_loss_diy_OK_BUT_BAD(y_true, y_pred):
    """
    :param y_true: list of true labels.
    :param y_pred: list of predicted values.
    :return:
    """
    N = len(y_true)
    if N < 1:
        return -1  # ERROR

    sums = 0
    for y, p in zip(y_true, y_pred):
        print "y: {}, (1 - y): {}, p: {}".format(y, 1-y, p)
        sums += (y * np.log(p) + (1 - y) * np.log(1 - p))

    return -sums / N


def log_loss_diy(y_true, y_pred, epsilon=1e-15):
    """
    :param y_true: list of true labels.
    :param y_pred: list of predicted values.
    :param epsilon: 加入参数epsilon是为了避免因预测概率输出为0或1,而导致的计算错误的情况
    :return:
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    assert len(y_true) and len(y_true) == len(y_pred)

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)) / len(y_true)


def main():
    y_true = [0, 0, 1, 1]
    y_pred = [0.01, 0.2, 0.7, 0.99]

    print("log_loss_diy(): {}".format(log_loss_diy(y_true, y_pred)))
    print("sklearn.metrics.log_loss(): {} ".format(log_loss(y_true, y_pred)))


if __name__ == "__main__":
    main()
