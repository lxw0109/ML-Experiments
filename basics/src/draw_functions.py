#!/usr/bin/python
# -*- coding: utf-8 -*-
# file: draw_functions.py
# author: liu.xiaowei
# date: 2019/7/7

import math
import matplotlib.pyplot as plt
import numpy as np


def entropy(X):
    Y = []
    for x in X:
        y = -x * math.log(x, 2) - (1 - x) * math.log(1 - x, 2)
        Y.append(y)
    return Y


def main():
    x = 0.1
    X = [x]
    while 1:
        x += 0.1
        if x > 1:
            break
        X.append(x)
    print X
    Y = entropy(X)
    print Y
    plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":
    main()