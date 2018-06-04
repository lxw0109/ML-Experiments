#!/usr/bin/env python3
# coding: utf-8
# File: normalization.py
# Author: lxw
# Date: 4/10/18 9:29 AM

import math
import matplotlib.pyplot as plt
import numpy as np
import random


# 1. max-min(Min-Max Normalization)
def max_min(data):
    """
    标准化数据，结果值映射到[0, 1]之间
    :param data: list of int/float.
    :return: list of int/float.
    """
    min_v = min(data)
    max_v = max(data)
    return [(num - min_v) / (max_v - min_v) for num in data]


# 1'. max-min(Min-Max Normalization)
def max_min_1(data):
    """
    标准化数据，结果值映射到[-1, 1]之间
    :param data: list of int/float.
    :return: list of int/float.
    """
    min_v = min(data)
    max_v = max(data)
    mean_v = np.mean(data)
    # NOTE: 并不是严格的[-1, 1], 只是近似的[-1, 1]
    return [(num - mean_v) / (max_v - min_v) for num in data]


# 1". max-min(Min-Max Normalization)
def max_min_2(data):
    """
    标准化数据，结果值映射到[-1, 1]之间
    另一种简单的解决方法是[0, 1]公式 * 2 - 1得到的范围就是[-1, 1]了
    :param data: list of int/float.
    :return: list of int/float.
    """
    result = max_min(data)
    return [item * 2 - 1 for item in result]


# 2. z-score Normalization
def z_score(data):
    """
    :param data: list
    :return: list
    """
    # 均值
    mean_miu = np.mean(data)
    # 标准差
    # z_score_numpy()与下面一行的结果是不太一样的
    std_sigma = math.sqrt(sum([(num - np.mean(data)) * (num - np.mean(data)) for num in data]) / (len(data) - 1))
    # z_score_numpy()与下面一行的结果完全相同
    #std_sigma = math.sqrt(sum([(num - np.mean(data)) * (num - np.mean(data)) for num in data]) / len(data))
    return [(num - mean_miu) / std_sigma for num in data]


# 2'. z-score Normalization with numpy
def z_score_numpy(data):
    """
    :param data: list
    :return: list
    """
    data_np = np.array(data).astype(float)
    mean_miu = np.mean(data_np)    # 均值
    std_sigma = np.std(data_np)    # 标准差    # NOTE: 注意这一行的代码不能与下一行的代码交换顺序
    data_np -= mean_miu
    data_np /= std_sigma
    return data_np.tolist()


if __name__ == "__main__":
    # 从结果来看, max-min normalization 和 z-score normalization 的结果都保持了原有数据的分布
    # random.seed(1)
    data = [random.randint(1, 100) for _ in range(10)]
    x = [i for i in range(10)]
    print("Original Data:", data, "\n")
    plt.subplot(321)
    plt.plot(x, data, label="original", color="red")
    plt.legend()

    # 1. max-min（Min-Max Normalization）
    normalization = max_min(data)
    print("max-min Normalization([0, 1]):", normalization, "\n")
    plt.subplot(322)
    plt.plot(x, normalization, label="[0, 1]", color="green")
    plt.legend()

    # 1'. max-min（Min-Max Normalization）
    normalization = max_min_1(data)
    print("max-min Normalization([-1, 1]):", normalization, "\n")
    plt.subplot(323)
    plt.plot(x, normalization, label="[-1, 1](mean)", color="blue")
    plt.legend()

    # 1". max-min(Min-Max Normalization)
    normalization = max_min_2(data)
    print("max-min Normalization([-1, 1]):", normalization, "\n")
    plt.subplot(324)
    plt.plot(x, normalization, label="[-1, 1](linear transformation)", color="black")
    plt.legend()

    # 2. z-score Normalization
    normalization = z_score(data)
    print("z-score Normalization:", normalization, "\n")
    plt.subplot(325)
    plt.plot(x, normalization, label="z-score", color="yellow")
    plt.legend()

    # 2'. z-score Normalization with numpy
    normalization = z_score_numpy(data)
    print("z-score Normalization(numpy):", normalization, "\n")
    plt.subplot(326)
    plt.plot(x, normalization, label="z-score(numpy)", color="purple")
    plt.legend()

    plt.show()

