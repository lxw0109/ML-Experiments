#!/usr/bin/env python3
# coding: utf-8
# File: linear_regresion_demo.py
# Author: lxw
# Date: 4/8/18 8:00 PM
"""
Reference:
1. [用scikit-learn和pandas学习线性回归](https://www.cnblogs.com/pinard/p/6016029.html)
2. [Plotting Cross-Validated Predictions](http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html#sphx-glr-auto-examples-plot-cv-predict-py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
# from sklearn.cross_validation import train_test_split    # deprecated
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict


def main():
    ccpp_df = pd.read_excel("../data/CCPP/Folds5x2_pp.xlsx")
    print(ccpp_df.head())
    print(ccpp_df.shape)    # (9568, 5)
    # AT, V, AP, RH 4列作为样本特征
    x = ccpp_df[["AT", "V", "AP", "RH"]]
    # PE作为样本输出
    y = ccpp_df[["PE"]]

    # 把X和Y的样本划分成两部分:训练集, 测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)    # 训练集:测试集 = 3:1
    # print(type(x_train), type(y_train), type(x_test), type(y_test))
    # <class 'pandas.core.frame.DataFrame'> .. <class 'pandas.core.frame.DataFrame'>
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 运行scikit-learn的线性模型
    # scikit-learn的线性回归算法使用的是最小二乘法来实现的
    line_regr = linear_model.LinearRegression()
    line_regr.fit(x_train, y_train)

    # 拟合完毕后，我们看看我们需要的模型系数结果
    print(line_regr.intercept_)
    print(line_regr.coef_)
    """
    Output: 
    [460.05727267]
    [[-1.96865472 -0.2392946   0.0568509  -0.15861467]]
    这样我们就得到了在步骤1里面需要求得的5个值。也就是说PE和其他4个变量的关系如下：
    PE = 460.05727267 - 1.96865472 ∗ AT - 0.2392946 ∗ V + 0.0568509 ∗ AP - 0.15861467 ∗ RH
    """

    # 模型评价
    # 对于线性回归来说，一般用均方差(Mean Squared Error, MSE)或均方根差(Root Mean Squared Error, RMSE)在测试集上的
    # 表现来评价模型的好坏
    y_pred = line_regr.predict(x_test)
    print("MSE: {}".format(metrics.mean_squared_error(y_test, y_pred)))    # 20.837191547220346
    print("RMSE: {}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))    # 4.564777272465804

    # 画真实的点(这个例子的输入是多维的, 无法直接画)
    plt.scatter(x_test["V"][:100], y_test[:100], color="blue", edgecolors=(0, 0, 0))
    # 画拟合的直线
    plt.plot(x_test[:100], y_pred[:100], color="red", linewidth=4)    # 预测曲线
    plt.show()
    """
    """

    # 交叉验证: 可以通过交叉验证来持续优化模型
    predicted = cross_val_predict(line_regr, x, y, cv=10)    # 采用10折交叉验证
    print("MSE: {}".format(metrics.mean_squared_error(y, predicted)))    # 20.79367250985753
    print("RMSE: {}".format(np.sqrt(metrics.mean_squared_error(y, predicted))))    # 4.560007950635342
    # 发现两次结果不同:本次是对所有折的样本做测试集对应的预测值的MSE，而上一次的评价结果仅仅对25%的测试集做了MSE

    # 画图观察结果
    """
    # 这里画出真实值和预测值的变化关系，离中间的直线越近的点代表预测损失越低
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", color="black", lw=4)    # y=x
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()
    """


def draw_3D():
    from mpl_toolkits.mplot3d import Axes3D
    xx, yy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 100, 10))
    zz = 1.0 * xx + 3.5 * yy + np.random.randint(0, 100, (10, 10))

    # 构建成特征、值的形式
    X, Z = np.column_stack((xx.flatten(), yy.flatten())), zz.flatten()

    # 建立线性回归模型
    regr = linear_model.LinearRegression()

    # 拟合
    regr.fit(X, Z)

    # 不难得到平面的系数、截距
    a, b = regr.coef_, regr.intercept_

    # 给出待预测的一个特征
    x = np.array([[5.8, 78.3]])

    # 方式1：根据线性方程计算待预测的特征x对应的值z（注意：np.sum）
    print(np.sum(a * x) + b)

    # 方式2：根据predict方法预测的值z
    print(regr.predict(x))

    # 画图
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # 1.画出真实的点
    ax.scatter(xx, yy, zz)

    # 2.画出拟合的平面
    ax.plot_wireframe(xx, yy, regr.predict(X).reshape(10, 10))
    ax.plot_surface(xx, yy, regr.predict(X).reshape(10, 10), alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
    # draw_3D()