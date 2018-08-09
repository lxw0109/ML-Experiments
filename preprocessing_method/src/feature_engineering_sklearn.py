#!/usr/bin/env python3
# coding: utf-8
# File: preprocessing_sklearn.py
# Author: lxw
# Date: 5/28/18 8:09 AM
"""
使用sklearn进行特征工程的示例很不错
References:
1. [使用sklearn做单机特征工程](http://www.cnblogs.com/jasonfreak/p/5448385.html)
"""

import numpy as np

from numpy import vstack, array, nan

# packages for preprocessing
from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# packages for feature extraction
from minepy import MINE
from scipy.stats import pearsonr

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# packages for dimensionality reduction
from sklearn.decomposition import PCA
# from sklearn.lda import LDA    # ModuleNotFoundError: No module named 'sklearn.lda'


class FeatureEngineering:
    def __init__(self):
        # 导入IRIS数据集
        self.iris = load_iris()

    def preprocessing(self):
        # 1. 无量纲化
        # 常见的无量纲化方法有标准化和区间缩放法
        # 1.1 标准化，返回值为标准化后的数据.
        std_features = StandardScaler().fit_transform(self.iris.data)    # NOTE: 逐**列**进行标准化

        # 1.2 区间缩放，返回值为缩放到[0, 1]区间的数据.
        std_features = MinMaxScaler().fit_transform(self.iris.data)    # NOTE: 逐**列**进行标准化

        """
        标准化 vs 归一化的区别:
        标准化是依照特征矩阵的**列**处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下
        归一化是依照特征矩阵的**行**处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，
        也就是都转化为"单位向量", 例如使用l2范数进行归一化
        """
        # 1.3 归一化，返回值为归一化后的数据
        normal_features = Normalizer().fit_transform(self.iris.data)    # NOTE: 逐**行**进行标准化

        # 2. 对定量特征二值化
        # 二值化，阈值设置为3，返回值为二值化后的数据
        bin_features = Binarizer(threshold=3).fit_transform(self.iris.data)

        # 3. 对定性特征哑编码
        # 目标值进行哑编码(实际上不需要进行哑编码，这里只是为了举一个哑编码的例子).
        # 使用preproccessing库的OneHotEncoder类对数据进行哑编码
        # encoded_features = OneHotEncoder().fit_transform(self.iris.target)    # NO
        encoded_features = OneHotEncoder().fit_transform(self.iris.target.reshape(-1, 1))

        # 4. 对数据集新增一个样本，4个特征均赋值为NaN，表示数据缺失
        # 缺失值计算，返回值为计算缺失值后的数据
        # Imputer()的参数missing_value为缺失值的表示形式，默认为NaN, 参数strategy为缺失值填充方式，默认为mean(均值)
        stacked_data = vstack((array([nan, nan, nan, nan]), self.iris.data))
        imputed_data = Imputer().fit_transform(stacked_data)

        print()

    # 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
    def _mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)

    def feature_extraction(self):
        # 1. Filter
        # 1.1 方差选择法
        # 方差选择法，返回值为特征选择后的数据
        # Features with a training-set variance lower than `threshold` will be removed.
        extracted_features_data = VarianceThreshold(threshold=3).fit_transform(self.iris.data)  # shape: (150, 1)

        # 1.2 相关系数法
        # 选择K个最好的特征，返回选择特征后的数据
        # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组(评分, P值)的数组，
        # 数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
        # 参数k为选择的特征个数
        skb = SelectKBest(lambda X, Y: list(array(list(map(lambda x: pearsonr(x, Y), X.T))).T), k=2)
        extracted_features_data = skb.fit_transform(self.iris.data, self.iris.target)

        # 1.3 卡方检验
        # 选择K个最好的特征，返回选择特征后的数据
        extracted_features_data = SelectKBest(chi2, k=2).fit_transform(self.iris.data, self.iris.target)

        # 1.4 互信息法
        # 为了处理定量数据，最大信息系数法被提出，使用feature_selection库的SelectKBest类结合最大信息系数法来选择特征
        """
        # NOT WORKING
        extracted_features_data = SelectKBest(lambda X, Y: list(array(list(map(lambda x: self._mic(x, Y), X.T))).T),
                                              k=2).fit_transform(self.iris.data, self.iris.target)
        print(extracted_features_data)
        """

        # 2 Wrapper
        # 2.1 递归特征消除法, 返回特征选择后的数据
        # 参数estimator为基模型, 参数n_features_to_select为选择的特征个数
        extracted_features_data = RFE(estimator=LogisticRegression(), n_features_to_select=2).\
            fit_transform(self.iris.data, self.iris.target)

        # 3 Embedded
        # 3.1 基于惩罚项的特征选择法
        # 略
        # 3.2 GBDT作为基模型的特征选择
        extracted_features_data = SelectFromModel(GradientBoostingClassifier()).\
            fit_transform(self.iris.data, self.iris.target)
        # print(extracted_features_data)

    def dimensionality_reduction(self):
        # 1. 主成分分析法，返回降维后的数据
        # 参数n_components为主成分数目
        reduced_data = PCA(n_components=2).fit_transform(self.iris.data)
        # 使用PCA降维后的数据与原来的数据是完全不同的(没有哪一个列是与原来的列对应的)

        # 2. 线性判别分析法(LDA)，返回降维后的数据
        # 参数n_components为降维后的维数
        reduced_data = LDA(n_components=2).fit_transform(self.iris.data, self.iris.target)

        print(reduced_data)


class FeatureEngineering1:
    """
    Reference:
    [几种常用的特征选择方法](https://blog.csdn.net/kebu12345678/article/details/78437118)
    """
    def __init__(self):
        pass

    def univariate_feature_selection(self):
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import ShuffleSplit
        from sklearn.datasets import load_boston
        from sklearn.ensemble import RandomForestRegressor

        # Load boston housing dataset as an example
        boston = load_boston()
        X = boston["data"]
        Y = boston["target"]
        names = boston["feature_names"]
        print(f"X.shape: {X.shape}\nY.shape: {Y.shape}")

        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        scores = []
        for i in range(X.shape[1]):
            score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2", cv=ShuffleSplit(len(X), 3, 0.3))
            scores.append((round(np.mean(score), 3), names[i]))
        print(sorted(scores, reverse=True))


if __name__ == "__main__":
    """
    fe = FeatureEngineering()
    # fe.preprocessing()
    fe.feature_extraction()
    # fe.dimensionality_reduction()
    """

    # 1.
