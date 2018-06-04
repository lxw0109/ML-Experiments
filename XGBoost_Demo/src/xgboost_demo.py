#!/usr/bin/env python3
# coding: utf-8
# File: xgboost_demo.py
# Author: lxw
# Date: 5/29/18 9:05 AM


import sys
sys.path.append("/home/lxw/Software/xgboost/python-package")

import numpy as np
import xgboost as xgb


def xgboost_demo():
    train_data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
    label = np.random.randint(2, size=5)  # binary target
    label = label.reshape(-1, 1)

    # 当需要给样本设置权重和处理缺失值时，可以用如下方式
    w = np.random.rand(5, 1)  # 5 entities, each contains 1 features. shape: (5, 1)
    # dtrain = xgb.DMatrix(train_data, label=label)  # OK
    dtrain = xgb.DMatrix(train_data, label=label, missing=-999.0, weight=w)
    # 将 DMatrix 格式的数据保存成XGBoost的二进制格式，在下次加载时可以提高加载速度
    dtrain.save_binary("../data/train.buffer")

    eval_data = np.random.rand(7, 10)  # 7 entities, each contains 10 features
    deval = xgb.DMatrix(eval_data, missing=-999.0)

    # 参数设置
    param = {"bst:max_depth": 2, "bst:eta": 1, "silent": 1, "objective": "binary:logistic" }
    param["nthread"] = 4
    plst = list(param.items())
    plst += [("eval_metric", "ams@0")]
    # plst += [("eval_metric", "auc")]  # NOTE: Multiple evals can be handled in this way

    # 定义验证数据集，验证算法的性能
    watchlist = [(dtrain, "train"), (deval, "val")]

    num_round = 10
    bst = xgb.train(plst, dtrain, num_round, watchlist)
    bst.save_model("../data/1.model")
    # Dump Model and Feature Map: dump the model to txt and review the meaning of model
    bst.dump_model("../data/model_dump1.raw.txt")
    # bst.dump_model("../data/model_dump2.raw.txt", "../data/feature_map.txt")

    bst = xgb.Booster({"nthread": 4})  # init model
    bst.load_model("../data/1.model")

    test_data = np.random.rand(7, 10)  # 7 entities, each contains 10 features
    dtest = xgb.DMatrix(eval_data, missing=-999.0)
    ypred = bst.predict(dtest)
    # ypred = bst.predict(dtest, ntree_limit=bst.best_iteration)
    print("ypred:", ypred)  # ypred: [0.5 0.5 0.5 0.5 0.5 0.5 0.5]


if __name__ == "__main__":
    xgboost_demo()
