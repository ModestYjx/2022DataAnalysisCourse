#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from tools.datasets.getData import getSplitFinacialData
from time import time
import numpy as np
import pandas as pd
import roc

if __name__ == "__main__":
    # 读取Mnist数据集, 测试GBDT的分类模型
    # mnistSet = mnist.loadLecunMnistSet()
    # train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]
    train_data_path = "data/train_data.csv"
    test_data_path = "data/test_data.csv"
    decision_tree_predict_test_data_path = "data/decision_tree_predict_test_data.csv"
    train_data_X, train_data_Y = getSplitFinacialData(train_data_path)
    train_X, test_X, train_Y, test_Y = train_test_split(train_data_X, train_data_Y, test_size=0.2, random_state=0)


    m, n = np.shape(train_X)
    idx = list(range(m))
    np.random.shuffle(idx)

    # 使用PCA降维
    # num = 30000
    # pca = PCA(n_components=0.9, whiten=True, random_state=0)
    # for i in range(int(np.ceil(1.0 * m / num))):
    #     minEnd = min((i + 1) * num, m)
    #     sub_idx = idx[i * num:minEnd]
    #     train_pca_X = pca.fit_transform(train_X[sub_idx])
    #     print np.shape(train_pca_X)

    print("**********测试GradientBoostingClassifier类**********")
    t = time()
    param_grid1 = {"n_estimators": range(1000, 2001, 100)}
    param_grid2 = {'max_depth': range(30, 71, 10), 'min_samples_split': range(4, 9, 2)}
    param_grid3 = {'min_samples_split': range(4, 9, 2), 'min_samples_leaf': range(3, 12, 2)}
    param_grid4 = {'subsample': np.arange(0.6, 1.0, 0.05)}
    model = GridSearchCV(
        estimator=GradientBoostingClassifier(max_features=90, max_depth=40, min_samples_split=8, learning_rate=0.1,
                                             n_estimators=1800),
        param_grid=param_grid4, cv=3)
    # 拟合训练数据集
    model.fit(train_X, train_Y)
    print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))
    model = GradientBoostingClassifier(max_features=90, max_depth=40, min_samples_split=8, min_samples_leaf=3,
                                       n_estimators=1200, learning_rate=0.05, subsample=0.95)
    # 拟合训练数据集
    model.fit(train_X, train_Y)
    # 预测训练集
    train_Y_hat = model.predict(train_X[idx])
    print("训练集精确度: ", accuracy_score(train_Y[idx], train_Y_hat))
    # 预测测试集
    test_Y_hat = model.predict(test_X)
    print("测试集精确度: ", accuracy_score(test_Y, test_Y_hat))
    print("总耗时:", time() - t, "秒")
    # 绘制ROC曲线
    n_class = len(np.unique(train_Y))
    roc.drawROC(n_class, test_Y, test_Y_hat)

    # 读取CCPP数据集, 测试GBDT的回归模型
    data = pd.read_excel("data/CCPP/Folds5x2_pp.xlsx")
    # AT:温度, V:压力, AP:湿度, RH:压强, PE:输出电力
    # 样本特征X
    X = data[['AT', 'V', 'AP', 'RH']]
    # 数据归一化
    X = StandardScaler().fit_transform(X)
    # 样本输出Y
    Y = data[['PE']]
    # 划分训练集和测试集，将数据集的70%划入训练集，30%划入测试集
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)

    m, n = np.shape(train_X)
    idx = range(m)
    np.random.shuffle(idx)

    print("**********测试GradientBoostingRegressor类**********")
    t = time()
    # param_grid1 = {"n_estimators": range(700, 1501, 100)}
    # param_grid2 = {'max_depth': range(3, 10, 2), 'min_samples_split': range(4, 9, 2)}
    # param_grid3 = {'min_samples_split': (4, 9, 2), 'min_samples_leaf': range(2, 11, 2)}
    # param_grid4 = {'subsample': np.arange(0.6, 1.0, 0.05)}
    # model = GridSearchCV(
    #     estimator=GradientBoostingRegressor(learning_rate=0.1, n_estimators=1500, max_depth=5, min_samples_split=9,
    #                                         min_samples_leaf=2), param_grid=param_grid4, cv=5)
    # # 拟合训练数据集
    # model.fit(train_X, train_Y.values.ravel())
    # print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))
    model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1500, max_depth=5, min_samples_split=9,
                                      min_samples_leaf=2, subsample=0.95)
    # 拟合训练数据集
    model.fit(train_X, train_Y.values.ravel())
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print("测试集MSE:", mean_squared_error(test_Y, test_Y_pred))
    print("测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred)))
    print("总耗时:", time() - t, "秒")
