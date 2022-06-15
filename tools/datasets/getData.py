import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib as mpl


def getSplitFinacialData(path):
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    train_data = pd.DataFrame(pd.read_csv(path))
    train_data.drop(["extra_return_next_month"], axis=1, inplace=True)
    # test_data = pd.DataFrame(pd.read_csv("data/test_data.csv"))
    # test_data.drop(["extra_return_next_month"], axis=1, inplace=True)
    X_columns = list(train_data.columns)
    X_columns.remove("label")
    X_columns.remove("code")
    X_columns.remove("name")
    X_columns.remove("time")
    # 样本内训练集
    train_data_X = train_data[X_columns]
    train_data_Y = train_data[["label"]]

    # 样本外测试集
    # test_data_X = test_data[X_columns]
    # test_data_Y = test_data[["label"]]

    # 样本内训练集切分

    # print("数据如下：\n", X_train, X_test, Y_train, Y_test)
    return train_data_X, train_data_Y
    # return X_train, X_test, Y_train, Y_test


class Dataset:
    def __init__(self):
        return
