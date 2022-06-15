from time import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tools.datasets.getData import getFinacialData

if __name__ == "__main__":
    train_X, test_X, train_Y, test_Y = getFinacialData()
    print("train_X:\n", train_X)
    m, n = np.shape(train_X)
    print("m, n:\n", m, ",", n)

    p, q = np.shape(train_Y)
    print("p, q:\n", p, ",", q)

    idx = list(range(m))
    np.random.shuffle(idx)
    # print("train_X:\n", train_X.size)
    print("**********测试DecisionTreeRegressor类**********")
    t = time()
    # param_grid1 = {'max_depth': range(10, 31, 5), 'min_samples_split': range(3, 12, 2)}
    # param_grid2 = {'min_samples_split': range(3, 12, 2), 'min_samples_leaf': range(2, 6, 1)}
    # model = GridSearchCV(DecisionTreeRegressor(splitter='random', max_depth=20, min_samples_split=5),
    #                      param_grid=param_grid2, cv=5)
    # # 拟合训练数据集
    # model.fit(train_X, train_Y.values.ravel())
    # print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))
    model = DecisionTreeRegressor(splitter='random', max_depth=30, min_samples_split=5, min_samples_leaf=2)
    # 拟合训练数据集
    model.fit(train_X, train_Y.values.ravel())
    # 预测测试集
    test_Y_pred = model.predict(test_X)
    print("测试集得分:", model.score(test_X, test_Y))
    print("测试集MSE:", mean_squared_error(test_Y, test_Y_pred))
    print("测试集RMSE:", np.sqrt(mean_squared_error(test_Y, test_Y_pred)))
    print("总耗时:", time() - t, "秒")
