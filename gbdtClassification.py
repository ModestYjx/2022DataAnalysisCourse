from time import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from tools.datasets.getData import getSplitFinacialData

if __name__ == "__main__":
    # 读取Mnist数据集
    # mnistSet = mnist.loadLecunMnistSet()
    # train_X, train_Y, test_X, test_Y = mnistSet[0], mnistSet[1], mnistSet[2], mnistSet[3]
    train_data_path = "data/train_data.csv"
    test_data_path = "data/test_data.csv"
    decision_tree_predict_test_data_path = "data/decision_tree_predict_test_data.csv"
    gbdt_predict_test_data_path = "data/gbdt_predict_test_data.csv"
    train_data_X, train_data_Y = getSplitFinacialData(train_data_path)
    train_X, test_X, train_Y, test_Y = train_test_split(train_data_X, train_data_Y, test_size=0.2, random_state=0)

    m, n = np.shape(train_X)
    idx = list(range(m))
    # 分批训练数据时每次拟合的样本数
    num = 30000

    # 使用PCA降维, 看看占样本特征方差90%的特征数目, 可以根据这个数目指定DecisionTreeClassifier类的参数max_features
    # print(np.shape(train_X))
    # pca = PCA(n_components=0.9, whiten=True, random_state=0)
    # for i in range(int(np.ceil(1.0*m/num))):
    #     minEnd = min((i+1)*num, m)
    #     sub_idx = list(idx[i*num:minEnd])
    #     train_pca_X = pca.fit_transform(train_X[sub_idx])
    #     print(np.shape(train_pca_X))
    #
    # pca = PCA(n_components=90, whiten=True, svd_solver='randomized', random_state=0)
    # train_X = pca.fit_transform(train_X)
    # test_X = pca.transform(test_X)

    print("**********测试GradientBoostingClassifier类**********")
    t = time()
    param_grid1 = {"n_estimators": range(1000, 2001, 100)}
    param_grid2 = {'max_depth': range(30, 71, 10), 'min_samples_split': range(4, 9, 2)}
    param_grid3 = {'min_samples_split': range(4, 9, 2), 'min_samples_leaf': range(3, 12, 2)}
    param_grid4 = {'subsample': np.arange(0.6, 1.0, 0.05)}
    model = GridSearchCV(
        estimator=GradientBoostingClassifier(max_features=38, max_depth=40, min_samples_split=8, learning_rate=0.1,
                                             n_estimators=1800),
        param_grid=param_grid4, cv=3)

    m, n = np.shape(train_X)
    print("m, n:\n", m, ",", n)

    p, q = np.shape(train_Y)
    print("p, q:\n", p, ",", q)

    m1 = len(train_Y.values.ravel())
    print("m1\n", m1)

    # 拟合训练数据集
    model.fit(train_X, train_Y.values.ravel())
    print("最好的参数是:%s, 此时的得分是:%0.2f" % (model.best_params_, model.best_score_))
    model = GradientBoostingClassifier(max_features=38, max_depth=40, min_samples_split=8, min_samples_leaf=3,
                                       n_estimators=1200, learning_rate=0.05, subsample=0.95)
    # 拟合训练集数据
    model.fit(train_X, train_Y.ravel())

    # dotData = export_graphviz(model, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dotData)
    # graph.write_pdf("mnist.pdf")

    np.random.shuffle(idx)
    # 预测训练集

    train_Y_hat = model.predict(train_X)

    print("train_Y_hat\n", train_Y_hat)

    print('训练集精确度: ', accuracy_score(train_Y, train_Y_hat))

    # 2. 预测测试集
    # test_data_X, test_data_Y = getSplitFinacialData(test_data_path)
    # test_data_Y_hat = model.predict(test_data_X)
    #
    # t1, t2 = np.shape(test_data_X)
    # print("t1, t2:\n", p, ",", q)
    #
    # proba_Y = model.predict_proba(test_data_X)[:,1]
    #
    # auc = roc_auc_score(test_data_Y, proba_Y)
    # print("decisition tree的AUC:", auc)
    # print(classification_report(test_data_Y, test_data_Y_hat, digits=4))
    #
    # # p1, p2 = np.shape(proba_Y)
    # # print("p1, p2:\n", p, ",", q)
    #
    # print("proba_Y:\n", proba_Y)
    # print("proba_Y's length:\n", len(proba_Y))
    # # score = model.score(test_X, test_Y)
    # # print("score:", score)
    #
    # print('测试集精确度: ', accuracy_score(test_data_Y, test_data_Y_hat))
    # print("总耗时:", time() - t, "秒")
    #
    # # 输出预测文件
    # test_data = pd.DataFrame(pd.read_csv("data/test_data.csv"))
    # label_data = pd.DataFrame({"predict_label": test_data_Y_hat, "probability": proba_Y})
    # # label_data = pd.DataFrame({"predict_label": test_Y_hat})
    # data = pd.concat([test_data, label_data], axis=1)
    # data.to_csv("data/decision_tree_predict_test_data.csv", encoding="utf-8-sig", index=False)
    #
    #
    # # 绘制ROC曲线
    # n_class = len(np.unique(train_Y))
    # roc.drawROC(n_class, test_data_Y, test_data_Y_hat)
    #
    # getAucPerMonth(gbdt_predict_test_data_path)
