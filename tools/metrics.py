# 样本内训练
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd


def getAucPerMonth(path):
    # 按月统计评估指标
    data = pd.DataFrame(pd.read_csv(path))

    time_group = data.groupby("time")
    time = []
    accuracy = []
    auc = []
    for name, group in time_group:
        time.append(name)
        accuracy.append(accuracy_score(group["label"], group["predict_label"]))
        auc.append(roc_auc_score(group["label"], group["probability"]))
    result = pd.DataFrame({"time": time, "Accuracy": accuracy, "AUC": auc})
    print(result)
    result.to_csv(path.replace('predict_test_data.csv', 'evaluation.csv'), encoding="utf-8-sig")
