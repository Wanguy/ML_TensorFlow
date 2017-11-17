# coding=utf-8
import pandas as pd


def CSVSaver(testdata, predictions):
    # 构建提交结果的数据结构，并将结果存储为csv文件
    submission = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("titanic-submission.csv", index=False)
