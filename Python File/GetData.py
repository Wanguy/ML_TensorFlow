# coding=utf-8
import re
from tensorflow_basic import *


def get_title(name):
    if pd.isnull(name):
        return 'NULL'
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1).lower()
    else:
        return 'None'


# 转换矩阵
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()

title = {
    'mr': 1,
    'mrs': 2, 'mme': 2,
    'ms': 3, 'miss': 3, 'mlle': 3,
    'don': 4, 'sir': 4, 'jonkheer': 4, 'major': 4, 'col': 4, 'dr': 4, 'master': 4, 'capt': 4,
    'dona': 5, 'lady': 5, 'countess': 5,
    'rev': 7
}
data['Title'] = data['Name'].apply(lambda name: title.get(get_title(name)))
data['Honor'] = data['Title'].apply(lambda title: 1 if title == 4 or title == 5 else 0)