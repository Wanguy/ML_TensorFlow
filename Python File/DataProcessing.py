# coding=utf-8
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from Path import *

v1 = tf.Variable(tf.zeros([200]))
saver = tf.train.Saver()
v2 = tf.Variable(tf.ones([100]))

# 读取训练数据
data = pd.read_csv(train_path)
# 取部分特征字段用于分类，并将所有却是的字段设为 0
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
mean_age = data['Age'].mean()
data['Age'][data.Age.isnull()] = mean_age
data = data.fillna(0)

# 转换矩阵
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()

# 两种分类分别是幸存和死亡，'Survived' 字段是其中一种分类的标签
# 新添 'Deceased' 字段表示第二种分类的标签，取值为 'Survived' 字段取非
data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()

X_train, X_test, Y_train, Y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)


# 声明输入的占位符
# shape 参数的第一个元素为 None，表示可以同时放入任意条记录
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 6])
    Y = tf.placeholder(tf.float32, shape=[None, 2])


# 声明变量，使用逻辑回归算法
with tf.name_scope('classifier'):
    W = tf.Variable(tf.random_normal([6, 2]), name='weights')
    b = tf.Variable(tf.zeros([2]), name='bias')
    saver = tf.train.Saver()
    Y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
    tf.summary.histogram('weight', W)
    tf.summary.histogram('bias', b)

# 使用交叉熵作为代价函数
with tf.name_scope('cost'):
    cross_entropy = - tf.reduce_sum(Y * tf.log(Y_pred + 1e-10), reduction_indices=1)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cost)

# 使用随机梯度下降算法优化器来最小化代价，系统自动构建反向传播部分的计算图
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_pred, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', acc_op)

# 使用梯度下降算法最小化代价，系统自动构建反向传播部分的计算图
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
