# coding=utf-8
import pandas as pd  # 数据分析

import numpy as np

train_path = '/Users/sephiroth/Developer/PycharmProject/TensorFlowTest/Data Source/train.csv'
test_path = "/Users/sephiroth/Developer/PycharmProject/TensorFlowTest/Data Source/test.csv"
model_path = "/Users/sephiroth/Developer/PycharmProject/TensorFlowTest/Data Model/model.ckpt"

# from setuptools.sandbox import save_path

data = pd.read_csv(train_path)  # 读入的数据为一个DataFrame类型的对象

# DataFrame是一个二维数据类型

data.info()  # 查看数据概况

# 取部分特征字段用于分类，并将所有缺失的字段补充为0

# 对Sex字段进行正规化处理

data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)

# 对年龄字段补全均值

mean_age = data['Age'].mean()  # 29.69

# print mean_age

data['Age'][data.Age.isnull()] = mean_age

data = data.fillna(0)

dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

dataset_X = dataset_X.as_matrix()  # 转换成矩阵

# print dataset_X

# 两种分类分别是幸存和死亡，即'Survived'和'Deceased'

data['Deceased'] = data['Survived'].apply(lambda s: int(not s))

dataset_Y = data[['Deceased', 'Survived']]

dataset_Y = dataset_Y.as_matrix()

# print dataset_Y

# scikit-learn库中提供了用于切分数据集的工具函数train_test_split，随机打乱数据集后按比列拆分数据集

from sklearn.model_selection import train_test_split

# 使用函数train_test_split将标记数据切分为训练数据集和验证数据集，其中验证数据集占20%

X_train, X_validation, Y_train, Y_validation = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# print Y_validation



# 接下来，使用TensorFlow构建计算图

# 使用placeholder声明占位符

# 声明输入数据占位符

# shape参数的第一个元素为none,表示可以同时放入任意条记录

import tensorflow as tf

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 6])

    Y = tf.placeholder(tf.float32, shape=[None, 2])

with tf.name_scope('classifier'):
    # 声明参数变量

    W = tf.Variable(tf.random_normal([6, 2]), name='weights')

    b = tf.Variable(tf.zeros([2]), name='bias')

    # use saver to save and restore model

    saver = tf.train.Saver()

    # 构造前向传播计算图

    y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

    # add histogram summaries for weights, view on tensorboard

    tf.summary.histogram('weights', W)

    tf.summary.histogram('bias', b)

with tf.name_scope('cost'):
    # 声明代价函数

    cross_entropy = - tf.reduce_sum(Y * tf.log(y_pred + 1e-10), reduction_indices=1)

    cost = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cost', cost)

# 验证集合上的正确率

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1))

    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    tf.summary.scalar('accuracy', acc_op)

# 使用梯度下降算法最小化代价，系统自动构建反向传播部分的计算图

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# 计算图的声明完成



# 构建训练迭代过程

with tf.Session() as sess1:
    # create a log writer. run 'tensorboard--logdir=./logs'

    sum_ops = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter('/tmp/log', sess1.graph)

    # 初始化所有变量

    tf.global_variables_initializer().run()

    # training loop

    for epoch in range(10):

        total_loss = 0.

        for i in range(len(X_train)):
            # prepare feed data and run,这里相关于是使用随机梯度下降

            feed_dict = {X: [X_train[i]], Y: [Y_train[i]]}

            #             _, loss = sess1.run([sum_ops, train_op, cost], feed_dict = feed_dict)

            _, loss = sess1.run([train_op, cost], feed_dict=feed_dict)

            total_loss += loss

            # display loss per epoch

        #         summary_writer.add_summary(summary, epoch)

        print('Epoch: %04d, total loss = %-9f' % (epoch + 1, total_loss))

        # 用验证数据集合评估模型的表现

        #     pred = sess1.run(y_pred, feed_dict = {X: X_validation})

        # argmax是找最大值的位置，后面的1指的是轴

        #     correct = np.equal(np.argmax(pred, 1), np.argmax(Y_validation, 1))

        #     print correct

        #     cao = correct.astype(np.float32)

        #     print cao

        # astype指的是类型转换,mean表示加起来，除以个数，这里是1 和 0， 所以可以

        #     accuracy = np.mean(correct.astype(np.float32))

        summary, accuracy = sess1.run([sum_ops, acc_op], feed_dict={X: X_validation, Y: Y_validation})

        summary_writer.add_summary(summary, epoch)

        print ('Accuracy on validation set: %.9f' % accuracy)

        save_path = saver.save(sess1, model_path)

    print 'Training complete!'

with tf.Session() as sess2:
    # predict on test data

    testdata = pd.read_csv(test_path)

    testdata = testdata.fillna(0)

    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)

    X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

    saver.restore(sess2, model_path)

    predictions = np.argmax(sess2.run(y_pred, feed_dict={X: X_test}), 1)

    print predictions

    submission = pd.DataFrame({

        'PassengerId': testdata['PassengerId'],

        'Survived': predictions

    })

    submission.to_csv('Titanic-submission-miao.csv', index=False)
