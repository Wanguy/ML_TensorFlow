# coding=utf-8
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

train_path = '/Users/sephiroth/Documents/ML_TensorFlow/Data Source/train.csv'
test_path = "/Users/sephiroth/Documents/ML_TensorFlow/Data Source/test.csv"
model_path = "/Users/sephiroth/Documents/ML_TensorFlow/Data Model/model.ckpt"

v1 = tf.Variable(tf.zeros([200]))
saver = tf.train.Saver()
v2 = tf.Variable(tf.ones([100]))

# 读取训练数据
data = pd.read_csv(train_path)
# 取部分特征字段用于分类，并将所有却是的字段设为 0
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data = data.fillna(0)
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()

# 两种分类分别是幸存和死亡，'Survived' 字段是其中一种分类的标签
# 新添 'Deceased' 字段表示第二种分类的标签，取值为 'Survived' 字段取非
data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()

X_train, X_test, Y_train, Y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2,
                                                    random_state=42)
# 声明输入的占位符
# shape 参数的第一个元素为 None，表示可以同时放入任意条记录
X = tf.placeholder(tf.float32, shape=[None, 6])
Y = tf.placeholder(tf.float32, shape=[None, 2])

# 声明变量，使用逻辑回归算法
W = tf.Variable(tf.random_normal([6, 2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')
Y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

# 使用交叉熵作为代价函数
cross_entropy = - tf.reduce_sum(Y * tf.log(Y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

# 使用随机梯度下降算法优化器来最小化代价，系统自动构建反向传播部分的计算图
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    # 初始化所有变量，必须最想执行
    tf.global_variables_initializer().run()

    # 训练迭代10轮
    for epoch in range(10):
        total_loss = 0
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], Y: [Y_train[i]]}
            # 通过 session.run 接口触发执行
            _, loss = sess.run([train_op, cost], feed_dict=feed)
            total_loss += loss
        print ("Epoch: %04d, total loss = %.9f" % (epoch + 1, total_loss))
        # 评估校验数据集上的准确率
        pred = sess.run(Y_pred, feed_dict={X: X_test})
        correct = np.equal(np.argmax(pred, 1), np.argmax(Y_test, 1))
        accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set: %.9f" % accuracy)
    save_path = saver.save(sess, model_path)
    print ('Training complete')






# 开启 Session 进行预测
with tf.Session() as sess2:
    tf.global_variables_initializer().run()
    # 读入测试数据集并完成预处理
    testdata = pd.read_csv(test_path).fillna(0)
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    # 加载模型存档
    saver.restore(sess2, model_path)
    # 正向传播计算
    predictions = np.argmax(sess2.run(Y_pred, feed_dict={X: X_test}), 1)

# 构建提交结果的数据结构，并将结果存储为csv文件
submission = pd.DataFrame({
    "PassengerId": testdata["PassengerId"],
    "Survived": predictions
})
submission.to_csv("/Users/sephiroth/Documents/ML_TensorFlow/titanic-submission.csv", index=False)
