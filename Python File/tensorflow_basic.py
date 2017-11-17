# coding=utf-8
import numpy as np
from DataProcessing import *
from CSVSever import CSVSaver

with tf.Session() as sess1:
    merged = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter('./logs', sess1.graph)

    # 初始化所有变量
    tf.global_variables_initializer().run()
    # 训练迭代10轮
    for epoch in range(10):
        total_loss = 0
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], Y: [Y_train[i]]}
            # 通过 session.run 接口触发执行
            _, loss = sess1.run([train_op, cost], feed_dict=feed)
            total_loss += loss

        print ("Epoch: %04d, total loss = %.9f" % (epoch + 1, total_loss))
        summary, accuracy = sess1.run([merged, acc_op], feed_dict={X: X_test, Y: Y_test})
        summary_writer.add_summary(summary, epoch)
        print ("Accuracy on validation set: %.9f" % accuracy)

    save_path = saver.save(sess1, model_path)
    print ('Training complete')

    # # 评估校验数据集上的准确率
    # pred = sess.run(Y_pred, feed_dict={X: X_test})
    #
    # # argmax是找最大值的位置，后面的1指的是轴
    # correct = np.equal(np.argmax(pred, 1), np.argmax(Y_test, 1))
    #
    # # astype指的是类型转换,mean表示加起来，除以个数，这里是1 和 0， 所以可以
    # accuracy = np.mean(correct.astype(np.float32))

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
    CSVSaver(testdata, predictions)
