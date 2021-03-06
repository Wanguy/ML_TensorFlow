# coding=utf-8
import pandas as pd
import tensorflow as tf
from Path import *

# 将 train.csv 转换为 train.tfrecords
def transform_to_tfrecord():
    data = pd.read_csv(train_path)
    tfrecord_file = tfrecord_path

    def int_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

    def float_feature(value):
        return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(len(data)):
        features = tf.train.Features(feature={
            'Age':float_feature(data['Age'][i]),
            'Survived':int_feature(data['Survived'][i]),
            'Pclass':int_feature(data['Pclass'][i]),
            'Parch':int_feature(data['Parch'][i]),
            'SibSp':int_feature(data['SibSp'][i]),
            'Sex':int_feature(1 if data['Sex'][i] == 'male' else 0),
            'Fare':float_feature(data['Fare'][i])
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()