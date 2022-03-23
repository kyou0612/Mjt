import tensorflow as tf
from tensorflow import keras
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def serialize_sample(x, y1,y2,y3,y4,sutelen,lichi,pn):
def mask_sample(x):
    x_binary = (x.astype(np.int32)).tobytes()

    x_list = _bytes_feature(x_binary)

    proto = tf.train.Example(features=tf.train.Features(feature={
        "mask": x_list,  # int32, (100, 100)

    }))
    return proto.SerializeToString()


def deserialize_example_4p(serialized_string):
    image_feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y1': tf.io.FixedLenFeature([], tf.string),
        'y2': tf.io.FixedLenFeature([], tf.string),
        'y3': tf.io.FixedLenFeature([], tf.string),
        'y4': tf.io.FixedLenFeature([], tf.string),

    }
    example = tf.io.parse_single_example(serialized_string, image_feature_description)
    image = tf.reshape(tf.io.decode_raw(example["x"], tf.int32), (96, 37, 1))
    label1 = tf.io.decode_raw(example["y1"], tf.int32)
    label2 = tf.io.decode_raw(example["y2"], tf.int32)
    label3 = tf.io.decode_raw(example["y3"], tf.int32)
    label4 = tf.io.decode_raw(example["y4"], tf.int32)

    return image, label1, label2, label3, label4


# m1 = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# m2 = tf.constant(0, shape=[96, 4])
# m3 = tf.concat([m1, m2], 0)
# m4 = tf.constant(1, shape=[100, 96])
#
# m = tf.concat([m3, m4], 1)


m1 = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
m2 = tf.constant(0, shape=[96, 4])
m3 = tf.concat([m1, m2], 0)


m4 = tf.constant(1, shape=[96, 96])
m5 = tf.constant(1, shape=[1, 24])
m6 = tf.constant(0, shape=[1, 24])
m7 = tf.concat([m5, m6,m6,m6], 0)
m8 = tf.concat([m6, m5,m6,m6], 0)
m9 = tf.concat([m6, m6,m5,m6], 0)
m10 = tf.concat([m6, m6,m6,m5], 0)
m11 = tf.concat([m7, m8,m9,m10], 1)
m12 = tf.concat([m11,m4], 0)

m = tf.concat([m3,m12], 1)


m = tf.cast(m, tf.bool)

for i in range(20):
    dataset = tf.data.TFRecordDataset(
        './learning_data/tnsp4p/data_train_2013_MjT_tnsp_96_37_4_' + str(
            i + 1) + '.tfrecords').map(deserialize_example_4p).batch(
        100000)

    for data in dataset:
        head = np.zeros([len(data[0]), 4, 37, 1])
        head = head + 0.1
        x_head = np.concatenate([head, data[0]], axis=1)

        mask = np.empty((0, 100, 100), bool)

        for j in range(len(x_head)):
            b = x_head[j]
            c = tf.math.not_equal(b, 0)

            # d = tf.math.reduce_all(c, 2)

            e = tf.math.reduce_any(c, 1)

            e1 = tf.reshape(e, [100])

            h = tf.logical_and(e1, e)
            h = tf.logical_and(h, m)

            h = tf.reshape(h, [1, 100, 100])

            mask = np.concatenate([mask, h], 0)

            if j % 1000 == 0:
                print(j)

    with tf.io.TFRecordWriter(
            './learning_data/tnsp4p/mask2_train_2013_MjT_tnsp_96_37_4_' + str(
                    i + 1) + '.tfrecords') as writer:
        for i in range(mask.shape[0]):
            example = mask_sample(mask)
            writer.write(example)
    print('file' + str(i) + 'over')