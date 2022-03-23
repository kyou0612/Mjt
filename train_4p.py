import tensorflow as tf
from tensorflow import keras
import numpy as np
from readtfrecord import deserialize_example_4p
from modeltenpai import create_mjt_4player

train_path = './learning_data/tnsp/data_train_2013_MjT_tnsp_96_37_4_'


def run_experiment(model):
    optimizer = tf.optimizers.Adam(
        learning_rate=0.000001
        # , decay=0.000003
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.binary_accuracy
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model.summary()

    t = 0
    epochs = num_epochs
    trainloss_list = []
    trainacc_list = []
    valloss_list = []
    valacc_list = []
    testloss_list = []
    testacc_list = []
    x_predict = []
    # trainpart = [2,3,4]
    # testpart = 1

    ###マスクのCLS部分のマス###
    m1 = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    m2 = tf.constant(0, shape=[96, 4])
    m3 = tf.concat([m1, m2], 0)
    m4 = tf.constant(1, shape=[100, 96])

    m = tf.concat([m3, m4], 1)
    m = tf.cast(m, tf.bool)

    while t < epochs:
        for i in range(20):
            dataset = tf.data.TFRecordDataset(train_path + str(i + 1) + '.tfrecords').map(deserialize_example_4p).batch(
                100000)

            for data in dataset:
                head = np.zeros([len(data[0]), 4, 37, 1])
                # head = head + 0.1
                x_head = np.concatenate([head, data[0]], axis=1)

                mask = np.empty((0, 100, 100), bool)
                for i in range(len(x_head)):
                    b = x_head[i]
                    c = tf.math.not_equal(b, 0)

                    # d = tf.math.reduce_all(c, 2)

                    e = tf.math.reduce_any(c, 1)

                    e1 = tf.reshape(e, [100])

                    h = tf.logical_and(e1, e)
                    h = tf.logical_and(h, m)

                    h = tf.reshape(h, [1, 100, 100])


                    mask = np.concatenate([mask, h], 0)


                history = model.fit(
                    x=[x_head, mask],
                    y=[data[1],data[2],data[3],data[4]],
                    epochs=1,
                    class_weight={0: 1, 1: 1.25},
                    verbose=1,
                    validation_split=0.1
                    # callbacks=[checkpoint_callback],
                )
                trainloss_list.append(history.history['loss'])
                trainacc_list.append(history.history['binary_accuracy'])
                # valloss_list.append(history.history['val_loss'])
                # valacc_list.append(history.history['val_binary_accuracy'])
        t += 1
        model.save_weights("/content/drive/MyDrive/experiment/model/checkpoint/hitoriformaskwithCLS.ckpt" + str(t))

    return history


num_classes = 1
input_shape = (100, 37, 1)
mask_shape = (100, 100)
# learning_rate = 0.001
# weight_decay = 0.0001
# batch_size = 256
num_epochs = 50
# image_size = 72  # We'll resize input images to this size
# patch_size = 6  # Size of the patches to be extract from the input images
num_patches = 100
projection_dim = 104
hai_dim = 37
num_heads = 5
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 6
mlp_head_units = [104, 32]  # Size of the dense layers of the final classifier
train_path = './learning_data/tnsp/data_train_2013_MjT_tnsp_24_36_1_'

vit_classifier = create_mjt_4player(input_shape, mask_shape, num_patches, projection_dim,
                                    transformer_layers, num_heads, transformer_units, mlp_head_units, num_classes,hai_dim)
history = run_experiment(vit_classifier)