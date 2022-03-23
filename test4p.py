import tensorflow as tf
from tensorflow import keras
import numpy as np
from readtfrecord import deserialize_example_4p
from modeltenpai import create_mjt_4player
from readtfrecord import mask_example_4p
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


def test_experiment(model):
    optimizer = tf.optimizers.Adam(
        learning_rate=0.00005
        # , decay=0.000005
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        # loss=weight_binary_crossentropy,
        metrics=[
            # binary_acc
            keras.metrics.binary_accuracy
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model.summary()

    checkpoint_filepath = "../checkpoint/4p_mask_CLS.ckpt24"


    model.load_weights(checkpoint_filepath)

    testtesttestx = np.empty((0, 100, 37, 1), int)
    testtesttesty1 = np.empty((0, 1), int)
    testtesttesty2 = np.empty((0, 1), int)
    testtesttesty3 = np.empty((0, 1), int)
    testtesttesty4 = np.empty((0, 1), int)
    playernum = np.empty((0, 1), int)
    lichilichi = np.empty((0, 1), int)
    sutelenlen = np.empty((0, 1), int)
    mask = np.empty((0, 100, 100), bool)
    head = np.zeros([50000, 4, 37, 1])
    head = head + 0.1
    result1 = np.empty((0),float)
    result2 = np.empty((0), float)
    result3 = np.empty((0), float)
    result4 = np.empty((0), float)

    for i in range(2):
        testdataset = tf.data.TFRecordDataset(test_path+ str(i + 1) + '.tfrecords').map(deserialize_example_4p).batch(
            10000)
        maskset = tf.data.TFRecordDataset(mask_path + str(i + 1) + '.tfrecords').map(mask_example_4p).batch(
            10000)

        for (data,m) in zip(testdataset,maskset):
            # testtesttestx = np.concatenate([testtesttestx, data[0]],axis=0)
            x_head = np.concatenate([head, data[0]], axis=1)





            result = model.predict([x_head[0],m])
            testtesttesty1 = np.concatenate([testtesttesty1, data[1]])
            testtesttesty2 = np.concatenate([testtesttesty2, data[2]])
            testtesttesty3 = np.concatenate([testtesttesty3, data[3]])
            testtesttesty4 = np.concatenate([testtesttesty4, data[4]])

            result1 = np.concatenate([result1, result[0]])
            result2 = np.concatenate([result2, result[1]])
            result3 = np.concatenate([result3, result[2]])
            result4 = np.concatenate([result4, result[3]])
            # playernum = np.concatenate([playernum, data[7]])
            # lichilichi = np.concatenate([lichilichi, data[6]])
            # sutelenlen = np.concatenate([sutelenlen, data[5]])

            # plt.figure(figsize=(8,6))
            # plt.plot(fpr, tpr)
            # plt.title("ROC curve", fontsize=18)
            # plt.xlabel("false positive rate", fontsize=18)
            # plt.ylabel("true positive rate", fontsize=18)
            # plt.tick_params(labelsize=18)
            # plt.show()
            # print(auc(fpr, tpr))
    # score = model.evaluate(testtesttestx, [testtesttesty1,testtesttesty2,testtesttesty3,testtesttesty4])
    # print(score)
    score = 0



    # fpr, tpr, thresholds = roc_curve(y_test, result)
    result_1 = [int((s + 0.5) // 1) for s in result1]
    print(classification_report(testtesttesty1, result_1, digits=4))

    result_2 = [int((s + 0.5) // 1) for s in result2]
    print(classification_report(testtesttesty2, result_2, digits=4))

    result_3 = [int((s + 0.5) // 1) for s in result3]
    print(classification_report(testtesttesty3, result_3, digits=4))

    result_4 = [int((s + 0.5) // 1) for s in result4]
    print(classification_report(testtesttesty4, result_4, digits=4))

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
train_path = '../tnsp4p/data_train_2013_MjT_tnsp_96_37_4_'
test_path = '../tnsp4p/data_test_2015_MjT_tnsp_96_37_4_'
mask_path = '../tnsp4p/mask_train_2013_MjT_tnsp_96_37_4_'

vit_classifier = create_mjt_4player(input_shape, mask_shape, num_patches, projection_dim,
                                    transformer_layers, num_heads, transformer_units, mlp_head_units, num_classes,hai_dim)
history = test_experiment(vit_classifier)