import tensorflow as tf
from tensorflow import keras
import numpy as np
from readtfrecord import deserialize_example_1p
from modeltenpai import create_mjt_1player
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


def test_experiment(model):
    optimizer = tf.optimizers.Adam(
        learning_rate=0.000002
        # , decay=0.000005
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        # loss = weight_binary_crossentropy,
        metrics=[
            # binary_acc
            keras.metrics.binary_accuracy
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model.summary()

    checkpoint_filepath = "/content/drive/MyDrive/experiment/backup/hitoriformasknoCLS.ckpt9"

    model.load_weights(checkpoint_filepath)

    testtesttestx = np.empty((0, 24, 36, 1), int)
    testtesttesty1 = np.empty((0, 1), int)
    playernum = np.empty((0, 1), int)
    lichilichi = np.empty((0, 1), int)
    sutelenlen = np.empty((0, 1), int)

    for i in range(2):
        testdataset = tf.data.TFRecordDataset(test_path + str(i + 1) + '.tfrecords').map(deserialize_example_1p).batch(
            100000)

        for data in testdataset:
            # testtesttestx = np.concatenate([testtesttestx, data[0]],axis=0)

            testtesttestx = np.append(testtesttestx, data[0], axis=0)
            testtesttesty1 = np.concatenate([testtesttesty1, data[1]])

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
    head = np.zeros([len(testtesttestx), 1, 36, 1])
    head = head + 0.1
    x_head = np.concatenate([head, testtesttestx], axis=1)
    mask = np.empty((0, 25, 25), bool)
    for i in range(len(x_head)):
        b = x_head[i]
        c = tf.math.not_equal(b, 0)

        d = tf.math.reduce_all(c, 2)

        e = tf.math.reduce_any(c, 1)

        e1 = tf.reshape(e, [25])

        h = tf.logical_and(e1, e)

        h = tf.reshape(h, [1, 25, 25])
        mask = np.concatenate([mask, h], 0)

    actmodel = Model(inputs=model.input, outputs=model.get_layer('multi_head_attention_5').output)

    activations = actmodel.predict([x_head, mask])
    print(len(activations))
    print(len(activations[0]))
    print(len(activations[0][0]))

    for i in range(len(activations)):
        print(activations[i])
        input()

    score = 0
    result = model.predict([x_head, mask])
    # fpr, tpr, thresholds = roc_curve(y_test, result)
    result_1 = [int((s + 0.5) // 1) for s in result]
    print(classification_report(testtesttesty1, result_1, digits=4))

    fpr, tpr, thresholds = roc_curve(testtesttesty1, result)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.title("ROC curve", fontsize=18)
    plt.xlabel("false positive rate", fontsize=18)
    plt.ylabel("true positive rate", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.savefig('LSTM_pretenpai.eps')
    plt.show()
    print(auc(fpr, tpr))

    # lichiflag = 0
    # tenpaiflag = 0
    # lasttenpaiflag = 0
    # listenflag = 0
    # layer_name = 'lambda'
    # hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # hidden_output = hidden_layer_model.predict(testtesttestx)

    # for i in range(len(result[0])):
    #   if playernum[i][0] == 0 and sutelenlen[i][0] == 1:
    #     lichiflag = [0,0,0,0]
    #     tenpaiflag = 0
    #     listenflag = 0
    #     lasttenpaiflag = 0
    #   lichiflag[playernum[i][0]] = lichilichi[i][0]
    #   print(str(result[0][i])+' '+str(result[1][i])+' '+str(result[2][i])+' '+str(result[3][i])+' '+str(result_1[i])+' '+str(result_2[i])+' '+str(result_3[i])+' '+str(result_4[i])+' label:'+str(testtesttesty1[i])+' '+str(testtesttesty2[i])+' '+str(testtesttesty3[i])+' '+str(testtesttesty4[i])+' player:'+str(playernum[i])+' reach:'+str(lichiflag[0])+' '+str(lichiflag[1])+' '+str(lichiflag[2])+' '+str(lichiflag[3])+' round:'+str(sutelenlen[i]))
    #   tenpaiflag = testtesttesty1[i][0]+testtesttesty1[i][0]+testtesttesty1[i][0]+testtesttesty1[i][0]
    #   if listenflag == 0 and tenpaiflag >0 and np.sum(lichiflag) == 0:
    #     listenflag = 1
    #   if listenflag == 1 and np.sum(lichiflag) > 0:
    #     listenflag = 2
    #   if listenflag == 2 and tenpaiflag < lasttenpaiflag:
    #     listenflag = 3
    #   if listenflag == 3:
    #     input()

    # lasttenpaiflag = tenpaiflag
    # if i >= 4000:
    #   break

    # for i in range(result):
    #   print(str(result[i])+' '+str(result_1[i])+' '+str(testtesttesty1[i])+' '+str(hidden_output[i]))
    return score

num_classes = 1
input_shape = (25,36,1)
mask_shape =(25,25)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
# image_size = 72  # We'll resize input images to this size
# patch_size = 6  # Size of the patches to be extract from the input images
num_patches = 25
projection_dim = 104
num_heads = 5
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 6
mlp_head_units = [104, 32]  # Size of the dense layers of the final classifier
train_path = './learning_data/tnsp/data_test_2015_MjT_tnsp_24_36_1_'

vit_classifier = create_mjt_1player(input_shape,mask_shape,num_patches,projection_dim,
                          transformer_layers,num_heads,transformer_units,mlp_head_units,num_classes)
history = test_experiment(vit_classifier)