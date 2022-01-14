#!/usr/bin/env python3
# coding = utf-8

# This turned out to be a failed attempt!

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from mspnet import MSP

train_path = './TextDis_benchmark/'
train_txt = './TextDis_benchmark/trainList.txt'
x_train_savepath = './TextDis_benchmark/x_train.npy'
y_train_savepath = './TextDis_benchmark/y_train.npy'

# test_path = './mnist_image_label/mnist_test_jpg_10000/'
# test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'
# x_test_savepath = './mnist_image_label/mnist_x_test.npy'
# y_test_savepath = './mnist_image_label/mnist_y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[1] , 标签为value[2] , 存入列表
        img_path = path + value[1]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = img.resize((224, 224))
        img = np.array(img.convert("L"))  #.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        # img = img /255. # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[2])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_


if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 224, 224, 1))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)

# np.random.seed(116)
# np.random.shuffle(x_train)
# np.random.seed(116)
# np.random.shuffle(y_train)
# tf.random.set_seed(116)
# x_train = x_train[:200]
# y_train = y_train[:200]
x_train = x_train / 255.0
# print(x_train.shape)

tf.compat.v1.disable_eager_execution()
model = MSP()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            # optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./TextDis_benchmark/MSP_Net.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=2 , validation_split=0.1, validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()