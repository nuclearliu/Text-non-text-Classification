#!/usr/bin/env python3
# coding = utf-8

import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
from ResNet18 import ResNet18

# train_path = './TextDis_benchmark/'
# train_txt = './TextDis_benchmark/trainList.txt'
# x_train_savepath = './TextDis_benchmark/x_train.npy'
# y_train_savepath = './TextDis_benchmark/y_train.npy'

test_path = './TextDis_benchmark/'
test_txt = './TextDis_benchmark/testList.txt'
x_test_savepath = './TextDis_benchmark/x_test.npy'
y_test_savepath = './TextDis_benchmark/y_test.npy'


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


if os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_test = np.reshape(x_test_save, (len(x_test_save), 224, 224, 1))
else:
    print('-------------Generate Datasets-----------------')
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

x_test = x_test / 255.0
print(x_test.shape)

model = ResNet18([3, 4, 6, 3])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./ResNet34model/ResNet34.ckpt"
model.load_weights(checkpoint_save_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)
start = time.time()
result = model.predict(x_test)
print("Duration = " + str(time.time() - start))
# result = tf.argmax(result, axis=1)
# dif = result - y_test
# print(result)
# print(y_test)
# print(dif)
# total_count = len(result)
# accurate_count = 0
# for i in dif:
#     if i == 0:
#         accurate_count += 1
# print(accurate_count/total_count)

result = tf.argmax(result, axis=1)
dif = result - y_test
# print(dif.shape)
total_count = len(result)
accurate_count = 0
TP = FP = TN = FN = 0
for idx, r in enumerate(result):
    if r == 1 and y_test[idx] == 1:
        TP += 1
    if r == 1 and y_test[idx] == 0:
        FP += 1
    if r == 0 and y_test[idx] == 1:
        FN += 1
    if r == 0 and y_test[idx] == 0:
        TN += 1
    if r == y_test[idx]:
        accurate_count += 1
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print("precision: {}".format(precision))
print("recall: {}".format(recall))
print("F-Measure: {}".format(2*precision*recall/(precision+recall)))
print("accuracy: {}".format(accurate_count/total_count))


# history = model.fit(x_train, y_train, batch_size=32, epochs=1, validation_split=0.2, validation_freq=1,
#                     callbacks=[cp_callback])
# model.summary()

# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()