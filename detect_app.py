#!/usr/bin/env python3
# coding = utf-8

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

checkpoint_save_path = "./TextDis_benchmark/ResNet18.ckpt"
model.load_weights(checkpoint_save_path)

out_path = "./output_images/"
text_image_count = 0
non_text_image_count = 0
img_path = input("\x1b[0mPath: ")
img_path = img_path.replace("\\","")
img_path = img_path.strip()
while img_path not in ["quit", "q"]:
    img_orig = Image.open(img_path)  # 读入图片
    img = img_orig.resize((224, 224))
    img = np.array(img.convert("L"))
    img = img[tf.newaxis, ...]
    img = np.reshape(img, (1, 224, 224, 1))
    img = img / 255.
    result = model.predict(img)
    print(result)
    result = tf.argmax(result, axis=1)
    img_out = Image.new('RGB', (img_orig.size[0], img_orig.size[1] + 35), (20, 136, 173))
    img_out.paste(img_orig, (0, 0))
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    draw = ImageDraw.Draw(img_out)
    if result:
        text_image_count += 1
        print("\x1b[35mtext detected")
        draw.text((10, img_orig.size[1] + 5), "text detected", (255, 255, 255), font=font)
        img_out.save(out_path + "text/" + str(text_image_count) + ".jpg")
    else:
        non_text_image_count += 1
        print("\x1b[35mtext not detected")
        draw.text((10, img_orig.size[1] + 5), "text not detected", (255, 255, 255), font=font)
        img_out.save(out_path + "nonText/" + str(non_text_image_count) + ".jpg")
    img_path = input("\x1b[0mPath: ")
    img_path = img_path.replace("\\", "")
    img_path = img_path.strip()