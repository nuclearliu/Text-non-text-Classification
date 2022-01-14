import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model, layers


class SPP_layer(tf.keras.layers.Layer):
    def __init__(self, num_levels=4, pool_type='max_pool'):
        super(SPP_layer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def build(self, input_shape):
        self.num_levels = 4
        self.pool_type = 'max_pool'

    def call(self, inputs, **kwargs):

        shape = inputs.shape
        num, h, w, c = shape[0], shape[1], shape[2], shape[3]
        # if h == None:
        #     h = 255
        # if w == None:
        #     w = 255

        # print(shape)

        for i in range(self.num_levels):
            level = (2 * i + 1) * 2
            channel = level * level * c
            kernel_size = [1, np.ceil(h / level + 1).astype(np.int), np.ceil(w / level + 1).astype(np.int), 1]
            stride_size = [1, np.floor(h / level + 1).astype(np.int), np.floor(w / level + 1).astype(np.int), 1]

            if self.pool_type == 'max_pool':
                pool = tf.nn.max_pool(inputs, ksize=kernel_size, strides=stride_size, padding='SAME')
                # print('after pooling :' , pool.shape)
                pool = tf.reshape(pool, (-1, channel))

            else:
                pool = tf.nn.avg_pool(inputs, ksize=kernel_size, strides=stride_size, padding='SAME')
                # print('after pooling :',pool.shape )
                pool = tf.reshape(pool, (-1, channel))

            # print('after reshape:',pool.shape)

            if level == 2:
                x_flatten = tf.reshape(pool, (-1, channel))
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)

            # x_flatten.get_shape().as_list()

        # x_flatten = tf.reshape(x_flatten, (-1, 1920))

        return x_flatten


class MSP(Model):
    def __init__(self):
        super(MSP, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.deconv_3 = layers.Conv2DTranspose(filters=128, kernel_size=(1, 1), strides=1, padding='same')
        self.deconv_3_b = BatchNormalization()
        self.deconv_3_a = Activation("relu")
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)


        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.deconv_4 = layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=2, padding='same')
        self.deconv_4_b = BatchNormalization()
        self.deconv_4_a = Activation("relu")
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.deconv_5 = layers.Conv2DTranspose(filters=256, kernel_size=(8, 8), strides=4, padding='same')
        self.deconv_5_b = BatchNormalization()
        self.deconv_5_a = Activation("relu")
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.spp = SPP_layer()

        self.fc_1 = Dense(8192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.fc_2 = Dense(8192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.out = Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())


    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        deconv_layer_3 = self.deconv_3(x)
        deconv_layer_3 = self.deconv_3_b(deconv_layer_3)
        deconv_layer_3 = self.deconv_3_a(deconv_layer_3)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        deconv_layer_4 = self.deconv_4(x)
        deconv_layer_4 = self.deconv_4_b(deconv_layer_4)
        deconv_layer_4 = self.deconv_4_a(deconv_layer_4)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        deconv_layer_5 = self.deconv_5(x)
        deconv_layer_5 = self.deconv_5_b(deconv_layer_5)
        deconv_layer_5 = self.deconv_5_a(deconv_layer_5)
        # x = self.p5(x)
        # x = self.d5(x)

        multi_level = tf.concat([deconv_layer_3, deconv_layer_4, deconv_layer_5], axis=3)
        afterSPP = self.spp(multi_level)
        afterSPP = self.flatten(afterSPP)
        y = self.fc_1(afterSPP)
        y = self.fc_2(y)
        y = self.out(y)


        return y
