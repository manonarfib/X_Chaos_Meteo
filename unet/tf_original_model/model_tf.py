import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dropout, concatenate, BatchNormalization, Activation, multiply, Lambda, Reshape, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Permute, Concatenate, Conv2D, Add
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dot
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, DepthwiseConv2D, LayerNormalization, Softmax


def Late_WF_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    def UNet(x, lags, features_output, filters, dropout, kernel_init):

        # --- Contracting part / encoder ---#
        conv1 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(x)
        conv1 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv1)
        pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)

        conv2 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool1)
        conv2 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv2)
        pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)

        conv3 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool2)
        conv3 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv3)
        pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)

        conv4 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool3)
        conv4 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv4)
        drop4 = Dropout(dropout)(conv4)

        # --- Bottleneck part ---#
        pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
        conv5 = Conv3D(16*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool4)
        compressLags = Conv3D(16*filters, (lags, 1, 1),
                              activation='relu', padding='valid')(conv5)
        conv5 = Conv3D(16*filters, 3, activation='relu', padding='same',
                       kernel_initializer=kernel_init)(compressLags)
        drop5 = Dropout(dropout)(conv5)

        # --- Expanding part / decoder ---#
        up6 = Conv3D(8*filters, 2, activation='relu', padding='same',
                     kernel_initializer=kernel_init)(UpSampling3D(size=(1, 2, 2))(drop5))
        compressLags = Conv3D(8*filters, (lags, 1, 1),
                              activation='relu', padding='valid')(drop4)
        merge6 = concatenate([compressLags, up6], axis=-1)
        conv6 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge6)
        conv6 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv6)

        up7 = Conv3D(4*filters, 2, activation='relu', padding='same',
                     kernel_initializer=kernel_init)(UpSampling3D(size=(1, 2, 2))(conv6))
        compressLags = Conv3D(4*filters, (lags, 1, 1),
                              activation='relu', padding='valid')(conv3)
        merge7 = concatenate([compressLags, up7], axis=-1)
        conv7 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge7)
        conv7 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv7)

        up8 = Conv3D(2*filters, 2, activation='relu', padding='same',
                     kernel_initializer=kernel_init)(UpSampling3D(size=(1, 2, 2))(conv7))
        compressLags = Conv3D(2*filters, (lags, 1, 1),
                              activation='relu', padding='valid')(conv2)
        merge8 = concatenate([compressLags, up8], axis=-1)
        conv8 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge8)
        conv8 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv8)

        up9 = Conv3D(filters, 2, activation='relu', padding='same',
                     kernel_initializer=kernel_init)(UpSampling3D(size=(1, 2, 2))(conv8))
        compressLags = Conv3D(filters, (lags, 1, 1),
                              activation='relu', padding='valid')(conv1)
        merge9 = concatenate([compressLags, up9], axis=-1)
        conv9 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge9)
        conv9 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv9)
        conv9 = Conv3D(2*features_output, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv9)

        conv10 = Conv3D(features_output, 1, activation='relu')(
            conv9)  # Reduce last dimension

        return Model(inputs=x, outputs=conv10)

    inputA = Input(shape=(lags, latitude, longitude, features))
    inputB = Input(shape=(lags, latitude, longitude, features))

    streamA = UNet(inputA, lags, features_output,
                   filters, dropout, kernel_init)
    streamB = UNet(inputB, lags, features_output,
                   filters, dropout, kernel_init)

    fusion = concatenate([streamA.output, streamB.output])

    out = Conv3D(features_output, 1, activation='linear',
                 padding='same')(fusion)  # Reduce last dimension
    return Model(inputs=[inputA, inputB], outputs=out)
