import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, concatenate, Conv3D, MaxPool3D, UpSampling3D, Lambda, Cropping3D
import numpy as np
from keras.layers import Layer


class PrintShape(Layer):
    def __init__(self, name_prefix="", **kwargs):
        super().__init__(**kwargs)
        self.name_prefix = name_prefix

    def call(self, x):
        tf.print(self.name_prefix, tf.shape(x))
        return x


def Late_WF_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):

    def UNet(x, lags, features_output, filters, dropout, kernel_init):
        # --- Encoder ---
        print("x shape", x.shape)

        conv1 = Conv3D(filters, 3, activation='relu',
               padding='same', kernel_initializer=kernel_init)(x)
        conv1 = PrintShape("conv1")(conv1)

        conv1 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv1)
        conv1 = PrintShape("conv1_2")(conv1)

        pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
        pool1 = PrintShape("pool1")(pool1)

        conv2 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool1)
        conv2 = PrintShape("conv2")(conv2)

        conv2 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv2)
        conv2 = PrintShape("conv2_2")(conv2)

        pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
        pool2 = PrintShape("pool2")(pool2)

        conv3 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool2)
        conv3 = PrintShape("conv3")(conv3)

        conv3 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv3)
        conv3 = PrintShape("conv3_2")(conv3)

        pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
        pool3 = PrintShape("pool3")(pool3)

        conv4 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool3)
        conv4 = PrintShape("conv4")(conv4)

        conv4 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv4)
        conv4 = PrintShape("conv4_2")(conv4)

        drop4 = Dropout(dropout)(conv4)
        drop4 = PrintShape("drop4")(drop4)

        # --- Bottleneck ---
        pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
        pool4 = PrintShape("pool4")(pool4)

        conv5 = Conv3D(16*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(pool4)
        conv5 = PrintShape("conv5")(conv5)

        compressLags = Conv3D(16*filters, (lags, 1, 1),
                              activation='relu', padding='valid')(conv5)
        compressLags = PrintShape("compressLags")(compressLags)

        conv5 = Conv3D(16*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(compressLags)
        conv5 = PrintShape("conv5_post")(conv5)

        drop5 = Dropout(dropout)(conv5)
        drop5 = PrintShape("drop5")(drop5)

        # --- Decoder ---
        up6 = Conv3D(8*filters, 2, activation='relu', padding='same',
                     kernel_initializer=kernel_init)(UpSampling3D(size=(1, 2, 2))(drop5))
        up6 = PrintShape("up6")(up6)

        compressLags6 = Conv3D(8*filters, (lags, 1, 1),
                               activation='relu', padding='valid')(drop4)
        compressLags6 = PrintShape("compressLags6")(compressLags6)

        merge6 = concatenate([compressLags6, up6], axis=-1)
        merge6 = PrintShape("merge6")(merge6)

        conv6 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge6)
        conv6 = PrintShape("conv6")(conv6)

        conv6 = Conv3D(8*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv6)
        conv6 = PrintShape("conv6_2")(conv6)

        up7 = UpSampling3D(size=(1, 2, 2))(conv6)
        up7 = Conv3D(4*filters, 2, activation='relu',
                     padding='same', kernel_initializer=kernel_init)(up7)
        up7 = PrintShape("up7")(up7)

        compressLags7 = Conv3D(4*filters, (lags, 1, 1),
                               activation='relu', padding='valid')(conv3)
        compressLags7 = PrintShape("compressLags7")(compressLags7)

        merge7 = concatenate([compressLags7, up7], axis=-1)
        merge7 = PrintShape("merge7")(merge7)

        conv7 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge7)
        conv7 = PrintShape("conv7")(conv7)

        conv7 = Conv3D(4*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv7)
        conv7 = PrintShape("conv7_2")(conv7)


        up8 = UpSampling3D(size=(1, 2, 2))(conv7)
        up8 = Conv3D(2*filters, 2, activation='relu',
                     padding='same', kernel_initializer=kernel_init)(up8)
        up8 = PrintShape("up8")(up8)

        compressLags8 = Conv3D(2*filters, (lags, 1, 1),
                               activation='relu', padding='valid')(conv2)
        compressLags8 = PrintShape("compressLags8")(compressLags8)


        merge8 = concatenate([compressLags8, up8], axis=-1)
        merge8 = PrintShape("merge8")(merge8)

        conv8 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge8)
        conv8 = PrintShape("conv8")(conv8)

        conv8 = Conv3D(2*filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv8)
        conv8 = PrintShape("conv8_2")(conv8)


        up9 = UpSampling3D(size=(1, 2, 2))(conv8)
        up9 = Conv3D(filters, 2, activation='relu',
                     padding='same', kernel_initializer=kernel_init)(up9)
        up9 = PrintShape("up9")(up9)

        compressLags9 = Conv3D(filters, (lags, 1, 1),
                               activation='relu', padding='valid')(conv1)
        compressLags9 = PrintShape("compressLags9")(compressLags9)


        merge9 = concatenate([compressLags9, up9], axis=-1)
        merge9 = PrintShape("merge9")(merge9)

        conv9 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(merge9)
        conv9 = PrintShape("conv9")(conv9)

        conv9 = Conv3D(filters, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv9)
        conv9 = PrintShape("conv9_2")(conv9)

        conv9 = Conv3D(2*features_output, 3, activation='relu',
                       padding='same', kernel_initializer=kernel_init)(conv9)
        conv9 = PrintShape("conv9_out")(conv9)


        conv10 = Conv3D(features_output, 1, activation='relu')(conv9)
        conv10 = PrintShape("conv10")(conv10)

        return conv10

    inputA = Input(shape=(lags, latitude, longitude, features))
    inputB = Input(shape=(lags, latitude, longitude, features))

    outA = UNet(inputA, lags, features_output, filters, dropout, kernel_init)
    outB = UNet(inputB, lags, features_output, filters, dropout, kernel_init)

    fusion = concatenate([outA, outB], axis=-1)
    fusion = PrintShape("fusion")(fusion)

    out = Conv3D(features_output, 1, activation='linear',
                 padding='same')(fusion)
    out = PrintShape("out_final")(out)

    return Model(inputs=[inputA, inputB], outputs=out)


# Paramètres du modèle
lags = 9
latitude = 149
longitude = 221
features = 29
features_output = 1
filters = 16
dropout = 0.1

def next_multiple_of_16(n):
    """
    Retourne le multiple de 16 le plus proche supérieur ou égal à n
    """
    if n % 16 == 0:
        return n
    return ((n // 16) + 1) * 16

print(next_multiple_of_16(latitude), next_multiple_of_16(longitude))

# Import du modèle (assure-toi que Late_WF_UNet est défini ou importé)
# from ton_module import Late_WF_UNet

# Création du modèle
# model = Late_WF_UNet(
#     lags=lags,
#     latitude=latitude,
#     longitude=longitude,
#     features=features,
#     features_output=features_output,
#     filters=filters,
#     dropout=dropout
# )
model = Late_WF_UNet(
    lags=lags,
    latitude=next_multiple_of_16(latitude),
    longitude=next_multiple_of_16(longitude),
    features=features,
    features_output=features_output,
    filters=filters,
    dropout=dropout
)

# Affiche le résumé du modèle
model.summary()

# Crée un batch de données aléatoires
batch_size = 2
# xA = np.random.rand(batch_size, lags, latitude, longitude,
#                     features).astype(np.float32)
# xB = np.random.rand(batch_size, lags, latitude, longitude,
#                     features).astype(np.float32)
xA = np.random.rand(batch_size, lags, next_multiple_of_16(latitude), next_multiple_of_16(longitude),
                    features).astype(np.float32)
xB = np.random.rand(batch_size, lags, next_multiple_of_16(latitude), next_multiple_of_16(longitude),
                    features).astype(np.float32)

# Passe les données dans le modèle
output = model([xA, xB])

# Affiche la shape de sortie
print("Output shape:", output.shape)

# Test de compilation
model.compile(optimizer='adam', loss='mse')

# Test d'une étape d'entraînement
# y_dummy = np.random.rand(batch_size, lags, latitude,
#                          longitude, features_output).astype(np.float32)
y_dummy = np.random.rand(batch_size, lags, next_multiple_of_16(latitude),
                         next_multiple_of_16(longitude), features_output).astype(np.float32)
loss = model.train_on_batch([xA, xB], y_dummy)
print("Loss:", loss)
