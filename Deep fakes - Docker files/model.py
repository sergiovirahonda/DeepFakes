from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np


def autoencoder_model():
    #Making encoder:

    input_img = layers.Input(shape=(120, 120, 3))
    x = layers.Conv2D(256,kernel_size=5, strides=2, padding='same',activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(512,kernel_size=5, strides=2, padding='same',activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(1024,kernel_size=5, strides=2, padding='same',activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(9216)(x)
    encoded = layers.Reshape((3,3,1024))(x)

    encoder = Model(input_img, encoded,name="encoder")
    encoder.summary()

    #Making decoder:
    decoder_input= layers.Input(shape=((3,3,1024)))
    x = layers.Conv2D(1024,kernel_size=5, strides=2, padding='same',activation='relu')(decoder_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(512,kernel_size=5, strides=2, padding='same',activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256,kernel_size=5, strides=2, padding='same',activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(np.prod((120, 120, 3)))(x)
    decoded = layers.Reshape((120, 120, 3))(x)

    decoder = Model(decoder_input, decoded,name="decoder")
    decoder.summary()

    #Making autoencoder
    auto_input = layers.Input(shape=(120,120,3))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    autoencoder = Model(auto_input, decoded,name="autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss='mae')
    #autoencoder.compile(optimizer=optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss='mse')
    autoencoder.summary()

    return autoencoder
