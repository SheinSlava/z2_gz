import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def my_model(num_classes=2, img_height=80, img_width=800):


    input_img = keras.Input(shape=(80, 800, 1))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mse'])

    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.Huber(),
    #               metrics=['mse'])

    return autoencoder


def model_fit(model, train_X, test_X, train_Y, test_Y, checkpoint_path, epochs=10):
    #     checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_path = "hys_res_1/cp-{epoch:04d}.ckpt"

    checkpoint_dir = os.path.dirname(checkpoint_path)


    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(train_X,
                        train_Y,
                        epochs=10,
                        validation_data=(test_X, test_Y),
                        validation_steps=30,
                        callbacks=[cp_callback])
    os.listdir(checkpoint_dir)
    return history


if __name__ == "__main__":

    model = my_model()
    print(model.summary())