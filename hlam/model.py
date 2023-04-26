import tensorflow as tf
from keras.models import Sequential
import os
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer


def my_model():
    model = Sequential([

        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') # 3 classes

    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def model_fit(model, train_X, test_X, train_Y, test_Y):
    #     checkpoint_path = "training_1/cp.ckpt"
    checkpoint_path = "model_test_1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(train_X,
                        train_Y,
                        epochs=150,
                        validation_data=(test_X, test_Y),
                        validation_steps=30,
                        callbacks=[cp_callback])
    os.listdir(checkpoint_dir)
    return history
