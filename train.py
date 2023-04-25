import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import my_model, model_fit
from processing import get_arr_data

import matplotlib.pyplot as plt

import sys

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

if __name__ == "__main__":

    dataset_path = '/home/sheins/z2_gz/dataset/train/train'

    x, y = get_arr_data(dataset_path)
    print(x[2])
    print(type(x))
    print(type(x[0]))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    model = my_model()
    print(model)
    # hys = model_fit(model, X_train, X_test, y_train, y_test)
    #
    #
    # model.save("./model_test5/my_model5.h5")
    #
    # history = hys
    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # plot_graphs(history, 'accuracy')
    # plt.ylim(None, 1)
    # plt.subplot(1, 2, 2)
    # plot_graphs(history, 'loss')
    # plt.ylim(0, None)