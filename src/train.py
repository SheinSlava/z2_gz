from sklearn.model_selection import train_test_split
from hlam.processing import get_arr_data

import matplotlib.pyplot as plt
from model import my_model, model_fit
from process import process_image_data


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

if __name__ == "__main__":

    image_data_dir = "/home/sheins/z2_gz/dataset/image"
    train_ds, val_ds = process_image_data(image_data_dir)
    print(train_ds)
    model = my_model()
    print(model)
    hys = model_fit(model, train_ds, val_ds)

    model.save("./model_test1/my_model1.h5")

    history = hys
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)


###########################################



