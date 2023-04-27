from sklearn.model_selection import train_test_split
from process import processing_mel
from model import my_model, model_fit
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

if __name__ == "__main__":

    INPUT_DIR = '/home/sheins/z2_gz/dataset/train/train'
    checkpoint_path = "hys_res/cp-{epoch:04d}.ckpt"
    save_path = "models/my_model2.h5"


    x, y = processing_mel(INPUT_DIR)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    model = my_model()
    print(model)
    hys = model_fit(model, X_train, X_test, y_train, y_test, checkpoint_path=checkpoint_path)

    model.save(save_path)

    history = hys
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)