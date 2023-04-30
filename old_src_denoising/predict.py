import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":

    mel_dir_no = '/home/sheins/z2_gz/dataset/val/val/noisy/82/*'
    mel_dir_cl = '/home/sheins/z2_gz/dataset/val/val/clean/82/*'


    model_dir = '/old_src_denoising/models/my_model_denoise_1.h5'

    x_noisy = []
    for i in glob.glob(mel_dir_no):
        arr_item = np.load(i)
        arr_item = arr_item.T
        plt.imsave('1.jpg', arr_item)

        im = Image.open('1.jpg')
        im = im.resize((800, 80), Image.LANCZOS)
        im = np.array(im)

        x_noisy.append(im)

    x_noisy = np.array(x_noisy)
    print(x_noisy.shape)


    x_clear = []
    for i in glob.glob(mel_dir_cl):
        arr_item = np.load(i)
        arr_item = arr_item.T
        plt.imsave('1.jpg', arr_item)

        im = Image.open('1.jpg')
        im = im.resize((800, 80), Image.LANCZOS)
        im = np.array(im)

        x_clear.append(im)

    x_clear = np.array(x_clear)
    print(x_clear.shape)

    model = tf.keras.models.load_model(model_dir)
    clear_pred = model.predict(x_noisy)

    print(clear_pred.shape)

    from sklearn.metrics import f1_score
    f1 = f1_score(x_clear, clear_pred, average='macro')
    print(f1)

    # res_mse = mean_squared_error(x_clear, clear_pred)
    # print(res_mse)