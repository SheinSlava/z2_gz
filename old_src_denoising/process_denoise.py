import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



def processing_mel_denoise(data_path):

    path_cl = data_path + '/clean/20/*'
    path_no = data_path + '/noisy/20/*'

    # print(path_cl)

    x_noisy = []
    for i in glob.glob(path_no):
        # print(i)
        arr_item = np.load(i)
        plt.imsave('1.jpg', arr_item.T)
        im = Image.open('1.jpg')
        im = im.resize((800, 80), Image.LANCZOS)
        im = np.array(im)
        x_noisy.append(im)

    y_clear = []
    for i in glob.glob(path_cl):
        # print(i)
        arr_item = np.load(i)
        plt.imsave('1.jpg', arr_item.T)
        im = Image.open('1.jpg')
        im = im.resize((800, 80), Image.LANCZOS)
        im = np.array(im)
        y_clear.append(im)




    x_data = x_noisy
    x_data = np.array(x_data)
    np.save('x_no.npy', x_data)
    y_data = y_clear
    y_data = np.array(y_data)
    np.save('y_cl.npy', y_data)

    return x_data, y_data

if __name__ == "__main__":
    INPUT_DIR = '/home/sheins/z2_gz/dataset/train/train'
    x, y = processing_mel_denoise(INPUT_DIR)
    print(len(x))
    for i in range(len(x)):
        print(x[i].shape)

    print(type(x))
    print(x.shape)

    print(type(y))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print(len(X_train))
    print(len(y_train))
