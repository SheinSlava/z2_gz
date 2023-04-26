import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab
import glob
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_arr_data(data_path):

    path_cl = data_path + '/clean/20/*'
    path_no = data_path + '/noisy/20/*'

    if not os.path.exists(os.path.join(data_path+'/audio-images', 'clean')):
        os.mkdir(os.path.join(data_path, 'clean'))

    x_clear = []
    for i in glob.glob(path_cl):
        item = np.load(i)
        plt.imshow(item)
        plt.savefig('123.png')
        # item = item.T
        # item = tf.keras.utils.pad_sequences(item, maxlen=10)
        x_clear.append(item)

    x_clear = np.array(x_clear, dtype=object)
    print(type(x_clear))
    print(x_clear[1])
    print(x_clear[1].shape)


    x_clear = np.empty(len(glob.glob(path_cl)))
    ddd = glob.glob(path_cl)
    for i in range(len(ddd)):
        a = np.load(ddd[i])
        print(a)
        # x_clear.append(np.load(i))
        x_clear[i] = np.load(ddd[i])



    print("RAB 1")
    y_clear = [0] * len(x_clear)
    # print(y_clear)
    print(len(y_clear))

    x_noisy = []
    for i in glob.glob(path_no):
        # print(i)
        x_noisy.append(np.load(i))
    print("RAB 2")
    y_noisy = [1] * len(x_noisy)
    # print(y_noisy)
    print(len(y_noisy))

    x_data_list = x_clear + x_noisy
    x_data = np.array(x_data_list, dtype=object)
    y_data_list = y_clear + y_noisy
    y_data = np.array(y_data_list, dtype=object)

    return x_data, y_data

# def get_train_test_data(data_path):
#
#     return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    INPUT_DIR = '/dataset/train/train'
    OUTPUT_DIR = '/dataset/'
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):
        os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))


    x, y = get_arr_data(OUTPUT_DIR)
    print(x[2])
    print(type(x))
    print(type(x[0]))

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    # print(X_train)
    # print(y_train)
    # print(X_train.shape)
    # print(y_train.shape)