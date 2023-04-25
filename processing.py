import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.model_selection import train_test_split

def get_arr_data(data_path):

    path_cl = data_path + '/clean/20/*'
    path_no = data_path + '/noisy/20/*'

    x_clear = []
    for i in glob.glob(path_cl):
        x_clear.append(np.load(i))
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

    dataset_path = '/home/sheins/z2_gz/dataset/train/train'

    x, y = get_arr_data(dataset_path)
    print(x[2])
    print(type(x))
    print(type(x[0]))

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    # print(X_train)
    # print(y_train)
    # print(X_train.shape)
    # print(y_train.shape)