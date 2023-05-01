import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Тренировочный скрипт модели для классификации спектрограмм "
    )

    parser.add_argument('-m', '--model_dir',
                        type=str,
                        default='models/my_model_denoise_7.h5',
                        help="Путь до файла модели"
                        )

    parser.add_argument('-d', '--data_dir',
                        type=str,
                        default='/home/sheins/z2_gz/dataset/val/val',
                        help="Путь до папок clean и noise"
                        )

    return parser.parse_args()

def processing_funnc(mel_path):
    list_mels = []
    for i in glob.glob(mel_path):
        arr_item = np.load(i)
        arr_item = arr_item.T
        arr_item = arr_item*1000
        arr_item = tf.keras.preprocessing.sequence.pad_sequences(arr_item, maxlen=800)
        arr_item = np.expand_dims(arr_item, axis=-1)
        list_mels.append(arr_item)
    list_mels = np.array(list_mels)
    list_mels = list_mels/1000
    return list_mels

def mse_evaluate(true, pred):
    # true = true[0]
    true = np.squeeze(true, axis=-1)

    # pred = pred[0]
    pred = np.squeeze(pred, axis=-1)

    res_pred = mean_squared_error(true, pred)
    return res_pred

if __name__ == "__main__":

    arguments = parse_arguments()

    model_dir = arguments.model_dir
    data_dir = arguments.data_dir

    mel_dir_cl = data_dir + '/clean/*/*'
    mel_dir_no = data_dir + '/noisy/*/*'

    print(mel_dir_cl)

    proc_mel_cl = processing_funnc(mel_dir_cl)
    print(proc_mel_cl.shape)

    proc_mel_no = processing_funnc(mel_dir_no)
    print(proc_mel_no.shape)

    model = tf.keras.models.load_model(model_dir)
    predict = model.predict(proc_mel_no)

    print(predict.shape)
    plt.imshow(predict[0])

    all_mse_e = list()
    all_mse_e_pred = list()

    for i in range(len(predict)):
        # print(predict[i].shape)
        mse_e = mse_evaluate(proc_mel_cl[i], proc_mel_no[i])
        all_mse_e.append(mse_e)
        print("MSE actual (clear and noisy): ", mse_e)
        mse_e_pred = mse_evaluate(proc_mel_cl[i], predict[i])
        all_mse_e_pred.append(mse_e_pred)
        print("MSE predict (clear and denoise): ", mse_e_pred)
        print('___________________________________________')

    print("ALL RESULTS")
    mean_actual = sum(all_mse_e)/len(all_mse_e)
    print("MSE mean actual : ", mean_actual)
    mean_pred = sum(all_mse_e_pred)/len(all_mse_e_pred)
    print("MSE mean predict : ", mean_pred)