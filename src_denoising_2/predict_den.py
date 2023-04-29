import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error


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
    true = true[0]
    true = np.squeeze(true, axis=-1)

    pred = pred[0]
    pred = np.squeeze(pred, axis=-1)

    res_pred = mean_squared_error(true, pred)
    return res_pred

if __name__ == "__main__":


    mel_dir_cl_1 = '/s/ls4/users/slava1195/z2_gz/dataset/train/train/clean/20/20_5360_20-5360-0059.npy'
    mel_dir_no = '/s/ls4/users/slava1195/z2_gz/dataset/train/train/noisy/20/20_5360_20-5360-0059.npy'

    model_dir = '/s/ls4/users/slava1195/z2_gz/src_denoising_2/models/my_model_denoise_6.h5'

    proc_mel_cl = processing_funnc(mel_dir_cl_1)
    print(proc_mel_cl.shape)

    proc_mel_no = processing_funnc(mel_dir_no)
    print(proc_mel_no.shape)

    model = tf.keras.models.load_model(model_dir)
    predict = model.predict(proc_mel_no)

    mse_e = mse_evaluate(proc_mel_cl, proc_mel_no)
    print("MSE: ", mse_e)