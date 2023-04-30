from predict import processing_spectogramm
import tensorflow as tf
import glob
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Тренировочный скрипт модели для классификации спектрограмм "
    )

    parser.add_argument('-m', '--model_dir',
                        type=str,
                        default='/home/sheins/z2_gz/src_class_2/models/my_model3.h5',
                        help="Путь до файла"
                        )

    parser.add_argument('-d', '--data_dir',
                        type=str,
                        default='/home/sheins/z2_gz/dataset/val/val',
                        help="Путь до файла"
                        )

    return parser.parse_args()

if __name__ == "__main__":

    arguments = parse_arguments()

    model_dir = arguments.model_dir
    data_dir = arguments.data_dir

    # model_dir = '/home/sheins/z2_gz/src_class_2/models/my_model3.h5'
    model = tf.keras.models.load_model(model_dir)
    print(model.summary())

    # clear_dir = '/home/sheins/z2_gz/dataset/val/val/clean/*/*'
    clear_dir = data_dir + '/clean/*/*'
    clear_mel = processing_spectogramm(clear_dir)
    print(len(clear_mel))

    # noise_dir = '/home/sheins/z2_gz/dataset/val/val/noisy/*/*'
    noise_dir = data_dir + '/noisy/*/*'
    noisy_mel = processing_spectogramm(noise_dir)
    print(len(noisy_mel))

    y_clear = [0] * len(clear_mel)
    y_noise = [1] * len(noisy_mel)
    y = y_clear + y_noise

    class_names = [0, 1]

    prediction_cl = model.predict(clear_mel)
    results_cl = []
    for i in prediction_cl:
        score = tf.nn.softmax(i)
        score_value_cl = class_names[np.argmax(score)]
        results_cl.append(score_value_cl)

    predictions_no = model.predict(noisy_mel)
    results_no = []
    for i in predictions_no:
        score = tf.nn.softmax(i)
        score_value_no = class_names[np.argmax(score)]
        results_no.append(score_value_no)

    all_results = results_cl + results_no
    print("Accuracy: ", accuracy_score(y, all_results))
    print("F1 MACRO: ", f1_score(y, all_results, average='macro'))
    # print("F1 MICRO: ", f1_score(all_results, y, average='micro'))
    # print("F1: ", f1_score(all_results, y, average='macro'))