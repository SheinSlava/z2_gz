import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import argparse
from PIL import Image
import glob


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Тренировочный скрипт модели для классификации спектрограмм "
    )

    parser.add_argument('-f', '--file_dir',
                        type=str,
                        default='/home/sheins/z2_gz/dataset/val/val/clean/5765/5765_231845_5765-231845-0029.npy',
                        help="Путь до файла"
                        )

    return parser.parse_args()


def processing_spectogramm(pred_dir):
    res = []
    for i in glob.glob(pred_dir):
        arr_item = np.load(i)
        plt.imsave('pred.jpg', arr_item.T)
        im = Image.open('pred.jpg')
        im = im.resize((800, 80), Image.LANCZOS)
        im = np.array(im)
        res.append(im)
    res = np.array(res)
    return res


if __name__ == "__main__":

    model_dir = 'models/my_model3.h5'

    arguments = parse_arguments()
    mel_diagram_dir = arguments.file_dir

    process_mel = processing_spectogramm(mel_diagram_dir)

    model = tf.keras.models.load_model(model_dir)
    predictions = model.predict(process_mel)
    class_names = ['clear', 'noisy']

    print("Results: ", predictions)

    for i in range(len(predictions)):
        score = tf.nn.softmax(predictions[i])
        print(
            "This diagramm most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
