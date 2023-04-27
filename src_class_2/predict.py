import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import argparse
from PIL import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Тренировочный скрипт модели для классификации спектрограмм "
    )

    parser.add_argument('-f', '--file_dir',
                        type=str,
                        default='/home/sheins/z2_gz/dataset/train/train/noisy/161/161_121897_161-121897-0041.npy',
                        help="Путь до файла"
                        )

    return parser.parse_args()
def processing_spectogramm(pred_dir):
    res = []
    arr_item = np.load(pred_dir)
    plt.imsave('pred.jpg', arr_item.T)
    im = Image.open('pred.jpg')
    im = im.resize((800, 80), Image.LANCZOS)
    im = np.array(im)
    res.append(im)
    res = np.array(res)
    return res

if __name__ == "__main__":

    model_dir = '/home/sheins/z2_gz/src_class_2/models/my_model2.h5'


    arguments = parse_arguments()
    mel_diagram_dir = arguments.file_dir

    process_mel = processing_spectogramm(mel_diagram_dir)


    model = tf.keras.models.load_model(model_dir)
    predictions = model.predict(process_mel)
    score = tf.nn.softmax(predictions[0])

    print("Results: ", predictions)

    class_names = ['clear', 'noisy']
    print(
        "This diagramm most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
