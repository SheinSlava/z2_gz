import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import argparse


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


if __name__ == "__main__":

    arguments = parse_arguments()
    mel_diagram_dir = arguments.file_dir

    item = np.load(mel_diagram_dir)
    plt.imshow(item.T)
    plt.axis('off')
    plt.savefig('predict.png', bbox_inches='tight')

    img_height = 360
    img_width = 360
    class_names = ['clean', 'noisy']

    img = tf.keras.utils.load_img(
        'predict.png', target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    model = tf.keras.models.load_model('/home/sheins/z2_gz/models/my_model1.h5')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )