import argparse
import matplotlib.pyplot as plt
from model import my_model, model_fit
from process import process_image_data
from pathlib import Path
import os


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Тренировочный скрипт модели для классификации спектрограмм "
    )

    # parser.add_argument('-c', '--config',
    #                     type=str,
    #                     default='configs/RuSentNE/rut5.conf',
    #                     help="Путь до файла конфигурации относительно корня проекта"
    #                     )

    # parser.add_argument('-n', '--namespace',
    #                     type=str,
    #                     default='default',
    #                     help="Пространство имен конфигурации"
    #                     )

    parser.add_argument('-d', '--data',
                        type=str,
                        default='/home/sheins/z2_gz/dataset/image',
                        help="Путь до тренировочного файла относительно корня проекта"
                        )

    # parser.add_argument('-v', '--valid',
    #                     type=str,
    #                     default='data/RuSentNE/interim/valid_part_rut5_v0.csv',
    #                     help="Путь до валидационного файла относительно корня проекта"
    #                     )
    #
    # parser.add_argument('-l', '--local',
    #                     action='store_true',
    #                     default=False,
    #                     help='Использование путей относительно корня проекта'
    #                     )

    return parser.parse_args()

if __name__ == "__main__":


    arguments = parse_arguments()
    # project_path = Path(__file__).parents[2]
    # project_path = '~/z2_gz'
    # print("!!!!", project_path)
    #
    # image_data_dir = project_path.joinpath(arguments.data)

    image_data_dir = arguments.data
    print("!!!!", image_data_dir)


    train_ds, val_ds = process_image_data(image_data_dir)
    print(train_ds)
    model = my_model()
    print(model)
    hys = model_fit(model, train_ds, val_ds)

    model.save("~/z2_gz/models/my_model1.h5")

    history = hys
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)


###########################################



