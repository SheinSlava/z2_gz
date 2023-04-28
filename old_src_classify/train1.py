
import argparse

from pyhocon import ConfigFactory
from pathlib import Path
import model

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Тренировочный скрипт модели определения тональности текстов на базе RuT5"
    )

    parser.add_argument('-c', '--config',
                        type=str,
                        default='configs/RuSentNE/rut5.conf',
                        help="Путь до файла конфигурации относительно корня проекта"
                        )

    parser.add_argument('-n', '--namespace',
                        type=str,
                        default='default',
                        help="Пространство имен конфигурации"
                        )

    parser.add_argument('-d', '--data',
                        type=str,
                        default='data/RuSentNE/interim/train_part_rut5_v0.csv',
                        help="Путь до тренировочного файла относительно корня проекта"
                        )

    parser.add_argument('-v', '--valid',
                        type=str,
                        default='data/RuSentNE/interim/valid_part_rut5_v0.csv',
                        help="Путь до валидационного файла относительно корня проекта"
                        )

    parser.add_argument('-l', '--local',
                        action='store_true',
                        default=False,
                        help='Использование путей относительно корня проекта'
                        )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()

    project_path = Path(__file__).parents[3]

    config_path = project_path.joinpath(arguments.config)
    config = ConfigFactory.parse_file(config_path)[arguments.namespace]

    if arguments.local:
        train_data_path = project_path.joinpath(arguments.data)
        valid_data_path = project_path.joinpath(arguments.valid)
    else:
        train_data_path = arguments.data
        valid_data_path = arguments.valid

    import os
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config["mlflow_experiment_name"]

    t5_model = RuT5SentimentModel(config, config["pretrained_dir"])
    t5_model.fit(train_data_path, valid_data_path)