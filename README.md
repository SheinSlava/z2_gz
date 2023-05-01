# z2_gz

# Данный репозиторий посвящен задаче классификации и фильтрации спектограмм аудио файлов.

## Описание проекта:
- 'hlam' - папка с первыми тестовыми вариантами;

- 'old_src_classify' - папка с первыми тестовым вариантом классификации (не рабочий вариант)
- 'old_src_denoising' - папка с первыми тестовым вариантом классификации (не рабочий вариант)


#### - 'src_class_2' - папка с решением задачи классификации
    - 'models' - папка с моделью
      - 'download_model_classify.sh' - файл для скачивания модели классификации
    - 'model.py' - файл с моделью
    - 'process.py' - файл с предобработкой датасета
    - 'train.py' - скрипт с обучением 
    - 'predict.py' - скрипт с получением предсказанием
    - 'evaluation.py' - скрипт с оценкой точности для проверочного набора данных 
    - 'README.md' - инструкция для использования

#### - 'src_denoising_2' - папка с решением задачи фильтрации
    - 'models' - папка с моделью
      - 'download_model_denoise.sh' - файл для скачивания модели фильтрации
    - 'model.py' - файл с моделью (не используется)
    - 'unn_model.py' - файл с моделью (используется)
    - 'process_den.py' - файл с предобработкой датасета
    - 'train_den.py' - скрипт с обучением 
    - 'predict_den.py' - скрипт с получением предсказанием с оценкой точности для проверочного набора данных
    - 'README.md' - инструкция для использования

### Файл для создания окружения: 
  - gz_env.yml


# TODO:
- [X] Реализовать заготовку
- [X] Добавить arg parser
- [X] Написать скрипты для запуска
- [X] Добавить скрпт на скачивание модели
- [ ] Добавить CONFIG file
- [ ] Добавить нормальные sh скрипты для запусков
- [ ] Добавить корректное взаимодейтсвие с путями, проверки и т.д.
- [ ] Добавить MakeFile для ведения экспериментов и тд
