import pathlib
import tensorflow as tf


def process_image_data(image_data_dir, batch_size=32, img_height=360, img_width=360):

    # data_dir = "/home/sheins/z2_gz/dataset/image"
    data_dir = pathlib.Path(image_data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds

if __name__ == "__main__":

    image_data_dir = "/home/sheins/z2_gz/dataset/image"

    train_ds, val_ds = process_image_data(image_data_dir)

    class_names = train_ds.class_names
    print(class_names)
