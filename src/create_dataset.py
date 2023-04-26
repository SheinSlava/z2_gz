from matplotlib import pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def create_image_dataset(train_data_dir, save_dir):

    path_cl = '/clear'
    path_no = '/noisy'

    for i in glob.glob(train_data_dir+path_cl+'/*/*'):
        file_stem = Path(i).stem
        save_path = save_dir + path_cl + '/' + file_stem
        item = np.load(i)
        plt.imshow(item.T)
        plt.axis('off')
        plt.savefig(f'{save_path}.jpg', bbox_inches='tight')

    for i in glob.glob(train_data_dir+path_no+'/*/*'):
        file_stem = Path(i).stem
        save_path = save_dir + path_no + '/' + file_stem
        item = np.load(i)
        plt.imshow(item.T)
        plt.axis('off')
        plt.savefig(f'{save_path}.jpg', bbox_inches='tight')



if __name__ == "__main__":

    INPUT_DIR = '~/z2_gz/dataset/train/train'
    OUTPUT_DIR = '~/z2_gz/dataset/image_data'

    create_image_dataset(INPUT_DIR, OUTPUT_DIR)


