import pickle as pkl
from os.path import join

import numpy as np
from PIL import Image
from streamAPI.utility import csv_itr


def load_mnist_data(mnist_dir):
    """
    reads data from mnist_dir.
    :param mnist_dir:
    :return:
    """
    x, y = [], []

    def load_data(file):
        for doc in csv_itr(join(mnist_dir, file)):
            with Image.open(join(mnist_dir, doc['file'])) as img:
                x.append(np.array(img))

                label = np.zeros(10)
                _class_label = int(doc['class'])
                label[_class_label] = 1

                y.append(label)

    load_data('test-labels.csv')
    load_data('train-labels.csv')

    return np.array(x), np.array(y)


def create_pickle(mnist_dir, pkl_file):
    x, y = load_mnist_data(mnist_dir)

    with open(pkl_file, 'wb') as f:
        pkl.dump(dict(x=x, y=y), f)


if __name__ == '__main__':
    # download data from https://drive.google.com/open?id=1ULv8fv58DqZUKz7b7NdP-tQ05m74CkiV&authuser=0
    # put it in "ann" folder after extracting
    
    create_pickle('mnist', 'data.pkl')
