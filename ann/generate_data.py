import pickle as pkl
from csv import DictReader
from functools import partial
from os.path import join

import numpy as np
from PIL import Image


def load_file(mnist_dir, doc):
    with Image.open(join(mnist_dir, doc['file'])) as img:
        label = np.zeros(10)
        _class_label = int(doc['class'])
        label[_class_label] = 1

        return np.array(img), label


def load_mnist_data(mnist_dir, parallel=False):
    """
    reads data from mnist_dir.
    :param mnist_dir:
    :param parallel
    :return:
    """
    x, y = [], []

    _load_file = partial(load_file, mnist_dir)

    def load_data(file):
        with open(join(mnist_dir, file)) as f:
            reader = DictReader(f)

            if parallel:
                from streamAPI.stream import ParallelStream
                itr = ParallelStream(reader).batch_processor(_load_file, 1000)
            else:
                itr = (_load_file(doc) for doc in reader)

            for data, label in itr:
                x.append(data)
                y.append(label)

    load_data('test-labels.csv')
    load_data('train-labels.csv')

    return np.array(x), np.array(y)


def create_pickle(mnist_dir, pkl_file, parallel=False):
    x, y = load_mnist_data(mnist_dir, parallel=parallel)

    with open(pkl_file, 'wb') as f:
        pkl.dump(dict(x=x, y=y), f)


if __name__ == '__main__':
    # download data from https://drive.google.com/open?id=1ULv8fv58DqZUKz7b7NdP-tQ05m74CkiV&authuser=0
    # put it in "ann" folder after extracting

    create_pickle('mnist', 'data.pkl')
