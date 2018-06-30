from functools import partial
from typing import List, Sequence, Tuple

import tensorflow as tf
from PIL import Image
from numpy import array, ndarray, zeros
from streamAPI.stream import ParallelStream, Stream
from streamAPI.utility import (csv_itr, execution_time, files_inside_dir,
                               get_file_name)


class Data:
    def __init__(self, file: str, clazz: int):
        self.file = file

        with Image.open(file) as f:
            self.input = array(f).ravel()

        self.clazz = Data._label(clazz)

    @classmethod
    def create_data(cls, _clazz: dict, file: str):
        return Data(file, _clazz[get_file_name(file)])

    def feature(self) -> ndarray:
        return self.input

    def label(self):
        return self.clazz

    @staticmethod
    def _label(label: int) -> ndarray:
        q = zeros(10)  # 0 - 9 digits
        q[label] = 1

        return q

    def __str__(self):
        return get_file_name(self.file) + ' -> ' + str(self.clazz)


@execution_time()
def get_data(training: bool) -> List[Data]:
    if training:
        label_file = '../mnist/train-labels.csv'
        image_dir = '../mnist/train-images'
    else:
        label_file = '../mnist/test-labels.csv'
        image_dir = '../mnist/test-images'

    key_mapper = lambda doc: get_file_name(doc['file'])
    value_mapper = lambda doc: int(doc['class'])

    _clazz = Stream(csv_itr(label_file)).mapping(key_mapper, value_mapper)

    worker = 8
    multiprocessing = True
    dispatch_size = 1000

    return (files_inside_dir(image_dir,
                             as_type=lambda x: ParallelStream(x, worker, multiprocessing))
            .batch_processor(partial(Data.create_data, _clazz), dispatch_size)
            .as_seq())


@execution_time(prefix='ANN')
def main():
    training_data = get_data(training=True)
    testing_data = get_data(training=False)

    feature_size = 28 * 28
    hidden_nu = 200
    output_size = 10

    x = tf.placeholder(tf.float32, [None, feature_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')

    W1 = tf.Variable(tf.random_normal([feature_size, hidden_nu]), name='W1')
    b1 = tf.Variable(tf.random_normal([hidden_nu]), name='b1')

    W2 = tf.Variable(tf.random_normal([hidden_nu, output_size]), name='W2')
    b2 = tf.Variable(tf.random_normal([output_size]), name='b2')

    L0_L1 = x @ W1 + b1
    L1_L1 = tf.nn.relu(L0_L1)

    L1_L2 = L1_L1 @ W2 + b2
    L2_L2 = tf.nn.softmax(L1_L2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=L2_L2, labels=y),
                          name='cost')
    optimization = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, name='optimization')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./mnist/visualize', graph=sess.graph)
        sess.run(init)

        def img_clazz(_data: Sequence[Data]):
            img = array([pt.feature() for pt in _data])
            clazz = array([pt.label() for pt in _data])

            return img, clazz

        epochs = 100
        batch_size = 1000

        # ---------------------------------------------------------------------------------
        def process_batch(_chunk: Tuple[Data, ...]):
            img, clazz = img_clazz(_chunk)

            _, c = sess.run([optimization, cost], feed_dict={x: img, y: clazz})

            return c / batch_size

        def process_training_data():
            return (Stream(training_data)
                    .batch(batch_size)
                    .map(process_batch)
                    .reduce(lambda x, y: x + y)
                    .get())

        # -----------------------------------------------------------------------------------

        (Stream.from_supplier(process_training_data)
         .enumerate(start=1)
         .limit(epochs)
         .for_each(print))

        pred_temp = tf.equal(tf.argmax(L2_L2, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, dtype=tf.float32))

        img, clazz = img_clazz(testing_data)

        print(accuracy.eval({x: img, y: clazz}))


if __name__ == '__main__':
    main()
