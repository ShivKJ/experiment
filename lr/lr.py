import tensorflow as tf
from numpy import array, linspace
from numpy.random import random
from streamAPI.utility import csv_itr


def generate_batch(n: int):
    x_batch = linspace(-1, 1, n)
    y_batch = 2 * x_batch + random(x_batch.shape) * 0.3
    return x_batch, y_batch


def vehicle_data():
    itr = csv_itr('data/refined_mile_per_gallon.csv')
    data = []

    for doc in itr:
        data.append((
            int(doc['cylinders']),
            float(doc['displacement']),
            int(doc['horsepower']),
            int(doc['weight']),
            float(doc['acceleration']),
            float(doc['mpg'])
        ))

    data = array(data)
    size = 200
    x_train, y_train = data[:200, :-1], data[:200, -1].reshape((size, 1))
    x_test, y_test = data[200:, :-1], data[200:, -1].reshape((len(data) - size, 1))

    return x_train, y_train, x_test, y_test


def linear_regression(alpha):
    with tf.name_scope('Graph') as params:
        feature_size = 5
        output_size = 1

        x = tf.placeholder(dtype=tf.float64, shape=[None, feature_size], name='x')
        y = tf.placeholder(dtype=tf.float64, shape=[None, output_size], name='y')

        w = tf.Variable(tf.random_normal([feature_size, output_size], dtype=tf.float64),
                        name='W', dtype=tf.float64)

        b = tf.Variable(tf.random_normal([output_size], dtype=tf.float64),
                        name='b', dtype=tf.float64)

        y_pred = x @ w + b

        # cost = tf.reduce_mean(tf.square(y_pred - y)) + alpha * tf.reduce_mean(w * w)
        cost = tf.reduce_mean(tf.square(y_pred - y)) + alpha * tf.reduce_mean(abs(w))
        # cost = tf.reduce_mean(tf.square(y_pred - y)) + alpha * (tf.transpose(w) @ w)

        return x, y, y_pred, cost


def run():
    x_batch, y_batch, x_test, y_test = vehicle_data()
    x, y, y_pred, cost = linear_regression(0.001)

    learning = 0.1

    optimizer = tf.train.AdamOptimizer(learning_rate=learning).minimize(cost)
    init = tf.global_variables_initializer()

    writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(50000):
            _, batch_cost = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
            print(epoch, batch_cost)
        y_t = sess.run(cost, {x: x_test, y: y_test})

        print(y_t)


if __name__ == '__main__':
    # vehicle_data()
    # linear_regression()
    run()
