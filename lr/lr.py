import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def generate_batch(n: int):
    x_batch = np.linspace(-1, 1, n)
    y_batch = 2 * x_batch + np.random.random(x_batch.shape) * 0.3
    return x_batch, y_batch


def linear_regression():
    with tf.name_scope('Graph') as params:
        x = tf.placeholder(dtype=tf.float32, shape=[None], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        w = tf.Variable(np.random.normal(), name='W')
        b = tf.Variable(np.random.normal(), name='b')

        y_pred = w * x + b

        cost = tf.reduce_mean(tf.square(y_pred - y))

        return x, y, y_pred, cost


def run():
    x_batch, y_batch = generate_batch(1000)
    x, y, y_pred, cost = linear_regression()
    learning = 0.1

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning).minimize(cost)
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1000):
            _, batch_cost = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
            print(epoch, batch_cost)

        y_t = sess.run(y_pred, {x: x_batch})

        plt.scatter(x_batch, y_batch)
        plt.plot(x_batch, y_t)

        plt.show()


if __name__ == '__main__':
    run()
