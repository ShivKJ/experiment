import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from tensorflow import (Session, Variable, argmax, cast, equal, float32, global_variables_initializer, nn, placeholder,
                        random_normal, reduce_mean, summary, train)


def main(x, y, training_fraction=0.80,
         learning_rate=0.001,
         epochs=1000, batch_size=1000, print_at=100):
    """
    :param x: shape = m * 786
    :param y: shape = m * 10
    :param training_fraction:
    :param epochs:
    :param batch_size:
    :param print_at:
    :return:
    """
    training_size = int(len(x) * training_fraction)

    # if last batch size is less than half of desired batch size then throwing exception.
    # In future, instead of throwing exception we may avoid using this last batch.

    assert training_size % batch_size == 0 or training_size % batch_size > batch_size / 2
    last_batch_size = training_size % batch_size

    _data = train_test_split(x, y, train_size=training_fraction, stratify=y.argmax(1), random_state=0)

    # training_data_x, training_data_y = x[:training_size], y[:training_size]
    # testing_data_x, testing_data_y = x[training_size:], y[training_size:]

    training_data_x, training_data_y = _data[0], _data[2]
    testing_data_x, testing_data_y = _data[1], _data[3]

    feature_size = training_data_x.shape[1]
    hidden_nu = 20
    output_size = training_data_y.shape[1]

    x = placeholder(float32, [None, feature_size], name='x')
    y = placeholder(float32, [None, output_size], name='y')

    # also check xavier_initializer
    W1 = Variable(random_normal([feature_size, hidden_nu], seed=1, dtype=float32), name='W1')
    b1 = Variable(random_normal([hidden_nu], dtype=float32, seed=2), name='b1')  # use zeros also

    W2 = Variable(random_normal([hidden_nu, output_size], seed=3, dtype=float32), name='W2')
    b2 = Variable(random_normal([output_size], dtype=float32, seed=4), name='b2')

    L0_L1 = x @ W1 + b1
    L1_L1 = nn.relu(L0_L1)

    L1_L2 = L1_L1 @ W2 + b2
    L2_L2 = nn.softmax(L1_L2)

    cost = reduce_mean(nn.softmax_cross_entropy_with_logits_v2(logits=L2_L2, labels=y),
                       name='cost')

    optimization = train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name='optimization')

    init = global_variables_initializer()

    currect_predictions = equal(argmax(L2_L2, axis=1), argmax(y, axis=1))

    accuracy = reduce_mean(cast(currect_predictions, float32))

    with Session() as sess:
        writer = summary.FileWriter('mnist/visualize', graph=sess.graph)
        sess.run(init)

        _cost_array = []
        _training_accuracy_array = []
        _testing_accuracy_array = []
        # ---------------------------------------------------------------------------------
        for e in range(epochs):

            _idx = RandomState(e).permutation(training_size)  # check how much does it matter to add
            # uniformity of data in each batch.

            total_cost = 0

            def mini_batch(start_idx, end_idx):
                curr_idx = _idx[start_idx:end_idx]

                _x = training_data_x[curr_idx]
                _y = training_data_y[curr_idx]

                _, c = sess.run([optimization, cost],
                                feed_dict={x: _x, y: _y})

                return (end_idx - start_idx) * c

            for i in range(0, training_size, batch_size):
                total_cost += mini_batch(i, i + batch_size)

            if last_batch_size != 0:
                total_cost += mini_batch(training_size - last_batch_size, training_size)

            mean_cost = round(total_cost / training_size, 3)

            _training_accuracy = round(100 * sess.run(accuracy, feed_dict={x: training_data_x,
                                                                           y: training_data_y}), 2)
            _testing_accuracy = round(100 * sess.run(accuracy, feed_dict={x: testing_data_x,
                                                                          y: testing_data_y}), 2)

            _cost_array.append(mean_cost)
            _training_accuracy_array.append(_training_accuracy)
            _testing_accuracy_array.append(_testing_accuracy)

            if e % print_at == 0:
                print('epoch:', e,
                      'mean_cost:', mean_cost,
                      'training_accuracy:', _training_accuracy, '%',
                      'testing_accuracy:', _testing_accuracy, '%')

        return _cost_array, _training_accuracy_array, _testing_accuracy_array


if __name__ == '__main__':
    import pickle as pkl
    from matplotlib import pyplot as plt

    with open('data.pkl', 'rb') as f:  # to generate this file, see generate_data.py file.
        data = pkl.load(f)
        x, y = data['x'], data['y']
        x = x.reshape(len(x), -1)
        x = x / 255  # check other normalization techniques
        epochs = 1000
        batch_size = 1000

        # experiment with:
        # 1) Initialization methods
        # 2) batch size
        # 3) batch normalizations
        # 4) number of hidden neurons
        # 5) Regularization techniques
        # 6) learning rate
        # 7) Optimizer

        cost, training, testing = main(x, y, epochs=epochs, batch_size=batch_size)

        es = np.arange(epochs)

        plt.plot(es, training, label='training')
        plt.plot(es, testing, label='testing')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')

        plt.legend()
        plt.show()

        plt.plot(es, cost)
        plt.xlabel('epoch')
        plt.ylabel('cost')

        plt.show()
