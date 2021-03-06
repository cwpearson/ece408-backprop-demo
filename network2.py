import numpy as np
import numpy.random as npr
import random

import matplotlib.pyplot as plt

DATA_TYPE = np.float32


def dataset_get_sin():
    NUM = 100
    RATIO = 0.8
    SPLIT = int(NUM * RATIO)
    data = np.zeros((NUM, 2), DATA_TYPE)
    data[:, 0] = np.linspace(0.0, 2 * np.pi, num=NUM)  # inputs
    data[:, 1] = np.sin(data[:, 0])  # outputs
    npr.shuffle(data)
    training, test = data[:SPLIT, :], data[SPLIT:, :]
    return training, test


def dataset_get_linear():
    NUM = 100
    RATIO = 0.8
    SPLIT = int(NUM * RATIO)
    data = np.zeros((NUM, 2), DATA_TYPE)
    data[:, 0] = np.linspace(0.0, 2 * np.pi, num=NUM)  # inputs
    data[:, 1] = 2 * data[:, 0]  # outputs
    npr.shuffle(data)
    training, test = data[:SPLIT, :], data[SPLIT:, :]
    return training, test


def relu(x):
    """Apply a rectified linear unit to x"""
    return np.maximum(0, x)


def d_relu(x):
    res = x
    res[res >= 0] = 1
    res[res < 0] = 0
    return res


def sigmoid(vec):
    """Apply sigmoid to vec"""
    return 1.0 / (1.0 + np.exp(-1 * vec))


def d_sigmoid(vec):
    s = sigmoid(vec)
    return s * (1 - s)


def L(x, y):
    return (x - y) * (x - y)


class Model(object):

    def __init__(self, layer_sizes, h, dh, data_type):
        self.w1 = npr.rand(layer_sizes[0]).astype(data_type)
        self.b1 = npr.rand(layer_sizes[0]).astype(data_type)
        self.w2 = npr.rand(layer_sizes[1], layer_sizes[0]).astype(data_type)
        self.b2 = npr.rand(layer_sizes[1]).astype(data_type)
        self.w3 = npr.rand(1, layer_sizes[1]).astype(data_type)
        self.b3 = npr.rand(1).astype(data_type)

        self.w1 /= np.sum(self.w1)
        self.w2 /= np.sum(self.w2)
        self.w3 /= np.sum(self.w3)
        self.b1 /= np.sum(self.b1)
        self.b2 /= np.sum(self.b2)
        self.b3 /= np.sum(self.b3)

        self.h = h
        self.dh = dh

    def z1(self, x):
        return self.w1 * x + self.b1

    def a1(self, x):
        return self.h(self.z1(x))

    def z2(self, x):
        return self.w2.dot(self.a1(x)) + self.b2

    def a2(self, x):
        return self.h(self.z2(x))

    def f(self, x):
        return self.w3.dot(self.a2(x)) + self.b3

# Last layer updates

    def dLdf(self, x, y):
        return 2.0 * (self.f(x) - y)

    def dLdb3(self, x, y):
        return self.dLdf(x, y)

    def dLdw3(self, x, y):
        return self.dLdf(x, y) * np.sum(self.a2(x))

# Second layer updates

    def da2db2(self, x):
        return self.dh(self.z2(x)) * 1.0

    def dfdb2(self, x):
        return np.dot(self.w3, self.da2db2(x))

    def dLdb2(self, x, y):
        return self.dLdf(x, y) * self.dfdb2(x)

    def dz2dw2(self, x):  # how z2 changes with a row of w2
        return np.sum(self.a1(x))

    def da2dw2(self, x):
        return self.dh(self.z2(x)) * self.dz2dw2(x)

    def dfdw2(self, x):
        # print self.dfdz2(x).shape
        return np.dot(self.dfdz2(x), self.dz2dw2(x))

    def dLdw2(self, x, y):
        return self.dLdf(x, y) * np.sum(self.dfdw2(x))

# First layer updates

    def dz1db1(self):
        return np.ones(self.b1.shape)

    def dfda2(self):  # how f changes with the a2[i]
        return np.sum(self.w3)

    def dfdz2(self, x):  # how f changes wrt each entry of z2
        return self.dfda2() * self.dh(self.z2(x))

    def dz2dz1(self, x):  # how z2 entries affected by z1
        return self.w2 * self.dh(self.z1(x))

    def dfdz1(self, x):
        # print self.dfdz2(x).shape
        # print self.dz2dz1(x).shape
        # print np.dot(self.dfdz2(x), self.dz2dz1(x)).shape
        return np.dot(self.dfdz2(x), self.dz2dz1(x))

    def dLdb1(self, x, y):
        return self.dLdf(x, y) * np.dot(self.dfdz1(x), self.dz1db1())

    def da1dw1(self, x):
        return self.dh(self.z1(x)) * x

    def dz2dw1(self, x):  # how z2 changes with the ith entry of w1
        ret = np.zeros(self.w2.shape)
        for j in range(len(self.b2)):
            ret[j] = np.dot(self.w2[j], self.da1dw1(x))
        return ret

    def dfdw1(self, x):  # how f changes with the ith entry of w1
        # print self.dfdz2(x).shape
        # print self.dz2dw1(x).shape
        return np.dot(self.dfdz2(x), self.dz2dw1(x))

    def dLdw1(self, x, y):
        return self.dLdf(x, y) * np.sum(self.dfdw1(x))

    def backward(self, training_samples, ETA):
        """Do backpropagation with stochastic gradient descent on the model using training_samples"""
        for sample in training_samples:
            sample_input = sample[0]
            sample_output = sample[1]

            b3_grad = self.dLdb3(sample_input, sample_output)
            b2_grad = self.dLdb2(sample_input, sample_output)
            b1_grad = self.dLdb1(sample_input, sample_output)
            w3_grad = self.dLdw3(sample_input, sample_output)
            w2_grad = self.dLdw2(sample_input, sample_output)
            w1_grad = self.dLdw1(sample_input, sample_output)
            self.b3 -= ETA * b3_grad
            self.b2 -= ETA * b2_grad
            self.b1 -= ETA * b1_grad
            self.w3 -= ETA * w3_grad
            self.w2 -= ETA * w2_grad
            self.w1 -= ETA * w1_grad
        return


def evaluate(model, samples):
    """Report the loss function over the data"""
    loss_acc = 0.0
    for sample in samples:
        guess = model.f(sample[0])
        actual = sample[1]
        loss_acc += L(guess, actual)
    return loss_acc / len(samples)

# TRAIN_DATA, TEST_DATA = dataset_get_sin()
TRAIN_DATA, TEST_DATA = dataset_get_linear()

MODEL = Model([10, 6], sigmoid, d_sigmoid, DATA_TYPE)
# MODEL = Model(10, relu, d_relu, DATA_TYPE)

# Train the model with some training data
TRAINING_ITERS = 500
LEARNING_RATE = 0.001
TRAINING_SUBSET_SIZE = len(TRAIN_DATA)

print TRAINING_SUBSET_SIZE

best_rate = np.inf
rates = [["iter", "training_rate", "test_rate"]]
for training_iter in range(TRAINING_ITERS):
    # Create a training sample
    training_subset_indices = npr.choice(
        range(len(TRAIN_DATA)), size=TRAINING_SUBSET_SIZE, replace=False)
    training_subset = [TRAIN_DATA[i] for i in training_subset_indices]
    random.shuffle(training_subset)

    # Apply backpropagation
    MODEL.backward(training_subset, LEARNING_RATE)

    # Evaluate accuracy against training data
    training_rate = evaluate(MODEL, training_subset)
    test_rate = evaluate(MODEL, TEST_DATA)
    rates += [[training_iter, training_rate, test_rate]]

    print training_iter, "positive rates:", training_rate, test_rate,

    # If it's the best one so far, store it
    if training_rate < best_rate:
        print "(new best)"
        best_rate = training_rate
    else:
        print ""

TEST_OUTPUT = np.vectorize(MODEL.f)(TEST_DATA[:, 0])
TRAIN_OUTPUT = np.vectorize(MODEL.f)(TRAIN_DATA[:, 0])

scatter_train, = plt.plot(
    TRAIN_DATA[:, 0], TRAIN_DATA[:, 1], 'ro', label="Training data")
scatter_train_out, = plt.plot(
    TRAIN_DATA[:, 0], TRAIN_OUTPUT, 'go', label="Training output")
scatter_test_out, = plt.plot(
    TEST_DATA[:, 0], TEST_OUTPUT, 'bo', label="Test output")
plt.legend(handles=[scatter_train, scatter_train_out, scatter_test_out])

plt.show()
