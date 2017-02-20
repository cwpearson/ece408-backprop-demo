import copy
import numpy as np
import numpy.random as npr
import random
from sklearn import preprocessing

import matplotlib.pyplot as plt

DATA_TYPE = np.float32


def dataset_get_sin():
    NUM = 200
    RATIO = 0.7
    SPLIT = int(NUM * RATIO)
    data = np.zeros((NUM, 2), DATA_TYPE)
    data[:, 0] = np.linspace(0.0, 1 * np.pi, num=NUM)  # inputs
    data[:, 1] = np.sin(data[:, 0])  # outputs
    npr.shuffle(data)
    training, test = data[:SPLIT, :], data[SPLIT:, :]
    return training, test


def dataset_get_linear():
    NUM = 1000
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
    res[res < 0] = 0.01
    return res


def sigmoid(vec):
    """Apply sigmoid to vec"""
    return 1.0 / (1.0 + np.exp(-1 * vec))


def d_sigmoid(vec):
    s = sigmoid(vec)
    return s * (1 - s)


def L(x, y):
    return 0.5 * (x - y) * (x - y)


class Model(object):

    def __init__(self, layer_size, h, dh, data_type):
        self.w1 = npr.uniform(0, 1, layer_size)
        self.w2 = npr.uniform(0, 1, (1, layer_size))
        self.b1 = npr.uniform(0, 1, layer_size)
        self.b2 = npr.uniform(0, 1, 1)

        # self.w1 = preprocessing.scale(self.w1)
        # self.w2 = preprocessing.scale(self.w2)
        # self.b1 = preprocessing.scale(self.b1)
        # self.b2 = preprocessing.scale(self.b2)

        self.h = h
        self.dh = dh

        self.w1_v = np.zeros(self.w1.shape)
        self.w2_v = np.zeros(self.w2.shape)
        self.b1_v = np.zeros(self.b1.shape)
        self.b2_v = np.zeros(self.b2.shape)

    def L(self, x, y):
        f_x = self.f(x)
        return 0.5 * (f_x - y) * (f_x - y)

    def z1(self, x):
        return self.w1 * x + self.b1

    def a(self, x):
        return self.h(self.z1(x))

    def f(self, x):
        return np.dot(self.w2, self.a(x)) + self.b2

    def dLdf(self, x, y):
        return self.f(x) - y

    def dLdb2(self, x, y):
        return self.dLdf(x, y)

    def dfda(self):  # how f changes with ith element of a
        return self.w2

    def dadz1(self, x):  # how a[i] changes with z1[i]
        """Compute da/dz1 for an input x"""
        return self.dh(self.z1(x))

    def dLdz1(self, x, y):
        """Compute dL/dz1 for an input x and expected output y"""
        return self.dLdf(x, y) * np.dot(self.dfda(), self.dadz1(x))

    def dLdw1(self, x, y):
        """Compute dL/dw1 for an input x and expected output y"""
        return self.dLdf(x, y) * np.dot(self.dfda(), self.dadz1(x) * x)

    def dLdw2(self, x, y):
        """Compute dL/dw2 for an input x and expected output y"""
        return self.dLdf(x, y) * np.sum(self.a(x))  # df/dw2

    def dLdb1(self, x, y):
        return self.dLdf(x, y) * np.dot(self.dfda(), self.dadz1(x))

    def backward(self, training_samples, ETA):
        for sample in training_samples:
            sample_input = sample[0]
            sample_output = sample[1]

            b2_grad = self.dLdb2(sample_input, sample_output)
            w2_grad = self.dLdw2(sample_input, sample_output)
            b1_grad = self.dLdb1(sample_input, sample_output)
            w1_grad = self.dLdw1(sample_input, sample_output)
            self.b2 -= ETA * b2_grad
            self.b1 -= ETA * b1_grad
            self.w2 -= ETA * w2_grad
            self.w1 -= ETA * w1_grad
        return

    def backward_minibatch(self, batch, ETA):
        b2_grad = np.zeros(self.b2.shape)
        b1_grad = np.zeros(self.b1.shape)
        w2_grad = np.zeros(self.w2.shape)
        w1_grad = np.zeros(self.w1.shape)

        for sample in batch:
            sample_input = sample[0]
            sample_output = sample[1]

            # self.grad_checker(10e-4, sample_input, sample_output)

            b2_grad += self.dLdb2(sample_input, sample_output)
            w2_grad += self.dLdw2(sample_input, sample_output)
            b1_grad += self.dLdb1(sample_input, sample_output)
            w1_grad += self.dLdw1(sample_input, sample_output)

        self.b2 -= ETA * b2_grad / len(batch)
        self.b1 -= ETA * b1_grad / len(batch)
        self.w2 -= ETA * w2_grad / len(batch)
        self.w1 -= ETA * w1_grad / len(batch)
        return

    def SGDm(self, training_samples, ETA):
        alpha = 0.99
        for sample in training_samples:
            sample_input = sample[0]
            sample_output = sample[1]

            self.b2_v = alpha * self.b2_v + ETA * \
                self.dLdb2(sample_input, sample_output)
            self.w2_v = alpha * self.w2_v + ETA * \
                self.dLdw2(sample_input, sample_output)
            self.b1_v = alpha * self.b1_v + ETA * \
                self.dLdb1(sample_input, sample_output)
            self.w1_v = alpha * self.w1_v + ETA * \
                self.dLdw1(sample_input, sample_output)
            self.b2 -= self.b2_v
            self.b1 -= self.b1_v
            self.w2 -= self.w2_v
            self.w1 -= self.w1_v
        return

    def grad_checker(self, eps, x, y):
        # Check b2
        inc_model = copy.deepcopy(self)
        dec_model = copy.deepcopy(self)
        inc_model.b2 = self.b2 + eps
        dec_model.b2 = self.b2 - eps
        grad_estimate = (inc_model.L(x, y) - dec_model.L(x, y)) / (2 * eps)
        grad_actual = self.dLdb2(x, y)
        if np.linalg.norm(grad_estimate - grad_actual) > 10e-5:
            print "b2"

        # Check b1
        inc_model = copy.deepcopy(self)
        dec_model = copy.deepcopy(self)
        inc_model.b1 = self.b1 + eps
        dec_model.b1 = self.b1 - eps
        grad_estimate = (inc_model.L(x, y) - dec_model.L(x, y)) / (2 * eps)
        grad_actual = self.dLdb1(x, y)
        if np.linalg.norm(grad_estimate - grad_actual) > 10e-5:
            print "b1"

        # Check w2
        inc_model = copy.deepcopy(self)
        dec_model = copy.deepcopy(self)
        inc_model.w2 = self.w2 + eps
        dec_model.w2 = self.w2 - eps
        grad_estimate = (inc_model.L(x, y) - dec_model.L(x, y)) / (2 * eps)
        grad_actual = self.dLdw2(x, y)
        if np.linalg.norm(grad_estimate - grad_actual) > 10e-5:
            print "w2"

        # Check w1
        inc_model = copy.deepcopy(self)
        dec_model = copy.deepcopy(self)
        inc_model.w1 = self.w1 + eps
        dec_model.w1 = self.w1 - eps
        grad_estimate = (inc_model.L(x, y) - dec_model.L(x, y)) / (2 * eps)
        grad_actual = self.dLdw1(x, y)
        if np.linalg.norm(grad_estimate - grad_actual) > 10e-5:
            print "w1"


def evaluate(model, samples):
    """Report the average loss function over the data"""
    cost_acc = 0.0
    for sample in samples:
        cost_acc += model.L(sample[0], sample[1])
    return cost_acc / len(samples)

TRAIN_DATA, TEST_DATA = dataset_get_sin()
# TRAIN_DATA, TEST_DATA = dataset_get_linear()

MODEL = Model(8, sigmoid, d_sigmoid, DATA_TYPE)
# MODEL = Model(20, relu, d_relu, DATA_TYPE)

# Train the model with some training data
MAX_EPOCHS = 2000
TRAINING_SUBSET_SIZE = len(TRAIN_DATA)
PATIENCE = 200

print TRAINING_SUBSET_SIZE

print "Epoch\tTraining Cost Function\tTest Cost Function"

best_rate = np.inf
best_model = None
for epoch in range(MAX_EPOCHS):
    # Create a training sample
    training_subset_indices = npr.choice(
        range(len(TRAIN_DATA)), size=TRAINING_SUBSET_SIZE, replace=False)
    training_subset = [TRAIN_DATA[i] for i in training_subset_indices]
    random.shuffle(training_subset)

    # Apply backpropagation
    # MODEL.backward(training_subset, LEARNING_RATE)

    # Apply backpropagation
    # MODEL.SGDm(training_subset, 0.00004)

    # Apply backprop with minibatch
    BATCH_SIZE = 4
    LEARNING_RATE = 0.05
    for i in range(0, len(training_subset), BATCH_SIZE):
        batch = training_subset[i:min(i + BATCH_SIZE, len(training_subset))]
        MODEL.backward_minibatch(batch, LEARNING_RATE)

    # Evaluate accuracy against training data and test data
    training_rate = evaluate(MODEL, training_subset)
    test_rate = evaluate(MODEL, TEST_DATA)

    print epoch, training_rate, test_rate,

    # If it's the best one so far, store it
    if training_rate < best_rate:
        print "(new best)"
        best_rate = training_rate
        best_model = copy.deepcopy(MODEL)
        patience = PATIENCE
    else:
        patience -= 1
        print patience

    if patience <= 0:
        print PATIENCE, "iterations without improvement"
        break

test_rate = evaluate(MODEL, TEST_DATA)
print "Test cost:", test_rate

TEST_OUTPUT = np.vectorize(best_model.f)(TEST_DATA[:, 0])
TRAIN_OUTPUT = np.vectorize(best_model.f)(TRAIN_DATA[:, 0])

scatter_train, = plt.plot(
    TRAIN_DATA[:, 0], TRAIN_DATA[:, 1], 'ro', markersize=2, label="Real Data")
scatter_train_out, = plt.plot(
    TRAIN_DATA[:, 0], TRAIN_OUTPUT, 'go', label="Network output on training data")
scatter_test_out, = plt.plot(
    TEST_DATA[:, 0], TEST_OUTPUT, 'bo', label="Network output on test data")
plt.legend(handles=[scatter_train, scatter_train_out, scatter_test_out])
plt.savefig("results.png", bbox_inches="tight")
plt.show()
