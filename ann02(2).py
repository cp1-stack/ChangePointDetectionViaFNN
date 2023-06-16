import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score
import numpy.random as r
import struct
import random as rd
import math
from threading import Thread
import os
import pandas as pd
from time import *
import random

random.seed(200)

rho = 0.9
sigma = 2

x1 = np.random.normal(loc=0, scale=2, size=100)
x = np.append(x1, np.random.normal(loc=0, scale=2, size=10))
for i in range(4):
    t = np.random.normal(loc=i, scale=2 * i, size=10)
    x = np.append(x, np.append(x1, t))  # 550

x_diff = np.diff(x)  # 549

x_ma5 = np.empty(shape=[0, 550])  # 550
for i in range(550):
    a = np.mean(x[i:i + 5])
    x_ma5 = np.append(x_ma5, a)

x_ma5 = x_ma5[:549]

x_ma20 = np.empty(shape=[0, 550])  # 550
for i in range(550):
    a = np.mean(x[i:i + 20])
    x_ma20 = np.append(x_ma20, a)

x_ma20 = x_ma20[:549]

x_ma60 = np.empty(shape=[0, 550])  # 550
for i in range(550):
    a = np.mean(x[i:i + 60])
    x_ma60 = np.append(x_ma60, a)

x_ma60 = x_ma60[:549]

x_mv5 = np.empty(shape=[0, 550])  # 551
for i in range(550):
    a = np.var(x[i:i + 5])
    x_mv5 = np.append(x_ma5, a)

x_mv5 = x_mv5[:549]

x_mv20 = np.empty(shape=[0, 550])  # 551
for i in range(547):
    a = np.var(x[i:i + 20])
    x_mv20 = np.append(x_ma20, a)

x_mv20 = x_mv20[:549]

x_mv60 = np.empty(shape=[0, 550])  # 551
for i in range(550):
    a = np.var(x[i:i + 60])
    x_mv60 = np.append(x_ma60, a)

x_mv60 = x_mv60[:549]

data = np.vstack((x_diff, x_ma5, x_ma20, x_ma60, x_mv5, x_mv20, x_mv60))
print(np.shape(data))
print(type(data))
y1 = np.ones((1, 100))
y0 = np.zeros((1, 10))
y10 = np.append(y1, y0)
y = np.empty(shape=[0, 550])
for i in range(5):
    y = np.append(y, y10)
y = (y[:549]).reshape(1, -1)
print(len(y))

nn_structure = [7, 9, 1]


def f(x):
    return 1.0 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def ReluFunc(x):
    x = (np.abs(x) + x) / 2.0
    return x


def ReluPrime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))  # np.random.permutation()随机排列序列
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def stratify(y, C=10):
    category = {}
    for i in range(y.shape[1]):
        for j in range(C):
            if np.argmax(y[:, i]) == j:
                category.setdefault(j, []).append(i)  # 储存类别索引
    return category


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    # np.random.seed(3)
    for l in range(1, len(nn_structure)):
        # W[l] = np.random.randn(nn_structure[l], nn_structure[l-1])*np.sqrt(1/nn_structure[l-1])
        # b[l] = np.zeros((nn_structure[l],))
        W[l] = np.random.randn(nn_structure[l], nn_structure[l - 1]) * np.sqrt(1 / nn_structure[l - 1])
        b[l] = np.zeros((nn_structure[l], 1))
    return W, b


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        tri_b[l] = np.zeros((nn_structure[l], 1))
    return tri_W, tri_b


def initialize_with_ssad(nn_structure, C):
    v = {}
    s = {}
    for l in range(1, len(nn_structure)):
        for j in range(0, C):
            v["dW" + str(l) + str(j)] = np.zeros((nn_structure[l], nn_structure[l - 1]))
            v["db" + str(l) + str(j)] = np.zeros((nn_structure[l], 1))
        s["dW" + str(l)] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        s["db" + str(l)] = np.zeros((nn_structure[l], 1))
    return v, s


def update_parameters_with_ssag(W, b, tri_W, tri_b, v, s, alpha, C, idx):
    for l in range(len(nn_structure) - 1, 0, -1):
        # update parameters
        s["dW" + str(l)] = (s["dW" + str(l)] - v["dW" + str(l) + str(idx)]) + tri_W[l]
        s["db" + str(l)] = (s["db" + str(l)] - v["db" + str(l) + str(idx)]) + tri_b[l]
        v["dW" + str(l) + str(idx)] = tri_W[l]
        v["db" + str(l) + str(idx)] = tri_b[l]
        W[l] += -alpha * s["dW" + str(l)] / C
        b[l] += -alpha * s["db" + str(l)] / C
    return W, b, v, s


def update_parameters_with_bsgd(W, b, tri_W, tri_b, alpha, lamb):
    # Update rule for each parameter
    for l in range(len(nn_structure) - 1, 0, -1):
        W[l] += -alpha * (tri_W[l] + lamb * W[l])
        b[l] += -alpha * tri_b[l]
    return W, b


def initialize_with_momentum(nn_structure):
    # number of layers in the neural networks
    v = {}
    for l in range(1, len(nn_structure)):
        v["dW" + str(l)] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        v["db" + str(l)] = np.zeros((nn_structure[l], 1))
    return v


def update_parameters_with_momentum(W, b, tri_W, tri_b, v, beta, alpha):
    for l in range(len(nn_structure) - 1, 0, -1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * tri_W[l]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * tri_b[l]
        # update parameters
        W[l] += -alpha * tri_W[l]
        b[l] += -alpha * tri_b[l]
    return W, b, v


def initialize_with_adam(nn_structure):
    # number of layers in the neural networks
    v = {}
    s = {}
    for l in range(1, len(nn_structure)):
        ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l)] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        v["db" + str(l)] = np.zeros((nn_structure[l], 1))
        s["dW" + str(l)] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        s["db" + str(l)] = np.zeros((nn_structure[l], 1))
    return v, s


def update_parameters_with_adam(W, b, tri_W, tri_b, v, s, t, alpha=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(len(nn_structure) - 1, 0, -1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * tri_W[l]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * tri_b[l]
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (tri_W[l] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (tri_b[l] ** 2)
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
        W[l] += - alpha * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        b[l] += - alpha * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
    return W, b, v, s


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        # z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        # z[l + 1] = np.dot(W[l],node_in) + b[l]
        z[l + 1] = W[l].dot(node_in) + b[l]
        if l == len(W):
            h[l + 1] = f(z[l + 1])
        else:
            h[l + 1] = ReluFunc(z[l + 1])  # h^(l) = f(z^(l))
    return h, z


# h, z = feed_forward(x_batch, W, b)

def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y - h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * ReluPrime(z_l)
    # return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn_model(nn_structure, X, y, optimizer, num_epochs, bsize=549, alpha=0.0008, C=10, lamb=0.005, beta=0.9,
                   beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(nn_structure)
    begin_time = time()
    W, b = setup_and_init_weights(nn_structure)
    avg_cost_func = []
    train_accuracy = []
    t = 0
    if optimizer == "bsgd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_with_momentum(nn_structure)
    elif optimizer == "adam":
        v, s = initialize_with_adam(nn_structure)
    elif optimizer == "ssag":
        v, s = initialize_with_ssad(nn_structure, C)
        category = stratify(y, C=10)
        for i in range(num_epochs):
            tri_W, tri_b = init_tri_values(nn_structure)
            avg_cost = 0
            idx = np.random.randint(C)
            slice = rd.sample(list(category[idx]), bsize)  # 分类别随机抽取数据
            x_batch = X[:, slice]
            y_batch = y[:, slice]
            delta = {}
            h, z = feed_forward(x_batch, W, b)
            avg_cost = np.linalg.norm((y_batch - h[L]), axis=0).sum()
            train_accuracy = accuracy_score(y, h[l])
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y_batch, h[l], z[l])
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l + 1], W[l], z[l])
                    tri_W[l] += np.dot(delta[l + 1], np.transpose(h[l])) * 1.0 / bsize
                    tri_b[l] += delta[l + 1] * 1.0 / bsize
            W, b, v, s = update_parameters_with_ssag(W, b, tri_W, tri_b, v, s, alpha, C, idx)
            avg_cost = 1.0 / bsize * avg_cost
            # avg_cost_func.append(avg_cost)
            if i % 100 == 0:
                print("Cost after iter %i of %i: %f" % (i, num_epochs, avg_cost))
                avg_cost_func.append(avg_cost)

    # print('Starting gradient descent for {} iterations'.format(iter_num))
    if optimizer != "ssag":
        # while cnt < iter_num:
        for i in range(num_epochs):
            # rid = np.random.randint(0, len(y) - bsize, dtype=np.int64)
            # x_batch = X[rid:rid + bsize, ...]
            # y_batch = y[rid:rid + bsize, ...]
            mini_batches = random_mini_batches(X, y, bsize)
            for minibatch in mini_batches:
                (x_batch, y_batch) = minibatch
                tri_W, tri_b = init_tri_values(nn_structure)
                avg_cost = 0
                delta = {}
                # 前向传播
                h, z = feed_forward(x_batch, W, b)
                # loop from nl-1 to 1 backpropagating the errors
                # 损失值

                avg_cost = 1.0 / bsize * np.linalg.norm((y_batch - h[L]), axis=0).sum()
                # 反向传播)
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        delta[l] = calculate_out_layer_delta(y_batch, h[l], z[l])
                    else:
                        if l > 1:
                            delta[l] = calculate_hidden_delta(delta[l + 1], W[l], z[l])
                        tri_W[l] = np.dot(delta[l + 1], np.transpose(h[l])) * 1.0 / bsize
                        tri_b[l] = np.sum(delta[l + 1], axis=1).reshape(delta[l + 1].shape[0], 1) * 1.0 / bsize
                        # grads["db" + str(L)] = np.sum(dZ, axis=1).reshape(dZ.shape[0], 1) / m
                # 损失值
                # avg_cost = 1.0 / bsize * avg_cost
                # avg_cost_func.append(avg_cost)
                # 参数更新
                if optimizer == "bsgd":
                    W, b = update_parameters_with_bsgd(W, b, tri_W, tri_b, alpha, lamb)
                elif optimizer == "momentum":
                    W, b, v = update_parameters_with_momentum(W, b, tri_W, tri_b, v, beta, alpha)
                elif optimizer == "adam":
                    t = t + 1  # Adam counter
                    W, b, v, s = update_parameters_with_adam(W, b, tri_W, tri_b, v, s, t, alpha, beta1, beta2, epsilon)
            # 计算训练精度
            h, z = feed_forward(X, W, b)
            m = X.shape[1]
            Y_prediction = np.zeros((m,))
            for j in range((h[L]).shape[1]):

                # Convert probabilities A[0,i] to actual predictions p[0,i]
                ### START CODE HERE ### (≈ 4 lines of code)
                if (h[L])[0, j] <= 0.5:
                    Y_prediction[j] = 0
                else:
                    Y_prediction[j] = 1
                ### END CODE HERE ###
            train_acc = accuracy_score(y.reshape(m, 1), Y_prediction)
            if i % 100 == 0:
                print("Cost after epochs %i  of %i: %f" % (i, num_epochs, avg_cost))
                print("train_accuracy after epochs %i  of %i: %f" % (i, num_epochs, train_acc))
            avg_cost_func.append(avg_cost)
            train_accuracy.append(train_acc)

            # x_axix.append(i)
            # print('obj value:{} after {}th iterations at alpha:{},lamb:{}'.format(avg_cost,cnq    t,alpha,lamb))
        end_time = time()
        run_time = end_time - begin_time
        print("运行时间为：%f" % (run_time))
    return W, b, avg_cost_func, train_accuracy


# select_parameters(X_trains_norm,y_v_train,X_tests_norm,test_labels)
# W, b, avg_cost_func=train_nn_model(nn_structure,X_trains_norm,y_v_train,400,0.04,0.001)
W, b, avg_cost_func, train_accuracy = train_nn_model(nn_structure, data, y, optimizer="bsgd", num_epochs=10000, bsize=549, alpha=0.0008,
                                     lamb=0.008, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)


# W, b, avg_cost_func=train_nn_model(nn_structure,X_trains_norm, y_v_train,optimizer= "momentum",num_epochs=1000,bsize=1,alpha=0.04, lamb=0.001, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)
# W, b, avg_cost_func, x_axix = train_nn_model(nn_structure,data, y, optimizer="adam", num_epochs=10,bsize=64, alpha=0.0007, lamb=0.001, beta=0.9, beta1=0.9, beta2=0.999,epsilon=1e-8)

def predict(W, b, X):
    L = len(W) + 1
    m = X.shape[1]
    Y_prediction = np.zeros((m,))
    h, z = feed_forward(X, W, b)
    for i in range((h[L]).shape[1]):
        if (h[L])[0, i] <= 0.5:
            Y_prediction[i] = 0
        else:
            Y_prediction[i] = 1
    return Y_prediction


y_p = predict(W, b, data)
# y_pred = predict_y(W, b, X_tests_norm, 4)

print(accuracy_score(y.reshape(-1, ), y_p) * 100)
plt.figure(figsize=(9, 6))
plt.plot(avg_cost_func, label='lr=0.008')
# plt.plot(train_accuracy, label='lr=0.008')
plt.legend()


# select_parameters(X_trains_norm,y_v_train,X_tests_norm,test_labels)
# W, b, avg_cost_func=train_nn_model(nn_structure,X_trains_norm,y_v_train,400,0.04,0.001)
W, b, avg_cost_func, train_accuracy = train_nn_model(nn_structure, data, y, optimizer="bsgd", num_epochs=10000, bsize=549, alpha=0.0008,
                                     lamb=0.005, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)

plt.plot(avg_cost_func, label='lr=0.005')
# plt.plot(train_accuracy, label='lr=0.005')
plt.legend()


W, b, avg_cost_func, train_accuracy = train_nn_model(nn_structure, data, y, optimizer="bsgd", num_epochs=10000, bsize=549, alpha=0.0008,
                                     lamb=0.003, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)

plt.plot(avg_cost_func, label='lr=0.003')
# plt.plot(train_accuracy, label='lr=0.003')
plt.legend()

W, b, avg_cost_func, train_accuracy = train_nn_model(nn_structure, data, y, optimizer="bsgd", num_epochs=10000, bsize=549, alpha=0.0008,
                                     lamb=0.001, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)

plt.plot(avg_cost_func, label='lr=0.001')

# plt.plot(train_accuracy, label='lr=0.001')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average cost')
# plt.savefig("train_accuracy.png")

plt.savefig("avg_cost2.png")
plt.show()





# plt.figure(figsize=(9, 6))
# plt.plot(train_accuracy)
# plt.savefig("train_accuracy.png")
# plt.show()
