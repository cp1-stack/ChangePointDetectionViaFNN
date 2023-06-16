import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas
from sklearn.metrics import accuracy_score
import numpy.random as r
import struct
import random as rd
from threading import Thread
import os
import pandas as pd
from time import *
import random
import time
import csv

time_start = time.time()

# n = 1000
# delta_n = 1
# m = 50
# mu = 1
#
# M = np.zeros(n)
# V_id = np.eye(n)
#
# kesi = np.random.multivariate_normal(M, V_id, 1)
# I_m = np.ones(n)
# I_m[0:500] = 0
#
# X = mu + delta_n * I_m + kesi

p = 1
cps_data = pd.read_csv('ChangePoints_data_test_betas.csv', header=None, sep=',')
X = cps_data[0].values
X = X.reshape(1, -1)


def cusum(X):
    X_cum = np.cumsum(X, axis=1)

    X_avgcum = X_cum / np.mat(np.arange(1, X_cum.shape[1] + 1).reshape(1, X_cum.shape[1]))

    def X_avgcum_reverse_dn():
        X_avgcum_reverse_dn = (np.arange(0, X_cum.shape[1])[::-1].reshape(1, X_cum.shape[1]))
        X_avgcum_reverse_dn[:, -1] = 1
        return X_avgcum_reverse_dn

    X_avgcum_reverse = (np.sum(X, axis=1).reshape(X_cum.shape[0], 1) - X_cum) / X_avgcum_reverse_dn()

    X_num = np.array(np.arange(1, X_cum.shape[1] + 1))

    Z_n_c = np.sqrt(X_num * (len(X_num) - X_num) / len(X_num))
    Z_n_c_1 = Z_n_c
    for i in range(p - 1):
        Z_n_c = np.vstack((Z_n_c, Z_n_c_1))

    X_cum_df = X_avgcum - X_avgcum_reverse
    Z_n = np.multiply(Z_n_c, X_cum_df)
    T_n = abs(Z_n).argmax()
    print(T_n)
    return T_n


cusum(X)
cusum(X[:, 0:597])
cusum(X[:, 597:1000])

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
