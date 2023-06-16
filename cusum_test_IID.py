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

time_start = time.time()

n = 500
p = 1

mu = 1
delta_vector = [0.1, 0.3, 0.5, 0.7, 1]
Rmse = []
for delta in delta_vector:
    delta_n = delta
    RMSE_CUSUM = []
    for epoch in range(200):
        M = np.zeros(n)
        V_id = np.eye(n)

        kesi = np.random.normal(0, 1, size=(1,n))
        I_m = np.ones(n)
        I_m[0:int(n / 2)] = 0

        X = mu + delta_n * I_m + kesi

        X_cum = np.cumsum(X, axis=1)

        X_avgcum = X_cum / np.mat(np.arange(1, X_cum.shape[1] + 1).reshape(1, X_cum.shape[1]))


        def X_avgcum_reverse_dn():
            X_avgcum_reverse_dn = (np.arange(0, X_cum.shape[1])[::-1].reshape(1, X_cum.shape[1]))
            X_avgcum_reverse_dn[:, -1] = 1
            return X_avgcum_reverse_dn


        X_avgcum_reverse = (np.sum(X, axis=1).reshape(X_cum.shape[0], 1) - X_cum) / X_avgcum_reverse_dn()

        X_num = np.array(np.arange(1, X_cum.shape[1] + 1))

        Z_n_c = np.power(X_num * (len(X_num) - X_num) / len(X_num),0.9)
        Z_n_c_1 = Z_n_c
        for i in range(p - 1):
            Z_n_c = np.vstack((Z_n_c, Z_n_c_1))

        X_cum_df = X_avgcum - X_avgcum_reverse
        Z_n = np.multiply(Z_n_c, X_cum_df)
        T_n = abs(Z_n).argmax()
        # print(T_n)
        RMSE_CUSUM.append(((T_n - (n / 2))/n) ** 2)
    # plt.plot(np.arange(500),abs(Z_n).reshape(500,-1))
    # plt.show()
    # print(np.sqrt(sum(RMSE_CUSUM) / 200))
    Rmse.append(np.sqrt(sum(RMSE_CUSUM) / 200))
print(Rmse)
time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
