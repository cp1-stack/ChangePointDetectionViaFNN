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

p = 500
n = 1
tao_star01 = 0.4
tao_star02 = 0.6
# tao_star03 = 0.9
theta_01 = 0.3
theta_02 = 0.7
# theta_03 = 0.7



class Dependent:

    def __init__(self):
        pass

    def M(self):
        M = np.zeros(p)
        return M

    def V_id(self):
        V_id = np.eye(p)
        return V_id

    def V_sd(self):
        V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)
        return V_sd

    def V_md(self):
        V_md = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                V_md[i, j] = 0.8 ** (abs(i - j))
        return V_md


dependent = Dependent()

M = dependent.M()
V_id = dependent.V_id()

X = np.random.multivariate_normal(M, V_id, n)[0]

Theta1 = np.zeros(p)
Theta1[int(tao_star01 * p):int(tao_star02 * p)] = 0.3
Theta1[int(tao_star02 * p):] = 1
# Theta1[int(tao_star03 * p):] = 2

Theta2 = np.zeros(p)
Theta2[:int(tao_star01 * p)] = 0
Theta2[int(tao_star01 * p):int(tao_star02 * p)] = 1
Theta2[int(tao_star02 * p):] = 2
# Theta2[int(tao_star03 * p):] = 3

y = Theta2
Z = Theta1 + X
# plt.plot(np.arange(1,1001),Z)
# plt.plot(np.arange(1,1001),y)
# plt.show()

class Feature:
    def Diff(self):
        Z_diff = np.diff(Z)
        Z_diff = np.append(Z_diff, 0)
        return Z_diff

    def MA(self, t):
        Z_MA_array = []
        for i in range(p):
            Z_MA = np.mean(Z[i:i + t])
            Z_MA_array = np.append(Z_MA_array, Z_MA)
        return Z_MA_array

    def MV(self, t):
        Z_MV_array = []
        for i in range(p):
            Z_MV = np.var(Z[i:i + t])
            Z_MV_array = np.append(Z_MV_array, Z_MV)
        return Z_MV_array


feature = Feature()
Z_Diff = feature.Diff()

Z_MA_5 = feature.MA(5)
Z_MA_10 = feature.MA(10)
Z_MA_20 = feature.MA(20)

Z_MV_5 = feature.MV(5)
Z_MV_10 = feature.MV(10)
Z_MV_20 = feature.MV(20)

data = np.vstack((Z, Z_Diff, Z_MA_5, Z_MA_10, Z_MA_20, Z_MV_5, Z_MV_10, Z_MV_20, y)).T
dataframe = pd.DataFrame(data)
dataframe.to_csv("ChangePoints_data_train_betas.csv", mode='a', header= False, index=False, sep=',')
# dataframe.to_csv("ChangePoints_data_test_betas.csv", mode='w', header= False, index=False, sep=',')
