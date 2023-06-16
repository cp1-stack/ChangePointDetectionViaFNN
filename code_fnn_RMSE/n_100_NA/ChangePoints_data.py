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

p = 100
n = 1
tao_star = 0.5
theta_0 = 1



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
Theta1 = 1 * np.ones(p)
Theta1[:int(tao_star * p)] = 0

# Theta2 = np.zeros(p)
# Theta2[int(tao_star * p)] = 1

y = Theta1
Z = theta_0 + Theta1 + X


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
dataframe.to_csv("ChangePoints_data_train_beta10.csv", mode='a', header= False, index=False, sep=',')
# dataframe.to_csv("ChangePoints_data_test_beta10.csv", mode='w', header= False, index=False, sep=',')
