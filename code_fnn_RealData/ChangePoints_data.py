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

data = pd.read_csv('data_sh600000_turn04.csv')
print(data.shape)
Z = np.array(data).reshape(-1, )
print(Z.shape)


class Feature:
    def Diff(self):
        Z_diff = np.diff(Z)
        Z_diff = np.append(Z_diff, 0)
        return Z_diff

    def MA(self, t):
        Z_MA_array = []
        for i in range(len(Z)):
            Z_MA = np.mean(Z[i:i + t])
            Z_MA_array = np.append(Z_MA_array, Z_MA)
        return Z_MA_array

    def MV(self, t):
        Z_MV_array = []
        for i in range(len(Z)):
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
y = np.ones(len(Z))
y[:600]=0
data = np.vstack((Z, Z_Diff, Z_MA_5, Z_MA_10, Z_MA_20, Z_MV_5, Z_MV_10, Z_MV_20, y)).T
dataframe = pd.DataFrame(data)
# dataframe.to_csv("ChangePoints_data_train_realdata02.csv", mode='w', header=False, index=False, sep=',')
dataframe.to_csv("ChangePoints_data_test_realdata02.csv", mode='w', header= False, index=False, sep=',')
