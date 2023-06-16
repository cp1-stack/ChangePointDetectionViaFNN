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
from scipy import stats


data = np.loadtxt(open("statis_real.csv", "rb"), delimiter=",")

plt.figure(figsize=(10,8),dpi=300)
plt.tick_params(labelsize=20)

ind = np.array([20,40,60,80])

for j in ind:
    plt.plot(data[j,:], linestyle='-',label='{} epoch'.format(j))


# for a,b in zip(alpha,data[1,:]):
#     plt.text(a, b+0.001, '%.3f' % b, ha='center', va= 'bottom',fontsize=9)
plt.xlabel('Real data',fontsize=30)
plt.ylabel('statistics',fontsize=30)
plt.rcParams.update({'font.size': 18})
plt.legend()
plt.savefig("statis_real.png")
plt.show()