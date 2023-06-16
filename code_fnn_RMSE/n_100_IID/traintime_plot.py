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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline

# data = np.loadtxt(open("TrainTime.csv", "rb"), delimiter=",")

plt.figure(figsize=(10, 6), dpi=300)
plt.tick_params(labelsize=10)

x = np.array(['100', '200', '500'])
y = np.array([94.185, 102.015, 108.288])
y2 = np.array([95.026, 99.549, 106.402])
width=np.array([0.7, 0.7, 0.7])
# plt.plot(x, y_smoothed,linestyle='-', marker='^')


plt.bar(x, y2,width,color=['slateblue','darkslateblue','mediumslateblue'])

# for a,b in zip(alpha,data[1,:]):
#     plt.text(a, b+0.001, '%.3f' % b, ha='center', va= 'bottom',fontsize=9)
plt.xlabel('Sample Size', fontsize=18)
plt.ylabel('Time (s)', fontsize=18)
plt.rcParams.update({'font.size': 18})
# plt.legend()
plt.savefig("TrainTime_na.png")
plt.show()
