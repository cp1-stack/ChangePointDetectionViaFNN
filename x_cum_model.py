import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

import pandas
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
for k in range(4):
    np.random.seed(7)

    rho = 0.9
    sigma = 2
    n = 1200
    x1 = np.random.normal(loc=0, scale=2, size=500)
    x = np.empty((0, n))
    # x = np.append(x1, np.random.normal(loc=0, scale=2, size=20))

    for i in range(2):
        t = np.random.gamma(shape=2, scale=2 * (i + 1), size=100*(k+1))
        x = np.append(x, np.append(x1, t))

    plt.figure(figsize=(9, 6))
    # plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Values of random variables')
    plt.plot(x)
    plt.savefig("model_%d00.png"%(k+1))
plt.show()








