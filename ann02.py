import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

rho = 0.9
sigma = 2

x1 = np.random.normal(loc=0, scale=2, size=100)
x = np.append(x1, np.random.normal(loc=0, scale=2, size=50))
for i in range(5):
    t = np.random.normal(loc=i+1, scale=2*(i+1), size=50)
    x = np.append(x, np.append(x1, t))  # 550



# x_diff = np.diff(x)  # 549
#
# x_ma5 = np.empty(shape=[0, 550])  # 550
# for i in range(550):
#     a = np.mean(x[i:i+5])
#     x_ma5 = np.append(x_ma5, a)
#
# x_ma5 = x_ma5[:549]
#
# x_ma20 = np.empty(shape=[0, 550])  # 550
# for i in range(550):
#     a = np.mean(x[i:i+20])
#     x_ma20 = np.append(x_ma20, a)
#
# x_ma20 = x_ma20[:549]
#
# x_ma60 = np.empty(shape=[0, 550])  # 550
# for i in range(550):
#     a = np.mean(x[i:i+60])
#     x_ma60 = np.append(x_ma60, a)
#
# x_ma60 = x_ma60[:549]
#
# x_mv5 = np.empty(shape=[0, 550])  # 551
# for i in range(550):
#     a = np.var(x[i:i+5])
#     x_mv5 = np.append(x_ma5, a)
#
# x_mv5 = x_mv5[:549]
#
# x_mv20 = np.empty(shape=[0, 550])  # 551
# for i in range(547):
#     a = np.var(x[i:i+20])
#     x_mv20 = np.append(x_ma20, a)
#
# x_mv20 = x_mv20[:549]
#
# x_mv60 = np.empty(shape=[0, 550])  # 551
# for i in range(550):
#     a = np.var(x[i:i+60])
#     x_mv60 = np.append(x_ma60, a)
#
# x_mv60 = x_mv60[:549]
#
# data = np.vstack((x_diff, x_ma5, x_ma20, x_ma60, x_mv5, x_mv20, x_mv60))
# print(np.shape(data))
# print(type(data))
#
# y1 = np.ones((1, 100))
#
#
# y0 = np.zeros((1, 10))
#
#
# y10 = np.append(y1, y0)
#
# y = np.empty(shape=[0, 550])
# for i in range(5):
#     y = np.append(y, y10)
#
# y = y[:549]
# x = x[:549]
# set_01 = np.vstack((x, y))
# print(set_01)
#
# set_02 = pd.DataFrame(set_01)
# print(set_02)

plt.figure(figsize=(9, 6))
plt.plot(x)
plt.xlabel('Epoch')
plt.ylabel('Value of the random variable')
plt.savefig("model_m50.png")
plt.show()