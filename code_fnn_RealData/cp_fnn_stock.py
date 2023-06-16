# import akshare as ak
import matplotlib.pyplot as plt

# stock_01 = ak.stock_zh_a_daily(symbol="sh600000", adjust="hfq")
# stock_volume = stock_01['volume']
# print(stock_01['volume'])
# plt.plot(stock_01['volume'])
# plt.show()
#
# stock_01.to_csv('data_sh600000.csv', sep=',', index=False, header=True)
# import pandas as pd
#
# data = pd.read_csv('data_sh600000.csv')
# stock_sh = data['turnover']
# plt.plot(stock_sh)
# plt.show()
# plt.plot(stock_01['volume'])
# plt.show()
import numpy as np
import pandas as pd
import seaborn as sns

col_name = ['Close', 'DIFF of Close', 'MA(5) of Close', 'MA(10) of Close', 'MA(20) of Close', 'MV(5) of Close',
            'MV(10) of Close', 'MV(20) of Close', 'date']

data = pd.read_csv('CPreal1.csv', names=col_name, header=None)
data.set_index(['date'], inplace=True)

data.plot(y='Close', legend=True, fontsize='12', rot=25, figsize=(12, 6))
plt.ylabel('Values of Close', fontsize='15')
plt.savefig("RealData_close.png", dpi=300)
plt.show()

data.plot(legend=True, fontsize='12', rot=25, figsize=(12, 6))
plt.ylabel('Values of Close', fontsize='15')
plt.savefig("RealData0.png", dpi=300)
plt.show()

data.plot(y='Close', legend=True, fontsize='12', rot=25, figsize=(12, 6))
plt.ylabel('Values of Close', fontsize='15')
plt.axvline(x=552, ls="-", c="red", linewidth=1, label='FNN estimation')
plt.axvline(x=556, ls="-", c="purple", linewidth=1, label='CUSUM-type estimation')
plt.legend()
plt.savefig("RealData_03.png", dpi=300)
plt.show()

# stock_sh = data['volume'][1000:3000]
# stock_sh.to_csv('data_sh600000_volume.csv', sep=',', index=False, header=False)

# print(data.head())
# plt.plot(data['turnover'])
# plt.show()

# plt.plot(data['turnover'][2000:2100])
# plt.show()
#
# sns.set_style("darkgrid")

# dt1 = pd.date_range(start="19991110", end="20080611", freq="D")


# close_data = pd.Series(data.iloc[:, 0].to_numpy(), index=pd.date_range("10/11/1999", periods=1998))
# close_data.plot(label='Close')
# plt.plot(data.iloc[:, 0], label='Close')
# plt.plot(data.iloc[:, 1], label='DIFF of Close')
# plt.plot(data.iloc[:, 2], label='MA(5) of Close')
# plt.plot(data.iloc[:, 3], label='MA(10) of Close')
# plt.plot(data.iloc[:, 4], label='MA(20) of Close')
# plt.plot(data.iloc[:, 5], label='MV(5) of Close')
# plt.plot(data.iloc[:, 6], label='MV(10) of Close')
# plt.plot(data.iloc[:, 7], label='MV(20) of Close')
# plt.axvline(x=1705, ls="-", c="red", linewidth=1, label='FNN estimation')
# plt.axvline(x=1699, ls="-", c="purple", linewidth=1, label='CUSUM estimation')
# plt.xlabel('Trading day')
# plt.ylabel('Values of Close')
# plt.legend()
# plt.savefig("RealData__.png")
# plt.show()

# data02 = pd.read_csv('data_sh600000.csv')
# plt.plot(data02['close'])
# plt.show()

# data03 = pd.read_csv('ChangePoints_data_train_realdata02.csv')
# plt.plot(data03)
# plt.show()
# data01.to_csv('data_sh600000_turn04.csv', sep=',', index=False, header=False)
