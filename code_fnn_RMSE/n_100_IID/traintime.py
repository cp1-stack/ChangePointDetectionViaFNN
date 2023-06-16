import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
import seaborn as sns
import os

logger = logging.getLogger('outputs')
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(filename='test_pred.log')
fileHandler.setLevel(logging.INFO)

logger.addHandler(fileHandler)


class CPDatasets(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# xy = np.loadtxt('ChangePoints_data.csv', delimiter=',', dtype=np.float32)
# x_data = torch.from_numpy(xy[:, :-1])
# y_data = torch.from_numpy(xy[:, [-1]])


p = 500
n = 1
tao_star = 0.5
theta_0 = 0.7


RMSE = []
rmse_num =0
for _ in range(100):
    t1 = time.time()

    def df():

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

        dependent = Dependent()
        M = dependent.M()
        V_sd = dependent.V_sd()

        X = np.random.multivariate_normal(M, V_sd, n)[0]
        Theta1 = 1 * np.ones(p)
        Theta1[:int(tao_star * p)] = 0

        # Theta2 = np.zeros(p)
        # Theta2[int(tao_star * p)] = 1

        y = Theta1
        Z = theta_0 + Theta1 + X

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
        return dataframe


    # df().to_csv("ChangePoints_data_train_beta05.csv", mode='a', header=False, index=False, sep=',')
    # df().to_csv("ChangePoints_data_train_beta05.csv", mode='a', header=False, index=False, sep=',')
    # df().to_csv("ChangePoints_data_train_beta05.csv", mode='a', header=False, index=False, sep=',')
    # df().to_csv("ChangePoints_data_train_beta05.csv", mode='a', header=False, index=False, sep=',')
    # df().to_csv("ChangePoints_data_train_beta05.csv", mode='a', header=False, index=False, sep=',')

    df().to_csv("ChangePoints_data_test_beta05.csv", mode='w', header=False, index=False, sep=',')

    train_dataset = CPDatasets('ChangePoints_data_train_beta05.csv')
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)

    test_dataset = CPDatasets('ChangePoints_data_test_beta05.csv')
    test_loader = DataLoader(dataset=test_dataset, batch_size=p, shuffle=False, num_workers=0)


    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.liner1 = torch.nn.Linear(8, 1)
            # self.liner2 = torch.nn.Linear(7, 2)
            # self.liner3 = torch.nn.Linear(2, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.sigmoid(self.liner1(x))
            # x = self.sigmoid(self.liner2(x))
            # x = self.sigmoid(self.liner3(x))
            return x


    model = Model()
    # model = model.cuda()
    criterion = torch.nn.BCELoss()
    # criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    if __name__ == '__main__':

        Epoch = 100
        total_train_step = 0
        acc_list = []
        rmse = []
        Loc = []
        statis = []

        for epoch in range(Epoch):
            print('---------第{}轮训练开始---------'.format(epoch + 1))
            logger.info('---------第{}轮训练开始---------'.format(epoch + 1))
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # inputs = inputs.cuda()
                # labels = labels.cuda()
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                # print(epoch, loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_step = total_train_step + 1
                # if total_train_step % 1 == 0:
                # print('训练次数：{}， loss：{}'.format(total_train_step, loss.item()))

                # if True:
                #     y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
                #
                #     acc = torch.eq(y_pred_label, labels).sum().item() / labels.size(0)
                #     print("loss = ", loss.item(), "acc = ", acc)

            model.eval()
            total_accuracy = 0

            with torch.no_grad():

                for i, data in enumerate(test_loader, 0):
                    inputs, labels = data
                    # inputs = inputs.cuda()
                    # labels = labels.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    accuracy = (outputs.argmax(1) == labels).sum()
                    total_accuracy = total_accuracy + accuracy
                    y_pred_label = torch.where(outputs >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

                    acc = torch.eq(y_pred_label, labels).sum().item() / labels.size(0)

                    print("loss = ", loss.item(), "acc = ", acc)
                    # print(y_pred_label.sum().item())
                    # logger.info(f'loss = {loss.item()} acc =  {acc}')
                    # logger.info(f'{outputs}')

                    # result = [outputs for outputs in outputs if min(abs(outputs-0.5))<0.05]
                    # print(result)
                    # outputs_np = outputs.numpy()
                    # index = np.where(abs(outputs_np-0.5)<0.05)
                    # print(np.argmax(y_pred_label.numpy()))
                    print(y_pred_label.numpy().argmax())
                    # logger.info(f'{((np.argmax(y_pred_label.numpy())-50)/100)**2}')


                    # Loc.append(y_pred_label.numpy().argmax())
                    # print(outputs.numpy())
                    # print(np.shape(outputs.numpy().reshape(-1)))
                    # statis.append(outputs.numpy().reshape(-1))

                    # print(rmse)
                    # rmse.append((np.argmax(y_pred_label.numpy())-50)**2)
        # plt.plot(acc_list)
        # plt.show()

        # print('整体测试集上的正确率：{}%'.format(total_accuracy / len(test_dataset)))

        print(np.sqrt(sum(rmse[51:]) / 50))
        # print(Loc)
        # if acc > 0.9:
        #     RMSE.append(np.sqrt(sum(rmse[91:]) / 10))
        #     print(RMSE)

        # os.remove("ChangePoints_data_train_beta05.csv")
        os.remove("ChangePoints_data_test_beta05.csv")
        # print(np.shape(statis))
        RMSE.append(time.time() - t1)
        print(RMSE)
        # plt.hist(Loc)
        # plt.show()
        # np.savetxt('statis07.csv', statis, fmt='%4f',delimiter=',')
np.savetxt('TrainTime100.csv', RMSE, fmt='%4f', delimiter=',')
