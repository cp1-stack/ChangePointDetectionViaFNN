import matplotlib.pyplot as plt
import pandas as pd




import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging

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

train_dataset = CPDatasets('ChangePoints_data_train_realdata02.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2)

test_dataset = CPDatasets('ChangePoints_data_train_realdata03.csv')
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=2)


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
    pred = []
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
                # logger.info(f'{y_pred_label}')
                # print(y_pred_label.numpy())

                # result = [outputs for outputs in outputs if min(abs(outputs-0.5))<0.05]
                # print(result)
                # outputs_np = outputs.numpy()
                # index = np.where(abs(outputs_np-0.5)<0.05)
                # print(np.argmax(y_pred_label.numpy()))
                print(y_pred_label.numpy().argmax())
                pred.append(y_pred_label.numpy().argmax())
                statis.append(outputs.numpy().reshape(-1))
                # logger.info(f'{((np.argmax(y_pred_label.numpy())-50)/100)**2}')
                # rmse.append(((y_pred_label.numpy().argmax()-250)/500)**2)
                # print(rmse)
                # rmse.append((np.argmax(y_pred_label.numpy())-50)**2)
    # plt.plot(acc_list)
    # plt.show()
        # print('整体测试集上的正确率：{}%'.format(total_accuracy / len(test_dataset)))
        #         plt.plot(y_pred_label.numpy())
        #         plt.show()
    print(sum(pred[51:]) / 50)
    # np.savetxt('statis_real.csv', statis, fmt='%4f', delimiter=',')