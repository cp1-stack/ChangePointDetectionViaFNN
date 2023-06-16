import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
import torch.nn.functional as F

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

train_dataset = CPDatasets('ChangePoints_data_train_betas.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2)

test_dataset = CPDatasets('ChangePoints_data_test_betas.csv')
test_loader = DataLoader(dataset=test_dataset, batch_size=500, shuffle=False, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 10),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(10, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 4)
        )
        # self.liner1 = torch.nn.Linear(8, 4)
        # self.liner2 = torch.nn.Linear(6, 4)
        # self.liner3 = torch.nn.Linear(2, 1)
        # self.sigmoid = torch.nn.Sigmoid()
        # self.liner3 = torch.nn.Linear(4, 4)

    def forward(self, x):
        # x = self.sigmoid(self.liner1(x))
        # x = self.drop(x)
        # x = self.sigmoid(self.liner2(x))
        # x = self.sigmoid(self.liner3(x))
        x = self.net(x)
        return x


model = Model()
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.5)


if __name__ == '__main__':

    Epoch = 200
    total_train_step = 0
    acc_list = []
    # rmse = []
    for epoch in range(Epoch):
        print('---------第{}轮训练开始---------'.format(epoch + 1))
        logger.info('---------第{}轮训练开始---------'.format(epoch + 1))
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels.squeeze(1).long())
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
        correct = 0
        total = 0
        with torch.no_grad():

            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                # _, pred = torch.max(outputs.data, dim=1)
                # correct += (pred == labels).sum().item()
                # total += labels.size(0)
                # print('acc = {}'.format(correct/1000))
                loss = criterion(outputs, labels.squeeze(1).long())
                # accuracy = (outputs.argmax(1) == labels).sum()
                # total_accuracy = total_accuracy + accuracy
                y_pred_label = outputs.argmax(1)

                acc = torch.eq(y_pred_label, labels.squeeze(1).long()).sum().item() / len(labels)

                score = metrics.adjusted_rand_score(Tensor.cpu(labels).reshape(500, ), Tensor.cpu(y_pred_label))
                print("loss = ", loss.item(), "acc = ", acc, 'ARI=', score)
                # print(torch.eq(y_pred_label, labels.squeeze(1).long()).sum().item())
                # print(y_pred_label.sum().item())
                # logger.info(f'loss = {loss.item()} acc =  {acc}')
                logger.info(f'{y_pred_label}')

                # result = [outputs for outputs in outputs if min(abs(outputs-0.5))<0.05]
                # print(result)
                # outputs_np = outputs.numpy()
                # index = np.where(abs(outputs_np-0.5)<0.05)
                # print(np.argmax(y_pred_label.numpy()))
                # print(y_pred_label.numpy().argmax())
                # logger.info(f'{((np.argmax(y_pred_label.numpy())-50)/100)**2}')
                # rmse.append(((y_pred_label.numpy().argmax()-50)/100)**2)
                # print(rmse)
                # rmse.append((np.argmax(y_pred_label.numpy())-50)**2)
                acc_list.append(acc)
    # plt.plot(acc_list)
    # plt.show()z

    # print('整体测试集上的正确率：{}%'.format(total_accuracy / len(test_dataset)))
    # print(np.sqrt(sum(rmse[51:])/50))
    logger.info(f'{acc_list}')