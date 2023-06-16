import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


xy = np.loadtxt('ChangePoints_data.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

dataset = CPDatasets('ChangePoints_data_train.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.liner1 = torch.nn.Linear(8, 4)
        self.liner2 = torch.nn.Linear(4, 2)
        self.liner3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.liner1(x))
        x = self.sigmoid(self.liner2(x))
        x = self.sigmoid(self.liner3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

epoch_list = []
loss_list = []
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
        print("loss = ", loss.item(), "acc = ", acc)
