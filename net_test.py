#!/user/bin/env python
# coding=utf-8
"""
@project : PythonPytorch
@author  : shanyi
#@file   : net_test.py
#@ide    : PyCharm
#@time   : 2021-08-13 15:01:41
"""
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets


def get_data():
    bs = 64
    train_dataloder = DataLoader(
        datasets.MNIST('data', download=True, train=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])),
        batch_size=bs, shuffle=True)

    valid_dataloder = DataLoader(
        datasets.MNIST('data', download=True, train=False, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])),
        batch_size=bs * 2)
    return train_dataloder, valid_dataloder


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.Cov1 = nn.Sequential(
            nn.Conv2d(1, 6, [5, 5], 1),
            nn.Sigmoid(),
            nn.MaxPool2d([2, 2], 2),
        )
        self.Cov2 = nn.Sequential(
            nn.Conv2d(6, 16, [5, 5], 1),
            nn.Sigmoid(),
            nn.MaxPool2d([2, 2], 2),
        )
        self.Cov3 = nn.Sequential(
            nn.Conv2d(16, 120, [5, 5], 1),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        outlayer = F.sigmoid
        x = self.Cov1(x)
        x = self.Cov2(x)
        x = self.Cov3(x)
        x = x.view(-1, 120)
        x = outlayer(self.fc1(x))
        x = self.fc2(x)
        return x


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.Cov1 = nn.Sequential(
            nn.Conv2d(1, 96, [11, 11], 4, 2),
            nn.ReLU(True),
            nn.MaxPool2d([3, 3], 2),
        )
        self.Cov2 = nn.Sequential(
            nn.Conv2d(96, 256, [5, 5], 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d([3, 3], 2),
        )
        self.Cov3 = nn.Sequential(
            nn.Conv2d(256, 384, [3, 3], 1, 1),
            nn.ReLU(True),
        )
        self.Cov4 = nn.Sequential(
            nn.Conv2d(384, 384, [3, 3], 1, 1),
            nn.ReLU(True),
        )
        self.Cov5 = nn.Sequential(
            nn.Conv2d(384, 256, [3, 3], 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d([3, 3], 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.Cov1(x)
        x = self.Cov2(x)
        x = self.Cov3(x)
        x = self.Cov4(x)
        x = self.Cov5(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def get_model():
    model_temp = Alexnet()
    # LeNet优化器选择Adam
    opt_temp = optim.SGD(model_temp.parameters(), lr=1e-2)
    return model_temp, opt_temp


def loss_batch(model_temp, xb_temp, yb_temp, opt_temp=None):
    loss_func = F.cross_entropy
    loss = loss_func(model_temp(xb_temp), yb_temp)
    if opt_temp is not None:
        loss.backward()
        opt_temp.step()
        opt_temp.zero_grad()
    return loss.item(), len(xb_temp)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 5
    loss_func = F.cross_entropy
    train_dl, valid_dl = get_data()
    model, opt = get_model()
    model = model.to(device)
    for e in range(epoch):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss_batch(model, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb.to(device)), yb.to(device)) for xb, yb in valid_dl)
        print(e, ":", valid_loss / len(valid_dl))

print("损失：", loss_func(model(xb), yb))
print("准确度：", accuracy(model(xb), yb))
