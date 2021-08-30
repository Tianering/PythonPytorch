#!/user/bin/env python
# coding=utf-8
"""
@project : PythonPytorch
@author  : shanyi
#@file   : MobileNet_v1.py
#@ide    : PyCharm
#@time   : 2021-08-30 10:20:26
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


class MobileNetv1(nn.Module):
    def __init__(self):
        super(MobileNetv1, self).__init__()

        # 标准卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 深度卷积
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        self.model = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return x


def get_model():
    model_temp = MobileNetv1()
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
