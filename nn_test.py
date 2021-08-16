from pathlib import Path
import requests
import torch
import pickle
import gzip
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from IPython.core.debugger import set_trace
from matplotlib import pyplot
import pylab
import numpy as np

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/ "
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape


# 激活函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


# 损失函数（负对数似然函数）
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


# 使用nn.Functional重构，替代激活函数和损失函数
loss_func = F.cross_entropy


# def model(xb):
#     # @为点积
#     # return log_softmax(xb @ weights + bias)
#     return xb @ weights + bias


# 计算模型精确度
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


# 继承于nn.Module类建立对象类
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用nn.Linear建立线性层，替代手动定义和初始化权重、偏置值及其他工作
        self.lin = nn.Linear(784, 10)
        # self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        # self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return self.lin(xb)
    # 设置权重和偏置值
    # randn以正态分布随机生成
    # weights = torch.randn(784, 10) / math.sqrt(784)
    # weights.requires_grad_()
    # bias = torch.zeros(10, requires_grad=True)
    # print(weights.size())


# 使用optim重构，使用step方法进行前进步骤
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


# def fit():
#     for epoch in range(epochs):
#         for i in range((n - 1) // bs + 1):
#             xb, yb = train_ds[i * bs: i * bs + bs]
#             pred = model(xb)
#             loss = loss_func(pred, yb)
#
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#             with torch.no_grad():
#                 weights -= weights.grad * lr
#                 bias -= bias.grad * lr
#                 weights.grad.zero_()
#                 bias.grad.zero_()


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


# 实例化对象
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
model, opt = get_model()
bs = 64
# xb = x_train[0:bs]
# yb = y_train[0:bs]
# 使用Dataset和DataLoader重构
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

# 添加验证
for epoch in range(epochs):
    # 启用BatchNormalization和Dropout
    model.train()
    for xb, yb in train_dl:
        loss_batch(model, loss_func, xb, yb, opt)
    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, ":", valid_loss / len(valid_dl))

print("损失：", loss_func(model(xb), yb))
print("准确度：", accuracy(model(xb), yb))
