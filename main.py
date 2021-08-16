
import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import gzip
from pathlib import Path
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

# 设置数据位置
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

# 取数据并放入tensor
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape


# 计算模型精确度
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


# 构造对象类（继承于nn.Module并利用nn.linear优化）
class MnistData(nn.Module):
    def __init__(self):
        super(MnistData, self).__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, x):
        return self.lin(x)


# 实例化对象和构建一个optimizer对象
def getmodel():
    model = MnistData()
    return model, optim.SGD(model.parameters(), lr=lr)


# 使用nn.functional配置损失函数
loss_functional = F.cross_entropy

# 使用Dataset与DataLoader包装数据及加载
size_batch = 64
train_xy = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_xy, batch_size=size_batch, shuffle=True)
valid_xy = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_xy, batch_size=size_batch * 2)

lr = 0.5  # 学习率
model, opt = getmodel()
epochs = 2  # 周期
writer = SummaryWriter()
for e in range(epochs):
    # 启用BatchNormalization和Dropout
    model.train()
    for xt, yt in train_dl:
        loss = loss_functional(model(xt), yt)
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_functional(model(xb), yb) for xb, yb in valid_dl)
        print(e, ":", valid_loss / len(valid_dl))

print("loss:", loss)
print("acc:", accuracy(model(xt), yt))

torch.save(model.state_dict(), 'models/mnist.ckpt')

model.load_state_dict(torch.load('models/mnist.ckpt', map_location=lambda storage, loc: storage))
model.eval()
# predict = F.softmax(model(x_train[0]), dim=0)
for num in range(10):
    predict = F.softmax(model(x_train[4]), dim=0)[num]
    print(predict)

