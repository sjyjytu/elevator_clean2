from smec_ml.loss_model import *

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader # 方便数据的装载和加载
import json


class PretrainModel(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(PretrainModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(floor_num*2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, floor_num*2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y


class DatasetFromTXT(Dataset):
    def __init__(self, transforms=None):
        print('loading data...')
        self.data, self.labels = self.read_data_from_txt()
        print(f'load data successfully with {len(self.data)} piece of data!')
        print(self.labels.shape)
        self.transforms = transforms

    def __getitem__(self, index):
        single_data_label = self.labels[index]
        # data_as_dict = self.data[index]
        data_as_np = self.data[index]
        data_as_tensor = torch.from_numpy(data_as_np).float()
        return (data_as_tensor, single_data_label)
        # return (data_as_dict, single_data_label)

    def __len__(self):
        return len(self.data)

    def read_data_from_txt(self, datafile='./dataset2.txt'):
        Xs = []
        ys = []
        with open(datafile) as f:
            for line in f:
                if line == '' or line == '\n':
                    continue
                # line = line.replace('\'', '\"')
                # print(line)
                data = json.loads(line)
                x = data['x_prime']
                y = data['y']
                Xs.append(x)
                ys.append(y)
        return np.array(Xs), np.array(ys)


n_epochs = 10000  # 训练轮数epoch 的数目
batch_size = 32  # 决定每次读取多少条数据
log_interval = 100 # 间隔几次进行打印训练信息
test_interval = 10 # 间隔几次进行打印训练信息


dataset = DatasetFromTXT()
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

model = PretrainModel(5, 3)
model_index = 5325
model.load_state_dict(torch.load(f'./save_models2/model_{model_index}.pt'))


def train():
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # 定义损失函数和优化器
    lossfunc = torch.nn.MSELoss()
    best_loss = 10000
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_num = 0
        model.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            target = torch.tensor(target, dtype=torch.float32)
            mask = target > 0
            output = output * mask
            loss = lossfunc(output.squeeze(), target.squeeze())  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()
            train_num += target.shape[0]
            if i % log_interval == 0:  # 准备打印相关信息
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, loss.item()))
        print('Train Epoch: {} \t Average Loss: {:.6f}'.format(
                    epoch, train_loss / train_num))
        # if (epoch + 1) % 1 == 0:
        #     loss, test_num = 0.0, 0
        #     with torch.no_grad():  # 不会求梯度、反向传播
        #         model.eval()  # 不启用 BatchNormalization 和 Dropout
        #         for X, y in test_loader:
        #             loss += lossfunc(model(X).squeeze(), y.squeeze()).float().item()
        #             test_num += y.shape[0]
        #         print('test loss %.6f' % (loss / test_num))
        if train_loss < best_loss:
            best_loss = train_loss
            # torch.save(model.state_dict(), f'./save_models/model_{str(model_index + epoch + 1).zfill(4)}.pt')
            print(f'===========================Saving model {str(model_index + epoch + 1).zfill(4)}.pt with test loss {train_loss:.6f} ===========================')


# train()


def evaluate():
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    lossfunc = torch.nn.MSELoss()
    for i, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            # print(data)
            # print(target)
            y = model(data)
            mask = target > 0
            y = y * mask
            print("pred:", y.numpy())
            print("real:", target.numpy())
            loss = lossfunc(y, target)
            print("loss:", loss.item())
            delta = y - target
            print("delta", delta.numpy())
            a = input('')


evaluate()
