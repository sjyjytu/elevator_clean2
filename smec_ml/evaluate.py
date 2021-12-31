# 先用公式计算的loss生成一堆数据集进行预训练，
# 之后再根据平均到达时间和等待时间来调

# 生成数据集，电梯位置，速度，hallcall，carcall情况。
# 最好是在跑实际数据和搜索算法的时候生成数据？
# 数据格式：[{'pos': pos_vec, 'door_state': door_state, 'dir': dir_vec, 'vol': vol, 'car_call': car_call, 'up_call': up_call, 'dn_call': dn_call, 'loss'} for i in range(elev_num)] [loss for i in range(elev_num)]
# 怎么利用这个loss呢，应该有个baseline？
# 先训练Elevinfo?

from smec_ml.loss_model import *

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader # 方便数据的装载和加载
from sklearn.preprocessing import MinMaxScaler #对numpy数据进行标准化
import json


class DatasetFromTXT(Dataset):
    def __init__(self, transforms=None):
        print('loading data...')
        self.data, labels = self.read_data_from_txt()
        print(f'load data successfully with {len(self.data)} piece of data!')
        labels = np.array(labels)
        labels = labels.reshape((-1, 1))
        self.scaler = MinMaxScaler()

        labels = self.scaler.fit_transform(labels) # 对标签的每列进行标准化(映射到0-1)
        # print(labels.shape)
        labels = labels.reshape(-1, )
        self.labels = labels
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

    def _is_data_null(self, data):
        # if data['pos'] == 0 and data['dir'] == 0 and data['vol'] == 0 and data['door_state'] == 0 and\
        #         data['car_call'] == [] and data['up_call'] == [] and data['dn_call'] == []:
        if data['loss'] == 0:
            return True
        return False

    def read_data_from_txt(self, datafile='./dataset_small.txt'):
        all_data = []
        losses = []
        with open(datafile) as f:
            for line in f:
                if line == '' or line == '\n':
                    continue
                # line = line.replace('\'', '\"')
                # print(line)
                data = json.loads(line)
                # {'pos': , 'door_state': , 'dir': , 'vol': , 'car_call': ,
                #  'up_call': , 'dn_call': , 'loss': }
                pos_vec = ElevEncoder1.pos2vec(16, data['pos'], 3)
                dir_vec = ElevEncoder1.dir2vec(16, data['pos'], 3, data['dir'])
                car_call = ElevEncoder1.call2vec(16, data['car_call'])
                up_call = ElevEncoder1.call2vec(16, data['up_call'])
                dn_call = ElevEncoder1.call2vec(16, data['dn_call'])
                vol = data['vol']
                door_state = data['door_state']
                loss = data['loss']

                # if loss > 10000 or loss == 0:
                #     continue

                # elev_info = {}
                # elev_info['pos_vec'] = pos_vec
                # elev_info['dir_vec'] = dir_vec
                # elev_info['vol'] = vol
                # elev_info['door_state'] = door_state
                # elev_info['car_call'] = car_call
                # elev_info['up_call'] = up_call
                # elev_info['dn_call'] = dn_call
                elev_info = pos_vec + dir_vec + [vol, door_state] + car_call + up_call + dn_call

                all_data.append(elev_info)
                losses.append(loss)
        return np.array(all_data), losses


n_epochs = 1000  # 训练轮数epoch 的数目
batch_size = 64  # 决定每次读取多少条数据
log_interval = 100 # 间隔几次进行打印训练信息
test_interval = 10 # 间隔几次进行打印训练信息


dataset = DatasetFromTXT()
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

model = PretrainModel(16, 3)
model.load_state_dict(torch.load('./save_models/model_0016.pt'))


def train():
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # 定义损失函数和优化器
    lossfunc = torch.nn.MSELoss()
    best_loss = 10000
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_num = 0
        model.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            target = torch.tensor(target, dtype=torch.float32)
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()
            train_num += target.shape[0]
            if i % log_interval == 0:  # 准备打印相关信息
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, loss.item()))
        print('Train Epoch: {} \t Average Loss: {:.6f}'.format(
                    epoch, train_loss))
        if (epoch + 1) % 1 == 0:
            loss, test_num = 0.0, 0
            with torch.no_grad():  # 不会求梯度、反向传播
                model.eval()  # 不启用 BatchNormalization 和 Dropout
                for X, y in test_loader:
                    loss += lossfunc(model(X), y).float().item()
                    test_num += y.shape[0]
                print('test loss %.6f' % (loss))
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), f'./save_models/model_{str(epoch + 1).zfill(4)}.pt')
                print(f'Saving model with test loss {loss:.6f} ...')


# train()


def evaluate():
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for i, (data, target) in enumerate(test_loader):
        print(data)
        print('real: ', target, dataset.scaler.inverse_transform(target.numpy().reshape(-1, 1)))
        with torch.no_grad():
            y = model(data)
        print('predict: ', y, dataset.scaler.inverse_transform(y.numpy().reshape(-1, 1)))
        a = input('')


evaluate()



