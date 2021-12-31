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
        labels = np.array(labels) / 500
        # labels = np.array(labels)
        # labels = labels.reshape((-1, 1))
        # scaler = MinMaxScaler()

        # labels = scaler.fit_transform(labels) # 对标签的每列进行标准化(映射到0-1)
        # print(labels.shape)
        # labels = labels.reshape(-1, )
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

    def read_data_from_txt(self, datafile='./dataset_noweight_clean2.txt'):
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
                car_call = ElevEncoder1.call2vec(16, data['car_call'])
                up_call = ElevEncoder1.call2vec(16, data['up_call'])
                dn_call = ElevEncoder1.call2vec(16, data['dn_call'])
                vol = data['vol']
                dir = data['dir']
                door_state = data['door_state']
                loss = data['loss']

                # if loss > 500:
                #     loss = 500

                elev_info = pos_vec + [vol, dir, door_state] + car_call + up_call + dn_call

                all_data.append(elev_info)
                losses.append(loss)
        return np.array(all_data), losses


n_epochs = 1000  # 训练轮数epoch 的数目
batch_size = 32  # 决定每次读取多少条数据
log_interval = 100 # 间隔几次进行打印训练信息
test_interval = 10 # 间隔几次进行打印训练信息


dataset = DatasetFromTXT()
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

model = PretrainModel4(16, 3)
model.load_state_dict(torch.load('./save_models4/model_0550.pt'))


def train():
    train_loader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=False)
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
            torch.save(model.state_dict(), f'./save_models4/model_{str(epoch + 1).zfill(4)}.pt')
            print(f'===========================Saving model {str(epoch + 1).zfill(4)}.pt with test loss {train_loss:.6f} ===========================')


# train()


def evaluate():
    lf = torch.nn.MSELoss()
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            # print(data)
            print('actual: ', target)
            y = model(data).squeeze(1)
            print('predict:', y)
            loss = lf(y, target)
            print('loss: ', loss)
            a = input('')


evaluate()
