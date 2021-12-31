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
from sklearn.linear_model import LinearRegression
import json
from sklearn.model_selection import train_test_split


def read_data_from_txt(datafile='./dataset_noweight_clean2.txt'):
# def read_data_from_txt(datafile='./dataset_small_clean2.txt'):
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
            door_vec = [0,0,0,0]
            door_vec[door_state] = 1
            loss = data['loss']

            if loss > 500:
                loss = 500

            # elev_info = pos_vec + dir_vec + [vol, door_state] + car_call + up_call + dn_call
            # elev_info = pos_vec + [data['dir'], vol, door_state] + car_call + up_call + dn_call
            # elev_info = [data['pos'] / 45, data['dir'], vol, door_state] + car_call + up_call + dn_call
            # elev_info = dir_vec + [data['pos'] / 45, vol, door_state] + car_call + up_call + dn_call
            # elev_info = pos_vec + dir_vec + car_call + up_call + dn_call + [vol] + door_vec
            elev_info = [data['pos'] / 45, data['dir'], vol, door_state] + car_call + up_call + dn_call

            all_data.append(elev_info)
            losses.append(loss)
    return np.array(all_data), np.array(losses) / 500


X, y = read_data_from_txt()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

predict_value = model.predict(X_test)

import pickle
with open('clf.pickle', 'wb') as f:
    pickle.dump(model, f)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predict_value)
print("Diff: {} {}".format(np.sqrt(mse), mse))

print(X_test.shape)
print(model.coef_)
print(model.intercept_)

for i in range(X_test.shape[0]):
    x = X_test[i]
    pv = model.predict(x.reshape((1, -1)))
    print('predict: ', pv)
    print('real: ', y_test[i])
    a = input('')

