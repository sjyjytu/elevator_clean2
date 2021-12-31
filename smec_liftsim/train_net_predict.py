import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data

BATCH_SIZE = 32

model = nn.Sequential(
          nn.Linear(16, 64),
          nn.ReLU(),
          nn.Linear(64, 128),
          nn.ReLU(),
          nn.Linear(128, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 16)
        )
# model = nn.Sequential(
#           nn.Linear(16, 64),
#           nn.ReLU(),
#           nn.Linear(64, 64),
#           nn.ReLU(),
#           nn.Linear(64, 16)
#         )

# model.load_state_dict(torch.load('predict.pt'))

criterion = nn.MSELoss() # 损失函数为MSE
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.1)

dataX = np.load('dataX.npy').astype(np.float32)
dataY_wt = np.load('dataY_wt.npy').astype(np.float32)
dataY_tt = np.load('dataY_tt.npy').astype(np.float32)
dataY_at = np.load('dataY_at.npy').astype(np.float32)
# print(dataX.shape)
# print(dataX[0])
# print(dataY_wt[0])
# print(dataY_tt[0])
# print(dataY_at[0])

max_pnum = np.max(dataX)
print(max_pnum)

dataY = dataY_wt

norm_dataX = dataX / max_pnum
train_size = int(0.8 * len(norm_dataX))
# train_size = int(32)
trainX = norm_dataX[:train_size, :]
trainY = dataY[:train_size, :]
testX = norm_dataX[train_size:, :]
testY = dataY[train_size:, :]
testX_tensor = torch.from_numpy(testX)
testY_tensor = torch.from_numpy(testY)

train_dataset = Data.TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))  # 将x,y读取，转换成Tensor格式
loader = Data.DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # 最新批数据
    shuffle=True,  # 是否随机打乱数据
)

train_or_draw = True

if train_or_draw:
    # train
    for epoch in range(3000):
        model.train()
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
            out = model(batch_x)
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.item(), '| mean loss: ', total_loss / (step+1))

        # test
        model.eval()
        with torch.no_grad():
            out = model(testX_tensor)
            loss = criterion(out, testY_tensor)
            # print('Test loss: ', loss.item())
            print('Epoch %d | train loss: %.2f | Test loss: %.2f' % (epoch, total_loss / (step + 1), loss.item()))

    torch.save(model.state_dict(), 'predict.pt')

else:
    # draw
    model.load_state_dict(torch.load('predict.pt'))
    # for i in range(0, trainX.shape[0]):
    #     X = trainX[i]
    #     Y = trainY[i]
    for i in range(train_size, dataX.shape[0]):
        X = dataX[i]
        Y = dataY[i]
        X_tensor = torch.from_numpy(X).unsqueeze(0)
        predictY = model(X_tensor).cpu().detach().numpy().squeeze(0)
        # print(X)
        # print(Y)
        # print(predictY)

        plt.figure()
        plt.plot(Y, label='Y')
        plt.plot(predictY, label='predictY')
        plt.bar(range(len(X)), X * 50)

        plt.legend(loc='best')
        # plt.savefig('E:\\elevator\\predict_flow\\figures2\\%s.jpg' % (
        #             start_time[:2] + start_time[3:] + '-' + end_time[:2] + end_time[3:]))
        plt.show()
        actual_argmin = np.argsort(Y)
        predict_argmin = np.argsort(predictY)
        print('actual: ', actual_argmin)
        print('predic: ', predict_argmin)
        a = input('回车以继续...')
        plt.close()

# else:
#     # draw
#     model.load_state_dict(torch.load('predict.pt'))
#     for i in range(0, trainX.shape[0]):
#         X = trainX[i]
#         Y = trainY[i]
#
#         X_tensor = torch.from_numpy(X).unsqueeze(0)
#         predictY = model(X_tensor).cpu().detach().numpy().squeeze(0)
#         print(X)
#         print(Y)
#         print(predictY)
#
#         plt.figure()
#         plt.plot(Y, label='Y')
#         plt.plot(predictY, label='predictY')
#         plt.bar(range(len(X)), X * 50)
#
#         plt.legend(loc='best')
#         # plt.savefig('E:\\elevator\\predict_flow\\figures2\\%s.jpg' % (
#         #             start_time[:2] + start_time[3:] + '-' + end_time[:2] + end_time[3:]))
#         plt.show()
#         actual_argmin = np.argsort(Y)
#         predict_argmin = np.argsort(predictY)
#         # print('actual: ', actual_argmin)
#         # print('predic: ', predict_argmin)
#         a = input('回车以继续...')
#         plt.close()