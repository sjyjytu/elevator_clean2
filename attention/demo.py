import torch
from self_attention_cv import MultiHeadSelfAttention, SelfAttention
from offline_tools.generate_dataset import *
from draw_data import process2, dt as delta_t

# model = MultiHeadSelfAttention(dim=64)
# x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
# mask = torch.zeros(10, 10)  # tokens X tokens
# mask[5:8, 5:8] = 1
# y = model(x, mask)
# print(y)
# print(y.size())

# model = SelfAttention(8)
# x = torch.rand(1,2,8)
# print(x)
# y = model(x)
# print(y)

data_of_section = '00:00-06:00'
data_dir = '../train_data/new/lunchpeak/'

dataX = process2(data_dir=data_dir)  # file, t, s, d
dataX = np.average(dataX, axis=0)  # 均值 t,s,d  每分钟的人数
prob = dataX / delta_t  # 每分钟中，每秒的生成人的概率  t,s,d
print(prob.shape)
# print(prob)

# generate dataset
T, S, D = dataX.shape
generate_mask = np.random.random((delta_t, T, S, D))
generate_mask = generate_mask < prob  # delta_t, t, s, d
generate_mask = generate_mask.transpose((1,0,2,3))
generate_mask = generate_mask.reshape((-1, S, D))

print(generate_mask.shape)
