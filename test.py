# from copy import deepcopy
#
# class A:
#     def __init__(self):
#         self.best_scheme = {'d': [1], 's': 1}
#
#     def func(self):
#         best_scheme = {'d': self.best_scheme['d'], 's': self.best_scheme['s']}
#         new_dispatch = deepcopy(best_scheme['d'])
#         new_dispatch.append(2)
#         best_scheme = {'d': new_dispatch,
#                        's': 2}
#         self.best_scheme = best_scheme
#         new_dispatch = [3]
#         new_dispatch.append(3)
#
# a = A()
# print(a.best_scheme)
# a.func()
# print(a.best_scheme)
# for i in range(64):
#
#     print(i, 0.98**i)
# import numpy as np
# for i in range(10):
#
#     print(np.random.rand(1,2,3))

# a = {1:2, '3':4}
# b = {1:2, '3':4}
# print(a==b)
# def cal_cur_last_floor(self):
#     if self._current_velocity != 0:
#         flr = self._current_position // self._floor_height
#         if self._run_direction == -1:
#             flr += 1
#     else:
#         flr = self._current_position // self._floor_height
#     return flr
#
#
# def cal_cur_next_floor(self):
#     if self._current_velocity != 0:
#         flr = self._current_position // self._floor_height
#         if self._run_direction == 1:
#             flr += 1
#     else:
#         flr = self._current_position // self._floor_height
#     return flr
#
# class A:
#     def __init__(self):
#         self._current_velocity = 0.1
#         self._run_direction = -1
#         self._current_position = 3
#         self._floor_height = 3
#
# a = A()
# print(cal_cur_last_floor(a))
# print(cal_cur_next_floor(a))


# for i in range(10, 1):
#     print(i)

# import torch
#
# a = torch.arange(0, 24)
# a = a.view((2,3,4))
# print(a)
# a = a.permute(0,2,1).contiguous()
# print(a)

import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace

# Visualize gradients of simple MLP
# The method below is for building directed graphs of PyTorch operations, built during forward propagation and showing which operations will be called on backward. It omits subgraphs which do not require gradients.

# model=nn.Sequential()
# model.add_module("W0", nn.Linear(8, 16))
# model.add_module("tanh", nn.Tanh())
# model.add_module("W1", nn.Linear(16, 1))
#
# x = torch.randn(1, 8)
# y = model(x)
#
# make_dot(y.mean(), params=dict(model.named_parameters()))  # 直接在ipython notebook中显示
#
# dot=make_dot(y.mean(), params=dict(model.named_parameters()))
# dot.render("model.pdf")  #保存为pdf

# import numpy as np
# a = np.arange(0, 50)
# b = np.exp(-a / 20) + 1
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(b)
# plt.show()
# print(b)

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

time = np.arange(10)
temp = np.random.random(10)*30
Swdown = np.random.random(10)*100-10
Rn = np.random.random(10)*100-10

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(time, Swdown, '-', label = 'Swdown')
lns2 = ax.plot(time, Rn, '-', label = 'Rn')
ax2 = ax.twinx()
lns3 = ax2.plot(time, temp, '-r', label = 'temp')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Time (h)")
ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
ax2.set_ylabel(r"Temperature ($^\circ$C)")
ax2.set_ylim(0, 35)
ax.set_ylim(-20,100)
plt.show()