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


for i in range(10, 1):
    print(i)