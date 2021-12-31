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
for i in range(64):

    print(i, 0.98**i)
