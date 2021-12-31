# import itertools
# import copy
# 
# 
# def get_k_elev_group(k, elev_num):
#     tmp_result = [[i] for i in range(elev_num)]
#     for i in range(k-1):
#         result = []
#         for tr in tmp_result:
#             for j in range(elev_num):
#                 result.append(tr+[j])
#         tmp_result = result
#     return tmp_result
# 
# np_1_4 = get_k_elev_group(1, 4)
# np_2_4 = get_k_elev_group(2, 4)
# np_3_4 = get_k_elev_group(3, 4)
# 
# 
# 
# k = 3
# 
# # products = []
# elev_num = 4
# # pd = itertools.permutations([0, 1, 2, 3])
# # print(list(pd))
# 
# to_serve_calls = [3, 5, 12, 27, 31]
# choose_idxes = to_serve_calls
# combinations = itertools.combinations(choose_idxes, k)
# # 剪枝啊，对那些明显很差的，就不要排列组合了？
# for c in combinations:
#     print(c)
#     # new_dis = copy.deepcopy(to_serve_calls)
#     # for i in range(k):
#     #     idx = c[i]
#     #     for e in range(elev_num):
#     #         new_new_dis = copy.deepcopy(new_dis)
#     #         new_new_dis[idx] = e
# 
# 
# 
# 
# 
# 
# 
# 

import torch
a = torch.ones((4, 8))
b = torch.sum(a, dim=1)
c = a.repeat(4,1,1)
print(a)
print(b)
print(c)