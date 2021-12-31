import torch.nn as nn
import torch
import numpy as np


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))


class MLPModule(nn.Module):
    # def __init__(self, input_size, hidden_size, active_func=nn.Tanh):
    def __init__(self, input_size, hidden_size, active_func=nn.ReLU):
        super(MLPModule, self).__init__()

        self.input_size, self.hs = input_size, hidden_size
        self.encoder = nn.Sequential(
            init_(nn.Linear(self.input_size, hidden_size)), active_func(),
            init_(nn.Linear(hidden_size, hidden_size)), active_func())

    def forward(self, x):
        return self.encoder(x)


# 多试几种，一种直接把所有信息给网络自己处理，一种帮它分一下三段route，一种直接把路径给出来（但是就丢失了预测的信息？）
class ElevEncoder1(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(ElevEncoder1, self).__init__()
        self.floor_num = floor_num
        self.floor_height = floor_height
        self.up_wait_encoder = MLPModule(input_size=floor_num, hidden_size=32)
        self.dn_wait_encoder = MLPModule(input_size=floor_num, hidden_size=32)
        self.car_wait_encoder = MLPModule(input_size=floor_num, hidden_size=32)
        self.elev_srv_dir_encoder = MLPModule(input_size=floor_num, hidden_size=8)
        self.elev_position_encoder = MLPModule(input_size=floor_num, hidden_size=8)
        self.elev_door_enconder = nn.Embedding(4, 8)  # 0 close, 1 opening, 2 open, 3 closing
        self.hidden_dim = 120

    @staticmethod
    def pos2vec(floor_num, pos, step):
        res = [0 for i in range(floor_num)]
        begin = int(pos // step)
        pre = pos % step
        res[begin] = 1 - pre / step
        if begin + 1 < floor_num:
            res[begin + 1] = pre / step
        return res

    @staticmethod
    def dir2vec(floor_num, pos, step, srv_dir):
        res = [0 for i in range(floor_num)]
        flr = int(pos // step)
        is_half = pos % step != 0
        if srv_dir == 1:
            if is_half:
                flr += 1
            for i in range(flr, floor_num):
                res[i] = 1
            for i in range(0, flr):
                res[i] = -1
        elif srv_dir == -1:
            for i in range(flr+1, floor_num):
                res[i] = -1
            for i in range(0, flr+1):
                res[i] = 1
        return res

    @staticmethod
    def call2vec(floor_num, call):
        vec = [0 for i in range(floor_num)]
        for c in call:
            vec[c] = 1
        return vec

    def forward(self, elev_info):
        pos_vec, dir_vec, vol, door_state, car_call, up_call, dn_call = elev_info.split([16, 16, 1, 1, 16, 16, 16], dim=-1)
        door_state = door_state.long()

        encode_pos = self.elev_position_encoder(pos_vec)
        encode_door = self.elev_door_enconder(door_state)
        encode_door = encode_door.squeeze(dim=1)
        encode_dir = self.elev_srv_dir_encoder(dir_vec)
        encode_carcall = self.car_wait_encoder(car_call)
        encode_upcall = self.up_wait_encoder(up_call)
        encode_dncall = self.dn_wait_encoder(dn_call)

        features = torch.cat((encode_pos, encode_door, encode_dir, encode_carcall, encode_upcall, encode_dncall), dim=-1)
        return features


class PretrainModel(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(PretrainModel, self).__init__()
        self.floor_num = floor_num
        self.floor_height = floor_height
        self.elev_encoder = ElevEncoder1(floor_num, floor_height)
        self.linear1 = nn.Linear(self.elev_encoder.hidden_dim, 32)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, elev_infos):
        y = self.elev_encoder(elev_infos)
        y = self.sigmoid2(self.linear2(self.relu1(self.linear1(y))))
        return y


class PretrainModel2(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(PretrainModel2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(82, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, elev_infos):
        y = self.model(elev_infos)
        return y


class PretrainModel3(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(PretrainModel3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(82, 1),
            # nn.Sigmoid(),
        )

    def forward(self, elev_infos):
        y = self.model(elev_infos)
        return y


class ElevEncoder2(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(ElevEncoder2, self).__init__()
        self.floor_num = floor_num
        self.floor_height = floor_height
        self.elev_srv_dir_encoder = nn.Embedding(3, 8)
        self.elev_door_enconder = nn.Embedding(4, 8)  # 0 close, 1 opening, 2 open, 3 closing
        self.hidden_dim = 73

    def forward(self, elev_info):
        pos_vec, vol, dir, door_state, car_call, up_call, dn_call = elev_info.split([16, 1, 1, 1, 16, 16, 16], dim=-1)
        door_state = door_state.long()
        # dir = dir.long()
        # print(door_state)
        encode_door = self.elev_door_enconder(door_state).squeeze(dim=1)
        # print(dir)
        # encode_dir = self.elev_srv_dir_encoder(dir).squeeze(dim=1)

        features = torch.cat((pos_vec, dir, car_call, up_call, dn_call, encode_door), dim=-1)
        return features


class PretrainModel4(nn.Module):
    def __init__(self, floor_num, floor_height):
        super(PretrainModel4, self).__init__()
        self.floor_num = floor_num
        self.floor_height = floor_height
        self.elev_encoder = ElevEncoder2(floor_num, floor_height)
        self.linear = nn.Linear(self.elev_encoder.hidden_dim, 1)

    def forward(self, elev_infos):
        y = self.elev_encoder(elev_infos)
        y = self.linear(y)
        return y


class MLModel(nn.Module):
    def __init__(self, elev_num, floor_num, floor_height):
        super(MLModel, self).__init__()
        self.elev_num = elev_num
        self.floor_num = floor_num
        self.floor_height = floor_height
        # # 不同电梯不同参数
        # self.elev_encoders = [ElevEncoder1(floor_num, floor_height) for i in range(self.elev_num)]
        # self.elev_mlps = [MLPModule(input_size=self.elev_encoders[i].hidden_dim, hidden_size=32) for i in range(self.elev_num)]
        # 不同电梯共用参数
        self.elev_encoder = ElevEncoder1(floor_num, floor_height)
        self.elev_mlp = MLPModule(input_size=self.elev_encoder.hidden_dim, hidden_size=32)
        self.score_func = MLPModule(input_size=32*self.elev_num, hidden_size=1)

    def forward(self, elev_infos):
        elev_outputs = []
        for i in range(self.elev_num):
            # elev_outputs.append(self.elev_mlps[i](self.elev_encoders[i](elev_infos[i])))  # 不同电梯不同参数
            elev_outputs.append(self.elev_mlp(self.elev_encoder(elev_infos[i])))  # 不同电梯共用参数
        elev_output_cat = torch.cat(elev_outputs, dim=-1)
        score = self.score_func(elev_output_cat)
        return score



