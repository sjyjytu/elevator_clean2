import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = 'cpu'

#  Hyperparameters
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 5
T_horizon = 20
learning_rate = 2.5e-4
MAX_PASSENGERS_LENGTH = 40
MAX_ELV_LENGTH = 10

add_people_at_step = 25
add_people_prob = 0.8

print_interval = 20
global_step = 0


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, output_size=1, encoding_dim=128, head_num=16, normalization=True):
        super(SequenceEncoder, self).__init__()
        self.normalization = normalization
        self.linear_1 = nn.Linear(input_dim, encoding_dim)
        self.attention = nn.MultiheadAttention(encoding_dim, head_num)

        self.conv_1 = nn.Conv1d(encoding_dim, encoding_dim, 1)
        self.conv_2 = nn.Conv1d(encoding_dim, encoding_dim * 1, 1)
        self.norm = nn.BatchNorm1d(encoding_dim * 1)
        self.linear_2 = nn.Linear(seq_len, output_size)

    def forward(self, x):
        """
        # input shape: [batch_size, max_seq_len, hidden_size]
        """
        x = self.linear_1(x)  # (batch_size, max_seq_len, encoding_dim)
        x = x.permute(1, 0, 2)  # (max_seq_len, batch_size, encoding_dim)
        x, _ = self.attention(x, x, x)  # (max_seq_len, batch_size, encoding_dim)
        x = x.permute(1, 2, 0)  # (batch_size, encoding_dim, max_seq_len)

        x_ = F.relu(self.conv_1(x))  # (batch_size, encoding_dim, max_seq_len)
        x_ = self.conv_2(x_)  # (batch_size, encoding_dim, max_seq_len)

        x = x_ + x
        ####if self.normalization == True:
        ####    x = self.norm(x)
        # x(batch_size, encoding_dim,max_seq_len)
        ##x,_ = self.lstm(F.relu(x.transpose(2,1)))
        ##x = x[:,-1,:]
        ##x = self.linear_2(F.relu(x))
        x = self.linear_2(F.relu(x)).squeeze(-1)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size, self.hs, self.os = input_size, hidden_size, output_size
        self.encoder = nn.Linear(self.input_size, self.hs)
        self.decoder = nn.Linear(self.hs, self.os)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(F.relu(encoded))
        return decoded


# noinspection PyCallingNonCallable
class SmecAgent(nn.Module):
    def __init__(self, lift_num, floor_num, sm):
        super(SmecAgent, self).__init__()
        self.lift_num = lift_num
        self.floor_num = floor_num
        self.memory = []
        self.up_wait_encoder = MLP(input_size=floor_num, hidden_size=32, output_size=32)
        self.down_wait_encoder = MLP(input_size=floor_num, hidden_size=32, output_size=32)
        self.elv_pass_encoder_global = MLP(input_size=floor_num, hidden_size=32, output_size=32)
        self.elv_pass_encoder_local = MLP(input_size=floor_num, hidden_size=32, output_size=32)
        self.elv_location_dic = nn.Embedding(num_embeddings=floor_num, embedding_dim=16)
        self.elv_location_encoder_global = MLP(input_size=16, hidden_size=16, output_size=8)
        self.elv_location_encoder_local = MLP(input_size=16, hidden_size=16, output_size=8)

        self.action_mlp = MLP(input_size=152, hidden_size=128, output_size=floor_num * 2)
        self.value_mlp = MLP(input_size=152, hidden_size=128, output_size=1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.sm = sm

    def encode_state(self, upload_wait_nums, download_wait_nums, elevator_loading_maps, elevator_location_maps):
        # input shape: [BS, dim]; output shape: [ELV, BS, dim].
        upload_wait_f = self.up_wait_encoder(upload_wait_nums)
        download_wait_f = self.down_wait_encoder(download_wait_nums)
        all_elv_pass = self.elv_pass_encoder_global(elevator_loading_maps)
        elv_pass_global = torch.mean(all_elv_pass, dim=1)
        all_elv_loc = self.elv_location_dic(elevator_location_maps)
        elv_loc_global = self.elv_location_encoder_global(all_elv_loc)
        elv_loc_global = torch.mean(elv_loc_global, dim=1)
        common_f = torch.cat((upload_wait_f, download_wait_f, elv_pass_global, elv_loc_global), -1)  # 152
        all_agent_features = []
        for idx in range(self.lift_num):
            cur_elv_pass = all_elv_pass[:, idx, :]
            cur_elv_loc = all_elv_loc[:, idx, :]
            local_f = torch.cat((cur_elv_pass, cur_elv_loc), -1)
            all_f = torch.cat((common_f, local_f), -1)
            all_agent_features.append(all_f)
        all_agent_features = torch.stack(all_agent_features)
        return all_agent_features

    def get_action(self, upload_wait_nums, download_wait_nums, elevator_loading_maps, elevator_location_maps):
        state_f = self.encode_state(upload_wait_nums, download_wait_nums, elevator_loading_maps, elevator_location_maps)
        action = self.action_mlp(state_f)
        action_prob = F.softmax(action, dim=-1)  # [ELV, BS, A]
        return action_prob

    def get_value(self, upload_wait_nums, download_wait_nums, elevator_loading_maps, elevator_location_maps):
        state_f = self.encode_state(upload_wait_nums, download_wait_nums, elevator_loading_maps, elevator_location_maps)
        value = self.value_mlp(state_f)
        print("value norm:", value.norm())
        return value

    def put_data(self, data):
        self.memory.append(data)

    def make_batch(self):
        s1s, s2s, s3s, s4s, action_list, reward_list, ns1s, ns2s, ns3s, ns4s, prob_list, done_list = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for data in self.memory:
            state, action, reward, next_state, prob, done = data
            s1s.append(state[0])
            s2s.append(state[1])
            s3s.append(state[2])
            s4s.append(state[3])
            action_list.append(action)
            reward_list.append(reward)
            prob_list.append(prob)
            ns1s.append(next_state[0])
            ns2s.append(next_state[1])
            ns3s.append(next_state[2])
            ns4s.append(next_state[3])
            done_mask = 0 if done else 1
            done_list.append(done_mask)
        self.memory = []  # clear memory
        s1s, s2s, s3s, s4s, a, r, ns1s, ns2s, ns3s, ns4s, done_mask, prob = \
            torch.tensor(s1s, dtype=torch.float).to(device), \
            torch.tensor(s2s, dtype=torch.float).to(device), \
            torch.tensor(s3s, dtype=torch.float).to(device), \
            torch.tensor(s4s, dtype=torch.long).to(device), \
            torch.tensor(action_list).long().to(device), torch.tensor(reward_list).float().to(device), \
            torch.tensor(ns1s, dtype=torch.float).to(device), \
            torch.tensor(ns2s, dtype=torch.float).to(device), \
            torch.tensor(ns3s, dtype=torch.float).to(device), \
            torch.tensor(ns4s, dtype=torch.long).to(device), \
            torch.tensor(done_list, dtype=torch.float).to(device), \
            torch.stack(prob_list).to(device)

        return s1s.squeeze(1), s2s.squeeze(1), s3s.squeeze(1), s4s.squeeze(1), a, r, \
               ns1s.squeeze(1), ns2s.squeeze(1), ns3s.squeeze(1), ns4s.squeeze(1), done_mask, prob

    def train_epoch(self, epoch):
        s1, s2, s3, s4, action, reward, ns1, ns2, ns3, ns4, done_mask, action_prob = self.make_batch()
        # apply reward to all elevators
        reward = reward.unsqueeze(0).repeat_interleave(self.lift_num, dim=0)
        done_mask = done_mask.unsqueeze(0).repeat_interleave(self.lift_num, dim=0)
        for i in range(K_epoch):
            v_t, v_next = self.get_value(s1, s2, s3, s4).squeeze(), self.get_value(ns1, ns2, ns3, ns4).squeeze()
            td_target = reward + gamma * v_next * done_mask
            delta = td_target - v_t
            delta = delta.cpu().detach().numpy()
            agent_num, time_num = delta.shape
            advantage_list = []
            advantage = np.array([0.0 for a in range(agent_num)])
            for time_step in reversed(range(time_num)):
                advantage = gamma * lmbda * advantage + delta[:, time_step]
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype=torch.float).to(device)  # [T, ELV]

            now_action = self.get_action(s1, s2, s3, s4).permute(1, 0, 2)
            now_action = now_action.gather(dim=2, index=action.unsqueeze(-1)).squeeze()

            ratio = torch.exp(torch.log(now_action) - torch.log(action_prob))
            # problem, now action is not the updated action
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss_1 = - torch.min(surr1, surr2).mean()
            loss_2 = F.smooth_l1_loss(v_t, td_target.detach())
            loss = loss_1 + loss_2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("loss1:", loss_1.item(), "loss2:", loss_2.item())
            if i == 0:
                self.sm.add_scalar('loss/loss_1', loss_1.item(), epoch)
                self.sm.add_scalar('loss/loss_2', loss_2.item(), epoch)
