import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from smec_ml.train_dqn_env import SmecEnv, LocalSearch

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
N_STATES = 73
env = SmecEnv(data_dir='../train_data/new/lunchpeak', render=False,
                       config_file='../smec_liftsim/rl_config2.ini')
from draw_data import print_hallcall_along_time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    # s_d: state and dispatch, samples x elev_num x dims(73)
    def forward(self, s_d):
        # s is the state of the env, d is the dispatch scheme
        x = self.fc1(s_d)
        x = F.relu(x)
        actions_value = self.out(x)
        actions_value = torch.sum(actions_value, dim=1)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.elev_num = 4
        self.floor_num = 16
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.searcher = LocalSearch()

    def separate_dispatches(self, k_dispatches):
        separate_dispatches = np.zeros((len(k_dispatches), self.elev_num, self.floor_num*2))
        for i, dispatch in enumerate(k_dispatches):
            for f in range(self.floor_num*2):
                if dispatch[f] != -1:
                    separate_dispatches[i][dispatch[f]][f] = 1
        return separate_dispatches

    def choose_action(self, x, add_hallcalls):
        if add_hallcalls == []:
            return None

        k_dispatches = self.searcher.get_k_dispatches(add_hallcalls)  # k x 32
        k_dispatches = self.separate_dispatches(k_dispatches)  # -> k x 4 x 32
        k_dispatches_tensor = torch.from_numpy(k_dispatches)

        k = k_dispatches.shape[0]

        x = torch.unsqueeze(torch.tensor(x, dtype=torch.float), 0)  # 1 x 4  x 41 (pos_vec, dir, car_call, encode_door)

        repeat_x = x.repeat(k, 1, 1)  # k x 4  x 39

        s_ds = torch.cat((repeat_x, k_dispatches_tensor), dim=-1)  # k x 4 x 73
        ps = self.eval_net.forward(s_ds).detach().cpu().numpy()  # k x 1
        dispatch_idx = np.random.choice(k, p=ps)
        action = dqn.searcher.dispatch2hallcalls(k_dispatches[dispatch_idx])
        s_d = s_ds[dispatch_idx, :, :]
        return action, s_d

    def choose_action_greedy(self, x):
        pass

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
dqn.searcher.bind_env(env)

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    # 重置权重参数
    print_hallcall_along_time(data_dir_prefix='../train_data/new/',
                              fileid=env.mansion.person_generator.file_idx)
    ep_r = 0
    while True:
        add_hallcalls = env.get_unallocate()

        # env.render()
        a, s_d = dqn.choose_action(s, add_hallcalls)

        # take action
        s_, r, done, info = env.step(a)

        # TODO: modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        if a is not None:
            dqn.store_transition(s, a, r, s_)
            ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break

        s = s_

