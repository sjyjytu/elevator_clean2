import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from smec_liftsim.smec_state import SmecState
from collections import OrderedDict


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class SmecRolloutStorage(object):
    def __init__(self, num_steps, env_num, elevator_num, floor_num, use_graph=True, use_advice=False):
        self.num_steps, self.env_num, self.elev_num, self.floor_num = num_steps, env_num, elevator_num, floor_num
        self.use_graph = use_graph
        assert use_graph
        self.graph_node_num = (self.elev_num + self.floor_num) * 2
        self.agent_num = self.floor_num * 2
        self.rewards = torch.zeros(self.num_steps, self.env_num, self.agent_num)
        self.value_preds = torch.zeros(self.num_steps + 1, self.env_num, self.agent_num)
        self.returns = torch.zeros(self.num_steps + 1, self.env_num, self.agent_num)
        self.action_log_probs = torch.zeros(self.num_steps, self.env_num, self.agent_num)
        self.actions = torch.zeros(self.num_steps, self.env_num, self.agent_num)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, env_num)
        self.step = 0
        if not use_graph:
            self.zero_obs = OrderedDict({'upload_wait_nums': torch.zeros(env_num, floor_num, ),
                                         'download_wait_nums': torch.zeros(env_num, floor_num, ),
                                         'elevator_loading_maps': torch.zeros((env_num, elevator_num, floor_num)),
                                         'elv_up_call': torch.zeros((env_num, elevator_num, floor_num)),
                                         'elv_down_call': torch.zeros((env_num, elevator_num, floor_num)),
                                         'elv_load_up': torch.zeros((env_num, elevator_num)),
                                         'elv_load_down': torch.zeros((env_num, elevator_num)),
                                         'elevator_location_maps': torch.zeros((env_num, elevator_num,)).long(),
                                         'legal_masks': torch.zeros(env_num, (floor_num + 1) * 2, )})
        else:
            candidate_num = elevator_num
            if use_advice:
                candidate_num += 1
            self.zero_obs = OrderedDict({'adj_m': torch.zeros(env_num, self.graph_node_num, self.graph_node_num),
                                         'node_feature_m': torch.zeros(env_num, self.graph_node_num, 3),
                                         'legal_masks': torch.zeros(env_num, floor_num * 2, candidate_num),
                                         'distances': torch.zeros(env_num, floor_num * 2, elevator_num),
                                         'valid_action_mask': torch.zeros(env_num, floor_num * 2)
                                         })
        self.obs = [self.zero_obs.copy() for j in range(num_steps + 1)]
        self.reset()

    def reset(self):
        self.rewards = torch.zeros(self.num_steps, self.env_num, self.agent_num)
        self.value_preds = torch.zeros(self.num_steps + 1, self.env_num, self.agent_num)
        self.returns = torch.zeros(self.num_steps + 1, self.env_num, self.agent_num)
        self.action_log_probs = torch.zeros(self.num_steps, self.env_num, self.agent_num)
        self.actions = torch.zeros(self.num_steps, self.env_num, self.agent_num)
        self.actions = self.actions.long()
        self.masks = torch.ones(self.num_steps + 1, self.env_num)
        self.step = 0
        self.obs = [self.zero_obs.copy() for j in range(self.num_steps + 1)]

    def to(self, device):
        for step_obs in self.obs:
            for k in step_obs:
                step_obs[k] = step_obs[k].to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        for k in self.obs[self.step + 1]:
            self.obs[self.step + 1][k].copy_(torch.tensor(obs[k]))
        self.actions[self.step].copy_(actions.squeeze())
        self.action_log_probs[self.step].copy_(action_log_probs.squeeze())
        self.value_preds[self.step].copy_(value_preds.squeeze())
        self.rewards[self.step].copy_(rewards).squeeze()
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for k in self.obs[0]:
            self.obs[0][k].copy_(self.obs[-1][k])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
        if use_proper_time_limits:
            raise NotImplementedError("Using proper time limit is not supported yet!")
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_target = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[
                        step + 1].unsqueeze(-1)
                    delta = td_target - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1].unsqueeze(-1) * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1].T *
                                          gamma * self.masks[step + 1]).T + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for indices in sampler:
            obs_batch = {}
            for k in self.obs[0]:
                cur_shape = list(self.obs[0][k].shape)
                cur_shape = [batch_size] + cur_shape[1:]
                obs_batch[k] = torch.stack([o[k] for o in self.obs[:-1]]).view(cur_shape)[indices]
            actions_batch = self.actions.view(batch_size, -1)[indices]
            value_preds_batch = self.value_preds[:-1].view(batch_size, -1)[indices]
            return_batch = self.returns[:-1].view(batch_size, -1)[indices]
            masks_batch = self.masks[:-1].view(batch_size, -1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(batch_size, -1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(batch_size, -1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                  old_action_log_probs_batch, adv_targ
