import random
from smec_rl_components.smec_sampler import *
from smec_rl_components.gnn_module import GraphCNN
from pytorch_rl.a2c_ppo_acktr.base_modules import *
from smec_rl_components.smec_graph_build import *
from smec_rl_components.optimization import *
import time


class SmecMLPEncoder(nn.Module):
    def __init__(self, lift_num, floor_num):
        super(SmecMLPEncoder, self).__init__()
        self.lift_num = lift_num
        self.floor_num = floor_num
        self.up_wait_encoder = MLPModule(input_size=floor_num, hidden_size=32)
        self.down_wait_encoder = MLPModule(input_size=floor_num, hidden_size=32)
        self.elv_pass_encoder = MLPModule(input_size=floor_num, hidden_size=32)
        self.elv_location_dic = nn.Embedding(num_embeddings=floor_num, embedding_dim=16)
        self.elv_location_encoder = MLPModule(input_size=16, hidden_size=8)
        self.hidden_dim = 144

    def forward(self, batch_fused_state):
        # input shape: [SAMPLE, dim]; output shape: [SAMPLE, ELV, dim].
        b_up_wait_nums = batch_fused_state['upload_wait_nums']
        b_down_wait_nums = batch_fused_state['download_wait_nums']
        b_loading = batch_fused_state['elevator_loading_maps']
        b_location = batch_fused_state['elevator_location_maps']
        upload_wait_f = self.up_wait_encoder(b_up_wait_nums)
        download_wait_f = self.down_wait_encoder(b_down_wait_nums)
        all_elv_pass = self.elv_pass_encoder(b_loading)
        elv_pass_global = torch.mean(all_elv_pass, dim=1)
        all_elv_loc = self.elv_location_dic(b_location)
        all_elv_loc = self.elv_location_encoder(all_elv_loc)
        elv_loc_global = torch.mean(all_elv_loc, dim=1)
        common_f = torch.cat((upload_wait_f, download_wait_f, elv_pass_global, elv_loc_global), -1)  # 152
        all_agent_features = []
        for idx in range(self.lift_num):
            cur_elv_pass = all_elv_pass[:, idx, :]
            cur_elv_loc = all_elv_loc[:, idx, :]
            local_f = torch.cat((cur_elv_pass, cur_elv_loc), -1)
            all_f = torch.cat((common_f, local_f), -1)
            all_agent_features.append(all_f)
        all_agent_features = torch.stack(all_agent_features, dim=1)
        return all_agent_features


class SmecGraphEncoder(nn.Module):
    def __init__(self, lift_num, floor_num, hidden_dim=128):
        super(SmecGraphEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gnn_encoder = GraphCNN(num_layers=3, num_mlp_layers=2, input_dim=3, hidden_dim=self.hidden_dim,
                                    learn_eps=False, neighbor_pooling_type='sum')
        self.relation_encoder = MLPModule(input_size=lift_num, hidden_size=self.hidden_dim)
        self.lift_num = lift_num
        self.floor_num = floor_num
        self.half_node = self.lift_num + self.floor_num

    def forward(self, batch_fused_state):
        # input shape: [SAMPLE, dim]; output shape: [SAMPLE, ELV, dim].
        adj, node_f = batch_fused_state['adj_m'], batch_fused_state['node_feature_m']
        distances = batch_fused_state['distances']
        node_hidden = self.gnn_encoder(node_f, padded_nei=None, adj=adj)
        up_floor_hidden = node_hidden[:, :self.floor_num, :]
        down_floor_hidden = node_hidden[:, self.half_node:self.half_node + self.floor_num, :]
        all_agent_features = torch.cat([up_floor_hidden, down_floor_hidden], dim=1)
        rel_agent_features = self.relation_encoder(distances)
        all_agent_features = all_agent_features + rel_agent_features
        return all_agent_features


class SmecBase(nn.Module):
    def __init__(self, lift_num, floor_num, use_graph=True):
        super(SmecBase, self).__init__()
        self.use_graph = use_graph
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.actor = SmecMLPEncoder(lift_num, floor_num) if not use_graph else SmecGraphEncoder(lift_num, floor_num)
        self.critic = SmecMLPEncoder(lift_num, floor_num) if not use_graph else SmecGraphEncoder(lift_num, floor_num)
        self.a_output = self.actor.hidden_dim
        self.critic_linear = init_(nn.Linear(self.critic.hidden_dim, 1))
        # self.rule_linear = init_(nn.Linear(self.actor.hidden_dim, 1))
        self.train()

    def forward(self, inputs):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        value = self.critic_linear(hidden_critic)
        # rule = self.rule_linear(hidden_actor).squeeze(2)
        rule = torch.argmax(value, dim=1, keepdim=True)
        return value, hidden_actor, rule


class SmecPolicy(nn.Module):
    def __init__(self, lift_num, floor_num, use_graph=True, open_mask=True, initialization=False, use_advice=False, device='cpu'):
        super(SmecPolicy, self).__init__()
        self.base = SmecBase(lift_num, floor_num, use_graph=use_graph)
        assert use_graph
        if not use_advice:
            self.dist = SmecSampler(self.base.a_output, lift_num, use_graph)
        else:
            # modified by JY, add advice choice.
            self.dist = SmecSampler(self.base.a_output, lift_num + 1, use_graph)

        self.open_mask = open_mask
        self.initialization = initialization

        # add by JY, predict with lstm
        # self.lstm_input_hidden_size = floor_num*2*self.base.a_output
        # lstm_output_hidden_size = 128
        # self.lstm = nn.LSTM(self.lstm_input_hidden_size, lstm_output_hidden_size)
        # self.lstm_linear = nn.Linear(lstm_output_hidden_size, floor_num)
        # self.hidden_cell = None
        # self.dist2 = SmecSampler(lstm_output_hidden_size, floor_num, use_graph)

        if initialization:
            initialize_weights(self, "orthogonal")

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def reset(self):
        self.hidden_cell = None

    def act(self, inputs_obs, deterministic=False):
        value, actor_features, rule = self.base(inputs_obs)
        # print(actor_features.shape)  batch * 32 * 128
        # rule 8*32*1
        # rule = rule.squeeze(2)
        # dist = self.dist(actor_features, legal_mask=inputs_obs['legal_masks'])
        legal_mask = inputs_obs['legal_masks'] if self.open_mask else None
        # dist = self.dist(actor_features, legal_mask=None)
        dist = self.dist(actor_features, legal_mask=legal_mask)

        # add by JY, elevator chooses floor with lstm.
        # lstm_out, self.hidden_cell = self.lstm(actor_features.view(1, -1, self.lstm_input_hidden_size), self.hidden_cell)
        # predict_floors = self.lstm_linear(lstm_out.squeeze(0))

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # print(action.shape)  # torch.Size([1, 34, 1])
        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs, rule

    def get_value(self, inputs, masks):
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs_obs, masks, action):
        value, actor_features, rule = self.base(inputs_obs)
        legal_mask = inputs_obs['legal_masks'] if self.open_mask else None
        # dist = self.dist(actor_features, legal_mask=None)
        dist = self.dist(actor_features, legal_mask=legal_mask)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        # add by JY, remove the effect of those not-taken action.
        valid_action_mask = inputs_obs['valid_action_mask']
        if valid_action_mask is not None:
            dist_entropy = dist_entropy * valid_action_mask
        # dist_entropy = dist_entropy.sum()
        dist_entropy = dist_entropy.mean()

        return value, action_log_probs, dist_entropy


class UniformPolicy(nn.Module):
    def __init__(self):
        super(UniformPolicy, self).__init__()

    def act(self, inputs_obs, deterministic=False):
        elevator_mask = inputs_obs['legal_masks'].squeeze()
        final_actions = []
        for idx in range(17 * 2):
            cur_mask = elevator_mask[idx]
            selection_list = []
            for index in range(6):
                if cur_mask[index] > 0.5:
                    selection_list.append(index)
            idx = random.choice(selection_list)
            final_actions.append(idx)
        final_actions = torch.tensor(final_actions)
        return None, final_actions, None


class DistancePolicy(nn.Module):
    def __init__(self):
        super(DistancePolicy, self).__init__()

    def act(self, inputs_obs, deterministic=False):
        elevator_mask = inputs_obs['legal_masks'].squeeze()
        distances = inputs_obs['distances'].squeeze()
        final_actions = []
        for idx in range(17 * 2):
            cur_mask = elevator_mask[idx]
            cur_distance = distances[idx]
            cur_distance = cur_distance * cur_mask + cur_distance * 1000000 * (1 - cur_mask)
            idx = torch.argmin(cur_distance)
            final_actions.append(idx)
        final_actions = torch.tensor(final_actions)
        return None, final_actions, None
