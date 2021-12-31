import random
from smec_rl_utils.smec_sampler import *
from pytorch_rl.a2c_ppo_acktr.base_modules import *


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
    def __init__(self, lift_num, floor_num):
        super(SmecGraphEncoder, self).__init__()
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


class SmecBase(nn.Module):
    def __init__(self, lift_num, floor_num, use_graph=True):
        super(SmecBase, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.actor = SmecMLPEncoder(lift_num, floor_num) if not use_graph else
        self.critic = SmecMLPEncoder(lift_num, floor_num)
        self.a_output = self.actor.hidden_dim
        self.critic_linear = init_(nn.Linear(self.critic.hidden_dim, 1))
        self.train()

    def forward(self, inputs):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        value = self.critic_linear(hidden_critic)
        return value, hidden_actor


class SmecPolicy(nn.Module):
    def __init__(self, lift_num, floor_num):
        super(SmecPolicy, self).__init__()
        self.base = SmecBase(lift_num, floor_num)
        self.dist = SmecSampler(self.base.a_output, floor_num * 2)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs_obs, deterministic=False):
        value, actor_features = self.base(inputs_obs)
        dist = self.dist(actor_features, legal_mask=inputs_obs['legal_masks'])

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs_obs, masks, action):
        value, actor_features = self.base(inputs_obs)
        dist = self.dist(actor_features, legal_mask=inputs_obs['legal_masks'])

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class UniformPolicy(nn.Module):
    def __init__(self):
        super(UniformPolicy, self).__init__()

    def act(self, inputs_obs, deterministic=False):
        legal_masks = inputs_obs['legal_masks']
        final_actions = []
        for env_mask in legal_masks:
            available_actions = []
            for jdx, bit in enumerate(env_mask):
                if bit > 0:
                    available_actions.append(jdx)
            act_vector = []
            for idx in range(6):
                if not available_actions:
                    cur_act = 0
                else:
                    cur_act = random.choice(available_actions)
                act_vector.append(cur_act)
            final_actions.append(act_vector)
        final_actions = torch.tensor(final_actions)
        return None, final_actions, None
