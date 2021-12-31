import torch
import torch.nn as nn
from pytorch_rl.a2c_ppo_acktr.utils import AddBias, init


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        father_log_probs = super().log_prob
        action_log = father_log_probs(actions.squeeze(-1))
        action_log = action_log.unsqueeze(-1)  # seems that don't need to sum
        # action_log = action_log.view(actions.size(0), -1).sum(-1).unsqueeze(-1)
        return action_log

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class SmecSampler(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SmecSampler, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, action_feature, legal_mask):
        x = self.linear(action_feature)
        # regularized_logits = x * legal_mask.unsqueeze(1) + 1e-9
        regularized_logits = x - 1e10 * (1 - legal_mask.unsqueeze(1))
        dist = FixedCategorical(logits=regularized_logits)
        return dist
