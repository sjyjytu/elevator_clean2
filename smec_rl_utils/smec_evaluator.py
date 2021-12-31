import numpy as np
import torch
from smec_liftsim.smec_rl_env import *
from smec_rl_utils.smec_policy import UniformPolicy


def evaluate(actor_critic, eval_env, device):
    eval_episode_rewards = []
    obs = eval_env.reset()
    for k in obs:
        obs[k] = obs[k].to(device).unsqueeze(0)
    sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0
    for time_step in range(5000):
        if eval_env.open_render:
            eval_env.render()
        with torch.no_grad():
            _, action, _ = actor_critic.act(obs, deterministic=True)
        # Observe reward and next obs
        obs, _, done, info = eval_env.step(action)
        for k in obs:
            obs[k] = obs[k].to(device).unsqueeze(0)
        if info['waiting_time']:
            eval_episode_rewards += info['waiting_time']
        sum_wait_rew += info['sum_wait_rew']
        sum_io_rew += info['sum_io_rew']
        sum_enter_rew += info['sum_enter_rew']

    eval_env.close()
    mean_time = np.mean(eval_episode_rewards) if eval_episode_rewards else -1
    print(f"----------------------------------evaluation result-------------------------------------------------------")
    print(f"[Evaluation] for {len(eval_episode_rewards)} people: mean waiting time {np.mean(eval_episode_rewards):.1f}.")
    print(f"[Evaluation] wait rew: {sum_wait_rew:.1f}; io rew: {sum_io_rew:.1f}; enter rew: {sum_enter_rew:.1f}.")
    return mean_time


def evaluate_uniform_baseline():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = 'cpu'

    test_env = make_env(seed=0, render=False)()
    actor_critic = UniformPolicy()
    evaluate(actor_critic, test_env, device)
    return


if __name__ == '__main__':
    evaluate_uniform_baseline()
