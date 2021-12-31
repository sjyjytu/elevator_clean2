import numpy as np


def pos_linear_reward(load_num, ratio=5):
    reward = load_num / ratio
    return reward


def neg_linear_reward(waiting_time_list, const=5):
    reward = const - np.mean(waiting_time_list) / 100 if waiting_time_list else 0
    return reward


def neg_exp_reward(waiting_time_list):
    reward = np.exp(- np.mean(waiting_time_list) / 60) if waiting_time_list else 0
    return reward


def neg_mean_reward(calling_wt, threshold=50):
    calling_wt = [wt for wt in calling_wt if wt > threshold]
    reward = - np.mean(calling_wt) / 10000 if calling_wt else 0
    return reward


def neg_sum_reward(calling_wt, threshold=50):
    calling_wt = [wt for wt in calling_wt if wt > threshold]
    reward = - np.sum(calling_wt) / 10000 if calling_wt else 0
    return reward


def hierarchical_neg_reward(calling_wt):
    all_rewards = []
    for wt in calling_wt:
        if wt < 50:
            continue
        elif wt < 100:
            all_rewards.append(- wt / 10000)
        elif wt < 200:
            all_rewards.append(- wt / 100)
        else:
            all_rewards.append(- wt / 10)
    reward = np.mean(all_rewards) if all_rewards else 0
    return reward


def concate_list(list_of_list):
    final_l = []
    for tl in list_of_list:
        final_l += tl
    return final_l
