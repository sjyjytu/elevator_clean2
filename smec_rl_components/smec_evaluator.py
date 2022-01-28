import numpy as np
import torch
from smec_liftsim.smec_rl_env import *
from smec_rl_components.smec_policy import UniformPolicy, DistancePolicy


def evaluate_general(eval_env, device, method, args, verbose=False):
    obs = eval_env.reset()
    for k in obs:
        obs[k] = obs[k].to(device).unsqueeze(0)
    dt = eval_env._config._delta_t
    total_energy = 0
    for time_step in range(int((3600 + 60) / dt)):
        # debug
        # if True and time_step > (865 / dt):
        # if eval_env.open_render:
        #     eval_env.render()
        # step method
        if method == 'rl':
            with torch.no_grad():
                _, action, _, rule = args["actor_critic"].act(obs, deterministic=True)
            # Observe reward and next obs
            actions = torch.cat((action.cpu(), rule.cpu()), dim=1)
            actions = actions.squeeze(0)
            # print(actions[0][0])
            obs, _, done, info = eval_env.step(actions)
        elif method == 'shortest':
            obs, _, done, _ = eval_env.step_shortest_elev(random_policy=False, use_rules=args["use_rules"])
        elif method == 'smec':
            obs, _, done, _ = eval_env.step_smec()
        elif method == 'hand':
            obs, _, done, _ = eval_env.step_hand(use_rules=args["use_rules"])
        elif method == 'random':
            obs, _, done, _ = eval_env.step_shortest_elev(random_policy=True, use_rules=args["use_rules"])
        else:
            print("Method not implemented!")
            return -1

        for k in obs:
            obs[k] = obs[k].to(device).unsqueeze(0)

        total_energy += info['total_energy']

        if done:
            break

    waiting_time = []
    transmit_time = []
    for k in eval_env.mansion.person_info.keys():
        info = eval_env.mansion.person_info[k]
        if verbose:
            print(k, end=' | ')
            print('ele %d' % (info[0] + 1), end=' | ')
            for p_t in info[1:]:
                print('%d : %.1f' % (p_t // 60, p_t % 60), end=' | ')
        waiting_time.append(info[2])
        transmit_time.append(info[4])

    eval_env.close()
    print(
        f"-------------------------------------------------evaluation result-------------------------------------------------")
    awt = np.mean(waiting_time)
    att = np.mean(transmit_time)
    print(
        f"[Evaluation] for {len(waiting_time)} people: mean waiting time {awt:.1f}, mean transmit time: {att:.1f}.")
    return awt + att, total_energy


def evaluate(actor_critic, eval_env, device, verbose=False):
    eval_episode_rewards = []
    obs = eval_env.reset()
    for k in obs:
        obs[k] = obs[k].to(device).unsqueeze(0)
    sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0
    dt = eval_env._config._delta_t
    break_delay = 120 / dt
    had_done = False
    awt = []
    for time_step in range(int((3600 + 60) / dt)):
        # if time_step > 240:
        #     print('debug')
        if eval_env.open_render:
            eval_env.render()
            # time.sleep(0.1)

        with torch.no_grad():
            _, action, _, rule = actor_critic.act(obs, deterministic=True)
        # Observe reward and next obs
        actions = torch.cat((action.cpu(), rule.cpu()), dim=1)
        actions = actions.squeeze(0)
        obs, _, done, info = eval_env.step(actions)

        for k in obs:
            obs[k] = obs[k].to(device).unsqueeze(0)
        if info['waiting_time']:
            eval_episode_rewards += info['waiting_time']
        # print(eval_episode_rewards)
        # if len(eval_episode_rewards) > 0:
        #     max_waiting_time = np.max(eval_episode_rewards)
        #     # print(max_waiting_time)
        #     # if max_waiting_time > 500:
        #     #     print('debug here')
        sum_wait_rew += info['sum_wait_rew']
        sum_io_rew += info['sum_io_rew']
        sum_enter_rew += info['sum_enter_rew']
        awt += info['awt']

        if done or had_done:
            had_done = True
            break_delay -= 1
            if break_delay == 0:
                break
    print(eval_env.evaluate_info)
    print(eval_env.mansion.evaluate_info)
    print(awt)
    print('awt: ', np.mean(awt))

    waiting_time = []
    transmit_time = []
    for k in eval_env.mansion.person_info.keys():
        info = eval_env.mansion.person_info[k]
        try:
            if verbose:
                print(k, end=' | ')
                print('ele %d' % (info[0] + 1), end=' | ')
                for p_t in info[1:]:
                    print('%d : %.1f' % (p_t // 60, p_t % 60), end=' | ')
                print('%.1f %.1f' % (info[2], info[3] - info[2] - info[1]))

            waiting_time.append(info[2])
            transmit_time.append(info[3] - info[2] - info[1])
            # print()
        except:
            pass

    mean_time = np.mean(eval_episode_rewards) if eval_episode_rewards else -1
    print(f"-------------------------------------------------evaluation result-------------------------------------------------")
    print(eval_env.mansion._person_generator.data_file)
    print(f"[Evaluation] for {len(waiting_time)} people: mean waiting time {np.mean(waiting_time):.1f}, mean transmit time: {np.mean(transmit_time):.1f}.")
    print(f"[Evaluation] wait rew: {sum_wait_rew:.1f}; io rew: {sum_io_rew:.1f}; enter rew: {sum_enter_rew:.1f}.")
    eval_env.close()
    return mean_time


def evaluate_shortest_first(eval_env, device, random_policy=False, use_rules=False):
    eval_episode_rewards = []
    obs = eval_env.reset()
    for k in obs:
        obs[k] = obs[k].to(device).unsqueeze(0)
    sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0
    dt = eval_env._config._delta_t
    break_delay = 180 / dt
    had_done = False
    for time_step in range(int((3600 + 60) / dt)):
        if eval_env.open_render:
            eval_env.render()
        # Observe reward and next obs
        obs, _, done, info = eval_env.step_shortest_elev(random_policy, use_rules=use_rules)

        for k in obs:
            obs[k] = obs[k].to(device).unsqueeze(0)
        if info['waiting_time']:
            eval_episode_rewards += info['waiting_time']
        sum_wait_rew += info['sum_wait_rew']
        sum_io_rew += info['sum_io_rew']
        sum_enter_rew += info['sum_enter_rew']

        if done or had_done:
            had_done = True
            break_delay -= 1
            if break_delay == 0:
                break
    print(eval_env.evaluate_info)
    print(eval_env.mansion.evaluate_info)
    print(eval_env.mansion.self_trip)

    waiting_time = []
    transmit_time = []
    for k in eval_env.mansion.person_info.keys():
        info = eval_env.mansion.person_info[k]
        # print(k, eval_env.mansion.person_info[k])
        print(k, end=' | ')
        print('ele %d' % (info[0] + 1), end=' | ')
        for p_t in info[1:]:
            print('%d : %.1f' % (p_t // 60, p_t % 60), end=' | ')
        waiting_time.append(info[2])
        transmit_time.append(info[3] - info[2] - info[1])
        print('%.1f %.1f' % (info[2], info[3] - info[2] - info[1]))
        # print()

    eval_env.close()
    mean_time = np.mean(eval_episode_rewards) if eval_episode_rewards else -1
    print(
        f"-------------------------------------------------evaluation result-------------------------------------------------")
    # print(f"[Evaluation] for {len(eval_episode_rewards)} people: mean waiting time {np.mean(eval_episode_rewards):.1f}.")
    print(
        f"[Evaluation] for {len(waiting_time)} people: mean waiting time {np.mean(waiting_time):.1f}, mean transmit time: {np.mean(transmit_time):.1f}.")
    print(f"[Evaluation] wait rew: {sum_wait_rew:.1f}; io rew: {sum_io_rew:.1f}; enter rew: {sum_enter_rew:.1f}.")
    return mean_time


def evaluate_smec(eval_env, device):
    eval_episode_rewards = []
    obs = eval_env.reset()
    for k in obs:
        obs[k] = obs[k].to(device).unsqueeze(0)
    sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0
    awt = []
    dt = eval_env._config._delta_t
    break_delay = 120 / dt
    had_done = False
    for time_step in range(int((3600 + 60) / dt)):
        if eval_env.open_render:
            eval_env.render()
        # Observe reward and next obs
        obs, _, done, info = eval_env.step_smec()
        for k in obs:
            obs[k] = obs[k].to(device).unsqueeze(0)
        if info['waiting_time']:
            eval_episode_rewards += info['waiting_time']
        sum_wait_rew += info['sum_wait_rew']
        sum_io_rew += info['sum_io_rew']
        sum_enter_rew += info['sum_enter_rew']
        awt += info['awt']

        if done or had_done:
            had_done = True
            break_delay -= 1
            if break_delay == 0:
                break

    print(awt)
    print('awt: ', np.mean(awt))
    waiting_time = []
    transmit_time = []
    for k in eval_env.mansion.person_info.keys():
        info = eval_env.mansion.person_info[k]
        info = eval_env.mansion.person_info[k]
        # print(k, eval_env.mansion.person_info[k])
        print(k, end=' | ')
        print('ele %d' % (info[0] + 1), end=' | ')
        for p_t in info[1:]:
            print('%d : %.1f' % (p_t // 60, p_t % 60), end=' | ')
        waiting_time.append(info[2])
        transmit_time.append(info[3] - info[2] - info[1])
        print('%.1f %.1f' % (info[2], info[3] - info[2] - info[1]))

    eval_env.close()
    mean_time = np.mean(eval_episode_rewards) if eval_episode_rewards else -1
    print(f"-------------------------------------------------evaluation result-------------------------------------------------")
    # print(f"[Evaluation] for {len(eval_episode_rewards)} people: mean waiting time {np.mean(eval_episode_rewards):.1f}, awt: {np.mean(awt):.1f}.")
    print(f"[Evaluation] for {len(waiting_time)} people: mean waiting time {np.mean(waiting_time):.1f}, mean transmit time: {np.mean(transmit_time):.1f}.")
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


def evaluate_distance_baseline():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = 'cpu'

    test_env = make_env(seed=0, render=True, real_data=True)()
    actor_critic = DistancePolicy()
    evaluate(actor_critic, test_env, device)
    return


if __name__ == '__main__':
    evaluate_distance_baseline()
