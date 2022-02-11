import numpy as np
import time
from smec_liftsim.generator_proxy import set_seed
from smec_liftsim.generator_proxy import PersonGenerator
from smec_liftsim.fixed_data_generator import FixedDataGenerator
from smec_liftsim.random_data_generator import RandomDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.utils import ElevatorHallCall
from smec_liftsim.smec_state import SmecState
from smec_rl_components.smec_graph_build import *
import argparse
import configparser
import random
import sys
import os
import torch
import gym
from gym.spaces import Discrete, Dict, Box
from gym.vector.async_vector_env import AsyncVectorEnv
from smec_rl_components.smec_reward import *
from smec_rl_components.normalization import *
from smec_liftsim.smec_constants import *

from offline_tools.generate_dataset import generate_dataset
from smec_liftsim.utils import PersonType


class SmecRLEnv(gym.Env):
    """
    RL environment for SMEC elevators.
    """
    def __init__(self, data_file='./smec_rl/simple_dataset_v2.csv', config_file=None, render=True, forbid_unrequired=True, seed=None, forbid_uncalled=False,
    # def __init__(self, data_file='train_data/new/lunchpeak/LunchPeak1_elvx.csv', config_file=None, render=True, forbid_unrequired=True, seed=None, forbid_uncalled=False,
                 use_graph=True, gamma=0.99, real_data=True, use_advice=False, special_reward=False, data_dir=None, file_begin_idx=None, dos=''):
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__) + '/rl_config2.ini')
        file_name = config_file
        self.forbid_uncalled = forbid_uncalled
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy.'
        # dos = '06:00-12:00'
        # dos = '00:00-06:00'
        # dos = '50:00-60:00'
        # dos = ''
        if dos == '':
            st = 0
        else:
            ts = dos.split('-')[0].split(':')
            st = int(ts[0]) * 60 + int(ts[1])

        if not real_data:
            # Create a different person generator
            gtype = config['PersonGenerator']['PersonGeneratorType']
            person_generator = PersonGenerator(gtype)
            person_generator.configure(config['PersonGenerator'])
        else:
            # person_generator = FixedDataGenerator(data_file=data_file, data_dir=data_dir, file_begin_idx=file_begin_idx, data_of_section=dos)
            person_generator = RandomDataGenerator(data_dir=data_dir, data_of_section=dos)
            # person_generator = RandomDataGenerator(data_dir=data_dir, data_of_section=dos, random_or_load_or_save=1)
        self._config = MansionConfig(
            dt=time_step,
            number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            floor_height=float(config['MansionInfo']['FloorHeight']),
            maximum_acceleration=float(config['MansionInfo']['Acceleration']),
            maximum_speed=float(config['MansionInfo']['RateSpeed']),
            person_entering_time=float(config['MansionInfo']['PersonEnterTime']),
            door_opening_time=float(config['MansionInfo']['DoorOpeningTime']),
            door_closing_time=float(config['MansionInfo']['DoorClosingTime']),
            keep_door_open_lag=float(config['MansionInfo']['DoorKeepOpenLagTime']),
            door_well_time2=float(config['MansionInfo']['DwellTime2']),
            maximum_parallel_entering_exiting_number=int(config['MansionInfo']['ParallelEnterNum']),
            rated_load=int(config['MansionInfo']['RateLoad']),
            start_time=st
        )

        self.mansion = MansionManager(int(config['MansionInfo']['ElevatorNumber']), person_generator, self._config,
                                      config['MansionInfo']['Name'])
        self.use_graph = use_graph
        self.viewer = None
        self.open_render = render
        if render:
        # if True:
            from smec_liftsim.rendering import Render
            self.viewer = Render(self.mansion)
        self.elevator_num = self.mansion.attribute.ElevatorNumber
        self.floor_num = int(config['MansionInfo']['NumberOfFloors'])
        self.waiting_times = []
        self.forbid_unrequired = forbid_unrequired

        if seed is not None:
            self.seed(seed)
        self.seed_c = seed

        # gym specific settings
        self.metadata = {'render.modes': []}
        self.gamma = gamma
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

        # Set these in ALL subclasses
        self.action_space = Box(low=0, high=self.floor_num * 2 - 1, shape=(self.elevator_num,), dtype=np.int)

        ele_f = (self.elevator_num, self.floor_num)
        self.graph_node_num = (self.elevator_num + self.floor_num) * 2
        self.gb = GraphBuilder(self.elevator_num, self.floor_num, 'cpu')
        self.empty_adj_matrix = self.gb.get_zero_adj_matrix()
        self.cur_adj_matrix = self.empty_adj_matrix.clone()
        self.empty_node_feature = self.gb.get_zero_node_feature()
        self.cur_node_feature = self.empty_node_feature.clone()
        assert self.use_graph

        self.use_advice = use_advice
        self.special_reward = special_reward
        candidate_num = self.elevator_num
        if use_advice:
            candidate_num += 1

        self.observation_space = Dict(
            {'adj_m': Box(low=-float('inf'), high=float('inf'), shape=(self.graph_node_num, self.graph_node_num)),
             'node_feature_m': Box(low=-float('inf'), high=float('inf'), shape=(self.graph_node_num, 3)),
             'legal_masks': Box(low=-float('inf'), high=float('inf'), shape=(self.floor_num * 2, candidate_num,)),
             'elevator_mask': Box(low=-float('inf'), high=float('inf'), shape=(self.elevator_num, self.floor_num * 2,)),
             'floor_mask': Box(low=-float('inf'), high=float('inf'), shape=(self.floor_num * 2,)),
             'distances': Box(low=-float('inf'), high=float('inf'), shape=(self.floor_num * 2, self.elevator_num,)),
             'valid_action_mask': Box(low=0, high=1, shape=(self.floor_num * 2,))
             })

        self.reward_filter = Identity()
        self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=gamma, clip=None)

        # state normalization
        self.state_filter = Identity()
        self.state_filter = ZFilter(self.state_filter, shape=[self.graph_node_num, 3], clip=None)

        # self.real_dataset = generate_dataset()
        # self.data_idx = 0
        # self.next_generate_person = self.real_dataset[self.data_idx]
        self.evaluate_info = {'valid_up_action': 0,
                              'advice_up_action': 0,
                              'valid_dn_action': 0,
                              'advice_dn_action': 0}

    @staticmethod
    def get_filter_by_list(list_len, query):
        cur_elv_mask = torch.tensor([0.0 for _ in range(list_len)])
        for elev in query:
            cur_elv_mask[elev] = 1.0
        return cur_elv_mask

    def get_action_mask(self, device):
        # M JY: add advice choice
        candidate_num = self.elevator_num + 1 if self.use_advice else self.elevator_num

        # get a list of action candidates by rules given pre-defined floors.
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        floor2elv_masks = []
        # handle up floors
        for idx in range(self.floor_num):
            if idx not in unallocated_up:
                cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            else:
                conv_elevators = self.mansion.get_convenience_elevators(up_or_down=True, floor_id=idx)
                if len(conv_elevators) > 0:  # convenient elevators exist
                    cur_elv_mask = self.get_filter_by_list(candidate_num, conv_elevators)
                    if self.use_advice:
                        cur_elv_mask[-1] = 1.0
                else:
                    cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            floor2elv_masks.append(cur_elv_mask)

        # handle down floors
        for idx in range(self.floor_num):
            if idx not in unallocated_dn:
                cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            else:
                conv_elevators = self.mansion.get_convenience_elevators(up_or_down=False, floor_id=idx)
                if len(conv_elevators) > 0:  # convenient elevators exist
                    cur_elv_mask = self.get_filter_by_list(candidate_num, conv_elevators)
                    if self.use_advice:
                        cur_elv_mask[-1] = 1.0
                # elif len(uncalled_elevators) > 0:  # non-called elevators exist
                #     cur_elv_mask = self.get_filter_by_list(self.elevator_num, uncalled_elevators)
                else:
                    cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            floor2elv_masks.append(cur_elv_mask)

        elevator_mask = torch.stack(floor2elv_masks).to(device)
        return elevator_mask

    def get_action_mask_plus(self, device):
        # get a list of action candidates by rules given pre-defined floors.
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()

        data = self.mansion._person_generator.prob[int(self._config.raw_time // 60)]  # 16*16
        floor_mask = np.zeros(self.floor_num*2)
        for src in range(self.floor_num):
            dn = data[src][:src]
            dn_sum = np.sum(dn)
            up = data[src][src:]
            up_sum = np.sum(up)

            floor_mask[src] = up_sum
            floor_mask[src+self.floor_num] = dn_sum
        floor_mask = torch.from_numpy(floor_mask)

        # 合并floor_mask 2f x 1
        for up in unallocated_up:
            floor_mask[up] += len(self.mansion._wait_upward_persons_queue[up])
        for dn in unallocated_dn:
            floor_mask[dn+self.floor_num] += len(self.mansion._wait_downward_persons_queue[dn])

        # 不管这个生成概率，只用当前的方便电梯
        convenience_mask = self.mansion.get_convenience_mask()  # e x 2f
        elevator_mask = torch.from_numpy(convenience_mask).to(device)  # e x 2f
        return elevator_mask, floor_mask

    def get_time(self):
        raw_time = self._config.raw_time
        cur_day = raw_time // (24 * 3600)
        cur_time = raw_time % (24 * 3600)
        return [cur_day, int(cur_time // 3600 + 7), int(cur_time % 3600 // 60), int(cur_time % 60)]

    def step(self, actions):
        # return self.step_rl_dt(actions)
        return self.step_rl_dp(actions)

    def step_rl_dt(self, actions):
        floor2elevators, advantage_floor = actions.split(32, 0)
        assert type(floor2elevators) == torch.Tensor, "only support tensor action"  # unwrapped raw action.

        # M JY: add advice choice
        if self.use_advice:
            advice_floor2elevators = self.get_floor2elevator_dis(floor2elevators.device).cpu().numpy()
            advice_floor2elevators = np.argmin(advice_floor2elevators, axis=1)

        floor2elevators = floor2elevators.squeeze()
        advantage_floor = advantage_floor.squeeze()
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]

        for up_floor in unallocated_up:
            self.evaluate_info['valid_up_action'] += 1
            cur_elev = floor2elevators[up_floor].item()
            if self.use_advice and cur_elev == self.elevator_num:
                # use advice
                self.evaluate_info['advice_up_action'] += 1
                cur_elev = advice_floor2elevators[up_floor]
            all_elv_up_fs[cur_elev].append(up_floor)
        for dn_floor in unallocated_dn:
            self.evaluate_info['valid_dn_action'] += 1
            cur_elev = floor2elevators[dn_floor + self.floor_num].item()
            if self.use_advice and cur_elev == self.elevator_num:
                # use advice
                self.evaluate_info['advice_dn_action'] += 1
                cur_elev = advice_floor2elevators[dn_floor + self.floor_num]
            all_elv_down_fs[cur_elev].append(dn_floor)
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))

        ###############################  use the reward from tnnls ################################
        if self.special_reward:
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt, hall_waiting_rewards, car_waiting_rewards, energy \
                = self.mansion.run_mansion(action_to_execute, special_reward=True, advantage_floor=advantage_floor)
            factor = 0.6
            reward = 0.02 * (-np.array(hall_waiting_rewards) - factor * np.array(car_waiting_rewards))
            info = {'waiting_time': concate_list(arrive_wt), 'sum_wait_rew': 0, 'sum_io_rew': 0,
                    'sum_enter_rew': 0, 'awt': awt}
        else:
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt, energy = \
                self.mansion.run_mansion(action_to_execute, advantage_floor=advantage_floor)

            final_reward, sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0, 0
            for idx in range(len(enter_num)):
                cur_wait_reward = neg_linear_reward(arrive_wt[idx])
                cur_io_reward = -1 * no_io_masks[idx]
                cur_enter_reward = pos_linear_reward(enter_num[idx])
                # change the next line to choose rewards
                cur_rew = cur_wait_reward
                sum_wait_rew += cur_wait_reward
                sum_io_rew += cur_io_reward
                sum_enter_rew += cur_enter_reward
                final_reward += cur_rew
            normalized_rew = self.reward_filter(final_reward)
            reward = [normalized_rew for _ in range(self.floor_num * 2)]
            reward = np.array(reward)
            info = {'waiting_time': concate_list(arrive_wt), 'sum_wait_rew': sum_wait_rew, 'sum_io_rew': sum_io_rew,
                    'sum_enter_rew': sum_enter_rew, 'awt': awt}
        print(f'energy: {energy}')
        new_obs = self.get_smec_state()
        self.mansion.generate_person()
        done = self.mansion.is_done
        return new_obs, reward, done, info

    def step_rl_dp(self, actions):
        floor2elevators, advantage_floor = actions.split(32, 0)
        # print(actions.shape, floor2elevators.shape, advantage_floor.shape)
        assert type(floor2elevators) == torch.Tensor, "only support tensor action"  # unwrapped raw action.

        # M JY: add advice choice
        if self.use_advice:
            advice_floor2elevators = self.get_floor2elevator_dis(floor2elevators.device).cpu().numpy()
            advice_floor2elevators = np.argmin(advice_floor2elevators, axis=1)

        floor2elevators = floor2elevators.squeeze()
        advantage_floor = advantage_floor.squeeze()
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]

        # for debug:
        j = advantage_floor.item()
        if j > 1:
            DEBUG = True
        else:
            DEBUG = False
        # print('j:', j)

        # verify the rl is trained
        # floor2elevators[0] = random.randint(0,3)
        # print(floor2elevators[0])
        # floor2elevators = self.get_floor2elevator_dis('cpu').cpu().numpy()
        # floor2elevators = np.argmin(floor2elevators, axis=1)

        for up_floor in unallocated_up:
            self.evaluate_info['valid_up_action'] += 1
            cur_elev = floor2elevators[up_floor].item()
            if self.use_advice and cur_elev == self.elevator_num:
                # use advice
                self.evaluate_info['advice_up_action'] += 1
                cur_elev = advice_floor2elevators[up_floor]
            all_elv_up_fs[cur_elev].append(up_floor)
        for dn_floor in unallocated_dn:
            self.evaluate_info['valid_dn_action'] += 1
            cur_elev = floor2elevators[dn_floor + self.floor_num].item()
            if self.use_advice and cur_elev == self.elevator_num:
                # use advice
                self.evaluate_info['advice_dn_action'] += 1
                cur_elev = advice_floor2elevators[dn_floor + self.floor_num]
            all_elv_down_fs[cur_elev].append(dn_floor)
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))

        # step until next person come
        next_call_come = False
        cur_time = self._config.raw_time
        reward = np.zeros((self.floor_num*2, ))
        arrive_wts = [[] for i in range(self.elevator_num)]
        total_energy = 0
        while not next_call_come and not self.mansion.is_done:
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt, hall_waiting_rewards, car_waiting_rewards, energy \
                = self.mansion.run_mansion(action_to_execute, special_reward=True, advantage_floor=advantage_floor)
            for i in range(self.elevator_num):
                arrive_wts[i] += arrive_wt[i]
            self.mansion.generate_person()
            if self.open_render:
                self.render()
            # time.sleep(0.05)
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            # print(unallocated_up, unallocated_dn)
            action_to_execute = [ElevatorHallCall([], []) for _ in range(self.elevator_num)]
            next_call_come = unallocated_up != [] or unallocated_dn != []
            # if DEBUG:
            #     print(action_to_execute, next_call_come, self.mansion.is_done)
            #     print(self.mansion._wait_upward_persons_queue)
            #     print(self.mansion._wait_downward_persons_queue)
            #     print(self.mansion.finish_person_num, self.mansion._person_generator.total_person_num)
            #     for idx, elev in enumerate(self.mansion._elevators):
            #         print(idx, elev._run_state, elev.state)


            # cal reward
            factor = 0
            reward += 0.01 * (-np.array(hall_waiting_rewards) - factor * np.array(car_waiting_rewards) - 5e-4 * energy)
            total_energy += energy
            # print(reward)
            # print(f'{hall_waiting_rewards[0]} {car_waiting_rewards[0]} {5e-4 * energy} {reward[0]}')

        # TODO: calculate reward, during the time interval between two person, finish how many person?
        finish_time = self._config.raw_time
        delta_t = finish_time - cur_time
        reward = reward * self._config._delta_t / delta_t
        info = {'waiting_time': concate_list(arrive_wts), 'sum_wait_rew': 0, 'sum_io_rew': 0,
                'sum_enter_rew': 0, 'awt': awt, 'total_energy': total_energy}
        new_obs = self.get_smec_state()
        self.mansion.generate_person()
        done = self.mansion.is_done

        # # TODO: tune the reward
        # if not done:
        #     new_reward = np.zeros_like(reward)
        # else:
        #     p_waiting_time = []
        #     p_transmit_time = []
        #     for k in self.mansion.person_info.keys():
        #         pinfo = self.mansion.person_info[k]
        #         p_waiting_time.append(pinfo[2])
        #         p_transmit_time.append(pinfo[4])
        #     p_awt = np.mean(p_waiting_time)
        #     p_att = np.mean(p_transmit_time)
        #     new_reward = np.ones_like(reward) * (-p_awt) / 60

        return new_obs, reward, done, info
        # return new_obs, new_reward, done, info

    # Implement by JY, just to simply compare with the RL agent
    def step_shortest_elev(self, random_policy=False, use_rules=True):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if not random_policy:
            floor2elevators = self.get_floor2elevator_dis(device).cpu().numpy()
            floor2elevators = np.argmin(floor2elevators, axis=1)
        else:
            floor2elevators = np.array([random.randint(0, self.elevator_num - 1) for _ in range(self.floor_num * 2)])
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        for up_floor in unallocated_up:
            cur_elev = floor2elevators[up_floor]
            all_elv_up_fs[cur_elev].append(up_floor)
        for dn_floor in unallocated_dn:
            cur_elev = floor2elevators[dn_floor + self.floor_num]
            all_elv_down_fs[cur_elev].append(dn_floor)
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        # if self.open_render:
        #     time.sleep(0.05)  # for accelerate simulate speed
        calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute, use_rules=use_rules)
        self.mansion.generate_person()
        final_reward, sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0, 0
        for idx in range(len(enter_num)):
            cur_wait_reward = neg_linear_reward(arrive_wt[idx])
            cur_io_reward = -1 * no_io_masks[idx]
            cur_enter_reward = pos_linear_reward(enter_num[idx])
            # change the next line to choose rewards
            cur_rew = cur_wait_reward
            sum_wait_rew += cur_wait_reward
            sum_io_rew += cur_io_reward
            sum_enter_rew += cur_enter_reward
            final_reward += cur_rew
        normalized_rew = self.reward_filter(final_reward)
        reward = [normalized_rew for _ in range(self.floor_num * 2)]
        # if self.open_render:
        #     print("all calls are", action_to_execute)
        new_obs = self.get_smec_state()
        # new_obs['node_feature_m'] = self.state_filter(new_obs['node_feature_m']).float()
        reward = np.array(reward)
        done = self.mansion.is_done
        info = {'waiting_time': concate_list(arrive_wt), 'sum_wait_rew': sum_wait_rew, 'sum_io_rew': sum_io_rew,
                'sum_enter_rew': sum_enter_rew, 'awt': awt}
        return new_obs, reward, done, info

    # Implement by JY, compare smec and RL
    def step_smec(self):
        # test with batch align
        # person_list = self.mansion._person_generator.generate_person()
        # unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        # all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        # done = False
        # for up_floor in unallocated_up:
        #     if not self.mansion._person_generator.used_ele.empty():
        #         cur_elev = self.mansion._person_generator.used_ele.get()
        #     else:
        #         print(self.mansion._wait_upward_persons_queue[up_floor][-1].AppearTime)
        #         done = True
        #         cur_elev = self.mansion._wait_upward_persons_queue[up_floor][-1].StatisticElev
        #     all_elv_up_fs[cur_elev].append(up_floor)
        # for dn_floor in unallocated_dn:
        #     cur_elev = self.mansion._wait_downward_persons_queue[dn_floor][-1].StatisticElev
        #     all_elv_down_fs[cur_elev].append(dn_floor)

        # test with person align
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        for up_floor in unallocated_up:
            for pop_idx in range(len(self.mansion._wait_upward_persons_queue[up_floor]) - 1, -1, -1):
                cur_elev = self.mansion._wait_upward_persons_queue[up_floor][pop_idx].StatisticElev
                if up_floor not in all_elv_up_fs[cur_elev]:
                    all_elv_up_fs[cur_elev].append(up_floor)
                    # break
        for dn_floor in unallocated_dn:
            for pop_idx in range(len(self.mansion._wait_downward_persons_queue[dn_floor]) - 1, -1, -1):
                cur_elev = self.mansion._wait_downward_persons_queue[dn_floor][pop_idx].StatisticElev
                if dn_floor not in all_elv_down_fs[cur_elev]:
                    all_elv_down_fs[cur_elev].append(dn_floor)
                    # break

        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        # if self.open_render:
        #     time.sleep(0.05)  # for accelerate simulate speed
        calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute)
        self.mansion.generate_person()
        final_reward, sum_wait_rew, sum_io_rew, sum_enter_rew = 0, 0, 0, 0
        for idx in range(len(enter_num)):
            cur_wait_reward = neg_linear_reward(arrive_wt[idx])
            cur_io_reward = -1 * no_io_masks[idx]
            cur_enter_reward = pos_linear_reward(enter_num[idx])
            # change the next line to choose rewards
            cur_rew = cur_wait_reward
            sum_wait_rew += cur_wait_reward
            sum_io_rew += cur_io_reward
            sum_enter_rew += cur_enter_reward
            final_reward += cur_rew
        normalized_rew = self.reward_filter(final_reward)
        reward = [normalized_rew for _ in range(self.floor_num * 2)]
        # if self.open_render:
        #     print("all calls are", action_to_execute)
        new_obs = self.get_smec_state()
        # new_obs['node_feature_m'] = self.state_filter(new_obs['node_feature_m']).float()
        reward = np.array(reward)
        # done = done or self.mansion.is_done
        done = self.mansion.is_done
        info = {'waiting_time': concate_list(arrive_wt), 'sum_wait_rew': sum_wait_rew, 'sum_io_rew': sum_io_rew,
                'sum_enter_rew': sum_enter_rew, 'awt': awt}
        return new_obs, reward, done, info

    # Implement by JY, choose elevate by hand for person.
    def step_hand(self, use_rules=False):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        for up_floor in unallocated_up:
            cur_elev = int(input(f'allocate for up {up_floor}: '))
            all_elv_up_fs[cur_elev].append(up_floor)
        for dn_floor in unallocated_dn:
            cur_elev = int(input(f'allocate for up {dn_floor}: '))
            all_elv_down_fs[cur_elev].append(dn_floor)
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute, use_rules=use_rules)
        self.mansion.generate_person()
        new_obs = self.get_smec_state()
        reward = 0
        done = self.mansion.is_done
        info = {}
        return new_obs, reward, done, info

    # def step_with_pure_rules(self):
    #     # for all elev, if it stay still, then move it to the first floor by allocating a car call. And allocate for the hall call with the simple shortest first algorithm.
    #     for elev


    def get_floor2elevator_dis(self, device):
        floor2elevator_dis = []
        for call_floor in range(self.floor_num):  # up calls
            cur_distance = []
            for elev in self.mansion._elevators:
                elevator_floor = elev._sync_floor
                # try by JY
                if call_floor == elevator_floor and \
                        (elev._run_state == ELEVATOR_STOP_DOOR_CLOSING or elev._run_state == ELEVATOR_RUN):
                    if elev._service_direction == 1:
                        elevator_floor += 0.01
                    elif elev._service_direction == -1:
                        elevator_floor -= 0.01
                going_up = elev._service_direction == 1  # going up
                if going_up and call_floor >= elevator_floor:
                    distance = call_floor - elevator_floor  # directly move up
                elif going_up and call_floor < elevator_floor:
                    distance = (self.floor_num - elevator_floor) + self.floor_num + call_floor  # move up + move to bottom + move to call
                else:
                    distance = elevator_floor + call_floor  # down to bottom and move up
                cur_distance.append(distance / self.floor_num)  # normalize
            floor2elevator_dis.append(cur_distance)

        for call_floor in range(self.floor_num):  # down calls
            cur_distance = []
            for elev in self.mansion._elevators:
                elevator_floor = elev._sync_floor
                # try by JY
                if call_floor == elevator_floor and \
                        (elev._run_state == ELEVATOR_STOP_DOOR_CLOSING or elev._run_state == ELEVATOR_RUN):
                    if elev._service_direction == 1:
                        elevator_floor += 0.01
                    elif elev._service_direction == -1:
                        elevator_floor -= 0.01
                going_down = elev._service_direction != 1  # going down
                if going_down and call_floor <= elevator_floor:
                    distance = elevator_floor - call_floor  # directly move down
                elif going_down and call_floor > elevator_floor:
                    distance = elevator_floor + self.floor_num + (
                                self.floor_num - call_floor)  # move down + move to top + move to call
                else:
                    distance = (self.floor_num - elevator_floor) + (self.floor_num - call_floor)  # to top and move down
                cur_distance.append(distance / self.floor_num)  # normalize
            floor2elevator_dis.append(cur_distance)
        floor2elevator_dis = torch.tensor(floor2elevator_dis).to(device)
        return floor2elevator_dis

    # no attention mask, pure convenience mask
    def get_smec_state(self):
        up_wait, down_wait, loading, location, up_call, down_call, load_up, load_down = self.mansion.get_rl_state(
            encode=True)
        up_wait, down_wait, loading, location = torch.tensor(up_wait), torch.tensor(down_wait), torch.tensor(
            loading), torch.tensor(location)
        self.cur_adj_matrix = self.gb.update_adj_matrix(self.cur_adj_matrix, up_call, down_call)
        self.cur_node_feature = self.gb.update_node_feature(self.cur_node_feature, up_wait, down_wait, load_up,
                                                            load_down, location)
        distances = self.get_floor2elevator_dis(up_wait.device)
        valid_action_mask = self.mansion.get_unallocated_floors_mask()
        valid_action_mask = torch.tensor(valid_action_mask).to(up_wait.device)

        legal_masks = self.get_action_mask(up_wait.device)
        elevator_mask, floor_mask = self.get_action_mask_plus(up_wait.device)
        ms = {'adj_m': self.cur_adj_matrix, 'node_feature_m': self.cur_node_feature,
              'legal_masks': legal_masks, 'elevator_mask': elevator_mask, 'floor_mask': floor_mask,
              'distances': distances, 'valid_action_mask': valid_action_mask}
        return ms

    def seed(self, seed=None):
        set_seed(seed)

    def reset(self):
        self.mansion.reset_env()
        self.cur_node_feature = self.empty_node_feature.clone()
        self.cur_adj_matrix = self.empty_adj_matrix.clone()
        state = self.get_smec_state()
        if self.seed_c:
            self.seed_c += 100
            self.seed(self.seed_c)
        self.reward_filter.reset()
        self.state_filter.reset()

        # self.data_idx = 0
        # self.next_generate_person = self.real_dataset[self.data_idx]
        # print(state)
        return state

    def render(self, **kwargs):
        self.viewer.view()

    def close(self):
        pass

    @property
    def attribute(self):
        return self.mansion.attribute

    @property
    def state(self):
        return self.mansion.state

    @property
    def statistics(self):
        return self.mansion.get_statistics()

    @property
    def log_debug(self):
        return self._config.log_notice

    @property
    def log_notice(self):
        return self._config.log_notice

    @property
    def log_warning(self):
        return self._config.log_warning

    @property
    def log_fatal(self):
        return self._config.log_fatal


iteration = 1000


def uniform_dispatch(up_floors, down_floors, elev_num, random_one=True):
    all_ups = [('up', ele) for ele in up_floors]
    all_downs = [('down', ele) for ele in down_floors]
    all_call_candidate = all_ups + all_downs
    candidate_num = len(all_call_candidate)
    split_interval = candidate_num // elev_num
    final_assignment = []
    for ele_id in range(elev_num):
        lb = ele_id * split_interval
        if ele_id == elev_num - 1:
            cur_assign = all_call_candidate[lb:]
        else:
            cur_assign = all_call_candidate[lb: lb + split_interval]
        up_floors, down_floors = [], []
        if random_one and cur_assign:
            selected_ele = random.choice(cur_assign)
            if selected_ele[0] == 'up':
                up_floors.append(selected_ele[1])
            else:
                down_floors.append(selected_ele[1])
        else:
            for ele in cur_assign:
                if ele[0] == 'up':
                    up_floors.append(ele[1])
                else:
                    down_floors.append(ele[1])
        final_assignment.append(ElevatorHallCall(up_floors, down_floors))
    return final_assignment


def identity_dispatch(up_floors, down_floors, elev_num):
    hallcall = ElevatorHallCall(up_floors, down_floors)
    return_ele = [hallcall for i in range(elev_num)]
    return return_ele


def make_env(seed=0, render=False, forbid_uncalled=False, use_graph=True, gamma=0.99, real_data=True,
             use_advice=False, special_reward=False, data_dir=None, file_begin_idx=None, dos=''):
    def _thunk():
        return SmecRLEnv(render=render, seed=seed, forbid_uncalled=forbid_uncalled, use_graph=use_graph, gamma=gamma,
                         real_data=real_data, use_advice=use_advice, special_reward=special_reward, data_dir=data_dir, file_begin_idx=file_begin_idx, dos=dos)

    return _thunk


def test_multi_env(num_processes):
    envs = [make_env(seed=i) for i in range(num_processes)]
    envs = AsyncVectorEnv(env_fns=envs)
    bo = envs.reset()
    batch_action = [torch.tensor([1 for i in range(6)]) for j in range(num_processes)]
    obs, rew, done, info = envs.step(batch_action)
    return


if __name__ == '__main__':
    # test_multi_env(8)
    eval_env = make_env(seed=0, render=False,
                        real_data=True, data_dir='../train_data/new/lunchpeak')()

    for t in range(360):

        a = eval_env.mansion._person_generator.generate_person()
        print(t, a)
        eval_env.step()
