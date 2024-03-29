# # 只根据当前状态作搜索选择
# # 另外每次应该记住topK个分配，下次有人来的时候，优先从这topK个方案开始搜索。
#
#
# # 只要没被serve就可以重新被重新分配
# # 但有一个问题，就是当前电梯的目的地、运行方向和速度可能与分配有关，分配方案变了会导致某些电梯陷入尴尬的运动状态，
# # 且又不能简单地假设电梯可以立刻停下，变向什么的。
# # 还有个问题就是假设1楼有20个人，派了0号梯去接，在evaluate的时候，0去完之后1楼还有人等着需要分配，那到底怎么分配，怎么算这些人。
# # 不一定要每个人都立刻分配，可以给分配none，下次某个dt再分配，也就是延迟分配，这样也可以干好多事情，只要达到整体效益最大就行。
# # 但是也是dt的分配，而不是de（event），难搞，别限太死，结合起来。
# def get_to_allocate():
#     return [0,1,3,6]
#
# def evaluate(allocation):


# 先别管那么多，仿照金奖写一个局部搜索；有一个必须管的事情是电梯的hall call被取消后怎么运动。（用parkcall写了一个尽快就近停靠，受最大加速度限制）
# 目前由于是会固定调整电梯的位置的，所以会把电梯弹回syn_floor；并且v会继续变化，然后变到最大。应该至少都应该让电梯移动到advance floor（如果电梯在动的话）
# 新的接人任务出现，先按最小cost（对人来说最近电梯（如果都没有去直接接这个人的话，也可以在某一个合适的时刻重新计算最近电梯来分配）
# or对电梯来说，加上这个人之后整体最优）；然后，找到能够交换的任务（只有接的任务能换了，在不同电梯之间交换？）


# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter

from smec_liftsim.generator_proxy import set_seed
from smec_liftsim.generator_proxy import PersonGenerator
from smec_liftsim.fixed_data_generator import FixedDataGenerator
from random_data_generator import RandomDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.utils import ElevatorHallCall
import configparser
import os
from smec_rl_components.smec_graph_build import *
from smec_liftsim.smec_constants import *

from copy import deepcopy
import random
import time


class SmecEnv:
    def __init__(self, data_file='smec_rl/simple_dataset_v2.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
                 use_graph=True, real_data=True, use_advice=False, special_reward=False, data_dir=None, file_begin_idx=None, dos=''):
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__) + '/smec_liftsim/rl_config2.ini')
        file_name = config_file
        self.forbid_uncalled = forbid_uncalled
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy.'
        # dos = '00:00-10:00'
        # dos = '00:00-20:00'
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
            person_generator = RandomDataGenerator(data_dir=data_dir, data_of_section=dos, random_or_load_or_save=1)

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
            from smec_liftsim.rendering import Render
            self.viewer = Render(self.mansion)
        self.elevator_num = self.mansion.attribute.ElevatorNumber
        self.floor_num = int(config['MansionInfo']['NumberOfFloors'])

        if seed is not None:
            self.seed(seed)
        self.seed_c = seed

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
                uncalled_elevators = self.mansion.get_uncalled_elevators()
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
                uncalled_elevators = self.mansion.get_uncalled_elevators()
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

    def get_time(self):
        raw_time = self._config.raw_time
        cur_day = raw_time // (24 * 3600)
        cur_time = raw_time % (24 * 3600)
        return [cur_day, int(cur_time // 3600 + 7), int(cur_time % 3600 // 60), int(cur_time % 60)]

    def step_dt(self, action, verbose=False):
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        is_valid = 0
        for up_floor in unallocated_up:
            cur_elev = action
            all_elv_up_fs[cur_elev].append(up_floor)
            is_valid += 1
        for dn_floor in unallocated_dn:
            cur_elev = action
            all_elv_down_fs[cur_elev].append(dn_floor)
            is_valid += 1
        if verbose and is_valid:
            print(f'Choosing {action} for up:{unallocated_up}, dn:{unallocated_dn}, valid:{is_valid}')
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute)
        self.mansion.generate_person()
        new_obs = self.get_smec_state()
        reward = 0
        done = self.mansion.person_generator.done
        info = {}
        return new_obs, reward, done, info

    def step_dp(self, action, dcar_call=True):
        next_call_come = False
        total_energy = 0
        rewards = []
        while not next_call_come and not self.is_end():
            # ret = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=False)
            ret = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=False, special_reward=True)
            energy = ret[-1]
            hall_waiting_rewards = ret[-3]
            car_waiting_rewards = ret[-2]
            total_energy += ret[-1]
            self.mansion.generate_person()
            if self.open_render:
                self.render()
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            action = [ElevatorHallCall([], []) for i in range(self.elevator_num)]
            next_call_come = unallocated_up != [] or unallocated_dn != []

            factor = 0
            reward = 0.01 * (-np.array(hall_waiting_rewards) - factor * np.array(car_waiting_rewards) - 5e-4 * energy)
            rewards.append(sum(reward))

        return total_energy, rewards

    def step(self, action):
        return self.step_dp(action)

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

    def get_smec_state(self):
        up_wait, down_wait, loading, location, up_call, down_call, load_up, load_down = self.mansion.get_rl_state(
            encode=True)
        up_wait, down_wait, loading, location = torch.tensor(up_wait), torch.tensor(down_wait), torch.tensor(
            loading), torch.tensor(location)
        legal_masks = self.get_action_mask(up_wait.device)
        self.cur_adj_matrix = self.gb.update_adj_matrix(self.cur_adj_matrix, up_call, down_call)
        self.cur_node_feature = self.gb.update_node_feature(self.cur_node_feature, up_wait, down_wait, load_up,
                                                            load_down, location)
        distances = self.get_floor2elevator_dis(up_wait.device)
        valid_action_mask = self.mansion.get_unallocated_floors_mask()
        valid_action_mask = torch.tensor(valid_action_mask).to(up_wait.device)
        ms = {'adj_m': self.cur_adj_matrix, 'node_feature_m': self.cur_node_feature, 'legal_masks': legal_masks,
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

        # self.data_idx = 0
        # self.next_generate_person = self.real_dataset[self.data_idx]
        return state

    def is_end(self):
        return self.mansion.is_done

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

    def get_reward(self):
        waiting_time = []
        transmit_time = []
        for k in self.mansion.person_info.keys():
            info = self.mansion.person_info[k]
            try:
                waiting_time.append(info[2])
                transmit_time.append(info[4])
            except:
                continue
        return np.mean(waiting_time), np.mean(transmit_time), len(waiting_time)


def up_queue_length(floor):
    # 上行等待队列长度
    waiting=elev_env.mansion._wait_upward_persons_queue
    return len(waiting[floor])


def dn_queue_length(floor):
    # 下行等待队列长度
    waiting = elev_env.mansion._wait_downward_persons_queue
    return len(waiting[floor])


def solve(hallcall, direction):
    global count
    if mode == 'nearest':
        # 选择距离最近的电梯，从距离相同的电梯中随机选择一个
        min_dist = 100
        target = []
        for i,elev in enumerate(elev_env.mansion._elevators):
            dist = abs(elev._sync_floor - hallcall)
            if dist < min_dist:
                min_dist = dist
                target = [i]
            elif dist == min_dist:
                target.append(i) # pass
        choice = random.randint(0, len(target) - 1)
        target = target[choice]

    elif mode == 'rr':
        # 轮流
        target = count
        count = (count+1) % elev_num

    elif mode == 'longest_first':
        # 选择对于已分配的hallcall，未接人数最少的电梯。对于人数相同的电梯中随机选择一个
        # 只计算hallcall人数，没有计算梯内人数
        weight = 1000
        target = []
        for i in range(elev_num):
            if weight > serving_num[i]:
                weight = serving_num[i]
                target = [i]
            elif weight == serving_num[i]:
                target.append(i)

        choice = random.randint(0, len(target) - 1)
        target = target[choice]
        # update
        if direction == 1:
            serving_num[target] += up_queue_length(hallcall)**2
        else:
            serving_num[target] += dn_queue_length(hallcall)**2

    elif mode == 'scan':
        # 四个电梯，采用相同运动方式
        target = 0

    elif mode == 'eta':
        # 选择预测到达时间最短的电梯，从时间相同的电梯中随机选择一个
        min_dist = 1000
        target = []
        #print('floor:', hallcall)
        for i,elev in enumerate(elev_env.mansion._elevators):
            # 计算时间：只是将需要转向的电梯，计算其通过的总楼层数（不计停止时间）
            # 对转向的电梯，假设它总是到达最远的楼层
            # 电梯停止
            if elev._run_state == ELEVATOR_STOP_DOOR_CLOSE:
                dist = abs(elev._sync_floor - hallcall)
            # 否则
            elif direction == 1:
                if elev._run_direction == 1:
                    if elev._sync_floor <= hallcall - 1:# 防止离得太近，停不下来
                        dist = hallcall - elev._sync_floor
                    else:
                        dist = (floor_num - elev._sync_floor - 1) + (floor_num - 1) + hallcall
                else:
                    dist = hallcall + elev._sync_floor
            else:
                if elev._run_direction == -1:
                    if elev._sync_floor >= hallcall + 1:
                        dist = elev._sync_floor -hallcall
                    else:
                        dist = elev._sync_floor + (floor_num - 1) + (floor_num - hallcall -1)
                else:
                    dist = 2 * floor_num - hallcall - elev._sync_floor + 2
            #print(dist)
            if dist < min_dist:
                min_dist = dist
                target = [i]
            elif dist == min_dist:
                target.append(i) # pass
        choice = random.randint(0, len(target) - 1)
        target = target[choice]

    return target

def prework(serving_num):
    # 计算每个电梯要接的总人数（权值为人数的平方）
    for elev in elev_env.mansion._elevators:
        sum=0
        for upcall in elev._hall_up_call:
            sum += up_queue_length(upcall)**2
        for downcall in elev._hall_dn_call:
            sum += dn_queue_length(downcall)**2
        serving_num.append(sum)
    #print(serving_num)


if __name__ == '__main__':
    dds = [
        ('./train_data/new/lunchpeak', '00:00-06:00'),
        ('./train_data/new/uppeak', '30:00-36:00'),
        ('./train_data/new/dnpeak', '06:00-12:00'),
        ('./train_data/new/notpeak', '00:00-06:00'),
    ]
    modes = ['nearest', 'rr', 'longest_first', 'scan', 'eta']
    # modes = ['eta']
    for mode in modes:
        print(mode)
        for dd in dds:
            pattern = dd[0].split('/')[-1]
            file = open(f'experiment_results/smec/{mode}-{pattern}.log', 'a')
            print('-' * 50, file=file)
            print(dd[0], dd[1], file=file)
            elev_env = SmecEnv(render=False, data_dir=dd[0], dos=dd[1])

            test_num = 20
            total_res = 0
            total_awt = 0
            total_att = 0
            total_energies = 0
            rs = []
            for tn in range(test_num):
                floor_num = elev_env.mansion._floor_number
                elev_num = 4
                best = 10000
                best_sol = None
                count = 0
                elev_env.reset()
                total_energy = 0

                while not elev_env.is_end():
                    # 问题是现在action是多维的，按理来说同一dt内也不会有多个楼层需要分配，但是可以重分配的话，就有了。所以这还是个组合优化的问题？但也可以先不考虑重分配，假设每次都是当前最优，就是贪心。
                    unallocated_up, unallocated_dn = elev_env.mansion.get_unallocated_floors()
                    action = [[[], []] for i in range(elev_num)]
                    if mode == 'longest_first':
                        # 将楼层按等待人数排序（降序），并对各电梯任务接人数进行计算
                        unallocated_up.sort(key=up_queue_length, reverse=True)
                        unallocated_dn.sort(key=dn_queue_length, reverse=True)
                        serving_num = []
                        prework(serving_num)

                    if unallocated_dn or unallocated_up:
                        for dn in unallocated_dn:
                            # print(dis_idx)
                            elev_idx = solve(dn, -1)
                            # scan:四个电梯相同运动
                            if mode == 'scan':
                                for i in range(elev_num):
                                    action[i][1].append(dn)
                            else:
                                action[elev_idx][1].append(dn)

                        for up in unallocated_up:
                            # print(dis_idx)
                            elev_idx = solve(up, 1)
                            if mode == 'scan':
                                for i in range(elev_num):
                                    action[i][0].append(up)
                            else:
                                action[elev_idx][0].append(up)
                    action = [ElevatorHallCall(hallcall[0], hallcall[1]) for hallcall in action]
                    # print(f'execute action: {action}')
                    # total_energy += elev_env.step(action)
                    energy, rewards = elev_env.step(action)
                    total_energy += energy
                    rs += rewards

                awt, att, pnum = elev_env.get_reward()
                total_res += awt + att
                total_awt += awt
                total_att += att
                total_energies += total_energy
                print(f'{awt:.2f} {att:.2f} {awt + att:.2f} {total_energy:.2f} for {pnum} people')
                print(f'{awt:.2f} {att:.2f} {awt + att:.2f} {total_energy:.2f} for {pnum} people', file=file)
            print(f'awt: {total_awt / test_num:.2f} att: {total_att / test_num:.2f} '
                  f'average time: {total_res / test_num:.2f} average energy: {total_energies / test_num:.2f}')
            print(f'awt: {total_awt / test_num:.2f} att: {total_att / test_num:.2f} '
                  f'average time: {total_res / test_num:.2f} average energy: {total_energies / test_num:.2f}', file=file)
            print(f'Reward list: {rs}', file=file)
            print(file=file)
            file.close()



