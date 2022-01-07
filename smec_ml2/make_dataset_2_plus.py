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
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.smec_elevator_new import SmecElevator
from smec_liftsim.utils import ElevatorHallCall
import configparser
import os
from smec_rl_components.smec_graph_build import *
from smec_liftsim.smec_constants import *

from copy import deepcopy
import random
import time
from smec_liftsim.utils import PersonType
import json


class SmecEnv:
    def __init__(self, render=False):
        config_file = '../smec_liftsim/rl_config2.ini'
        file_name = config_file
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy.'

        self._config = MansionConfig(
            dt=time_step,
            # number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            number_of_floors=16,
            floor_height=float(config['MansionInfo']['FloorHeight']),
            maximum_acceleration=float(config['MansionInfo']['Acceleration']),
            maximum_speed=float(config['MansionInfo']['RateSpeed']),
            person_entering_time=float(config['MansionInfo']['PersonEnterTime']),
            door_opening_time=float(config['MansionInfo']['DoorOpeningTime']),
            door_closing_time=float(config['MansionInfo']['DoorClosingTime']),
            keep_door_open_lag=float(config['MansionInfo']['DoorKeepOpenLagTime']),
            door_well_time2=float(config['MansionInfo']['DwellTime2']),
            maximum_parallel_entering_exiting_number=int(config['MansionInfo']['ParallelEnterNum']),
            rated_load=int(config['MansionInfo']['RateLoad'])
        )

        self.mansion = MansionManager(1, None, self._config, config['MansionInfo']['Name'])
        self.viewer = None
        self.open_render = render
        if render:
            from smec_liftsim.rendering import Render
            self.viewer = Render(self.mansion)
        self.elevator_num = self.mansion.attribute.ElevatorNumber
        self.floor_num = self._config.number_of_floors

        self.seed_c = None

        self.evaluate_info = {'valid_up_action': 0,
                              'advice_up_action': 0,
                              'valid_dn_action': 0,
                              'advice_dn_action': 0}

    def get_time(self):
        raw_time = self._config.raw_time
        cur_day = raw_time // (24 * 3600)
        cur_time = raw_time % (24 * 3600)
        return [cur_day, int(cur_time // 3600 + 7), int(cur_time % 3600 // 60), int(cur_time % 60)]

    def step_an_episode(self, person_list, init_floor, init_direction):
        self.mansion.generate_person(person_list=person_list)
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        action = [ElevatorHallCall(unallocated_up, unallocated_dn)]
        last_state = ELEVATOR_STOP_DOOR_CLOSE
        route = [self.mansion._elevators[0]._sync_floor]

        # 在模拟运行之前，可以按概率得出一个可能的停靠向量，代表在各个楼层停靠的概率
        vec_approxim = [0 for i in range(self.floor_num * 3)]
        vec_approxim[self.mansion._elevators[0]._sync_floor] = 1
        separate_post_prob = np.load('separate_post_prob.npy')
        if init_direction == 1:
            for uc in unallocated_up:
                # 1
                if uc >= init_floor:
                    vec_approxim[uc] = 1
                    for pred_carcall in range(uc, self.floor_num):
                        vec_approxim[pred_carcall] += separate_post_prob[uc][pred_carcall]
                # 3
                else:
                    vec_approxim[uc + 2 * self.floor_num] = 1
                    for pred_carcall in range(uc, self.floor_num):
                        vec_approxim[pred_carcall + 2 * self.floor_num] += separate_post_prob[uc][pred_carcall]
            for dc in unallocated_dn:
                vec_approxim[dc + self.floor_num] = 1
                for pred_carcall in range(0, dc):
                    vec_approxim[pred_carcall + self.floor_num] += separate_post_prob[dc][pred_carcall]

        elif (init_direction == -1):
            for uc in unallocated_up:
                vec_approxim[uc + self.floor_num] = 1
                for pred_carcall in range(uc, self.floor_num):
                    vec_approxim[pred_carcall + self.floor_num] += separate_post_prob[uc][pred_carcall]
            for dc in unallocated_dn:
                # 1
                if dc <= init_floor:
                    vec_approxim[dc] = 1
                    for pred_carcall in range(0, dc):
                        vec_approxim[pred_carcall] += separate_post_prob[dc][pred_carcall]
                # 3
                else:
                    vec_approxim[dc + 2 * self.floor_num] = 1
                    for pred_carcall in range(0, dc):
                        vec_approxim[pred_carcall + 2 * self.floor_num] += separate_post_prob[dc][pred_carcall]

        for carcall in self.mansion._elevators[0]._car_call:
            vec_approxim[carcall] = 1
        for i in range(self.floor_num * 3):
            vec_approxim[i] = min(1, vec_approxim[i])

        # real stop
        # print('action:', action)
        timeout = 300
        while self.mansion.finish_person_num != len(person_list) + len(init_persons) and timeout > 0:
            timeout -= self._config.delta_t
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt \
                = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=True)
            if self.viewer:
                self.render()
            action = None
            new_state = self.mansion._elevators[0]._run_state
            if last_state == ELEVATOR_RUN and new_state == ELEVATOR_STOP_DOOR_OPENING:
                flr = self.mansion._elevators[0]._sync_floor
                route.append(flr)
            last_state = new_state
        return route, vec_approxim

    def seed(self, seed=None):
        set_seed(seed)

    def reset(self):
        self.mansion.reset_env()
        if self.seed_c:
            self.seed_c += 100
            self.seed(self.seed_c)
        # return state

    def is_end(self):
        return self.mansion.is_done

    def render(self, **kwargs):
        self.viewer.view()

    def close(self):
        pass

    @property
    def state(self):
        return self.mansion.state

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
        return np.mean(waiting_time), np.mean(transmit_time)

    def get_reward_per_flr(self):
        ret = []
        # print('reward: ', len(init_persons), self.mansion.person_info.keys(), len(person_list))
        for id in self.mansion.person_info.keys():
            if id < len(init_persons):
                continue
            src = person_list[id - len(init_persons)].SourceFloor - 1
            dst = person_list[id - len(init_persons)].TargetFloor - 1
            info = self.mansion.person_info[id]
            if len(info) < 5:
                # 超载没处理的人
                continue
            waiting_time = info[2]
            transmit_time = info[4]
            ret.append({'id': id,
                        'src': src,
                        'dst': dst,
                        'waiting_time': waiting_time,
                        'transmit_time': transmit_time,
                        })
        return ret


def generate_person(flow_map):
    ret_persons = []
    # 按概率随机生成人，根据电梯速度和方向，指定方向和位置来生成？不生成第三趟的人
    id = len(init_persons)
    for i in range(1):
        samples = np.random.rand(*flow_map.shape)
        for src in range(samples.shape[0]):
            for dst in range(samples.shape[1]):
                if (samples[src, dst] < flow_map[src, dst]):
                    ret_persons.append(
                        PersonType(
                            id,
                            75,
                            src + 1,  # src
                            dst + 1,  # dst
                            0,  # cur time
                            0
                        ))
                    id += 1
    return ret_persons


def generate_person_init(flow_map, init_floor, init_direction):
    ret_persons = []
    # 按概率随机生成人，根据电梯速度和方向，指定方向和位置来生成？不生成第三趟的人
    id = 0
    for i in range(5):
        samples = np.random.rand(*flow_map.shape)
        for src in range(samples.shape[0]):
            for dst in range(samples.shape[1]):
                if (((init_direction == 1) and (dst > init_floor) and (src < init_floor)) or \
                    ((init_direction == -1) and (dst < init_floor) and (src > init_floor))) and (
                        samples[src, dst] < flow_map[src, dst]):
                    ret_persons.append(
                        PersonType(
                            id,
                            75,
                            src + 1,  # src
                            dst + 1,  # dst
                            0,  # cur time
                            0
                        ))
                    id += 1
                    if id == 5:
                        return ret_persons
    return ret_persons


def init_elevator(elev, init_floor, init_direction, init_persons=None):
    # 先设定电梯的方向速度位置，已载乘客和car call情况
    if init_persons is None:
        init_persons = []
    for ip in init_persons:
        elev._loaded_person[ip.TargetFloor - 1].append(deepcopy(ip))
        elev._load_weight += ip.Weight
        elev.press_button(ip.TargetFloor - 1)
        elev_env.mansion.person_info[ip.ID] = [0, 0, 0, 0]

    elev._run_direction = init_direction
    elev._service_direction = init_direction
    elev._current_position = init_floor * 3
    elev._sync_floor = init_floor

    pass


def save_data(df, init_floor, init_direction):
    # print(person_list, route, info)
    # for p in person_list:
    #     print(p.ID, p.SourceFloor, p.TargetFloor)
    # print(route)

    vec = [0 for i in range(elev_env.floor_num * 3)]
    vec_zero = [0 for i in range(elev_env.floor_num * 3)]

    # solve route
    # up
    if (init_direction == 1):
        top_idx = 0
        top_max = route[0]
        for r in range(1, len(route)):
            if route[r] > top_max:
                top_max = route[r]
                top_idx = r
            else:
                break

        down_min = top_max
        down_idx = top_idx
        for r in range(top_idx + 1, len(route)):
            if route[r] < down_min:
                down_min = route[r]
                down_idx = r
            else:
                break

        up_route = route[0:top_idx + 1]
        dn_route = route[top_idx:down_idx + 1]
        up_route2 = route[down_idx:]

        if len(up_route) == 1:
            up_route = []
        if len(dn_route) == 1:
            dn_route = []
        if len(up_route2) == 1:
            up_route2 = []

        for ur in up_route:
            vec[ur] = 1
            vec_zero[ur] = 1
        for dr in dn_route:
            vec[dr + elev_env.floor_num] = 1
            vec_zero[dr + elev_env.floor_num] = 1
        for ur2 in up_route2:
            vec[ur2 + elev_env.floor_num * 2] = 1
            if (ur2 < init_floor):
                vec_zero[ur2 + elev_env.floor_num * 2] = 1
    # down
    else:
        down_idx = init_floor
        down_min = route[0]
        for r in range(1, len(route)):
            if route[r] < down_min:
                down_min = route[r]
                down_idx = r
            else:
                break

        top_idx = down_idx
        top_max = down_min
        for r in range(down_idx + 1, len(route)):
            if route[r] > top_max:
                top_max = route[r]
                top_idx = r
            else:
                break

        dn_route = route[0:down_idx + 1]
        up_route = route[down_idx:top_idx + 1]
        dn_route2 = route[top_idx:]

        if len(dn_route) == 1:
            dn_route = []
        if len(up_route) == 1:
            up_route = []
        if len(dn_route2) == 1:
            dn_route2 = []

        for dr in dn_route:
            vec[dr] = 1
            vec_zero[dr] = 1
        for ur in up_route:
            vec[ur + elev_env.floor_num] = 1
            vec_zero[ur + elev_env.floor_num] = 1
        for dr2 in dn_route2:
            vec[dr2 + elev_env.floor_num * 2] = 1
            if (dr2 > init_floor):
                vec_zero[dr2 + elev_env.floor_num * 2] = 1

    # print('x:', vec)

    # solve time
    src2wt = [0 for i in range(elev_env.floor_num * 3)]
    src2pnum = [0 for i in range(elev_env.floor_num * 3)]
    # up
    if (init_direction == 1):
        for i in info:
            src = i['src']
            dst = i['dst']
            if src > dst:
                src2wt[src + elev_env.floor_num] += i['waiting_time']
                src2pnum[src + elev_env.floor_num] += 1
            elif src < init_floor:
                src2wt[src + elev_env.floor_num * 2] += i['waiting_time']
                src2pnum[src + elev_env.floor_num * 2] += 1
            else:
                src2wt[src] += i['waiting_time']
                src2pnum[src] += 1
    # down
    else:
        for i in info:
            src = i['src']
            dst = i['dst']
            if src < dst:
                src2wt[src + elev_env.floor_num] += i['waiting_time']
                src2pnum[src + elev_env.floor_num] += 1
            elif src > init_floor:
                src2wt[src + elev_env.floor_num * 2] += i['waiting_time']
                src2pnum[src + elev_env.floor_num * 2] += 1
            else:
                src2wt[src] += i['waiting_time']
                src2pnum[src] += 1

    for i in range(elev_env.floor_num * 3):
        if src2pnum[i] != 0:
            src2wt[i] /= src2pnum[i]
    # print('y:', src2wt)
    x = json.dumps({'x': vec, 'x_zero': vec_zero, 'y': src2wt, 'x_prime': vec_approxim})
    print(x, file=df)
    pass


def route_num(init, src, dst):
    if src < dst:
        if src >= init:
            return 1
        else:
            return 3
    else:
        return 2


if __name__ == '__main__':

    dataset_file = open('dataset16_all_up.txt', 'a')

    elev_env = SmecEnv(render=False)
    # flow_map = np.zeros((elev_env.floor_num, elev_env.floor_num))
    flow_map = np.load('independent_flow_map.npy') / 2
    print(flow_map)
    # init_persons = [PersonType(0, 75, 0+1, 2+1, 0, 0)]
    init_persons = []
    init_direction = 1

    for d in range(30000):
        print(d)
        elev_env.reset()

        # rand
        init_floor = random.randint(0, elev_env.floor_num - 1)
        init_persons = generate_person_init(flow_map, init_floor, init_direction)

        init_elevator(elev_env.mansion._elevators[0], init_floor, init_direction, init_persons)

        person_list = []
        while not person_list:
            person_list = generate_person(flow_map)
        # print('init:', init_floor, [ip.SourceFloor - 1 for ip in init_persons], [16*(route_num(init_floor, ip.SourceFloor - 1, ip.TargetFloor - 1)-1)+ ip.SourceFloor - 1 for ip in person_list])
        route, vec_approxim = elev_env.step_an_episode(person_list, init_floor, init_direction)

        info = elev_env.get_reward_per_flr()

        save_data(dataset_file, init_floor, init_direction)

        # print(elev_env.person_info)
        # print(elev_env.get_reward())  # (0.35, 3.8)
