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

from random_data_generator import RandomDataGenerator
from smec_liftsim.generator_proxy import set_seed
from smec_liftsim.generator_proxy import PersonGenerator
from smec_liftsim.fixed_data_generator import FixedDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.smec_elevator_new2 import SmecElevator
from smec_liftsim.utils import ElevatorHallCall
import configparser
import os
from smec_rl_components.smec_graph_build import *
from smec_liftsim.smec_constants import *

from copy import deepcopy
import random
import time


class SmecEnv:
    def __init__(self, data_file='smec_rl/simple_dataset_v2_test.csv', config_file='./smec_liftsim/rl_config2.ini', render=True, seed=10, forbid_uncalled=False,
    # def __init__(self, data_file='smec_rl/simple_dataset.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
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
        # dos = '30:00-40:00'
        # dos = ''
        # dos = '00:00-06:00'
        # dos = '06:00-12:00'
        # dos = '10:00-16:00'
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
            # person_generator = RandomDataGenerator(data_dir=data_dir, data_of_section=dos)

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
        self.mansion.use_old_unallocate_version = True
        self.use_graph = use_graph
        self.viewer = None
        self.open_render = render
        if render:
            from smec_liftsim.rendering import Render
            self.viewer = Render(self.mansion)
        self.elevator_num = self.mansion.attribute.ElevatorNumber
        self.floor_num = int(config['MansionInfo']['NumberOfFloors'])

        np.random.seed(seed)
        random.seed(seed)

    def step_dt(self, action, verbose=False):
        self.mansion.run_mansion(action, use_rules=False, replace_hallcall=True)
        self.mansion.generate_person()
        self.render()

    def step_dp(self, action, dcar_call=True):
        next_call_come = False
        total_energy = 0
        rewards = []
        while not next_call_come and not self.is_end():
            # ret = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=True)
            ret = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=True, special_reward=True)
            energy = ret[-1]
            hall_waiting_rewards = ret[-3]
            car_waiting_rewards = ret[-2]

            total_energy += ret[-1]
            self.mansion.generate_person()
            if self.open_render:
                self.render()
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            action = None
            next_call_come = unallocated_up != [] or unallocated_dn != []

            factor = 0
            reward = 0.01 * (-np.array(hall_waiting_rewards) - factor * np.array(car_waiting_rewards) - 5e-4 * energy)
            rewards.append(sum(reward))
        return total_energy, rewards

    def step(self, action):
        return self.step_dp(action)
        # self.step_dt(action)

    def reset(self):
        self.mansion.reset_env()

    def is_end(self):
        return self.mansion.is_done

    def render(self, **kwargs):
        self.viewer.view()

    def close(self):
        pass

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


class TimeCnt:
    def __init__(self):
        self.cnt = 0

    def step(self):
        self.cnt += 1

    def reset(self):
        self.cnt = 0

    def get_time(self, dt):
        return self.cnt * dt


class LocalSearch:
    def __init__(self, env):
        self.modes = ['lunchpeak', 'notpeak', 'uppeak', 'dnpeak']
        self.mode = self.modes[0]
        self.mansion = env.mansion
        self.test_elev = SmecElevator(mansion_config=self.mansion._config, name='TEST_ELEV')
        self.elev_num = self.mansion._elevator_number
        self.floor_num = self.mansion._floor_number

        self.candidate_schemes = []
        self.ELIMINATE_SIZE = 200
        self.SELECT_SIZE = 20
        self.NGEN = 50
        self.MAX_MUTATE_TIME = 2
        self.CROSS_PROB = 0.8
        self.MUTATE_PROB = 0.1
        self.best_scheme = {'dispatch': [-1 for _ in range(self.floor_num*2)], 'score': -10000}
        self.time_cnt = TimeCnt()

        self.upcall_weights = np.load('weights_%s_upcall.npy' % self.mode)
        self.dncall_weights = np.load('weights_%s_dncall.npy' % self.mode)
        self.carcall_weights = np.load('weights_%s_carcall.npy' % self.mode)
        self.weight_t = 0

        # 电梯能不能用，如果超载就不要选。
        self.elevator_mask = [1 for i in range(self.elev_num)]

        # 记录每个电梯的分派对应时间，每次决策之后都要重置更新，可以加速
        self.elevator_dispatch_time_table = [{} for i in range(self.elev_num)]

        # 用于计算累积人数、权重
        self.updn_last_serve_time = []

    # 根据delta floor计算需要时间。一个更真实的运动估计。
    def df2time(self, df, elev=None):
        if df == 0:
            if elev.is_fully_open or elev._is_door_opening:
                open_time = (1 - elev._door_open_rate) / elev._door_open_velocity
                lag_time = elev._keep_door_open_left
                close_time = 1 / elev._door_open_velocity
                return open_time + lag_time + close_time
            elif elev._is_door_closing:
                close_time = (1 - elev._door_open_rate) / elev._door_open_velocity
                return close_time
            return 7.3
        if df == 1:
            return 12
        elif df == 2:
            return 13.5
        else:
            return 13.5 + 1.2 * (df - 2)

    def dict_dispatch2hallcalls(self, dispatch):
        hallcalls = [[[], []] for _ in range(self.elev_num)]
        for i in dispatch.keys():
            elev_idx = dispatch[i]
            hallcalls[elev_idx][i // self.floor_num].append(i % self.floor_num)
        # return hallcalls
        return [ElevatorHallCall(hallcall[0], hallcall[1]) for hallcall in hallcalls]

    def dict_dispatch2hallcall_lists(self, dispatch):
        hallcall_lists = [[] for _ in range(self.elev_num)]
        for i in dispatch.keys():
            elev_idx = dispatch[i]
            hallcall_lists[elev_idx].append(i)
        return hallcall_lists

    def dispatch2hallcalls(self, dispatch):
        hallcalls = [[[], []] for _ in range(self.elev_num)]
        for i in range(len(dispatch)):
            if dispatch[i] == -1:
                continue
            elev_idx = dispatch[i]
            hallcalls[elev_idx][i // self.floor_num].append(i % self.floor_num)
        return [ElevatorHallCall(hallcall[0], hallcall[1]) for hallcall in hallcalls]

    def dispatch2hallcall_lists(self, dispatch):
        hallcall_lists = [[] for _ in range(self.elev_num)]
        for i in range(len(dispatch)):
            if dispatch[i] == -1:
                continue
            elev_idx = dispatch[i]
            hallcall_lists[elev_idx].append(i)
        return hallcall_lists

    # hallcall是ElevatorHallCall，都是0到16，而hallcall_list是0-32，car_call是0-16的
    def hallcall_list_carcall2key(self, hallcall_list, car_call):
        key = 0
        for call in hallcall_list:
            key += 2**call
        for call in car_call:
            key += 2**(self.floor_num*2+call)
        return key

    def add_hallcall_to_elev(self, hallcall, elev):
        floor_num = elev._number_of_floors
        if hallcall >= floor_num:
            if hallcall - floor_num not in elev._hall_dn_call:
                elev._hall_dn_call.append(hallcall - floor_num)
        else:
            if hallcall not in elev._hall_up_call:
                elev._hall_up_call.append(hallcall)

    # 计算电梯以当前速度和位置运行，要停下来至少要走的楼层。
    def cal_stop_floor(self, x, v, max_a, floor_height):
        if v > 0:
            run_dir = 1
        elif v < 0:
            run_dir = -1
        else:
            run_dir = 0
        least_run_dis = v * v / (2 * max_a)
        least_stop_pos = x + run_dir * least_run_dis
        stop_flr = max((least_stop_pos + run_dir * (floor_height - 0.001)), 0) // floor_height
        return int(stop_flr)

    # 计算电梯第一次停下来所需时间，因为可能有初速度，所以要特殊处理
    def cal_first_stop_time(self, elev, first_stop_flr, state):
        # self.elevator_mask[elev.elev_idx] = 1
        first_stop_pos = first_stop_flr * elev._floor_height
        # 如果第一落点是纯hallcall且满载了，那么这个电梯不能选
        if (state == 2 or state == 4) and elev._is_overloaded:
            # self.elevator_mask[elev.elev_idx] = 0
            return 10000
        cur_spd = elev._current_velocity
        df = abs(elev._current_position - first_stop_pos) / elev._floor_height
        df = int(round(df))
        consume_time = max(self.df2time(df, elev) - cur_spd, 0)  # 观察经验公式
        return consume_time
    
    def cal_accumulate_person(self, t, floor, type, delta_time=0):
        if type == 1:
            person_num = 1
        elif type == 2:
            person_num = self.upcall_weights[t, floor] * delta_time
        else:
            person_num = self.dncall_weights[t, floor] * delta_time
        
        if person_num < 1:
            person_num = 1
        return person_num

    # 由楼层和状态（carcall、upcall、dncall）得到call的权重，可以由时间变化。
    def floor_state2weight(self, floor, state):
        # 1 car 2 up 4 dn
        weight = 0
        if state % 2 == 1:
            weight += self.carcall_weights[self.weight_t, floor]
        if state // 2 % 2 == 1:
            weight += self.upcall_weights[self.weight_t, floor] * self.updn_delta_time[floor] / 60
        if state // 4 % 2 == 1:
            weight += self.dncall_weights[self.weight_t, floor] * self.updn_delta_time[floor+self.floor_num] / 60
        return weight
    
    def floor_state2weight2(self, floor, state):
        # 1 car 2 up 4 dn
        weight = 0
        if state % 2 == 1:
            # weight += self.cal_accumulate_person(self.weight_t, floor, 1)
            weight += 1
        if state // 2 % 2 == 1:
            # weight += self.cal_accumulate_person(self.weight_t, floor, 2, self.updn_delta_time[floor] / 60)
            weight += 1
        if state // 4 % 2 == 1:
            # weight += self.cal_accumulate_person(self.weight_t, floor, 3, self.updn_delta_time[floor+self.floor_num] / 60)
            weight += 1
        return weight

    def get_elev_route(self, elev, srv_dir, stp_flr, cur_pos, car_call, hall_up_dn_call):
        route = []
        # 正常来说，carcall只在r1
        f_1 = (srv_dir + 1) * floor_num // 2  # f when 1, 0 when -1
        f_m1 = (-srv_dir + 1) * floor_num // 2  # f when -1, 0 when 1
        one_1 = (srv_dir + 1) * 1 // 2  # 1 when 1, 0 when -1
        one_m1 = (-srv_dir + 1) * 1 // 2  # 1 when -1, 0 when 1
        # srv_dir=1 010, srv_dir=-1 101
        rparam = [(stp_flr, f_1, srv_dir, one_m1), (f_1, f_m1, -srv_dir, one_1), (f_m1, stp_flr, srv_dir, one_m1)]
        for rnum, rp in enumerate(rparam):
            for i in range(rp[0], rp[1], rp[2]):
                # state: 8 park call; 1 car, 2 up, 3 car and up, 4 dn, 5 car and dn.
                state = 0
                if rnum == 0 and i in car_call:
                    state += 1
                if i in hall_up_dn_call[rp[3]]:
                    state += 2 * (1 + rp[3])
                if state != 0:
                    route.append((i, state))
            if rnum == 1:
                # 电梯运行方向上必须要有一个目的地，如果没有call，就是被重新分配搞的，要手动加一个停靠位置
                if route == [] or (route[0][0] * elev._floor_height - cur_pos) * srv_dir < 0:
                    route.insert(0, (stp_flr, 8))
        return route

    # 不用模拟，用两个delta floor的距离来近似计算，计算公式由实验得出。
    def estimate_elev_route_loss(self, elev, hallcall=None):
        copy_elev = deepcopy(elev)
        if hallcall:
            copy_elev.replace_hall_call(hallcall)
        cur_flr = copy_elev._sync_floor
        cur_pos = copy_elev._current_position
        cur_spd = copy_elev._current_velocity
        srv_dir = copy_elev._service_direction
        car_call = copy_elev._car_call
        hall_up_dn_call = [copy_elev._hall_up_call, copy_elev._hall_dn_call]
        stp_flr = self.cal_stop_floor(cur_pos, cur_spd, 0.557, 3.0)

        # 如果电梯之前是空闲的，可能分配了hallcall之后srv_dir也是0没来得及更新，先运行一个dt给他更新一下。
        if srv_dir == 0:
            if hall_up_dn_call[0] + hall_up_dn_call[1] == []:
                return 0
            else:
                copy_elev.run_elevator()
                srv_dir = copy_elev._service_direction
                # print(hall_up_dn_call, cur_flr, cur_spd, srv_dir)

        route = self.get_elev_route(copy_elev, srv_dir, stp_flr, cur_pos, car_call, hall_up_dn_call)

        # 从cur_flr以cur_spd
        # 从cur_pos以cur_spd开始完成电梯的旅程route，特殊处理第一次停靠。
        loss = 0
        accumulate_time = 0
        # 加入floor_weights, TODO: carcall可能应该用前一时间片的权重呢。
        # route肯定不为空
        assert len(route) > 0
        first_stop_flr = route[0][0]

        # 第一次停靠因为可能有初速度，需要特殊处理，还要处理超载的问题
        consume_time = self.cal_first_stop_time(copy_elev, first_stop_flr, route[0][1])
        accumulate_time += consume_time
        loss += accumulate_time * self.floor_state2weight2(first_stop_flr, route[0][1])

        # 其他段路可以直接用实验公式计算。
        last_flr = first_stop_flr
        for stop_flr in route[1:]:
            df = abs(stop_flr[0] - last_flr)
            consume_time = self.df2time(df, copy_elev)
            accumulate_time += consume_time
            loss += accumulate_time * self.floor_state2weight2(stop_flr[0], stop_flr[1])
            last_flr = stop_flr[0]
        # maximum_weight = copy_elev._maximum_capacity
        # load_weight = copy_elev._load_weight
        # loss += load_weight / (maximum_weight * 0.8) * 100
        # print(route, loss)
        return loss

    def evaluate_dispatch_faster(self, dispatch):
        # print('dispatch: ', self.print_dispatch(dispatch))
        total_loss = 0
        hallcalls = self.dict_dispatch2hallcalls(dispatch)
        hallcall_lists = self.dict_dispatch2hallcall_lists(dispatch)
        for idx in range(self.elev_num):
            elev_dispatch_key = self.hallcall_list_carcall2key(hallcall_lists[idx], self.mansion._elevators[idx]._car_call)
            if elev_dispatch_key in self.elevator_dispatch_time_table[idx].keys():
                loss = self.elevator_dispatch_time_table[idx][elev_dispatch_key]
            else:
                loss = self.estimate_elev_route_loss(self.mansion._elevators[idx], hallcalls[idx])
                self.elevator_dispatch_time_table[idx][elev_dispatch_key] = loss
            # print(idx, hallcalls[idx], self.mansion._elevators[idx]._car_call, loss)
            total_loss += loss
        # print('total loss: ', total_loss)
        return 1 / (total_loss + 0.01)

    def evaluate_dispatch(self, dispatch):
        return self.evaluate_dispatch_faster(dispatch)

    def get_cur_dispatch(self):
        dispatch = {}
        elevs = self.mansion._elevators
        for idx, elev in enumerate(elevs):
            for uc in elev._hall_up_call:
                dispatch[uc] = idx
            for dc in elev._hall_dn_call:
                dispatch[dc + self.floor_num] = idx
        return dispatch

    def print_dispatch(self, dispatch):
        return [i for i in dispatch if i != -1]

    def clear_hallcall(self):
        reallocates = []
        for idx, elev in enumerate(self.mansion._elevators):
            reallocates += elev._hall_up_call
            reallocates += [i + self.floor_num for i in elev._hall_dn_call]
            elev._hall_up_call = []
            elev._hall_dn_call = []
        return reallocates

    # 返回的是一个所有未完成hallcall的一个分配，可能会改变颠覆之前的分配。
    def get_action(self, add_hallcalls):
        cur_time = self.mansion._config.raw_time
        self.weight_t = int(cur_time // 60)
        # print(f'cur time: {self.weight_t}, up weights: {self.upcall_weights[self.weight_t]} dn weights: {self.dncall_weights[self.weight_t]}')
        self.elevator_dispatch_time_table = [{} for i in range(self.elev_num)]
        self.updn_delta_time = [cur_time - last_time for last_time in self.mansion.updn_last_serve_time]

        # TODO:应该不清除快要到达的call
        hallcall_need_reallocate = self.clear_hallcall()
        hallcall_need_allocate = add_hallcalls + hallcall_need_reallocate

        # 先生成初始解
        cur_dispatch = self.get_cur_dispatch()  # 完成上述todo之前，这个都会是空的
        for i in hallcall_need_allocate:
            if i not in cur_dispatch.keys():
                cur_dispatch[i] = random.randint(0, self.elev_num - 1)

        while True and hallcall_need_allocate:
            best_change_delta = -1e6
            cur_score = self.evaluate_dispatch(cur_dispatch)
            best_dispatch_per_iter = None
            for j in range(len(hallcall_need_allocate)):
                for e in range(self.elev_num):
                    h = hallcall_need_allocate[j]
                    new_dispatch = deepcopy(cur_dispatch)
                    new_dispatch[h] = e
                    new_score = self.evaluate_dispatch(new_dispatch)
                    delta = new_score - cur_score
                    if delta > best_change_delta:
                        best_change_delta = delta
                        best_dispatch_per_iter = new_dispatch
            if best_change_delta <= 0:
                break
            cur_dispatch = best_dispatch_per_iter

        return cur_dispatch


def evaluate():
    import re
    file = './evaluate_log/localsearch_full_first_version.log'
    all_action = []
    with open(file) as f:
        for l in f:
            if l.startswith('execute action: '):
                action = l[len('execute action: '):].strip('\n')
                if action == 'None':
                    action = None
                else:
                    action = action[1:-1]
                    action = action.split('HallCall')[1:]
                    res = []
                    for a in action:
                        up_dn = a.split(', HallDnCall')
                        up = re.findall(r'[[](.+?)[]]', up_dn[0])
                        dn = re.findall(r'[[](.+?)[]]', up_dn[1])
                        if up:
                            up = [int(i) for i in up[0].split(',')]
                        if dn:
                            dn = [int(i) for i in dn[0].split(',')]
                        res.append(ElevatorHallCall(up, dn))
                    action = res

                all_action.append(action)
    return all_action


if __name__ == '__main__':
    dds = [
        # ('./train_data/new/lunchpeak', '00:00-06:00'),
        ('./train_data/new/uppeak', '30:00-36:00'),
        ('./train_data/new/dnpeak', '06:00-12:00'),
        ('./train_data/new/notpeak', '00:00-06:00'),
    ]

    for dd in dds:
        pattern = dd[0].split('/')[-1]
        file = open(f'experiment_results/rewards/sfm2-{pattern}.log', 'a')
        print('-'*50, file=file)
        print(dd[0], dd[1], file=file)
        elev_env = SmecEnv(render=False, data_dir=dd[0], dos=dd[1])

        test_num = 20
        total_res = 0
        total_energies = 0
        rs = []
        for tn in range(test_num):

            elev_env.reset()
            floor_num = elev_env.mansion._floor_number
            # print(elev_env.mansion.person_generator.data.data)
            solver = LocalSearch(elev_env)
            total_energy = 0
            while not elev_env.is_end():
                unallocated_up, unallocated_dn = elev_env.mansion.get_unallocated_floors()
                # print(elev_env.mansion.elevators_car_call_change, '+'*20, 'car call re dispatch...')
                if unallocated_dn or unallocated_up:
                    add_hallcalls = unallocated_up
                    for udn in unallocated_dn:
                        add_hallcalls.append(udn + floor_num)

                    dispatch = solver.get_action(add_hallcalls)
                    action = solver.dict_dispatch2hallcalls(dispatch)
                else:
                    dispatch = solver.get_action([])
                    action = solver.dict_dispatch2hallcalls(dispatch)
                # print(f'execute action: {action}')
                energy, rewards = elev_env.step(action)
                total_energy += energy
                rs += rewards

            # print(elev_env.person_info)
            awt, att, pnum = elev_env.get_reward()
            print(f'{awt:.2f} {att:.2f} {awt+att:.2f} {total_energy:.2f} for {pnum} people')
            print(f'{awt:.2f} {att:.2f} {awt+att:.2f} {total_energy:.2f} for {pnum} people', file=file)
            total_res += awt+att
            total_energies += total_energy
        print(f'average time: {total_res/test_num:.2f} average energy: {total_energies/test_num:.2f}')
        print(f'average time: {total_res/test_num:.2f} average energy: {total_energies/test_num:.2f}', file=file)
        print(f'Reward list: {rs}', file=file)
        print(file=file)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(rs)
        plt.show()
        plt.close()
        file.close()




