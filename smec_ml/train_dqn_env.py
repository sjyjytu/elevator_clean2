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
import itertools
import pickle


class SmecEnv:
    def __init__(self, data_file='train_data/new/lunchpeak/LunchPeak1_elvx.csv', config_file=None, render=True,
                 seed=None, forbid_uncalled=False,
                 use_graph=True, real_data=True, use_advice=False, special_reward=False, data_dir=None,
                 file_begin_idx=None):
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__) + '/smec_liftsim/rl_config2.ini')
        file_name = config_file
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy.'
        # dos = '00:00-10:00'
        # dos = '00:00-20:00'
        dos = ''
        if dos == '':
            st = 0
        else:
            ts = dos.split('-')[0].split(':')
            st = int(ts[0]) * 60 + int(ts[1])

        person_generator = FixedDataGenerator(data_file=data_file, data_dir=data_dir, file_begin_idx=file_begin_idx,
                                                  data_of_section=dos)
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

    def get_time(self):
        raw_time = self._config.raw_time
        cur_day = raw_time // (24 * 3600)
        cur_time = raw_time % (24 * 3600)
        return [cur_day, int(cur_time // 3600 + 7), int(cur_time % 3600 // 60), int(cur_time % 60)]

    def step(self, action, dcar_call=True):
        next_call_come = False
        while not next_call_come and not self.is_end():
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action,
                                                                                                      use_rules=False,
                                                                                                      replace_hallcall=True)
            # self.mansion.generate_person(byhand=True)
            self.mansion.generate_person()
            # self.render()
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            action = None
            next_call_come = unallocated_up != [] or unallocated_dn != []
            # if dcar_call:
            #     next_call_come = next_call_come or self.mansion.elevators_car_call_change
        new_obs = self.state
        reward = 0
        done = self.mansion.is_done
        info = {}
        return new_obs, reward, done, info

    def get_unallocate(self):
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        if unallocated_dn or unallocated_up:
            add_hallcalls = unallocated_up
            for udn in unallocated_dn:
                add_hallcalls.append(udn + self.floor_num)
            return add_hallcalls
        else:
            return []

    def seed(self, seed=None):
        set_seed(seed)

    def reset(self):
        self.mansion.reset_env()
        # state = self.get_smec_state()
        if self.seed_c:
            self.seed_c += 100
            self.seed(self.seed_c)
        return self.state

    def is_end(self):
        return self.mansion.is_done

    def render(self, **kwargs):
        self.viewer.view()

    @property
    def state(self):
        state = []
        for i, elev in enumerate(self.mansion._elevators):
            cur_pos = elev._current_position
            cur_spd = elev._current_velocity
            srv_dir = elev._service_direction
            car_call = elev._car_call

            door_state = 0
            if elev._is_door_opening:
                door_state = 1
            elif elev.is_fully_open:
                door_state = 2
            elif elev._is_door_closing:
                door_state = 3

            elev_info = {}
            elev_info['pos'] = cur_pos
            elev_info['door_state'] = door_state
            elev_info['vol'] = cur_spd
            elev_info['dir'] = srv_dir
            elev_info['car_call'] = car_call
            state.append(elev_info)
        return state

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
    def __init__(self, env=None):
        self.modes = ['lunchpeak', 'notpeak', 'uppeak', 'dnpeak']
        self.mode = self.modes[0]
        self.mansion = None
        if env is not None:
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
        self.best_scheme = {'dispatch': [-1 for _ in range(self.floor_num * 2)], 'score': -10000}
        self.best_k_schemes = []
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

        with open('./clf.pickle', 'rb') as f:
            self.ml_model = pickle.load(f)

        self.k2all_dispatches = [[]]
        for k in range(1, 4):
            self.k2all_dispatches.append(self.get_k_elev_group(k))
        self.K_ELE = 2

    def get_k_elev_group(self, k):
        tmp_result = [[i] for i in range(self.elev_num)]
        for i in range(k - 1):
            result = []
            for tr in tmp_result:
                for j in range(self.elev_num):
                    result.append(tr + [j])
            tmp_result = result
        return tmp_result

    def bind_env(self, env):
        self.mansion = env.mansion

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
            key += 2 ** call
        for call in car_call:
            key += 2 ** (self.floor_num * 2 + call)
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
            weight += self.dncall_weights[self.weight_t, floor] * self.updn_delta_time[floor + self.floor_num] / 60
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

    # 根据电梯的初速度，已分配的carcall和hallcall，可以把电梯接下来的运动分为三段：
    # 如电梯现在正在5楼往上行，那么
    # r1: 5楼到16楼之间的up call，以及car call。
    # r2: 16楼到1楼之间的dn call。
    # r3: 1楼到5楼之间的up call。
    # 按常理出牌的话，car call只会存在于r1。
    # 这里只记录电梯要停靠的楼层的位置以及state（便于结合权重）。
    # 在上面那个例子中，加入up call为[1,14], dn call为[16,7], car call为[8]的话，route就为：
    # [8, 14, 16, 7, 1]，这样就可以算出df来计算运行时间了。
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

    def estimate_elev_route_loss_ml(self, elev, hallcall=None):
        pos = elev._current_position / 45
        vol = elev._current_velocity
        srv_dir = elev._service_direction
        car_call = [1 if i in elev._car_call else 0 for i in range(self.floor_num)]
        up_call = [1 if i in hallcall.HallUpCall else 0 for i in range(self.floor_num)]
        dn_call = [1 if i in hallcall.HallDnCall else 0 for i in range(self.floor_num)]
        door_state = 0
        if elev._is_door_opening:
            door_state = 1
        elif elev.is_fully_open:
            door_state = 2
        elif elev._is_door_closing:
            door_state = 3
        elev_info = [pos, srv_dir, vol, door_state] + car_call + up_call + dn_call
        loss = self.ml_model.predict([elev_info])[0]
        if loss < 0:
            loss = 0
        # print('loss: ', loss)
        return loss

    def evaluate_dispatch(self, dispatch):
        # print('dispatch: ', self.print_dispatch(dispatch))
        total_loss = 0
        hallcalls = self.dispatch2hallcalls(dispatch)
        hallcall_lists = self.dispatch2hallcall_lists(dispatch)
        for idx in range(self.elev_num):
            elev_dispatch_key = self.hallcall_list_carcall2key(hallcall_lists[idx],
                                                               self.mansion._elevators[idx]._car_call)
            if elev_dispatch_key in self.elevator_dispatch_time_table[idx].keys():
                loss = self.elevator_dispatch_time_table[idx][elev_dispatch_key]
            else:
                loss = self.estimate_elev_route_loss(self.mansion._elevators[idx], hallcalls[idx])
                # loss = self.estimate_elev_route_loss_ml(self.mansion._elevators[idx], hallcalls[idx])
                self.elevator_dispatch_time_table[idx][elev_dispatch_key] = loss
            # print(idx, hallcalls[idx], self.mansion._elevators[idx]._car_call, loss)
            total_loss += loss
        # print('total loss: ', total_loss)
        return 1 / (total_loss + 0.01)

    # 针对单一hallcall，贪心选取loss最小的电梯。
    def get_elev_greedy(self, hallcall, cur_dispatch, origin_elev=-1):
        best_score = 0
        best_idx = -1
        best_dispatch = []
        # print(f'greedy for {add_hallcall}: ')
        for elev_idx in range(self.elev_num):
            if elev_idx == origin_elev:
                continue
            new_dispatch = [i for i in cur_dispatch]
            new_dispatch[hallcall] = elev_idx
            score = self.evaluate_dispatch(new_dispatch)
            if score > best_score:
                best_score = score
                best_idx = elev_idx
                best_dispatch = new_dispatch
        return best_idx, best_score, best_dispatch

    def get_cur_dispatch(self):
        dispatch = [-1 for _ in range(self.floor_num * 2)]  # -1表示没有hallcall
        elevs = self.mansion._elevators
        for idx, elev in enumerate(elevs):
            for uc in elev._hall_up_call:
                dispatch[uc] = idx
            for dc in elev._hall_dn_call:
                dispatch[dc + self.floor_num] = idx
        return dispatch

    def dispatch2to_serve_calls(self, dispatch):
        to_serve_calls = []
        for i, d in enumerate(dispatch):
            if d != -1:
                to_serve_calls.append(i)
        return to_serve_calls

    def print_dispatch(self, dispatch):
        return [i for i in dispatch if i != -1]

    def change_k_element(self, dispatch, k):
        to_serve_calls = self.dispatch2to_serve_calls(dispatch)
        if len(to_serve_calls) < k:
            return []
        res = []
        all_changes = self.k2all_dispatches[k]  # 4^k的所有可能
        combinations = itertools.combinations(to_serve_calls, k)  # 选出要改变的k个位置
        for c in combinations:
            for change in all_changes:
                new_dispatch = copy.deepcopy(dispatch)
                for i in range(k):
                    new_dispatch[c[i]] = change[i]
                res.append(new_dispatch)
        return res

    # 固定变异0-3个位置，遍历搜索所有，以当前解的score为阈值，效果不好，搜得慢
    def exploit_k_schemes_v1(self, k):
        st = time.time()
        self.best_k_schemes = [self.best_scheme]
        threshhold = self.best_scheme['score']
        begin_dispatch = self.best_scheme['dispatch']
        alternatives = self.change_k_element(begin_dispatch, self.K_ELE)
        for alternative in alternatives:
            score = self.evaluate_dispatch(alternative)
            if score > threshhold:
                self.best_k_schemes.append({'dispatch': alternative, 'score': score})
        et = time.time()
        print(f'search for {len(alternatives)}, get {len(self.best_k_schemes)}, consuming {et - st:.2f}')
        if len(self.best_k_schemes) > k:
            self.best_k_schemes = sorted(self.best_k_schemes, key=lambda x: x['score'], reverse=True)[:k]

    # 以当前解的score为阈值，只向后变异，效果更差，搜得还算快（搜得少）。
    def exploit_k_schemes_v2(self, k):
        best_scheme = self.best_scheme
        best_scheme['depth'] = 0
        best_scheme['last_sc'] = -1
        better_schemes = [best_scheme]
        st = time.time()
        search_num, find_better_time, deepest = 0, 0, 0
        to_serve_calls = self.dispatch2to_serve_calls(best_scheme['dispatch'])
        threshold = best_scheme['score']
        search_idx = 0
        while search_idx < len(better_schemes):
            cur_scheme = better_schemes[search_idx]
            search_idx += 1
            cur_dispatch = cur_scheme['dispatch']
            cur_score = cur_scheme['score']
            next_depth = cur_scheme['depth'] + 1
            last_sc = cur_scheme['last_sc']  # 只往后变异，效果不好
            for sc in to_serve_calls:
                if sc <= last_sc:
                    continue
                sc_other_best_idx, sc_other_best_score, sc_other_best_dispatch = self.get_elev_greedy(sc, cur_dispatch,
                                                                                                      cur_dispatch[sc])
                search_num += 3
                if sc_other_best_score > threshold:
                # if sc_other_best_score > cur_score:
                    better_schemes.append(
                        {'dispatch': sc_other_best_dispatch, 'score': sc_other_best_score, 'depth': next_depth, 'last_sc': sc})
                if sc_other_best_score > best_scheme['score']:
                    find_better_time += 1
                    best_scheme = {'dispatch': sc_other_best_dispatch,
                                   'score': sc_other_best_score}
                    if next_depth > deepest:
                        deepest = next_depth
        self.best_k_schemes = better_schemes
        et = time.time()
        print(f'search for {search_num}, get {len(self.best_k_schemes)}, '
              f'find better: {find_better_time}, depth: {deepest}, consuming {et - st:.2f}')
        if len(self.best_k_schemes) > k:
            self.best_k_schemes = sorted(self.best_k_schemes, key=lambda x: x['score'], reverse=True)[:k]
        self.best_scheme = best_scheme

    def dispatch2key(self, dispatch):
        key = []
        for i in dispatch:
            if i != -1:
                key.append(str(i))
        key = ''.join(key)
        return key

    # 以当前解为阈值，全位置变异，找到更优时，加入一个key避免重复（甚至陷入死循环），当然还是会重复搜索大量已经搜过的，但还好？因为每个电梯有table记录算过的。
    def exploit_k_schemes_v3(self, k):
        best_scheme = self.best_scheme
        best_scheme['depth'] = 0
        better_schemes = [best_scheme]
        st = time.time()
        search_num, find_better_time, deepest = 0, 0, 0
        to_serve_calls = self.dispatch2to_serve_calls(best_scheme['dispatch'])
        threshold = best_scheme['score']
        search_idx = 0
        searched = [self.dispatch2key(best_scheme['dispatch'])]
        # if len(to_serve_calls) == 10:
        #     print('debug')
        while search_idx < len(better_schemes) and len(better_schemes) < 100:
            cur_scheme = better_schemes[search_idx]
            search_idx += 1
            cur_dispatch = cur_scheme['dispatch']
            cur_score = cur_scheme['score']
            next_depth = cur_scheme['depth'] + 1
            for sc in to_serve_calls:
                sc_other_best_idx, sc_other_best_score, sc_other_best_dispatch = self.get_elev_greedy(sc, cur_dispatch,
                                                                                                      cur_dispatch[sc])
                search_num += 3
                # if sc_other_best_score - threshold > 0.01:
                # if sc_other_best_score > cur_score:
                if sc_other_best_score > best_scheme['score']:
                    key = self.dispatch2key(sc_other_best_dispatch)
                    if key in searched:
                        continue
                    else:
                        searched.append(key)
                    better_schemes.append(
                        {'dispatch': sc_other_best_dispatch, 'score': sc_other_best_score, 'depth': next_depth})
                    if next_depth > deepest:
                        deepest = next_depth

                    if sc_other_best_score > best_scheme['score']:
                        find_better_time += 1
                        best_scheme = {'dispatch': sc_other_best_dispatch,
                                       'score': sc_other_best_score}

        self.best_k_schemes = better_schemes
        et = time.time()
        print(f'search for {search_num}, get {len(self.best_k_schemes)}, '
              f'find better: {find_better_time}, depth: {deepest}, consuming {et - st:.2f}')
        if len(self.best_k_schemes) > k:
            self.best_k_schemes = sorted(self.best_k_schemes, key=lambda x: x['score'], reverse=True)[:k]
        self.best_scheme = best_scheme

    def get_k_dispatches(self, add_hallcalls, k=100):
        cur_time = self.mansion._config.raw_time
        self.weight_t = int(cur_time // 60)
        # print(
        #     f'cur time: {self.weight_t}, up weights: {self.upcall_weights[self.weight_t]} dn weights: {self.dncall_weights[self.weight_t]}')
        self.elevator_dispatch_time_table = [{} for i in range(self.elev_num)]
        self.updn_delta_time = [cur_time - last_time for last_time in self.mansion.updn_last_serve_time]
        # 先针对当前新加的hallcalls贪心，这会改变mansion的东西，每次都实时传？没事，改就改吧？
        cur_dispatch = self.get_cur_dispatch()
        for ah in add_hallcalls:
            choose_elev_idx, score, cur_dispatch = self.get_elev_greedy(ah, cur_dispatch)
            self.add_hallcall_to_elev(ah, self.mansion._elevators[choose_elev_idx])

        # 把当前的贪心的分派记作最优方案
        cur_dispatch = self.get_cur_dispatch()
        to_serve = self.dispatch2to_serve_calls(cur_dispatch)
        print()
        print(f'cur time: {self.weight_t}, {len(to_serve)} to serve: {to_serve}')

        self.best_scheme['dispatch'] = cur_dispatch
        self.best_scheme['score'] = self.evaluate_dispatch(cur_dispatch)

        # 选择合适算法基于当前情况寻找更优解。
        self.exploit_k_schemes_v3(k)

        return [s['dispatch'] for s in self.best_k_schemes]


if __name__ == '__main__':
    from draw_data import print_hallcall_along_time

    dataset_file = open('dataset_noweight.txt', 'a')

    elev_env = SmecEnv(data_dir='../train_data/new/lunchpeak', render=False,
                       config_file='../smec_liftsim/rl_config2.ini')
    for file in range(1):
        elev_env.reset()
        # 重置权重参数
        print_hallcall_along_time(data_dir_prefix='../train_data/new/',
                                  fileid=elev_env.mansion.person_generator.file_idx)
        floor_num = elev_env.mansion._floor_number
        # print(elev_env.mansion.person_generator.file_idx, elev_env.mansion.person_generator.data.data)
        solver = LocalSearch(elev_env)
        while not elev_env.is_end():
            add_hallcalls = elev_env.get_unallocate()
            if add_hallcalls != []:
                dispatch = solver.get_k_dispatches(add_hallcalls, 1)[0]
                # dispatch = solver.get_k_dispatches(add_hallcalls, 1)
                action = solver.dispatch2hallcalls(dispatch)
            else:
                action = None
            print(f'execute action: {action}')
            elev_env.step(action)

        # print(elev_env.person_info)
        print(elev_env.get_reward())  # (0.35, 3.8)




