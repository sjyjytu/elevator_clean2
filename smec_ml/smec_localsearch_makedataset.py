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


class SmecEnv:
    def __init__(self, data_file='train_data/new/lunchpeak/LunchPeak1_elvx.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
    # def __init__(self, data_file='train_data/new/uppeak/UpPeakFlow1_elvx.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
    # def __init__(self, data_file='train_data/new/dnpeak/DnPeakFlow1_elvx.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
    # def __init__(self, data_file='train_data/new/notpeak/NotPeak1_elvx.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
                 use_graph=True, real_data=True, use_advice=False, special_reward=False, data_dir=None, file_begin_idx=None):
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
        dos = ''
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
            person_generator = FixedDataGenerator(data_file=data_file, data_dir=data_dir, file_begin_idx=file_begin_idx, data_of_section=dos)
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
        while not next_call_come and not self.is_end():
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=True)
            # self.mansion.generate_person(byhand=True)
            self.mansion.generate_person()
            # self.render()
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            action = None
            next_call_come = unallocated_up != [] or unallocated_dn != []
            # if dcar_call:
            #     next_call_come = next_call_come or self.mansion.elevators_car_call_change
        # new_obs = self.get_smec_state()
        new_obs = None
        reward = 0
        done = self.mansion.person_generator.done
        info = {}
        return new_obs, reward, done, info

    def step(self, action):
        return self.step_dp(action)

    def seed(self, seed=None):
        set_seed(seed)

    def reset(self):
        self.mansion.reset_env()
        self.cur_node_feature = self.empty_node_feature.clone()
        self.cur_adj_matrix = self.empty_adj_matrix.clone()
        # state = self.get_smec_state()
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
        elev_info['up_call'] = copy_elev._hall_up_call
        elev_info['dn_call'] = copy_elev._hall_dn_call


        # 如果电梯之前是空闲的，可能分配了hallcall之后srv_dir也是0没来得及更新，先运行一个dt给他更新一下。
        if srv_dir == 0:
            if hall_up_dn_call[0] + hall_up_dn_call[1] == []:
                return 0, elev_info
            else:
                copy_elev.run_elevator()
                srv_dir = copy_elev._service_direction
                elev_info['dir'] = srv_dir
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
        return loss, elev_info

    # 评估一种分配方案：
    # 求所有电梯loss的和。
    def evaluate_dispatch_slower(self, dispatch):
        total_loss = 0
        hallcalls = self.dispatch2hallcalls(dispatch)
        for idx in range(self.elev_num):
            loss = self.estimate_elev_route_loss(self.mansion._elevators[idx], hallcalls[idx])
            total_loss += loss
        return 1 / (total_loss + 0.01)

    def evaluate_dispatch_faster(self, dispatch):
        # print('dispatch: ', self.print_dispatch(dispatch))
        total_loss = 0
        hallcalls = self.dispatch2hallcalls(dispatch)
        hallcall_lists = self.dispatch2hallcall_lists(dispatch)
        dataset = []
        for idx in range(self.elev_num):
            elev_dispatch_key = self.hallcall_list_carcall2key(hallcall_lists[idx], self.mansion._elevators[idx]._car_call)
            if elev_dispatch_key in self.elevator_dispatch_time_table[idx].keys():
                loss, elev_info = self.elevator_dispatch_time_table[idx][elev_dispatch_key]
            else:
                loss, elev_info = self.estimate_elev_route_loss(self.mansion._elevators[idx], hallcalls[idx])
                self.elevator_dispatch_time_table[idx][elev_dispatch_key] = loss, elev_info
            # print(idx, hallcalls[idx], self.mansion._elevators[idx]._car_call, loss)
            # 构造数据
            elev_info['loss'] = loss
            print(elev_info, file=dataset_file)
            total_loss += loss
        print(file=dataset_file)
        # print('total loss: ', total_loss)
        return 1 / (total_loss + 0.01)

    def evaluate_dispatch(self, dispatch):
        return self.evaluate_dispatch_faster(dispatch)

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
            # print(elev_idx, score)
            # 如果超载，则不考虑
            # if not self.elevator_mask[elev_idx]:
            #     continue
            # print(elev_idx, score)
            if score > best_score:
                best_score = score
                best_idx = elev_idx
                best_dispatch = new_dispatch
        return best_idx, best_score, best_dispatch

    def get_cur_dispatch(self):
        to_serve_calls = []
        dispatch = [-1 for _ in range(self.floor_num*2)]  # -1表示没有hallcall
        elevs = self.mansion._elevators
        for idx, elev in enumerate(elevs):
            for uc in elev._hall_up_call:
                dispatch[uc] = idx
                to_serve_calls.append(uc)
            for dc in elev._hall_dn_call:
                dispatch[dc + self.floor_num] = idx
                to_serve_calls.append(dc + self.floor_num)
        return dispatch, to_serve_calls

    def eploit_exhaust(self, to_serve_calls):
        solution_space = self.elev_num ** len(to_serve_calls)
        for i in range(solution_space):
            dispatch = [-1 for _ in range(self.floor_num * 2)]
            for j in range(len(to_serve_calls)):
                e = i % self.elev_num
                i = i // self.elev_num
                dispatch[to_serve_calls[j]] = e
            score = self.evaluate_dispatch(dispatch)
            if score > self.best_scheme['score']:
                self.best_scheme = {'dispatch': dispatch, 'score': score}
                print(f'Find better dispatch: {dispatch}, score: {score}')

    def print_dispatch(self, dispatch):
        return [i for i in dispatch if i != -1]

    def exploit_nearest(self, to_serve_calls):
        self.elevator_dispatch_time = [{} for i in range(self.elev_num)]
        better_schemes = [self.best_scheme]
        best_scheme = {'dispatch': self.best_scheme['dispatch'], 'score': self.best_scheme['score']}
        st = time.time()
        search_num = 0
        scheme_num = 0
        while len(better_schemes) > 0:
            scheme_num += 1
            cur_scheme = better_schemes.pop(0)
            cur_dispatch = cur_scheme['dispatch']
            cur_score = cur_scheme['score']
            print(f'olddis: {self.print_dispatch(cur_dispatch)} score: {cur_score}')
            print(f'to serve in exploit: {to_serve_calls}')
            for sc in to_serve_calls:
                # 找出替换cur_dispatch[sc]的最小loss的电梯，用这个方法会比之前少很多search
                sc_other_best_idx, sc_other_best_score, sc_other_best_dispatch = self.get_elev_greedy(sc, cur_dispatch, cur_dispatch[sc])
                search_num += 3
                if sc_other_best_score > best_scheme['score']:
                    better_schemes.append({'dispatch': sc_other_best_dispatch, 'score': sc_other_best_score})
                    best_scheme = {'dispatch': sc_other_best_dispatch,
                                   'score': sc_other_best_score}
                    print(f'better: {self.print_dispatch(sc_other_best_dispatch)}, score: {sc_other_best_score}')
                # for eidx in range(self.elev_num):
                #     if eidx != cur_dispatch[sc]:
                #         search_num += 1
                #         new_dispatch = deepcopy(cur_dispatch)
                #         new_dispatch[sc] = eidx
                #         new_score = self.evaluate_dispatch(new_dispatch)
                #         # print(new_score)
                #         # if new_score > cur_score:
                #         if new_score > best_scheme['score']:
                #             better_schemes.append({'dispatch': new_dispatch, 'score': new_score})
                #             print(f'better: {self.print_dispatch(new_dispatch)}, score: {new_score}')
                #             best_scheme = {'dispatch': new_dispatch,
                #                            'score': new_score}
                        # if new_score
        et = time.time()
        print(f'to serve: {len(to_serve_calls)}, search scheme: {scheme_num}, {search_num} times, {et-st:.2f}')
        self.best_scheme = best_scheme

    def exploit(self, to_serve_calls):
        # solution_space = self.elev_num ** len(to_serve_calls)
        # if solution_space < self.SELECT_SIZE:
        #     self.eploit_exhaust(to_serve_calls)
        # else:
        #     self.exploit_gene()
        self.exploit_nearest(to_serve_calls)

    # 返回的是一个所有未完成hallcall的一个分配，可能会改变颠覆之前的分配。
    def get_action(self, add_hallcalls):
        cur_time = self.mansion._config.raw_time
        self.weight_t = int(cur_time // 60)
        print(f'cur time: {self.weight_t}, up weights: {self.upcall_weights[self.weight_t]} dn weights: {self.dncall_weights[self.weight_t]}')
        self.elevator_dispatch_time_table = [{} for i in range(self.elev_num)]
        self.updn_delta_time = [cur_time - last_time for last_time in self.mansion.updn_last_serve_time]
        # 先针对当前新加的hallcalls贪心，这会改变mansion的东西，每次都实时传？没事，改就改吧？
        cur_dispatch, to_serve_calls = self.get_cur_dispatch()
        for ah in add_hallcalls:
            choose_elev_idx, score, cur_dispatch = self.get_elev_greedy(ah, cur_dispatch)
            self.add_hallcall_to_elev(ah, self.mansion._elevators[choose_elev_idx])

        # 把当前的贪心的分派记作最优方案
        cur_dispatch, to_serve_calls = self.get_cur_dispatch()
        self.best_scheme['dispatch'] = cur_dispatch
        self.best_scheme['score'] = self.evaluate_dispatch(cur_dispatch)

        # 选择合适算法基于当前情况寻找更优解。
        self.exploit(to_serve_calls)

        return self.best_scheme['dispatch']


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
    from draw_data import print_hallcall_along_time
    is_eval = False
    if is_eval:
        all_action = evaluate()
        aidx = 0

    dataset_file = open('dataset_noweight.txt', 'a')

    elev_env = SmecEnv(data_dir='../train_data/new/lunchpeak', render=False, config_file='../smec_liftsim/rl_config2.ini')
    for file in range(1):
        elev_env.reset()
        # 重置权重参数
        print_hallcall_along_time(data_dir_prefix='../train_data/new/', fileid=elev_env.mansion.person_generator.file_idx)
        floor_num = elev_env.mansion._floor_number
        print(elev_env.mansion.person_generator.file_idx, elev_env.mansion.person_generator.data.data)
        solver = LocalSearch(elev_env)
        while not elev_env.is_end():
            # 问题是现在action是多维的，按理来说同一dt内也不会有多个楼层需要分配，但是可以重分配的话，就有了。所以这还是个组合优化的问题？但也可以先不考虑重分配，假设每次都是当前最优，就是贪心。
            unallocated_up, unallocated_dn = elev_env.mansion.get_unallocated_floors()
            if not is_eval:
                # print(elev_env.mansion.elevators_car_call_change, '+'*20, 'car call re dispatch...')

                if unallocated_dn or unallocated_up:
                # if unallocated_dn or unallocated_up or elev_env.mansion.elevators_car_call_change:
                #     if elev_env.mansion.elevators_car_call_change:
                #         elev_env.mansion.elevators_car_call_change = False
                #         if not (unallocated_dn or unallocated_up):
                #             print('+'*20, 'car call re dispatch...')
                    add_hallcalls = unallocated_up
                    for udn in unallocated_dn:
                        add_hallcalls.append(udn + floor_num)

                    dispatch = solver.get_action(add_hallcalls)
                    action = solver.dispatch2hallcalls(dispatch)
                else:
                    action = None
            else:
                action = all_action[aidx]
                aidx += 1
            print(f'execute action: {action}')
            elev_env.step(action)

        # print(elev_env.person_info)
        print(elev_env.get_reward())  # (0.35, 3.8)




