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
        dos = '10:00-20:00'
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

    def step_dp(self, action):
        next_call_come = False
        while not next_call_come and not self.is_end():
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action, use_rules=False, replace_hallcall=True)
            self.mansion.generate_person()
            self.render()
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            action = None
            next_call_come = unallocated_up != [] or unallocated_dn != []
        new_obs = self.get_smec_state()
        reward = 0
        done = self.mansion.person_generator.done
        info = {}
        return new_obs, reward, done, info

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

        self.upcall_weights = np.load('weights_lunchpeak_upcall.npy')
        self.dncall_weights = np.load('weights_lunchpeak_dncall.npy')
        self.carcall_weights = np.load('weights_lunchpeak_carcall.npy')
        self.weight_t = 0

    # 根据delata floor计算需要时间。一个更真实的运动估计。
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

    def add_hallcall_to_elev(self, hallcall, elev):
        floor_num = elev._number_of_floors
        if hallcall >= floor_num:
            if hallcall - floor_num not in elev._hall_dn_call:
                elev._hall_dn_call.append(hallcall - floor_num)
        else:
            if hallcall not in elev._hall_up_call:
                elev._hall_up_call.append(hallcall)

    def simulate_elev_until_stop(self, elev, noNew=False):
        elev.arrive_info = []
        self.time_cnt.reset()
        while not elev.is_idle_stop:
            elev.run_elevator(time_cnt=self.time_cnt)
            if noNew:
                elev._car_call = []
        total_time = 0
        for finish_task in elev.arrive_info:
            # TODO: 加权和
            task = finish_task[0][0]  # c, u, d
            floor = int(finish_task[0][1:])
            consume_time = finish_task[1]
            total_time += consume_time
            # task2state = {'c': 1, 'u': 2, 'd': 4}
            # total_time += consume_time * self.floor_state2weight(floor, task2state[task])
        return total_time

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

    def cal_first_stop_time(self, elev, target_pos):
        cur_spd = elev._current_velocity
        df = abs(elev._current_position - target_pos) / elev._floor_height
        df = int(round(df))
        consume_time = max(self.df2time(df, elev) - cur_spd, 0)  # 观察经验公式
        return consume_time

    # 根据人流模式，预测出发层的目的地
    def upcall_to_car_dst(self, floor):
        return 15

    def dncall_to_car_dst(self, floor):
        return 0

    def floor_state2weight(self, floor, state):
        # 1 car 2 up 4 dn
        weight = 0
        if state % 2 == 1:
            weight += self.carcall_weights[self.weight_t, floor]
        if state // 2 % 2 == 1:
            weight += self.upcall_weights[self.weight_t, floor]
        if state // 4 % 2 == 1:
            weight += self.dncall_weights[self.weight_t, floor]
        return weight

    # 根据电梯运动情况和分配情况，计算电梯的运行时间？乘客的wt和tt？
    # 建议对每个节点使用累积时间并分别记录，然后乘以权重再相加。
    # hallcall怎么处理？而且一个hallcall很可能产生多个carcall。
    # TODO：暂时不管hallcall产生的carcall，只是执行hallcall和已有carcall。
    # 这样算是不对的，目标变成了尽快让所有电梯停下来，有点间接，没有考虑到所有人，应该还是算人的wt或tt，
    # 会导致的结果就是都分给同一台电梯，这样就不用别的电梯来跑了。
    # 应该算每个hallcall和carcall的完成时间。
    # 然后乘以wt_floor_weigths/tt_floor_weigths，再求和，这个weights是表示对该楼层上客人数和下客人数，因为不能预知所以只能预测。
    # 同时这个权重还可以是一个随时间变化的值。这样通用性（一个模型用在所有时间就有保证了。）
    # TODO: 这个算半真实值了，半在hallcall没处理，可以加入hallcall的loss，使得hallcall和carcall尽量重合。
    # 有个问题就是正在进来还没按电梯的人要去的楼层也会被算进来，其实应该是不知道的。
    def estimate_elev_route_loss_v1(self, elev, hallcall=None, need_copy=True):
        if need_copy:
            copy_elev = deepcopy(elev)
        else:
            copy_elev = elev
        if hallcall:
            copy_elev.replace_hall_call(hallcall)
        return self.simulate_elev_until_stop(copy_elev)

    # 用距离近似
    def estimate_elev_route_loss_v2(self, elev, hallcall=None, need_copy=True):
        if need_copy:
            copy_elev = deepcopy(elev)
        else:
            copy_elev = elev
        if hallcall:
            copy_elev.replace_hall_call(hallcall)
        cur_flr = copy_elev._sync_floor
        cur_pos = copy_elev._current_position
        cur_spd = copy_elev._current_velocity
        srv_dir = copy_elev._service_direction
        car_call = copy_elev._car_call
        hall_up_dn_call = [copy_elev._hall_up_call, copy_elev._hall_dn_call]
        stp_flr = self.cal_stop_floor(cur_pos, cur_spd, 0.557, 3.0)

        if srv_dir == 0:
            if hall_up_dn_call[0] + hall_up_dn_call[1] == []:
                return 0
            else:
                copy_elev.run_elevator()
                srv_dir = copy_elev._service_direction
                # print(hall_up_dn_call, cur_flr, cur_spd, srv_dir)

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
                if route == [] or (route[0][0]*copy_elev._floor_height - cur_pos) * srv_dir < 0:
                    route.insert(0, (stp_flr, 8))

        # 从cur_flr以cur_spd开始完成旅程route。
        # 不能从cur_flr开始，要从cur_pos开始，特殊处理第一段
        loss = 0
        accumulate_time = 0
        # 加入floor_weights, TODO: carcall可能应该用前一时间片的权重呢。
        # route肯定不为空
        assert len(route) > 0
        first_stop_flr = route[0][0]
        first_stop_pos = first_stop_flr * copy_elev._floor_height
        # TODO：精确点的话还要根据state决定第一次要不要开门。
        # 第一段用模拟，其他的用计算。
        # 还是不要用模拟。
        consume_time = self.cal_first_stop_time(copy_elev, first_stop_pos)
        accumulate_time += consume_time
        loss += accumulate_time * self.floor_state2weight(first_stop_flr, route[0][1])

        last_flr = first_stop_flr
        for stop_flr in route[1:]:
            df = abs(stop_flr[0] - last_flr)
            consume_time = self.df2time(df, copy_elev)
            accumulate_time += consume_time
            loss += accumulate_time * self.floor_state2weight(stop_flr[0], stop_flr[1])
            last_flr = stop_flr[0]
        # maximum_weight = copy_elev._maximum_capacity
        # load_weight = copy_elev._load_weight
        # loss += load_weight / (maximum_weight * 0.8) * 100
        # print(route, loss)
        return loss

    def estimate_elev_route_loss(self, elev, hallcall=None, need_copy=True, debugidx=None):
        # v = self.estimate_elev_route_loss_v1(elev, hallcall)
        v = self.estimate_elev_route_loss_v2(elev, hallcall)
        # if abs(v1 - v2) > 10:
        #     print(v1, v2)
            # v1 = self.estimate_elev_route_loss_v1(elev, hallcall)
            # v2 = self.estimate_elev_route_loss_v2(elev, hallcall)
        return v

    # 评估一种分配方案：
    # 可以用estimate所有电梯相加，也可以run完，run完又分为真实的和不管或者预测hallcall对应的carcall的，注意copy。
    def evaluate_dispatch(self, dispatch):
        loss = 0
        hallcalls = self.dispatch2hallcalls(dispatch)
        for idx in range(self.elev_num):
            loss += self.estimate_elev_route_loss(self.mansion._elevators[idx], hallcalls[idx], debugidx=idx)
        return 1 / (loss + 0.01)

    # TODO：之后所有改变都在这个loss里了，还要考虑电梯满载的问题，floor_weights还没用上
    def get_add_floor_loss(self, hallcall):
        floor_weights = [1 for _ in range(self.floor_num)]  # 每层楼的权重，根据人流模式来调整，如，用平均等待人数来设置
        floor_weights[0] = 10
        # 加一个人后对电梯运行时间的影响，这里运行时间考虑了正在执行的hallcall（对应的优化对象为wt）以及carcall（对应的优化对象为tt）
        # 考虑了新增这个人自己的时间以及对其他人的影响，算是考虑周到了。
        losses = []
        for idx, elev in enumerate(self.mansion._elevators):
            copy_elev = deepcopy(elev)
            # 处理超载
            if copy_elev._is_overloaded:
                loss = 10000
            else:
                origin_loss = self.estimate_elev_route_loss(copy_elev, debugidx=idx)
                # 把这个人添加到电梯
                self.add_hallcall_to_elev(hallcall, copy_elev)
                add_loss = self.estimate_elev_route_loss(copy_elev, need_copy=False, debugidx=idx)
                loss = add_loss - origin_loss
            losses.append((idx, loss))
        losses = sorted(losses, key=lambda x: x[1])
        return losses

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

    def mutate(self, dispatch, mutate_time=1):
        # 先随机变异，但是效率肯定很低吧，先试试，别管效果，先跑起来再优化
        avail_pos = []
        new_dispatch = deepcopy(dispatch)
        for i in range(len(new_dispatch)):
            if new_dispatch[i] != -1:
                avail_pos.append(i)
        for mt in range(mutate_time):
            pos = random.choice(avail_pos)  # chose a position in crossoff to perform mutation.
            new_elev = random.randint(0, self.elev_num-1)
            new_dispatch[pos] = new_elev
        return new_dispatch

    def crossoperate(self, schemes):
        """
        cross operation
        here we use two points crossoperate
        for example: gene1: [5, 2, 4, 7], gene2: [3, 6, 9, 2], if pos1=1, pos2=2
        5 | 2 | 4  7
        3 | 6 | 9  2
        =
        3 | 2 | 9  2
        5 | 6 | 4  7
        """

        dispatch1 = schemes[0]['dispatch']  # Gene's data of first offspring chosen from the selected pop
        dispatch2 = schemes[1]['dispatch']  # Gene's data of second offspring chosen from the selected pop

        dim = len(dispatch1)
        pos1 = random.randint(0, dim-1)  # select a position in the range from 0 to dim-1,
        pos2 = random.randint(0, dim-1)

        new_dispatch1 = []
        new_dispatch2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i <= max(pos1, pos2):
                new_dispatch1.append(dispatch2[i])
                new_dispatch2.append(dispatch1[i])
            else:
                new_dispatch1.append(dispatch1[i])
                new_dispatch2.append(dispatch2[i])

        return new_dispatch1, new_dispatch2

    # 保留好的，淘汰不好的
    def eliminate(self):
        self.candidate_schemes = sorted(self.candidate_schemes, key=itemgetter("score"), reverse=True)[:self.ELIMINATE_SIZE//4]

    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("score"), reverse=True)  # from large to small, return a pop
        return s_inds[0]

    def selection(self, individuals, k):
        """
        select some good individuals from pop, note that good individuals have greater probability to be choosen
        for example: a fitness list like that:[5, 4, 3, 2, 1], sum is 15,
        [-----|----|---|--|-]
        012345|6789|101112|1314|15
        we randomly choose a value in [0, 15],
        it belongs to first scale with greatest probability
        """
        s_inds = sorted(individuals, key=itemgetter("score"),
                        reverse=True)  # sort the pop by the reference of fitness
        sum_fits = sum(ind['score'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits], as threshold
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['score']  # sum up the fitness
                if sum_ >= u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of
                    # [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # from small to large, due to list.pop() method get the last element
        chosen = sorted(chosen, key=itemgetter("score"), reverse=False)
        return chosen

    def init_candidate_scheme(self):
        # 暂定方案：在best的基础上变异
        self.candidate_schemes = [self.best_scheme]
        while len(self.candidate_schemes) < self.SELECT_SIZE:
            chosen_scheme = random.choice(self.candidate_schemes)
            mutate_time = random.randint(1, self.MAX_MUTATE_TIME)
            new_dispatch = self.mutate(chosen_scheme['dispatch'], mutate_time=mutate_time)
            self.candidate_schemes.append({'dispatch': new_dispatch, 'score': self.evaluate_dispatch(new_dispatch)})

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

    def exploit_gene(self):
        # 找出电梯中未执行的hallcall，（剪枝的话，去掉一些明显会变差的交换），
        # 然后尝试交换，这个交换其实是不是可以直接改为重新分配，因为是每个分配只对应一个电梯
        # 然后可以加入随机扰动，随机抽一个随机给

        # 有个问题就是因为不连续，后代很有可能会不如前代，所以应该记录那些好的，其实就有点类似广度优先搜索，只不过随机了一点。
        # print("Start of evolution")

        # Begin the evolution
        # 将会执行 SELECT_SIZE * NGEN次，所以解空间小于这个的都可以直接遍历
        # TODO: 如果一段范围内收敛，可以结束。

        # 以当前最优解为起始发散，初始化candidate_scheme，至少填充至SELECT_SIZE那么大
        self.init_candidate_scheme()
        for g in range(self.NGEN):

            # print("############### Generation {} ###############".format(g))

            # Apply selection based on their converted fitness
            select_schemes = self.selection(self.candidate_schemes, self.SELECT_SIZE)

            new_schemes = []
            while len(select_schemes) > 1:
                # Apply crossover and mutation on the offspring

                # Select two individuals
                schemes = [select_schemes.pop() for _ in range(2)]

                if random.random() < self.CROSS_PROB:  # cross two individuals with probability CXPB
                    new_dispatch1, new_dispatch2 = self.crossoperate(schemes)
                    if random.random() < self.MUTATE_PROB:  # mutate an individual with probability MUTPB
                        new_dispatch1 = self.mutate(new_dispatch1)
                        new_dispatch2 = self.mutate(new_dispatch2)
                    elif schemes[0] == schemes[1]:  # don't add to new if the two selected schemes are the same.
                        continue
                    new_dispatch1_score = self.evaluate_dispatch(new_dispatch1)  # Evaluate the individuals
                    new_dispatch2_score = self.evaluate_dispatch(new_dispatch2)  # Evaluate the individuals
                    new_schemes.append({'dispatch': new_dispatch1, 'score': new_dispatch1_score})
                    new_schemes.append({'dispatch': new_dispatch2, 'score': new_dispatch2_score})
                # else:
                #     new_schemes.extend(schemes)

            # # The population is entirely replaced by the offspring
            # self.pop = nextoff

            # 合并旧的和新的，保留优秀的个体。
            # 但是这样会导致一直是局部优秀的个体之间繁殖，而没有机会给那些暂时不好的个体。可以把淘汰值设大一点
            self.candidate_schemes += new_schemes
            if len(self.candidate_schemes) >= self.ELIMINATE_SIZE:
                self.eliminate()
                best_scheme = self.candidate_schemes[0]
                if best_scheme['score'] > self.best_scheme['score']:
                    self.best_scheme = best_scheme
                    print(f'find better scheme: {self.best_scheme}')
                # print("Best individual found is {}, {}".format(self.best_scheme['dispatch'],
                #                                                self.best_scheme['score']))
        print("------ End of (successful) evolution ------")

    def exploit_nearest(self, to_serve_calls):
        better_schemes = [self.best_scheme]
        best_scheme = {'dispatch': self.best_scheme['dispatch'], 'score': self.best_scheme['score']}
        while len(better_schemes) > 0:
            cur_scheme = better_schemes.pop(0)
            cur_dispatch = cur_scheme['dispatch']
            cur_score = cur_scheme['score']
            for sc in to_serve_calls:
                for eidx in range(self.elev_num):
                    if eidx != cur_dispatch[sc]:
                        new_dispatch = deepcopy(cur_dispatch)
                        new_dispatch[sc] = eidx
                        new_score = self.evaluate_dispatch(new_dispatch)
                        # if new_score > cur_score:
                        if new_score > best_scheme['score']:
                            better_schemes.append({'dispatch': new_dispatch, 'score': new_score})
                            # if new_score > best_scheme['score']:
                            print('find better scheme...')
                            best_scheme = {'dispatch': new_dispatch,
                                           'score': new_score}
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
        self.weight_t = int(self.mansion._config.raw_time // 60)
        # 先针对当前新加的hallcalls贪心，这会改变mansion的东西，每次都实时传？没事，改就改吧？
        for ah in add_hallcalls:
            choose_elev_idx = self.get_add_floor_loss(ah)[0][0]
            self.add_hallcall_to_elev(ah, self.mansion._elevators[choose_elev_idx])

        # 把当前的最小loss的分派记作最优方案
        cur_dispatch, to_serve_calls = self.get_cur_dispatch()
        self.best_scheme['dispatch'] = cur_dispatch
        self.best_scheme['score'] = self.evaluate_dispatch(cur_dispatch)

        # 选择合适算法基于当前情况寻找更优解。
        self.exploit(to_serve_calls)

        return self.best_scheme['dispatch']


def evaluate():
    import re
    file = './evaluate_log/1216_lunch_v2nearest.log'
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
    is_eval = False
    if is_eval:
        all_action = evaluate()
        aidx = 0

    elev_env = SmecEnv()
    elev_env.reset()
    floor_num = elev_env.mansion._floor_number
    print(elev_env.mansion.person_generator.data.data)
    solver = LocalSearch(elev_env)
    while not elev_env.is_end():

        # 问题是现在action是多维的，按理来说同一dt内也不会有多个楼层需要分配，但是可以重分配的话，就有了。所以这还是个组合优化的问题？但也可以先不考虑重分配，假设每次都是当前最优，就是贪心。
        unallocated_up, unallocated_dn = elev_env.mansion.get_unallocated_floors()
        if not is_eval:
            if unallocated_dn or unallocated_up:
                add_hallcalls = unallocated_up
                for udn in unallocated_dn:
                    add_hallcalls.append(udn + floor_num)

                # action = solver.get_action(elev_env)

                # # 最短
                # floor2elevators = elev_env.get_floor2elevator_dis('cpu').cpu().numpy()
                # floor2elevators = np.argmin(floor2elevators, axis=1)
                # action = floor2elevators[0]
                # print(f'Choosing {action} for up:{unallocated_up}, dn:{unallocated_dn}, time:{elev_env._config.raw_time}, cur reward:{elev_env.get_reward()}')
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




