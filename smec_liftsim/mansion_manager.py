#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from collections import deque
from smec_liftsim.utils import *
# from smec_liftsim.smec_elevator import SmecElevator
from smec_liftsim.smec_elevator_new import SmecElevator
from smec_liftsim.utils import MansionAttribute, MansionState
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.generator_proxy import PersonGenerator


RUN_ELEVATOR_WITH_BUG = False


class MansionManager(object):
    """
    Mansion Class
    Mansion Randomly Generates Person that requiring elevators for a lift
    """

    def __init__(self, elevator_number, person_generator, mansion_config, name="Mansion"):
        """
        Initializing the Building
        Args:
          elevator_number: number of elevator in the building
          person_generator: PersonGenerator class that generates stochastic pattern of person flow
        Returns:
          None
        """
        # print(type(mansion_config))
        # assert isinstance(mansion_config, MansionConfig)
        self._name = name
        self._config = mansion_config
        self._floor_number = self._config.number_of_floors
        self._floor_height = self._config.floor_height
        self._dt = self._config.delta_t
        self._elevator_number = elevator_number
        self._person_generator = person_generator

        # if people are waiting for more than 300 seconds, he would give up!
        self._given_up_time_limit = None  # 300

        # used for statistics
        self._statistic_interval = int(600 / self._dt)
        self._delivered_person = deque()
        self._generated_person = deque()
        self._abandoned_person = deque()
        self._cumulative_waiting_time = deque()
        self._cumulative_energy_consumption = deque()

        self._elevators, self._button, self._wait_upward_persons_queue, self._wait_downward_persons_queue = \
            [], [], [], []
        self.reset_env()
        self.evaluate_info = {'reallocate_up': 0, 'reallocate_dn': 0}
        self.person_info = {}
        self.finish_person_num = 0
        self.self_trip = 0

        self.lock = [False for i in range(self._elevator_number)]
        self.lock_open_time = [0 for i in range(self._elevator_number)]

        self.not_in_hallcall_but_serving_by = [-1 for i in range(self._floor_number * 2)]
        self.elevators_car_call_change = False
        # 用于估计积累了多少人
        self.updn_last_serve_time = [self._config.raw_time for i in range(self._floor_number * 2)]
        # self.car_last_finish_time = [self._config.raw_time for i in self._floor_number]

    def reset_env(self):
        self._elevators = []
        for i in range(self._elevator_number):
            self._elevators.append(
                SmecElevator(start_position=0.0, mansion_config=self._config, name="%s_E%d" % (self._name, i + 1), _mansion=self, elev_index=i))

        self._config.reset()
        if self._person_generator:
            self._person_generator.link_mansion(self._config)
            self._person_generator.reset()

        self.lock = [False for i in range(self._elevator_number)]
        self.lock_open_time = [0 for i in range(self._elevator_number)]

        self.evaluate_info = {'reallocate_up': 0, 'reallocate_dn': 0}
        self.person_info = {}
        self.finish_person_num = 0
        self.self_trip = 0

        # whether the go up/down button is clicked
        self._button = [[False, False] for i in range(self._floor_number)]
        self._wait_upward_persons_queue = [deque() for i in range(self._floor_number)]
        self._wait_downward_persons_queue = [deque() for i in range(self._floor_number)]

    def get_rl_state(self, encode=True):
        pass_norm_number = 15
        upload_wait_nums = [len(ele) for ele in self._wait_upward_persons_queue]
        download_wait_nums = [len(ele) for ele in self._wait_downward_persons_queue]
        if encode:
            upload_wait_nums = [ele / pass_norm_number for ele in upload_wait_nums]
            download_wait_nums = [ele / pass_norm_number for ele in download_wait_nums]

        elevator_loading_maps = [[] for _ in self._elevators]
        for idx, elev in enumerate(self._elevators):
            cur_loading_person = elev.per_floor_loaded_people_num
            if encode:
                cur_loading_person = [ele / pass_norm_number for ele in cur_loading_person]
            elevator_loading_maps[idx] = cur_loading_person

        elevator_location_maps = [elv.cur_floor / self._floor_number for elv in self._elevators]
        elevator_up_calls = [elev._hall_up_call for elev in self._elevators]
        elevator_down_calls = [elev._hall_dn_call for elev in self._elevators]
        elv_load_up = [0 for _ in self._elevators]
        elv_load_down = [0 for _ in self._elevators]
        for idx, elev in enumerate(self._elevators):
            if elev._service_direction == 1:
                elv_load_up[idx] = sum(elevator_loading_maps[idx])
            else:
                elv_load_down[idx] = sum(elevator_loading_maps[idx])
        return upload_wait_nums, download_wait_nums, elevator_loading_maps, elevator_location_maps, \
               elevator_up_calls, elevator_down_calls, elv_load_up, elv_load_down

    def is_overload(self, elev, src, dst):
        loaded_person = elev._loaded_person
        cur_load_num = elev._load_weight // 75
        for i in range(max(0,src), min(dst+1, self._floor_number)):
            cur_load_num -= len(loaded_person[i])
        if cur_load_num < 10:
            return False
        return True

    def get_convenience_elevators(self, up_or_down, floor_id, include_uncalled=True):  # implemented by Zelin
        convenient_elevators = []
        if up_or_down:  # moving up
            for idx, elev in enumerate(self._elevators):
                if elev._service_direction == 1 and elev._sync_floor < floor_id:
                    # check if over load
                    if not self.is_overload(elev, elev.cal_cur_next_floor(), floor_id):
                        convenient_elevators.append(idx)
        else:
            for idx, elev in enumerate(self._elevators):
                if elev._service_direction == -1 and elev._sync_floor > floor_id:
                    # check if over load
                    if not self.is_overload(elev, floor_id, elev.cal_cur_next_floor()):
                        convenient_elevators.append(idx)
        if include_uncalled:
            uncalled = self.get_uncalled_elevators()
            for elev in uncalled:
                if elev not in convenient_elevators:
                    convenient_elevators.append(elev)
        return convenient_elevators

    def get_uncalled_elevators(self):  # implemented by Zelin
        not_called_elevators = []
        for i in range(len(self._elevators)):
            elv = self._elevators[i]
            all_car_call_len = len(elv._hall_up_call) + len(elv._car_call) + len(elv._hall_dn_call)
            if all_car_call_len == 0:
                not_called_elevators.append(i)
        return not_called_elevators

    # def get_unallocated_floors(self):  # implemented by Zelin
    #     up_called_floors = []
    #     for idx, floor_up in enumerate(self._wait_upward_persons_queue):
    #         if len(floor_up) > 0:
    #             up_called_floors.append(idx)
    #
    #     for idx, elev in enumerate(self._elevators):  # remove allocated
    #         cur_up_call = elev._hall_up_call
    #         for allocated_up in cur_up_call:
    #             if allocated_up in up_called_floors:
    #                 up_called_floors.remove(allocated_up)
    #
    #     dn_called_floors = []
    #     for idx, floor_dn in enumerate(self._wait_downward_persons_queue):
    #         if len(floor_dn) > 0:
    #             dn_called_floors.append(idx)
    #
    #     for idx, elev in enumerate(self._elevators):  # remove allocated
    #         cur_dn_call = elev._hall_dn_call
    #         for allocated_dn in cur_dn_call:
    #             if allocated_dn in dn_called_floors:
    #                 dn_called_floors.remove(allocated_dn)
    #     return up_called_floors, dn_called_floors

    # def get_load_weight_mask(self):
    #     mask = [[1 for j in range(self._floor_number * 2)] for i in range(self._elevator_number)]
    #     for i in range(self._elevator_number):
    #         srv_dir = self._elevators[i]._service_direction
    #         loaded_person = self._elevators[i]._loaded_person
    #         if self._elevators[i]._ser

    def get_unallocated_floors_v1(self):  # implemented by Jy
        allow_reallocation = False
        # if the floor is allocated to a car but the car is still not on the way, we allow to reallocate it.
        up_called_floors = []
        for idx, floor_up in enumerate(self._wait_upward_persons_queue):
            # 给人赋予意志，如果电梯装不下了，应该等当前满载电梯走了之后再按电梯
            if len(floor_up) > 0:
            # if len(floor_up) > 0 and self.not_in_hallcall_but_serving_by[idx] == -1:  # added by JY
                up_called_floors.append(idx)

        for idx, elev in enumerate(self._elevators):  # remove allocated
            cur_up_call = elev._hall_up_call
            for allocated_up in cur_up_call:
                is_serving = not (elev._service_direction == 1 and elev._sync_floor > allocated_up)
                if allocated_up in up_called_floors:
                    if is_serving or not allow_reallocation:
                        up_called_floors.remove(allocated_up)
                    # if allocated but not convenient or going to serve it, allow reallocation.
                    else:
                        self.evaluate_info['reallocate_up'] += 1
                        elev._hall_up_call.remove(allocated_up)  # reset and reallocate later.

        dn_called_floors = []
        for idx, floor_dn in enumerate(self._wait_downward_persons_queue):
            if len(floor_dn) > 0:
            # if len(floor_dn) > 0 and self.not_in_hallcall_but_serving_by[idx+self._floor_number] == -1:
                dn_called_floors.append(idx)

        for idx, elev in enumerate(self._elevators):  # remove allocated
            cur_dn_call = elev._hall_dn_call
            for allocated_dn in cur_dn_call:
                is_serving = not (elev._service_direction == -1 and elev._sync_floor < allocated_dn)
                if allocated_dn in dn_called_floors:
                    if is_serving or not allow_reallocation:
                        dn_called_floors.remove(allocated_dn)
                    else:
                        self.evaluate_info['reallocate_dn'] += 1
                        elev._hall_dn_call.remove(allocated_dn)
        return up_called_floors, dn_called_floors

    def get_unallocated_floors_v2(self):  # implemented by Jy
        allow_reallocation = True
        # if the floor is allocated to a car but the car is still not on the way, we allow to reallocate it.
        up_called_floors = []
        for idx, floor_up in enumerate(self._wait_upward_persons_queue):
            # 给人赋予意志，如果电梯装不下了，应该等当前满载电梯走了之后再按电梯
            if len(floor_up) > 0:
            # if len(floor_up) > 0 and self.not_in_hallcall_but_serving_by[idx] == -1:  # added by JY
                up_called_floors.append(idx)

        for idx, elev in enumerate(self._elevators):  # remove allocated
            cur_up_call = elev._hall_up_call
            for allocated_up in cur_up_call:
                # is_serving = not (elev._service_direction == 1 and elev._sync_floor > allocated_up)
                # is_serving = (elev._service_direction == 1 and elev._sync_floor <= allocated_up) or\
                #              (elev._service_direction == -1 and elev._target_floor == allocated_up)
                is_serving = elev._target_floor == allocated_up
                nf = elev.cal_cur_next_floor()
                is_overload = self.is_overload(elev, min(nf, allocated_up), max(nf, allocated_up))

                if allocated_up in up_called_floors:
                    # do not allocate this floor
                    if (is_serving or not allow_reallocation) and not is_overload:
                        up_called_floors.remove(allocated_up)
                    # if allocated but not convenient or going to serve it or is overload, allow reallocation.
                    else:
                        self.evaluate_info['reallocate_up'] += 1
                        elev._hall_up_call.remove(allocated_up)  # reset and reallocate later.

        dn_called_floors = []
        for idx, floor_dn in enumerate(self._wait_downward_persons_queue):
            if len(floor_dn) > 0:
            # if len(floor_dn) > 0 and self.not_in_hallcall_but_serving_by[idx+self._floor_number] == -1:
                dn_called_floors.append(idx)

        for idx, elev in enumerate(self._elevators):  # remove allocated
            cur_dn_call = elev._hall_dn_call
            for allocated_dn in cur_dn_call:
                # is_serving = not (elev._service_direction == -1 and elev._sync_floor < allocated_dn)
                # is_serving = (elev._service_direction == -1 and elev._sync_floor >= allocated_dn) or (
                #             elev._service_direction == 1 and elev._target_floor == allocated_dn)
                is_serving = elev._target_floor == allocated_dn
                nf = elev.cal_cur_next_floor()
                is_overload = self.is_overload(elev, min(nf, allocated_dn), max(nf, allocated_dn))
                if allocated_dn in dn_called_floors:
                    if (is_serving or not allow_reallocation) and not is_overload:
                        dn_called_floors.remove(allocated_dn)
                    else:
                        self.evaluate_info['reallocate_dn'] += 1
                        elev._hall_dn_call.remove(allocated_dn)
        return up_called_floors, dn_called_floors

    def get_unallocated_floors(self):  # implemented by Jy
        # return self.get_unallocated_floors_v1()
        return self.get_unallocated_floors_v2()

    def get_unallocated_floors_mask(self):
        unallocated_masks = [0 for _ in range(2 * self._floor_number)]
        up_unallocated, dn_unallocated = self.get_unallocated_floors()
        for up in up_unallocated:
            unallocated_masks[up] = 1
        for dn in dn_unallocated:
            unallocated_masks[dn + self._floor_number] = 1
        return unallocated_masks

    def generate_person(self, byhand=False, person_list=None):
        # modified by JY: move the generate person part from run_mansion to the env, so generate person after run mansion.
        if not person_list:
            if not byhand:
                person_list = self._person_generator.generate_person()
            else:
                # 3 9
                person_list = []
                a = input('产生人：')
                if a != '':
                    info = a.split(' ')
                    s = info[0]
                    e = info[1]
                    pnum = 1 if len(info) < 3 else int(info[2])
                    for pi in range(pnum):
                        person = PersonType(
                            pi,
                            75,
                            int(s),
                            int(e),
                            self._config.raw_time,
                            0
                        )
                        person_list.append(person)
        for person in person_list:
            if person.SourceFloor < person.TargetFloor:
                self._wait_upward_persons_queue[person.SourceFloor - 1].appendleft(person)
            elif person.SourceFloor > person.TargetFloor:
                self._wait_downward_persons_queue[person.SourceFloor - 1].appendleft(person)

    def run_mansion(self, hall_calls, special_reward=False, use_rules=False, advantage_floor=None, replace_hallcall=False):
        """
        Perform one step of simulations
        Args:
          hall_calls: A list of actions, e.g., action.add_target = [2, 6, 8], action.remove_target = [4]
          mark the target floor to be added into the queue or removed from the queue
        Returns:
          State, Cumulative waiting Time for Person, Energy Consumption of Elevator
        """
        # assert type(hall_calls) == list, "Type of input action should be list"
        # assert len(hall_calls) == self._elevator_number, "Number of hallcalls must equal to the number of elevators."
        self._config.step()  # update the current time

        energy_consumption = [0.0 for i in range(self._elevator_number)]

        # carry out actions on each elevator
        if not replace_hallcall:
            for idx in range(self._elevator_number):
                hall_calls[idx] = self._elevators[idx].restrict_hall_call(hall_calls[idx])  # check if overload
                self._elevators[idx].set_hall_call(hall_calls[idx])
        elif hall_calls:
            for idx in range(self._elevator_number):
                self._elevators[idx].replace_hall_call(hall_calls[idx])

        # # carry out rules
        # if advantage_floor is not None:
        #     if advantage_floor >= self._floor_number:
        #         advantage_floor -= self._floor_number
        #     for idx in range(self._elevator_number):
        #         elev = self._elevators[idx]
        #         if len(elev._hall_up_call) == 0 and len(elev._hall_dn_call) == 0 and len(elev._car_call) == 0:
        #             predict_floor = advantage_floor
        #             # if elev._sync_floor != predict_floor:
        #                 # elev.press_button(predict_floor)
        #             if elev._sync_floor != 0:
        #                 elev.press_button(0)
        #             # print('self go! ', advantage_floor)
        #                 # TODO: add an extra loss: the time to go to the predict floor and the time saved for people in a minute
        #
        # elif use_rules:
        if use_rules:
            # default rules for up peak mode: just go to the 1st floor.
            for idx in range(self._elevator_number):
                elev = self._elevators[idx]
                if elev.is_idle_stop and elev._sync_floor >= 1:
                    elev.set_park_call(0)
                    self.self_trip += 1
                    

        # make each elevator run one step
        loaded_person_num = 0
        all_enter_person_num = []
        delievered_person_num = 0
        all_people_waiting_time = []
        no_io_masks = []
        for idx in range(self._elevator_number):
            energy_consumption[idx], delivered_person_time, cur_loaded_num, cur_enter_num = \
                self._elevators[idx].run_elevator(RUN_ELEVATOR_WITH_BUG)  # !
            delievered_person_num += len(delivered_person_time)
            all_people_waiting_time.append(delivered_person_time)
            loaded_person_num += cur_loaded_num
            all_enter_person_num.append(cur_enter_num)
            is_open = self._elevators[idx].is_fully_open and (abs(self._elevators[idx]._remain_distance) < 0.05)
            no_io = len(self._elevators[idx]._entering_person) < 1 and len(self._elevators[idx]._exiting_person) < 1
            if len(delivered_person_time) < 1 and cur_enter_num < 1 and is_open and no_io:
                no_io_masks.append(1)
            else:
                no_io_masks.append(0)
        calling_wt = []
        for idx, cur_q in enumerate(self._wait_upward_persons_queue + self._wait_downward_persons_queue):
            for cur_p in cur_q:
                calling_wt.append(self._config.raw_time - cur_p.AppearTime)

        for floor in range(self._floor_number):
            if len(self._wait_upward_persons_queue[floor]) > 0:
                self._button[floor][0] = True
            else:
                self._button[floor][0] = False

            if len(self._wait_downward_persons_queue[floor]) > 0:
                self._button[floor][1] = True
            else:
                self._button[floor][1] = False

        ele_idxes = [i for i in range(self._elevator_number)]
        # random.shuffle(ele_idxes)  # random allocate elevators # removed by Zelin

        # average waiting time: from arrive to get in the car.
        awt = []
        for ele_idx in ele_idxes:  # !

            floor = self._elevators[ele_idx]._sync_floor
            delta_distance = self._elevators[ele_idx]._remain_distance
            # if ele_idx == 0:
            #     print(self._elevators[ele_idx]._target_floor)
            #     print(self._elevators[ele_idx]._service_direction)
            #     print('debug')
            is_open = self._elevators[ele_idx].is_fully_open and (abs(delta_distance) < 0.05)
            is_ready = self._elevators[ele_idx].ready_to_enter
            # Elevator stops at certain floor and the direction is consistent with the customers' target direction
            floor_idx = floor

            if is_open:
                if self._elevators[ele_idx]._service_direction == 1:
                    self._button[floor_idx][0] = False
                elif self._elevators[ele_idx]._service_direction == -1:
                    self._button[floor_idx][1] = False

                if not self.lock[ele_idx]:
                    self.lock_open_time[ele_idx] = self._config.raw_time
                    self.lock[ele_idx] = True
            else:
                self.lock[ele_idx] = False

            if is_ready and is_open:  # !
                self._config.log_debug(
                    "Floor: %d, Elevator: %s is open, %d persons are waiting to go upward, %d downward", floor,
                    self._elevators[ele_idx].name, len(self._wait_upward_persons_queue[floor_idx]),
                    len(self._wait_downward_persons_queue[floor_idx]))

                if self._elevators[ele_idx]._service_direction == -1:
                    for i in range(len(self._wait_downward_persons_queue[floor_idx]) - 1, -1, -1):
                        entering_person = self._wait_downward_persons_queue[floor_idx][i]
                        req_succ = self._elevators[ele_idx].person_request_in(entering_person)
                        if req_succ:
                            self._config.log_debug(
                                "Person %s is walking into the %s elevator",
                                entering_person,
                                self._elevators[ele_idx].name)
                            # awt.append(self._config.raw_time-entering_person.AppearTime)  # add by JY
                            # xzx:
                            tmp_awt = self.lock_open_time[ele_idx] - entering_person.AppearTime - (1 / self._elevators[ele_idx]._door_open_velocity)
                            if tmp_awt < 0:
                                tmp_awt = 0
                            awt.append(tmp_awt)  # add by JY
                            self.person_info[entering_person.ID] = [ele_idx, entering_person.AppearTime, tmp_awt, self.lock_open_time[ele_idx]]
                            del self._wait_downward_persons_queue[floor_idx][i]
                        else:  # if the reason of fail is overweighted, try next one
                            if not self._elevators[ele_idx]._is_overloaded:
                                break
                elif self._elevators[ele_idx]._service_direction == 1:  # if no one is entering
                    for i in range(len(self._wait_upward_persons_queue[floor_idx]) - 1, -1, -1):
                        entering_person = self._wait_upward_persons_queue[floor_idx][i]
                        req_succ = self._elevators[ele_idx].person_request_in(entering_person)
                        if req_succ:
                            self._config.log_debug(
                                "Person %s is walking into the %s elevator",
                                entering_person,
                                self._elevators[ele_idx].name)
                            # awt.append(self._config.raw_time - entering_person.AppearTime)  # add by JY
                            tmp_awt = self.lock_open_time[ele_idx] - entering_person.AppearTime - (1 / self._elevators[ele_idx]._door_open_velocity)
                            if tmp_awt < 0:
                                tmp_awt = 0
                            awt.append(tmp_awt)  # add by JY
                            self.person_info[entering_person.ID] = [ele_idx, entering_person.AppearTime, tmp_awt, self.lock_open_time[ele_idx]]
                            # print(tmp_awt)
                            del self._wait_upward_persons_queue[floor_idx][i]
                        else:
                            if not self._elevators[ele_idx]._is_overloaded:
                                break

                # add by JY, if the elevator is ready but nobody get in, or the elev is overload, then remove the hall call.
                # change to: if the elevator is going to leave or remove the hall call. Move to the elevator part.
                # if not RUN_ELEVATOR_WITH_BUG:
                #     cur_floor_idx = self._elevators[ele_idx]._sync_floor
                #     cur_dir = self._elevators[ele_idx]._service_direction
                #     if (cur_dir == -1 and len(self._wait_downward_persons_queue[cur_floor_idx]) == 0) \
                #             or (cur_dir == 1 and len(self._wait_upward_persons_queue[cur_floor_idx]) == 0) \
                #             or self._elevators[ele_idx]._is_overloaded:
                #         self._elevators[ele_idx].remove_cur_floor_hall_call()

        # Remove those who waited too long
        give_up_persons = 0
        if self._given_up_time_limit:
            for floor_idx in range(self._floor_number):
                for pop_idx in range(len(self._wait_upward_persons_queue[floor_idx]) - 1, -1, -1):
                    if self._config.raw_time - self._wait_upward_persons_queue[floor_idx][
                        pop_idx].AppearTime > self._given_up_time_limit:
                        self._wait_upward_persons_queue[floor_idx].pop()
                        give_up_persons += 1
                    else:
                        break
                for pop_idx in range(len(self._wait_downward_persons_queue[floor_idx]) - 1, -1, -1):
                    if self._config.raw_time - self._wait_downward_persons_queue[floor_idx][
                        pop_idx].AppearTime > self._given_up_time_limit:
                        self._wait_downward_persons_queue[floor_idx].pop()
                        give_up_persons += 1
                    else:
                        break

        cumulative_waiting_time = 0
        for i in range(self._floor_number):
            cumulative_waiting_time += self._dt * \
                                       len(self._wait_upward_persons_queue[i])
            cumulative_waiting_time += self._dt * \
                                       len(self._wait_downward_persons_queue[i])
        cumulative_waiting_time += loaded_person_num * self._dt
        cumulative_energy_consumption = float(sum(energy_consumption))

        self._delivered_person.appendleft(delievered_person_num)
        # self._generated_person.appendleft(tmp_generated_person)
        self._abandoned_person.appendleft(give_up_persons)
        self._cumulative_waiting_time.appendleft(cumulative_waiting_time)
        self._cumulative_energy_consumption.appendleft(cumulative_energy_consumption)
        if len(self._delivered_person) > self._statistic_interval:
            self._delivered_person.pop()
            # self._generated_person.pop()
            self._abandoned_person.pop()
            self._cumulative_waiting_time.pop()
            self._cumulative_energy_consumption.pop()

        # add by JY
        if special_reward:
            # waiting in the hall
            hall_waiting_rewards = [0 for _ in range(self._floor_number * 2)]
            waiting_queues = self._wait_upward_persons_queue + self._wait_downward_persons_queue
            cur_time = self._config.raw_time
            for floor_idx in range(len(hall_waiting_rewards)):
                # two way, accumulate or each step.
                cur_floor_reward = 0
                # way 1
                # for p in waiting_queues[floor_idx]:
                #     cur_floor_reward += (p.AppearTime - cur_time) ** 2
                # way 2
                cur_floor_reward = len(waiting_queues[floor_idx])

                hall_waiting_rewards[floor_idx] = cur_floor_reward

            # waiting in the elevator
            car_waiting_rewards = [0 for _ in range(self._floor_number * 2)]
            for ele_idx in range(self._elevator_number):
                # accounting for those floor which choose the elev, related to the action!
                cur_elev_reward = 0
                cur_elevator = self._elevators[ele_idx]
                for floor_idx in range(self._floor_number):
                    # way 1
                    # for p in cur_elevator._loaded_person[floor_idx]:
                    #     cur_elev_reward += (p.AppearTime - cur_time) ** 2
                    # way 2
                    cur_elev_reward = len(cur_elevator._loaded_person[floor_idx])

                for up_floor in cur_elevator._hall_up_call:
                    car_waiting_rewards[up_floor] += cur_elev_reward
                for dn_floor in cur_elevator._hall_dn_call:
                    car_waiting_rewards[dn_floor + self._floor_number] += cur_elev_reward

            return calling_wt, all_people_waiting_time, loaded_person_num, all_enter_person_num, no_io_masks, awt, \
                   hall_waiting_rewards, car_waiting_rewards, cumulative_energy_consumption

        return calling_wt, all_people_waiting_time, loaded_person_num, all_enter_person_num, no_io_masks, awt, cumulative_energy_consumption

    @property
    def up_served_call(self):  # added by Zelin
        floor2elev = [-1 for _ in range(self._floor_number)]
        for idx, elev in enumerate(self._elevators):
            cur_calls = elev._hall_up_call
            # print(idx, cur_calls)
            for call in cur_calls:
                if floor2elev[call] != -1:
                    floor2elev[call] = -2
                else:
                    floor2elev[call] = idx
        up_unserved, down_unserved = self.get_unallocated_floors()
        for floor in up_unserved:
            floor2elev[floor] = -3
        return floor2elev

    @property
    def dn_served_call(self):  # added by Zelin
        floor2elev = [-1 for _ in range(self._floor_number)]
        for idx, elev in enumerate(self._elevators):
            cur_calls = elev._hall_dn_call
            for call in cur_calls:
                if floor2elev[call] != -1:
                    floor2elev[call] = -2
                else:
                    floor2elev[call] = idx
        up_unserved, down_unserved = self.get_unallocated_floors()
        for floor in down_unserved:
            floor2elev[floor] = -3
        return floor2elev

    @property
    def state(self):
        """
        Return Current state of the building simulator
        """
        upward_req = []
        downward_req = []
        state_queue = []
        for idx in range(self._floor_number):
            if self._button[idx][0]:
                upward_req.append(idx)
            if self._button[idx][1]:
                downward_req.append(idx)
        for i in range(self._elevator_number):
            state_queue.append(self._elevators[i].state)

        ms = MansionState(state_queue, upward_req, downward_req)
        return ms

    @property
    def is_done(self):
        # consider better version for this function?
        return self._person_generator.done and self.finish_person_num == self._person_generator.total_person_num

    def get_statistics(self):
        """
        Get Mansion Statistics
        """
        return {
            "DeliveredPersons(10Minutes)": int(sum(self._delivered_person)),
            "GeneratedPersons(10Minutes)": int(sum(self._generated_person)),
            "AbandonedPersons(10Minutes)": int(sum(self._abandoned_person)),
            "EnergyConsumption(10Minutes)": float(sum(self._cumulative_energy_consumption)),
            "TotalWaitingTime(10Minutes)": float(sum(self._cumulative_waiting_time))}

    @property
    def attribute(self):
        """
        returns all kinds of attributes
        """
        return MansionAttribute(
            self._elevator_number,
            self._floor_number,
            self._floor_height)

    @property
    def config(self):
        """
        Returns config of the mansion
        """
        return self._config

    @property
    def waiting_queue(self):
        """
        Returns the waiting queue of each floor
        """
        return [self._wait_upward_persons_queue, self._wait_downward_persons_queue]

    @property
    def loaded_people(self):
        """
        Returns: the number of loaded people of each elevator
        """
        return [self._elevators[i].loaded_people_num for i in range(self._elevator_number)]

    @property
    def name(self):
        """
        Returns name of the mansion
        """
        return self._name

    @property
    def person_generator(self):
        return self._person_generator


if __name__ == '__main__':
    cfg = MansionConfig(dt=0.50, number_of_floors=12, floor_height=4)
    person_generator = PersonGenerator('UNIFORM')
    mm = MansionManager(1, person_generator, cfg, 'name')
    mm.run_mansion([ElevatorHallCall([3, 2], [])])
    print(mm.state)
