#   Reference on PaddlePaddle Liftsim, create a smec single elevator dispatch mode
#
#   Create by XuZhongxing on 2019/11/4, refactored by Zelin Zhao in 2021/02

import sys
from smec_liftsim.utils import *
from smec_liftsim.mansion_configs import MansionConfig
from copy import deepcopy
from smec_liftsim.smec_constants import *


class SmecElevator(object):
    """
    A simulator of elevator motion and power consumption
    Energy consumption calculated according to the paper

    Adak, M. Fatih, Nevcihan Duru, and H. Tarik Duru.
    "Elevator simulator design and estimating energy consumption of an elevator system."
    Energy and Buildings 65 (2013): 272-280.

    Created by Xu Zhongxing, reference on above simulator
    """

    def __init__(self, start_position=0.0, mansion_config=None, name="ELEVATOR", _mansion=None):
        # assert isinstance(mansion_config, MansionConfig)
        self._number_of_floors = mansion_config.number_of_floors
        self._floor_height = mansion_config.floor_height
        self._current_position = start_position
        self._maximum_speed = mansion_config.maximum_speed
        self._maximum_acceleration = mansion_config.maximum_acceleration
        self._maximum_capacity = mansion_config.maximum_capacity
        self._door_open_close_velocity = 1.0 / mansion_config.door_opening_closing_time
        self._person_entering_time = mansion_config._person_entering_time
        self._mpee_number = mansion_config._mpee_number  # maximum_parallel_entering_exiting_number
        self._dt = mansion_config.delta_t

        self._load_weight = 0.0
        self._current_velocity = 0.0
        self._door_open_rate = 0.0
        self._current_time = 0.0
        self._is_door_opening = False
        self._keep_door_open_left = 0.0
        self._keep_door_open_lag = mansion_config.keep_door_open_lag
        # actually the lag is 2.0s, as the value will decrease dt in the same time step
        self._is_door_closing = False
        self._config = mansion_config

        # A list of target floors in descending / ascending order
        self._car_call = list()
        self._hall_up_call = list()
        self._hall_dn_call = list()

        self._sync_floor = 0
        self._target_floor = 0
        self._advance_floor = self._sync_floor

        self._service_direction = 1
        self._run_direction = 0

        self._remain_distance = 0

        # Loaded Persons Queue
        self._loaded_person = [list() for i in range(self._number_of_floors)]
        # Entering Person
        self._entering_person = list()
        self._exiting_person = list()

        self._is_unloading = False
        self._is_overloaded = False
        self._run_state = ELEVATOR_STOP_DOOR_CLOSE

        self._name = name

        self._start_cmd = False
        self._stop_cmd = True

        self._mansion = _mansion

    def set_hall_call(self, hallcall):
        # assert isinstance(hallcall, ElevatorHallCall)
        # assert isinstance(hallcall.HallUpCall, list)
        # assert isinstance(hallcall.HallDnCall, list)
        for call in hallcall.HallUpCall:
            if call in self._hall_up_call:
                continue
            else:
                self._hall_up_call.append(call)

        for call in hallcall.HallDnCall:
            if call in self._hall_dn_call:
                continue
            else:
                self._hall_dn_call.append(call)

    def remove_cur_floor_hall_call(self):
        if (self._service_direction == 1) and self._sync_floor in self._hall_up_call:
                self._hall_up_call.remove(self._sync_floor)
        elif (self._service_direction == -1) and self._sync_floor in self._hall_dn_call:
            self._hall_dn_call.remove(self._sync_floor)

    def run_elevator(self):
        """
        run elevator for one step, simulate as smec elevator
        Returns:
          Energy Consumption in one single step
        """

        up_call = change_list_to_qtype_floor(self._hall_up_call, self._number_of_floors)
        dn_call = change_list_to_qtype_floor(self._hall_dn_call, self._number_of_floors)
        car_call = change_list_to_qtype_floor(self._car_call, self._number_of_floors)

        current_floor = self.calc_sync_floor(self._current_position)
        advance_floor = self.calc_advance_floor(self._current_position, self._current_velocity)
        movestate = (self._run_state == ELEVATOR_RUN)
        doorclose = (self._door_open_rate < EPSILON)
        srv_dir = self.service_direction_update(self._service_direction, self._run_direction, movestate, advance_floor,
                                                up_call, dn_call, doorclose, car_call)
        self._service_direction = srv_dir
        self._sync_floor = current_floor
        # Get the current immediate target floor
        self._advance_floor = advance_floor

        target_floor = self.go_floor_update(self._service_direction, advance_floor, up_call, dn_call, car_call)

        if self._run_state == ELEVATOR_STOP_DOOR_CLOSE:
            if target_floor != GO_FLOOR_NONE and target_floor != advance_floor:
            # if target_floor != GO_FLOOR_NONE and target_floor != advance_floor\
            #         and len(self._loaded_person[self._sync_floor]) == 0:  # add by JY, wait all person get off the car.
                self._run_state = ELEVATOR_RUN
                self._start_cmd = True
                self._stop_cmd = False
                self._target_floor = target_floor
                if current_floor < self._target_floor:
                    self._run_direction = 1
                elif current_floor > self._target_floor:
                    self._run_direction = -1
            elif target_floor != GO_FLOOR_NONE and target_floor == advance_floor and abs(self._current_velocity) < EPSILON:
                self._run_state = ELEVATOR_STOP_DOOR_OPENING
            self._current_position = self.get_floor_position(self._sync_floor)
        elif self._run_state == ELEVATOR_RUN:
            if target_floor != GO_FLOOR_NONE:
                self._target_floor = target_floor
            if abs(self._current_velocity) < EPSILON and (target_floor == advance_floor or target_floor == GO_FLOOR_NONE): # modified by Zelin
                self._run_state = ELEVATOR_STOP_DOOR_OPENING
        #         self._run_state = ELEVATOR_ARRIVE_TARGET
        # elif self._run_state == ELEVATOR_ARRIVE_TARGET:
        #     if self._service_direction == 1 and (
        #             (current_floor in self._hall_up_call) or (current_floor in self._car_call)):
        #         self._run_state = ELEVATOR_STOP_DOOR_OPENING
        #     elif self._service_direction == -1 and (
        #             (current_floor in self._hall_dn_call) or (current_floor in self._car_call)):
        #         self._run_state = ELEVATOR_STOP_DOOR_OPENING
            # else:  # this two lines added by Zelin
            #     self._run_state = -*
        elif self._run_state == ELEVATOR_STOP_DOOR_OPENING:
            self.require_door_opening()
            if self._door_open_rate > 1.0 - EPSILON:
                self._run_state = ELEVATOR_STOP_DOOR_OPEN
                if self._target_floor in self._car_call:
                    self._car_call.remove(self._target_floor)
        elif self._run_state == ELEVATOR_STOP_DOOR_OPEN:
            if self._is_door_closing:
                self._run_state = ELEVATOR_STOP_DOOR_CLOSING
        elif self._run_state == ELEVATOR_STOP_DOOR_CLOSING:
            if self._door_open_rate < EPSILON:
                self._run_state = ELEVATOR_STOP_DOOR_CLOSE
        else:
            pass

        target_position = self.get_floor_position(self._target_floor)
        remain_distance = self.calc_remain_distance(self._current_position, target_position)
        tmp_velocity, eff_dt = velocity_planner(self._current_velocity, remain_distance, self._maximum_acceleration,
                                                self._maximum_speed, self._dt)
        self._remain_distance = remain_distance
        # update the elevator position
        if abs(tmp_velocity) > EPSILON:
            delta = 0.5 * (tmp_velocity + self._current_velocity) * self._dt
        else:
            delta = 0.0

        self._current_position += delta

        # calculate the true acceleration
        acceleration = (tmp_velocity - self._current_velocity) / self._dt

        force_1 = (self._config.net_weight + self._load_weight) * \
                  (GRAVITY + acceleration)
        force_2 = self._config.rated_load * (GRAVITY - acceleration)
        net_force = force_1 - force_2
        m_load = abs(net_force) * self._config.pulley_radius / \
                 self._config.motor_gear_ratio / self._config.gear_efficiency
        energy_consumption = (m_load * abs(self._current_velocity) * self._config.gear_efficiency /
                              self._config.motor_efficiency * eff_dt + self._config.standby_power_consumption * self._dt)

        # update the elevator velocity
        self._current_velocity = tmp_velocity

        if self._is_door_opening or self._is_door_closing:
            energy_consumption += self._config.automatic_door_power * self._dt

        '''
        Door operate and call remove.
        '''

        # move the code to the state part
        # if self.is_stopped and target_floor == current_floor:
        #     self.require_door_opening()
        #     if self._door_open_rate > 1.0 - EPSILON:
        #         if target_floor in self._car_call:
        #             self._car_call.remove(target_floor)

        # modify by jy, only remove the hall call when all persons in that floor get in the car or the car is overload.
        if self.is_fully_open and self.is_stopped:
            if (self._service_direction == 1) and self._sync_floor in self._hall_up_call:
                self._hall_up_call.remove(self._sync_floor)
            elif (self._service_direction == -1) and self._sync_floor in self._hall_dn_call:
                self._hall_dn_call.remove(self._sync_floor)

        # release the control command to the elevator door
        if abs(self._current_velocity) > EPSILON:
            if self._is_door_opening:
                self._is_door_opening = False
            elif self._door_open_rate > EPSILON:
                self._is_door_closing = True

        # manage the control command of the door
        if self._door_open_rate < EPSILON:
            self._is_door_closing = False
        elif self._door_open_rate > 1.0 - EPSILON:
            self._is_door_opening = False
        if len(self._entering_person) > 0 or len(self._exiting_person) > 0:
            self._is_door_closing = False
            if self._door_open_rate < 1.0 - EPSILON:
                self._is_door_opening = True
                self._keep_door_open_left = self._keep_door_open_lag

        # manage the state of the door
        if self._is_door_opening:
            self._door_open_rate = min(
                # 1.0, self._door_open_rate + self._door_open_close_velocity)
                1.0, self._door_open_rate + self._door_open_close_velocity * self._dt)  # debug by JY
        elif self._is_door_closing:
            self._door_open_rate = max(
                0.0, self._door_open_rate - self._door_open_close_velocity * self._dt)

        # in case the door is open, print some log
        if self._door_open_rate < EPSILON and self._is_door_closing:
            self._config.log_debug(
                "Elevator: %s, Door is fully closed, Elevator at %2.2f floor",
                self._name,
                self._current_position / self._floor_height + 1.0)
        if self._door_open_rate > 1.0 - EPSILON and self._is_door_opening:
            self._config.log_debug(
                "Elevator: %s, Door is fully opening, Elevator at %2.2f floor",
                self._name,
                self._current_position / self._floor_height + 1.0)

        if self.is_fully_open:
            self._keep_door_open_left = max(0.0, self._keep_door_open_left - self._dt)

        arrived_waiting_time = []

        # # Unloading Persons
        # remove_tmp_idx = list()
        # for i in range(len(self._exiting_person)):
        #     self._exiting_person[i][1] -= self._dt
        #     if self._exiting_person[i][1] < EPSILON:
        #         remove_tmp_idx.append(i)
        # for i in sorted(remove_tmp_idx, reverse=True):
        #     out_person = self._exiting_person.pop(i)[0]
        #     self._load_weight -= out_person.Weight
        #     self._is_overloaded = False
        #     dt = self._config.raw_time - out_person.AppearTime
        #     # print('someone arrive from {} to {}, using {}'.format(out_person.SourceFloor, out_person.TargetFloor, dt))
        #     arrived_waiting_time.append(dt)
        #     self._mansion.person_info[out_person.ID].append(self._config.raw_time)  # add by JY
        if len(self._exiting_person) != 0:
            self._exiting_person[0][1] -= self._dt
            if self._exiting_person[0][1] < EPSILON:
                out_person = self._exiting_person.pop(0)[0]
                self._load_weight -= out_person.Weight
                self._is_overloaded = False


        #
        # # Loading Persons
        # remove_tmp_idx = list()
        floor_idx = self._sync_floor
        # delta_distance = remain_distance
        # for i in range(len(self._entering_person)):
        #     self._entering_person[i][1] -= self._dt
        #     if self._entering_person[i][1] < EPSILON:
        #         remove_tmp_idx.append(i)
        # for i in sorted(remove_tmp_idx, reverse=True):
        #     entering_person = self._entering_person.pop(i)[0]
        #     self._loaded_person[entering_person.TargetFloor - 1].append(deepcopy(entering_person))
        #     self._load_weight += entering_person.Weight
        #     self.press_button(entering_person.TargetFloor - 1)
        if len(self._entering_person) != 0:
            self._is_door_opening = True
            self._entering_person[0][1] -= self._dt
            if self._entering_person[0][1] < EPSILON:
                entering_person = self._entering_person.pop(0)[0]
                self._loaded_person[entering_person.TargetFloor - 1].append(deepcopy(entering_person))
                self._load_weight += entering_person.Weight
                self.press_button(entering_person.TargetFloor - 1)

        # add passengers to exiting queue
        if self.is_stopped and self.is_fully_open:
            # if abs(delta_distance) < EPSILON:
            if len(self._loaded_person[floor_idx]) > 0:
                # if(len(self._exiting_person) < self._config._mpee_number):
                tmp_unload_person = self._loaded_person[floor_idx].pop(0)
                # print(tmp_unload_person, self._sync_floor)
                self._exiting_person.append([tmp_unload_person, self._person_entering_time])
                self._mansion.person_info[tmp_unload_person.ID].append(self._config.raw_time - self._person_entering_time)  # add by JY
                self._config.log_debug("Person %s is walking out of the %s elevator", tmp_unload_person, self._name)

        if len(self._entering_person) > 0:
            self._is_entering = True
        else:
            self._is_entering = False

        if len(self._exiting_person) > 0 or len(self._loaded_person[floor_idx]) > 0:
            self._is_unloading = True
        else:
            self._is_unloading = False

        # always try closing the door
        if self.is_fully_open and not self._is_unloading and not self._is_entering:
            self.require_door_closing()

        self._config.log_debug(self.__repr__())
        loaded_num = sum(len(row) for row in self._loaded_person)
        entering_num = sum(len(row) for row in self._entering_person)
        return energy_consumption, arrived_waiting_time, loaded_num, entering_num

    def __repr__(self):
        return ("""Elevator Object: %s\n
            State\n\t
            Position: %f\n\t
            Floors: %d\n\t
            Velocity: %f\n\t
            Load: %f\n\t
            Run Direction: %d\n\t
            Target Floor: %d\n\t
            Service Direction: %d\n\t
            Car Calls: %s\n\t
            Is Overloaded: %d\n\t
            Door Open Rate: %f\n\t
            Is Door Opening: %d\n\t
            Is Door Closing: %d\n\t
            Loaded Persons: %s\n\t
            Entering Persons: %s\n\t
            Exiting Persons: %s\n\t"""
                ) % (self._name,
                     self._current_position,
                     self._sync_floor,
                     self._current_velocity,
                     self._load_weight,
                     self._run_direction,
                     self._target_floor,
                     self._service_direction,
                     self._car_call,
                     self._is_overloaded,
                     self._door_open_rate,
                     self._is_door_opening,
                     self._is_door_closing,
                     self._loaded_person,
                     self._entering_person,
                     self._exiting_person
                     )

    def door_fully_open(self, floor):
        """
        Returns whether the door is fully open in the corresponding floor
        Args:
          floor, the floor to be queried
        Returns:
          True if the elevator stops at the floor and opens the door, False in other case
        """
        cur_floor = self._sync_floor
        if (abs(cur_floor - float(floor)) < EPSILON
                and self._door_open_rate > 1.0 - EPSILON):
            return True
        else:
            return False

    @property
    def name(self):
        """
        Return Name of the Elevator
        """
        return self._name

    @property
    def state(self):
        """
        Return Formalized States
        """
        return ElevatorState(
            self._sync_floor, self._number_of_floors,
            self._current_velocity, self._maximum_speed,
            self._run_direction,
            self._door_open_rate,
            self._target_floor,
            self._service_direction,
            self._load_weight, self._maximum_capacity,
            self._car_call,
            self._is_overloaded,
            self._advance_floor,
            self._hall_up_call,
            self._hall_dn_call
        )

    @property
    def is_fully_open(self):
        return self._door_open_rate > 1.0 - EPSILON

    @property
    def ready_to_enter(self):
        """
        Returns:
          whether it is OK to enter the elevator
        """
        return (not self._is_unloading) and (not self._is_entering)

    @property
    def is_stopped(self):
        """
        Returns:
          whether the elevator stops
        """
        return abs(self._current_velocity) < EPSILON

    @property
    def loaded_people_num(self):
        """
        Returns:
            the number of loaded_people
        """
        return sum(len(self._loaded_person[i]) for i in range(self._number_of_floors))

    @property
    def per_floor_loaded_people_num(self):
        return [len(ele) for ele in self._loaded_person]

    @property
    def cur_floor(self):
        return self._sync_floor

    def _check_floor(self, floor):
        """
        check if the floor is in valid range
        """
        if floor < 0 or floor > (self._number_of_floors - 1):
            return False
        return True

    def _clip_v(self, v):
        """
        Clip the velocity
        """
        return max(-self._maximum_speed, min(self._maximum_speed, v))

    def _check_abnormal_state(self):
        """
        the function checks that the elevator is OK
        raise Exceptions if anything is wrong
        """
        cur_floor = self._sync_floor
        if (cur_floor < 0) or (cur_floor > self._number_of_floors - 1):
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, current_floor = %f",
                self._name,
                cur_floor)
        if abs(self._current_velocity) > self._maximum_speed:
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, current_velocity = %f exceeds the maximum velocity %f",
                self._name, self._current_velocity, self._maximum_speed)
        if self._load_weight > self._maximum_capacity:
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, load_weight(%f) > maximum capacity (%f)",
                self._name, self._load_weight, self._maximum_capacity)

        if self._run_direction * self._current_velocity < - 1.0:
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, direction(%d) dot not match current velocity %f, Elevator state: %s",
                self._name, self._run_direction, self._current_velocity, self)

    def restrict_hall_call(self, hall_call):  # added by Zelin.
        if self._is_overloaded and self._sync_floor in hall_call[0]:
            hall_call[0].remove(self._sync_floor)
        if self._is_overloaded and self._sync_floor in hall_call[1]:
            hall_call[1].remove(self._sync_floor)
        return hall_call

    def require_door_opening(self):
        """
        Requires the door to close for the elevator
        """
        # if (self.is_stopped and self._door_open_rate < 1.0 - EPSILON and self._is_overloaded == False):
        if self.is_stopped and self._door_open_rate < 1.0 - EPSILON:  # modified by Zelin
            self._is_door_opening = True
            self._keep_door_open_left = self._keep_door_open_lag
            self._is_door_closing = False

    def require_door_closing(self):
        """
        Requires the door to open for the elevator
        """
        if (self._door_open_rate > EPSILON and not self._is_door_opening
                and not self._is_unloading and not self._is_entering
                and self._keep_door_open_left < EPSILON):
            self._is_door_closing = True
            self._config.log_debug("Require door closing succeed")
        else:
            self._config.log_debug("Require door closing failed, door_open_rate = %f, is_unloading = %d, "
                                   "entering_person = %s", self._door_open_rate, self._is_unloading,
                                   self._entering_person)

    def person_request_in(self, person):
        """
        Load a person onto the elevator
        Args:
          Person Tuple
        Returns:
          True - if Success
          False - if Overloaded
        """
        # assert isinstance(person, PersonType)
        cur_floor = self.calc_sync_floor(self._current_position)
        if (abs(person.SourceFloor - cur_floor - 1) > EPSILON or
                abs(self._current_velocity) > EPSILON or
                abs(self._door_open_rate < 1.0 - 2.0 * EPSILON)):
            self._config.log_debug(
                "Refuse to accommadate person: elevator: %s, person: %s, illegal request",
                self,
                person)
            return False
        if len(self._entering_person) >= self._mpee_number:
            self._config.log_debug(
                "Refuse to accommadate person: elevator: %s, person: %s, maximum parallel stream arrived",
                self,
                person)
            return False

        # # initially commented out
        # if(self._is_overloaded_alarm > EPSILON):
        #   self._config.log_debug("Refuse to accommadate person: elevator: %s, person: %s, over_loaded", self, person)
        #   self.require_door_closing()
        #   return False

        # add all expected weight
        expected_weight = self._load_weight
        for iter_person, time in self._entering_person:
            expected_weight += iter_person.Weight
        # if expected_weight + person.Weight > self._maximum_capacity:
        if expected_weight + person.Weight > 0.8 * self._maximum_capacity:
            self._is_overloaded = True
            self.require_door_closing()
            return False
        else:  # modified by Zelin
            self._is_overloaded = False
        self._entering_person.append([deepcopy(person), self._person_entering_time])
        return True

    def press_button(self, button):
        """
        press a button in the elevator, might be valid or not valid
        Args:
          button: the button to be clicked
        Returns:
          None
        """
        if button >= 0 and button < self._number_of_floors and button not in self._car_call:
            self._car_call.append(button)

    # below is control calc
    def get_floor_position(self, flr):
        return flr * self._floor_height

    def calc_sync_floor(self, pos):
        lower_floor = 0
        upper_floor = self._number_of_floors - 1

        while upper_floor - lower_floor > 1:
            mid_floor = lower_floor + upper_floor
            mid_floor = mid_floor >> 1
            tmp_position = self.get_floor_position(mid_floor)
            if pos >= tmp_position:
                lower_floor = mid_floor
            else:
                upper_floor = mid_floor

        tmp_position = self.get_floor_position(lower_floor)
        tmp_position += self.get_floor_position(upper_floor)
        tmp_position /= 2

        if pos <= tmp_position:
            floor = lower_floor
        else:
            floor = upper_floor

        return floor

    def calc_advance_floor(self, pos, velocity):
        cur_floor = self.calc_sync_floor(pos)
        if abs(velocity) < EPSILON:
            adv_floor = cur_floor
        elif velocity >= EPSILON:
            adv_floor = self.calc_sync_floor(pos + 0.5 + 0.5 * velocity * velocity / self._maximum_acceleration)
        else:
            adv_floor = self.calc_sync_floor(pos - 0.5 - 0.5 * velocity * velocity / self._maximum_acceleration)

        return adv_floor

    def calc_remain_distance(self, current,
                             target):  # this remain distance is positive when run up, negative when run dn
        dist = target - current
        if abs(dist) < EPSILON:
            dist = 0.0

        return dist

    # Below is sa calc, service dir is
    def service_direction_update(self, srvdir, rundir, movestate, advance_floor, up_call, down_call, doorclose,
                                 car_call):
        if movestate == 0:
            if srvdir != 0:  # stop change to another service direction
                srv_dir = self.service_direction_stop_change(doorclose, srvdir, car_call, up_call, down_call,
                                                             advance_floor)
            else:  # stop and renew the service direction
                up_call = get_up_all_call(car_call, up_call)
                dn_call = get_dn_all_call(car_call, down_call)
                srv_dir = self.service_direction_stop_renew(advance_floor, up_call, dn_call)
        else:
            srv_dir = rundir  # if move, continue the service direction
        return srv_dir

    def service_direction_stop_change(self, doorclose, srvdir, car_call, up_call, down_call, advance_floor):
        allcall = get_all_updn_call(car_call, up_call, down_call)

        if allcall[advance_floor] == 1:
            allcall[advance_floor] = 0

        cbot = 0
        ctop = self._number_of_floors - 1
        up_call = get_up_all_call(car_call, up_call)
        dn_call = get_dn_all_call(car_call, down_call)

        if not has_forward_call(srvdir, allcall, advance_floor, ctop, cbot) and doorclose:
            srv_dir = self.service_direction_stop_renew(advance_floor, up_call, dn_call)
        else:
            srv_dir = self.service_direction_realtime_renew(srvdir, allcall, up_call, dn_call, advance_floor)

        return srv_dir

    def service_direction_realtime_renew(self, srvdir, allcall, up, dn, cadv):
        cbot = 0
        ctop = self._number_of_floors - 1
        allcall[cadv] = 0

        if has_forward_call(srvdir, allcall, cadv, ctop, cbot):
            srv_dir = srvdir
        elif (srvdir == 1 and get_qtype_floor(up, cadv)) or (srvdir == -1 and get_qtype_floor(dn, cadv)):
            srv_dir = srvdir
        elif (srvdir == 1 and get_qtype_floor(dn, cadv)) or (srvdir == -1 and get_qtype_floor(up, cadv)):
            srv_dir = self.reverse_service_direction(srvdir)
        elif has_backward_call(srvdir, allcall, cadv, ctop, cbot):
            srv_dir = self.reverse_service_direction(srvdir)
        elif (not has_forward_call(srvdir, allcall, cadv, ctop, cbot)) and (
        not has_backward_call(srvdir, allcall, cadv, ctop, cbot)):
            srv_dir = 0
        else:
            srv_dir = srvdir

        return srv_dir

    def reverse_service_direction(self, srvdir):
        if srvdir == 1:
            srv_dir = -1
        elif srvdir == -1:
            srv_dir = 1
        else:
            srv_dir = 0

        return srv_dir

    def service_direction_stop_renew(self, cadv, up, dn):
        cbot = 0
        ctop = self._number_of_floors - 1

        if get_qtype_floor(dn, cadv):
            srv_dir = -1
        elif get_qtype_floor(up, cadv):
            srv_dir = 1
        elif has_call_between(dn, cbot, cadv):
            srv_dir = -1
        elif has_call_between(up, cadv, ctop):
            srv_dir = 1
        elif has_call_between(up, cbot, cadv):
            srv_dir = -1
        elif has_call_between(dn, cadv, ctop):
            srv_dir = 1
        else:
            srv_dir = 0

        return srv_dir

    # service dir is not equal to run dir, input is service dir
    # if return 0, there is no target floor find, elevator should stop
    def go_floor_update(self, srvdir, cadv, up, dn, car_call):
        up_call = get_up_all_call(car_call, up)  # up all call
        dn_call = get_dn_all_call(car_call, dn)  # dn all call
        go_floor = GO_FLOOR_NONE
        srch_result = False
        cbot = 0
        ctop = self._number_of_floors - 1

        if srvdir == 1:
            for flr in range(cadv, ctop + 1, 1):  # this direction?
                if get_qtype_floor(up_call, flr):
                    go_floor = flr
                    srch_result = True
                    break
            if not srch_result:
                for flr in range(ctop, cadv - 1, -1):
                    if get_qtype_floor(dn_call, flr):
                        go_floor = flr
                    if flr == cadv or go_floor != GO_FLOOR_NONE:
                        break
        elif srvdir == -1:
            for flr in range(cadv, cbot - 1, -1):
                if get_qtype_floor(dn_call, flr):
                    go_floor = flr
                    srch_result = True
                if flr == cbot or go_floor != GO_FLOOR_NONE:
                    break
            if not srch_result:
                for flr in range(cbot, cadv + 1, 1):
                    if get_qtype_floor(up_call, flr):
                        go_floor = flr
                        break
        else:
            pass

        return go_floor

    def step(self, hall_call):
        self.set_hall_call(hall_call)
        energy, person_waiting_times, loaded_person_num, enter_num = self.run_elevator()
        return energy, person_waiting_times, person_waiting_times, enter_num


if __name__ == '__main__':
    cfg = MansionConfig(
            dt=0.50,
            number_of_floors=12,
            floor_height=4)
    se = SmecElevator(start_position=0.0, mansion_config=cfg)
    a = se.step(hall_call=ElevatorHallCall([3, 2], []))
    print(a)
