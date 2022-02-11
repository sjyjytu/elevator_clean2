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

# utils.py
#
# Utils for mansion and elevator simulator

import sys
import time
import collections

# Average Entering and Exiting Time of a person
ENTERING_TIME = 1.0
# A small number
EPSILON = 1.0e-2
# A huge number
HUGE = 1.0e+8
# Gravity Acceleration
GRAVITY = 9.80


# A person type
# ID : A unique ID for each person
# weight : weight of the person
# SourceFloor & TargetFloor : the person appears in the source floor and wants to get to the target floor
# AppearTime: appearing time
PersonType = collections.namedtuple(
    'Person', ['ID', 'Weight', 'SourceFloor', 'TargetFloor', 'AppearTime', 'StatisticElev'])
PersonType.__new__.__defaults__ = (-1, )
# The state of the elevator

ElevatorState = collections.namedtuple("ElevatorState",
                                       ["Floor", "MaximumFloor",
                                        "Velocity", "MaximumSpeed",
                                        "RunDirection",
                                        "DoorState",
                                        "TargetFloor",
                                        "ServiceDirection",
                                        "LoadWeight", "MaximumLoad",
                                        "CarCall",
                                        "Overloaded",
                                        "AdvanceFloor",
                                        "HallUpCalls",
                                        "HallDnCalls",
                                        "Position"])
'''
ElevatorState = collections.namedtuple("ElevatorState",
                                       ["Floor", "MaximumFloor",
                                        "Velocity", "MaximumSpeed",
                                        "Direction", "DoorState",
                                        "CurrentDispatchTarget", "DispatchTargetDirection",
                                        "LoadWeight", "MaximumLoad",
                                        "ReservedTargetFloors", "OverloadedAlarm",
                                        "DoorIsOpening", "DoorIsClosing"])
'''
# Action Specified for each elevator
# TargetFloor will temporarily add a target into the planned stop list
#     a -1 value will not change the current planned target
# DirectionIndicator specify the indicator of the direction when the elevator stop in that target floor
#     it is only useful when elevator is in completely free
ElevatorAction = collections.namedtuple(
    "Action", ["TargetFloor", "DirectionIndicator"])

ElevatorHallCall = collections.namedtuple(
    "HallCall", ["HallUpCall", "HallDnCall"])

# The states of the mansion
# ElevatorStates: A list of the ElevatorState tuple
# RequiringUpFloors & RequiringDownFloors: the floor that requires to go
# up / down
MansionState = collections.namedtuple(
    "MansionState", [
        "ElevatorStates", "RequiringUpwardFloors", "RequiringDownwardFloors"])
# Static attributes of the mansion
MansionAttribute = collections.namedtuple(
    'MansionAttribute', [
        "ElevatorNumber", "NumberOfFloor", "FloorHeight"])

# Formulated Simulate Time
SimulatedTime = collections.namedtuple(
    "SimulatedTime", ["Day", "Hour", "Min", "Sec"])


def formulate_simulation_time(time):
    """
    Turn Seconds Into Formulated Time
    Eg. 93671 = Day: 1, Hour: 2, Min: 1, Sec: 11
    """
    res = time
    day = int(res) // 86400
    res -= float(day) * 86400.0
    hour = int(res) // 3600
    res -= float(hour) * 3600
    minute = int(res) // 60
    res -= float(minute) * 60
    return SimulatedTime(day, hour, minute, res)


def simulation_time_to_str(time):
    """
    Turn simulated time to string
    """
    assert isinstance(time, SimulatedTime)
    return "(Day: %d, Hour: %d, Minute: %d, Second: %2.2f)" % (
        time.Day, time.Hour, time.Min, time.Sec)


def raw_time_to_str(time):
    """
    convert raw time to a string
    """
    formulated_time = formulate_simulation_time(time)
    return simulation_time_to_str(formulated_time)


def args_to_string(string, args):
    """
    Convert arguments to a string
    """
    if(len(args) < 1):
        return string
    elif(len(args) < 2):
        return string % args[0]
    else:
        return string % tuple(args)


# def select_max_acc(dist):
#     if dist >= 12.0 - 100*EPSILON:
#         acc = 0.6369
#     elif dist >= 9.0 - 100*EPSILON:
#         acc = 0.6305
#     elif dist >= 6.0 - 100*EPSILON:
#         acc = 0.5976
#     else:
#         acc = 0.5305
#
#     return acc*1.05   # coff from test
def select_max_acc(dist):
    if dist >= 12.0 - 100*EPSILON:
        acc = 0.668745
    elif dist >= 9.0 - 100*EPSILON:
        acc = 0.662025
    elif dist >= 6.0 - 100*EPSILON:
        acc = 0.62748
    else:
        acc = 0.557025

    return acc


def velocity_planner(start_v, target_x, max_acc, max_spd, dt):
    """
    Plan a trajectory toward target position, such that velocity = 0 when arriving at the position
    Args:
      start_v: initial velocity
      target_x: target position
      max_acc: maximum absolute value acceleration
      max_spd: maximum absolute value speed
      dt: time step
    Returns:
      des_vel, eff_dt
      the agent accelerate uniformly from start_v to des_vel from t = 0 to t = eff_dt
               keeps v = des_vel from t = eff_dt to t = dt
    """
    def clip_value(value, max_abs_val):
        return max(-max_abs_val, min(max_abs_val, value))

    if(start_v > 0):
        sign_v = 1.0
    else:
        sign_v = -1.0

    eff_dt = dt
    if(0.1 * EPSILON > abs(target_x)):
        if(abs(start_v) < abs(max_acc * dt)):
            eff_dt = abs(start_v) / abs(max_acc)
        des_vel = start_v - sign_v * max_acc * eff_dt
        return des_vel, eff_dt

    if(target_x > 0):
        sign_x = 1.0
    else:
        sign_x = -1.0

    num = 10
    # Firstly, find out whether it is ok to keep the current speed
    reaction_dist_s = 0.5 * start_v * abs(start_v) / max_acc + start_v * dt

    tmp_vel_e = clip_value(start_v + sign_x * max_acc * dt, max_spd)
    reaction_dist_e = 0.5 * tmp_vel_e * \
        abs(tmp_vel_e) / max_acc + 0.5 * (tmp_vel_e + start_v) * dt

    #print ("reaction dist:", reaction_dist_s, reaction_dist_e, "target x:", target_x)

    if((sign_x > 0 and target_x > reaction_dist_e - 0.1 * EPSILON) or
            (sign_x < 0 and target_x < reaction_dist_e + 0.1 * EPSILON)):
        # If it is, find the highest possible acceleration
        des_vel = tmp_vel_e
    elif((sign_x > 0 and target_x > reaction_dist_s - 0.1 * EPSILON) or
         (sign_x < 0 and target_x < reaction_dist_s + 0.1 * EPSILON)):
        # dichotomy search a proper acceleration
        fraction_s = 0.0
        fraction_e = 1.0
        fraction_m = 0.5
        iterate = 0
        des_vel = start_v
        while iterate < 5:
            tmp_vel = clip_value(
                start_v +
                sign_x *
                fraction_m *
                max_acc *
                dt,
                max_spd)
            reaction_dist = 0.5 * tmp_vel * \
                abs(tmp_vel) / max_acc + 0.5 * (tmp_vel + start_v) * dt

            if((target_x > 0 and target_x > reaction_dist) or
                    (target_x < 0 and target_x < reaction_dist)):
                fraction_s = fraction_m
                #print ("fraction_s", fraction_s)
                des_vel = tmp_vel
            else:
                fraction_e = fraction_m
            fraction_m = 0.5 * (fraction_s + fraction_e)
            iterate += 1
    else:
        if(abs(target_x) < 0.1 * EPSILON):
            if(abs(start_v) < abs(max_acc * dt)):
                eff_dt = abs(start_v) / abs(max_acc)
            des_vel = start_v - sign_v * max_acc * eff_dt
        else:
            req_acc = clip_value(- 0.5 * abs(start_v * \
                                 start_v / target_x) * sign_v, max_acc)
            if(abs(start_v) < abs(req_acc * dt)):
                eff_dt = abs(start_v) / abs(req_acc)
            des_vel = start_v + req_acc * eff_dt

    return des_vel, eff_dt


# call utils

# qtype is a list like this [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], it means floor 6 has a call
def change_list_to_qtype_floor(call, flrnum):
    qtype = []
    for flr in range(flrnum):
        if flr in call:
            qtype.append(1)
        else:
            qtype.append(0)

    return qtype


def merge_list_to_qtype_floor(call, park, flrnum):
    qtype = []
    for flr in range(flrnum):
        if (flr in call) or (flr in park):
            qtype.append(1)
        else:
            qtype.append(0)

    return qtype


# call is qtype, and cadv is advance floor
def get_qtype_floor(call, cadv):
    ans = False
    if call[cadv] == 1:
        ans = True
    return ans


def has_call_between(call, lower, upper):
    result = False
    for flr in range(lower, upper + 1):
        if get_qtype_floor(call, flr):
            result = True
            break

    return result


def has_forward_call(service_direction, call, floor_advance, top, bot):
    if service_direction == -1:
        result = has_call_between(call, bot, floor_advance)
    elif service_direction == 1:
        result = has_call_between(call, floor_advance, top)
    else:
        result = False

    return result


def has_backward_call(dir, call, cadv, top, bot):
    if dir == -1:
        result = has_call_between(call, cadv, top)
    elif dir == 1:
        result = has_call_between(call, bot, cadv)
    else:
        result = False

    return result


def get_up_all_call(car_call, hall_up_call):
    all_call = []

    for i in range(len(car_call)):
        if car_call[i] != 0 or hall_up_call[i] != 0:
            all_call.append(1)
        else:
            all_call.append(0)

    return all_call


def get_dn_all_call(car_call, hall_dn_call):
    all_call = []

    for i in range(len(car_call)):
        if car_call[i] != 0 or hall_dn_call[i] != 0:
            all_call.append(1)
        else:
            all_call.append(0)

    return all_call


def get_all_updn_call(car_call, up_call, dn_call):
    all_call = []

    for i in range(len(car_call)):
        if car_call[i] != 0 or up_call[i] != 0 or dn_call[i] != 0:
            all_call.append(1)
        else:
            all_call.append(0)

    return all_call
