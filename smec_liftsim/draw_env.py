import numpy as np
import time
from smec_liftsim.generator_proxy import set_seed
from smec_liftsim.generator_proxy import PersonGenerator
from smec_liftsim.fixed_data_generator import FixedDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.utils import ElevatorHallCall
from smec_liftsim.smec_elevator_new import SmecElevator
from smec_liftsim.utils import MansionAttribute, MansionState
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.generator_proxy import PersonGenerator
import configparser
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

config_file = os.path.join(os.path.dirname(__file__) + '/rl_config2.ini')
file_name = config_file
config = configparser.ConfigParser()
config.read(file_name)

time_step = float(config['Configuration']['RunningTimeStep'])
_config = MansionConfig(
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
        # start_time=900
    )


def run_one(calls):
    y = []
    elev = SmecElevator(start_position=0.0, mansion_config=_config, name="E%d" % (1), _mansion=None)
    _config.reset()
    for tt in calls:
        elev.press_button(tt)
    last_time_have_vel = False
    while True:
        _config.step()
        elev.run_elevator()
        if abs(elev._current_velocity) < 0.01:
            if last_time_have_vel:
                # print(_config.raw_time)
                y.append(_config.raw_time)
            last_time_have_vel = False
        else:
            last_time_have_vel = True
        if elev.is_idle_stop:
            # print(_config.raw_time)
            break
    return y

# plt.figure()
# plt.title('时间与楼层距离')
# plt.plot(range(1,16), y, color='blue')
# plt.grid(True)
# plt.xlabel('楼层')
# plt.ylabel('时间')
# plt.show()
# plt.close()


# # 间隔着跑的时间
# res = []
# for interval in range(1, 16):
#     res.append([])
#     for i in range(interval):
#         calls = range(i, 16, interval)
#         y = run_one(calls)
#         res[-1].append(y[-1])
#     print(f'interval: {interval}', res[-1])
# print(res)


def draw_mansion(init_call=8):
    data_file = '../train_data/new/lunchpeak/LunchPeak1_elvx.csv'
    person_generator = FixedDataGenerator(data_file=data_file, data_dir=None, file_begin_idx=None,
                                          data_of_section='')
    mansion = MansionManager(1, person_generator, _config,
                                  config['MansionInfo']['Name'])
    from smec_liftsim.rendering import Render
    viewer = Render(mansion)
    elev0 = mansion._elevators[0]
    elev0.set_hall_call(ElevatorHallCall([init_call],[]))
    for i in range(10000):
        a = input()
        if a != ' ':
            a = int(a)
            print(a)
            if a < 0:
                elev0.remove_hall_call(ElevatorHallCall([-a], []))
            else:
                elev0.set_hall_call(ElevatorHallCall([a], []))
        elev0.run_elevator()
        viewer.view()
        print(f'state:{elev0._run_state}, pos:{elev0._current_position}, dir:{elev0._service_direction}, '
              f'syn:{elev0._sync_floor}, adv:{elev0._advance_floor}, tar:{elev0._target_floor}, v:{elev0._current_velocity}')


def cal_dis2time(init_call=8):
    class TimeCnt:
        def __init__(self):
            self.cnt = 0

        def step(self):
            self.cnt += 1

        def reset(self):
            self.cnt = 0

        def get_time(self, dt):
            return self.cnt * dt
    time_cnt = TimeCnt()
    data_file = '../train_data/new/lunchpeak/LunchPeak1_elvx.csv'
    person_generator = FixedDataGenerator(data_file=data_file, data_dir=None, file_begin_idx=None,
                                          data_of_section='')
    mansion = MansionManager(1, person_generator, _config,
                                  config['MansionInfo']['Name'])
    time_cnt.reset()
    from smec_liftsim.rendering import Render
    viewer = Render(mansion)
    elev0 = mansion._elevators[0]
    elev0._current_position = 0.0
    elev0.set_hall_call(ElevatorHallCall([init_call],[]))
    while not elev0.is_idle_stop:
        elev0.run_elevator(time_cnt=time_cnt)
        viewer.view()
    for finish_task in elev0.arrive_info:
        task = finish_task[0][0]  # u, d, c
        floor = int(finish_task[0][1:])
        consume_time = finish_task[1]
        print(task, floor, f'{consume_time:.2f}')
        # total_time += consume_time


# draw_mansion()
for i in range(0, 16, 1):
    print(f'go {i} floor')
    cal_dis2time(i)


def dis2time(df):
    if df == 1:
        return 12
    elif df == 2:
        return 13.5
    else:
        return 13.5 + 1.2 * (df - 2)


def cal_dis2time_with_v(x, v, target_floor, door_open_rate=0):
    class TimeCnt:
        def __init__(self):
            self.cnt = 0

        def step(self):
            self.cnt += 1

        def reset(self):
            self.cnt = 0

        def get_time(self, dt):
            return self.cnt * dt
    time_cnt = TimeCnt()
    # data_file = '../train_data/new/lunchpeak/LunchPeak1_elvx.csv'
    # person_generator = FixedDataGenerator(data_file=data_file, data_dir=None, file_begin_idx=None,
    #                                       data_of_section='')
    # mansion = MansionManager(1, person_generator, _config,
    #                               config['MansionInfo']['Name'])
    time_cnt.reset()
    # from smec_liftsim.rendering import Render
    # viewer = Render(mansion)
    # elev0 = mansion._elevators[0]
    elev0 = SmecElevator(mansion_config=_config)
    elev0._current_position = x
    elev0._current_velocity = v
    elev0._door_open_rate=door_open_rate
    elev0.set_hall_call(ElevatorHallCall([target_floor],[]))
    while not elev0.is_idle_stop:
        elev0.run_elevator(time_cnt=time_cnt)
        # viewer.view()
    for finish_task in elev0.arrive_info:
        task = finish_task[0][0]  # u, d, c
        floor = int(finish_task[0][1:])
        consume_time = finish_task[1]
        # print(task, floor, f'{consume_time:.2f}')
    return consume_time


# for x in np.arange(0, 3.0, 0.1):
#     for v in np.arange(0, 2.5, 0.1):
#         target_floor = 3
#         # for target_floor in range(2, 16):
#         consume_time = cal_dis2time_with_v(x, v, target_floor)
#         print(f'{x:.1f} {v:.1f} {target_floor} {consume_time:.1f} ')
# for v in np.arange(0, 2.5, 0.1):
#     for x in np.arange(0, 3.0, 0.1):
#         target_floor = 3
#         # for target_floor in range(2, 16):
#         consume_time = cal_dis2time_with_v(x, v, target_floor)
#         print(f'{x:.1f} {v:.1f} {target_floor} {consume_time:.1f} ')
# for dr in np.arange(0, 1.0, 0.1):
#     # for target_floor in range(2, 16):
#     consume_time = cal_dis2time_with_v(0, 0, 1, dr)
#     print(f'{dr:.1f} {consume_time:.1f} ')