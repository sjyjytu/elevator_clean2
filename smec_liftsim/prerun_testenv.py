from smec_liftsim.fixed_data_generator import FixedDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.utils import ElevatorHallCall
import configparser
import os
import numpy as np
import matplotlib.pyplot as plt

config_file = os.path.join(os.path.dirname(__file__) + '/rl_config2.ini')
file_name = config_file
config = configparser.ConfigParser()
config.read(file_name)
dt = 30
dataX = []
dataY_wt = []
dataY_tt = []
dataY_at = []
draw = False
data_dir = 'E:\\elevator\\predict_flow\\down_data'
for file in os.listdir(data_dir):
    print(file)
    for st in range(0, 3300//dt):
        # start_time = '10:00'
        # end_time = '11:00'
        real_start_time = dt * st
        real_end_time = dt * (st+1)
        start_time = '%s:%s'%(str(real_start_time // 60).rjust(2, '0'), str(real_start_time % 60).rjust(2, '0'))
        end_time = '%s:%s'%(str(real_end_time // 60).rjust(2, '0'), str(real_end_time % 60).rjust(2, '0'))
        # print(start_time, end_time)
        _config = MansionConfig(start_time=int(start_time.split(':')[0])*60+int(start_time.split(':')[1])-1, dt=0.5, number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
                                             floor_height=float(config['MansionInfo']['FloorHeight']), maximum_capacity=1000,
                                             maximum_acceleration=0.8, maximum_speed=2.5)
        # data_file = 'E:\\elevator\\predict_flow\\down_data\\4Ele16FloorDnPeakFlow2_elvx.csv'
        data_file = os.path.join(data_dir, file)
        elevator_num = 1
        person_generator = FixedDataGenerator(data_file=data_file, data_of_section=start_time+'-'+end_time)
        mansion = MansionManager(elevator_num, person_generator, _config,
                                              config['MansionInfo']['Name'])
        # print person info
        p = person_generator.next_person
        if p is None:
            continue
        # print(p['start_level'], p['end_level'], p['appear_time'])
        level2pnum = [0 for i in range(16)]
        level2pnum[int(p['start_level'])-1] += 1
        while not person_generator.data.empty():
            p = person_generator.data.get()
            # print(p['start_level'], p['end_level'], p['appear_time'])
            level2pnum[int(p['start_level'])-1] += 1

        # render
        render = False
        if render:
            from smec_liftsim.rendering import Render
            viewer = Render(mansion)

        predict_run_time_up = [0 for x in range(16)]
        predict_run_time_dn = [0 for x in range(16)]
        mean_wts = [0 for x in range(16)]
        mean_tts = [0 for x in range(16)]
        actual_run_time = [0 for x in range(16)]
        for x in range(0, 16):
            mansion.reset_env()
            mansion._elevators[0]._current_position = mansion.config.floor_height * x
            mansion._elevators[0]._target_floor = x

            elevator = mansion._elevators[0]
            delay = 0
            while not mansion._person_generator.done or not (len(elevator._car_call+elevator._hall_up_call+elevator._hall_dn_call)==0) or delay < 20:
                if (mansion._person_generator.done and (len(elevator._car_call+elevator._hall_up_call+elevator._hall_dn_call)==0)):
                    delay += 1
                # print(mansion.config.raw_time, mansion._person_generator.done)
                unallocated_up, unallocated_dn = mansion.get_unallocated_floors()
                all_elv_up_fs, all_elv_down_fs = [[] for _ in range(elevator_num)], [[] for _ in range(elevator_num)]
                for up_floor in unallocated_up:
                    cur_elev = 0
                    all_elv_up_fs[cur_elev].append(up_floor)
                for dn_floor in unallocated_dn:
                    cur_elev = 0
                    all_elv_down_fs[cur_elev].append(dn_floor)
                action_to_execute = []
                for idx in range(elevator_num):
                    action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
                calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = mansion.run_mansion(action_to_execute)
                if render:
                    viewer.view()
            waiting_time = []
            transmit_time = []
            get_on_time = []
            get_off_time = []
            for k in mansion.person_info.keys():
                info = mansion.person_info[k]
                waiting_time.append(info[2] - info[1])
                transmit_time.append(info[3] - info[2])
                get_on_time.append(info[2])
                get_off_time.append(info[3])
            # print(x, waiting_time, transmit_time, np.mean(waiting_time), np.mean(transmit_time), np.mean(waiting_time)+np.mean(transmit_time))
            # print(f'elevator start at {x+1} floor, mean waiting time: {np.mean(waiting_time):.2f}, mean transmit time: {np.mean(transmit_time):.2f}, '
            #       f'mean total time: {np.mean(waiting_time) + np.mean(transmit_time):.2f}; {get_on_time}, {get_off_time}')

            # 求预测值
            # 以上为正方向
            E_all_up = 0
            for e in range(x, 16):
                E = (e-x) * level2pnum[e]
                E_all_up += E
            for e in range(0, x):
                E = ((15-x)*2+x-e) * level2pnum[e]
                E_all_up += E

            # 以下为正方向
            E_all_dn = 0
            for e in range(0, x):
                E = (x - e) * level2pnum[e]
                E_all_dn += E
            for e in range(x, 16):
                E = (x * 2 + e - x) * level2pnum[e]
                E_all_dn += E
            predict_run_time_up[x] = E_all_up
            predict_run_time_dn[x] = E_all_dn

            wt = np.mean(waiting_time)
            tt = np.mean(transmit_time)
            mean_wts[x] = wt
            mean_tts[x] = tt
            actual_run_time[x] = wt + tt
            # print(
            #     f'elevator start at {x + 1} floor, mean waiting time: {wt:.2f}, mean transmit time: {tt:.2f}, '
            #     f'mean total time: {wt + tt:.2f}; {min(E_all_up, E_all_dn)}, predict up: {E_all_up}, predict dn: {E_all_dn}')

        dataX.append(level2pnum)
        dataY_wt.append(mean_wts)
        dataY_tt.append(mean_tts)
        dataY_at.append(actual_run_time)

        # draw
        if draw:
            plt.figure()
            plt.plot(mean_wts, label='wt')
            plt.plot(mean_tts, label='tt')
            plt.plot(actual_run_time, label='actual')
            plt.plot(predict_run_time_up, label='predict_up')
            plt.plot(predict_run_time_dn, label='predict_dn')
            plt.bar(range(len(level2pnum)), np.array(level2pnum)*50)

            plt.legend(loc='best')
            plt.savefig('E:\\elevator\\predict_flow\\figures2\\%s.jpg'%(start_time[:2]+start_time[3:]+'-'+end_time[:2]+end_time[3:]))
            plt.close()
            # plt.show()


np.save('dataX.npy', np.array(dataX))
np.save('dataY_wt.npy', np.array(dataY_wt))
np.save('dataY_tt.npy', np.array(dataY_tt))
np.save('dataY_at.npy', np.array(dataY_at))

