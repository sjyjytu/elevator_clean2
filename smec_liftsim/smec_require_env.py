import numpy as np
import time
from smec_liftsim.generator_proxy import set_seed
from smec_liftsim.generator_proxy import PersonGenerator
from smec_liftsim.fixed_data_generator import FixedDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.utils import ElevatorHallCall
import configparser
import os


class SmecEnv():
    """
    environment for SMEC elevators.
    """

    def __init__(self, data_file, config_file=None, render=True, forbid_unrequired=True, seed=None, forbid_uncalled=False,
                 use_graph=True, gamma=0.99, real_data=True, use_advice=False, special_reward=False, data_dir=None, file_begin_idx=None):
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__) + '/rl_config2.ini')
        file_name = config_file
        self.forbid_uncalled = forbid_uncalled
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy.'

        if not real_data:
            # Create a different person generator
            gtype = config['PersonGenerator']['PersonGeneratorType']
            person_generator = PersonGenerator(gtype)
            person_generator.configure(config['PersonGenerator'])
        else:
            # person_generator = RealDataGenerator(data_folder='person_flow_data')
            person_generator = FixedDataGenerator(data_file=data_file, data_dir=data_dir, file_begin_idx=file_begin_idx)
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
            maximum_parallel_entering_exiting_number=int(config['MansionInfo']['ParallelEnterNum']),
            maximum_capacity=int(config['MansionInfo']['MaximumCapacity']),
            # start_time=900
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
        self.waiting_times = []
        self.forbid_unrequired = forbid_unrequired

        if seed is not None:
            self.seed(seed)
        self.seed_c = seed

        self.evaluate_info = {'valid_up_action': 0,
                              'advice_up_action': 0,
                              'valid_dn_action': 0,
                              'advice_dn_action': 0}


    # Implement by JY, compare smec and RL
    def step_smec(self):
        # test with batch align
        # person_list = self.mansion._person_generator.generate_person()
        # unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        # all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        # done = False
        # for up_floor in unallocated_up:
        #     if not self.mansion._person_generator.used_ele.empty():
        #         cur_elev = self.mansion._person_generator.used_ele.get()
        #     else:
        #         print(self.mansion._wait_upward_persons_queue[up_floor][-1].AppearTime)
        #         done = True
        #         cur_elev = self.mansion._wait_upward_persons_queue[up_floor][-1].StatisticElev
        #     all_elv_up_fs[cur_elev].append(up_floor)
        # for dn_floor in unallocated_dn:
        #     cur_elev = self.mansion._wait_downward_persons_queue[dn_floor][-1].StatisticElev
        #     all_elv_down_fs[cur_elev].append(dn_floor)

        # test with person align
        person_list = self.mansion._person_generator.generate_person()
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        for up_floor in unallocated_up:
            for pop_idx in range(len(self.mansion._wait_upward_persons_queue[up_floor]) - 1, -1, -1):
                cur_elev = self.mansion._wait_upward_persons_queue[up_floor][pop_idx].StatisticElev
                if up_floor not in all_elv_up_fs[cur_elev]:
                    all_elv_up_fs[cur_elev].append(up_floor)
                    break
        for dn_floor in unallocated_dn:
            for pop_idx in range(len(self.mansion._wait_downward_persons_queue[dn_floor]) - 1, -1, -1):
                cur_elev = self.mansion._wait_downward_persons_queue[dn_floor][pop_idx].StatisticElev
                if dn_floor not in all_elv_down_fs[cur_elev]:
                    all_elv_down_fs[cur_elev].append(dn_floor)
                    break

        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        # if self.open_render:
        #     time.sleep(0.01)  # for accelerate simulate speed
        calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute, person_list)

        done = self.mansion.is_done
        info = {'awt': awt}
        return None, None, done, info

    def seed(self, seed=None):
        set_seed(seed)



if __name__ == '__main__':
    # data_file ='E:\\elevator\\train_data\\up_peak_normal\\4Ele16FloorUpPeakFlow23_elvx.csv'
    data_file = '../train_data/up_peak_normal/4Ele16FloorUpPeakFlow28_elvx.csv'
    # data_file ='E:\\elevator\\person_flow_data\\1_elvx.csv'
    eval_env = SmecEnv(data_file=data_file, render=False)
    awt = []
    dt = eval_env._config._delta_t
    for time_step in range(int(3600 / dt) + 600):
        if eval_env.open_render:
            eval_env.viewer.view()
        # Observe reward and next obs
        _, _, done, info = eval_env.step_smec()
        awt += info['awt']

        if done:
            break

    print(awt)
    print('awt: ', np.mean(awt))
    waiting_time = []
    transmit_time = []
    for k in eval_env.mansion.person_info.keys():
        info = eval_env.mansion.person_info[k]
        # print(k, eval_env.mansion.person_info[k])
        print(k, end=' | ')
        print('ele %d' % (info[0] + 1), end=' | ')
        for p_t in info[1:]:
            print('%d : %.1f' % (p_t // 60, p_t % 60), end=' | ')
        try:
            waiting_time.append(info[2])
            transmit_time.append(info[3] - info[2] - info[1])
            print('%.1f %.1f' % (info[2], info[3] - info[2] - info[1]))
        except:
            pass

    print(
        f"[Evaluation] for {len(waiting_time)} people: mean waiting time {np.mean(waiting_time):.1f}, mean transmit time: {np.mean(transmit_time):.1f}.")

