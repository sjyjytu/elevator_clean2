import random
import numpy as np

import os


class Elev:
    def __init__(self, floor_num=10):
        self.pos = 0
        self.serve_dir = 0
        self.run_dir = 0
        self.is_unloading = 0
        self.is_loading = 0
        self.hall_call = {'up': [0 for i in range(floor_num)], 'dn': [0 for i in range(floor_num)]}
        self.car_call = [0 for i in range(floor_num)]
        self.load_pid_per_floor = [[] for i in range(floor_num)]


class Mansion:
    def __init__(self, floor_num=10):
        self.hall_waiting_personid = {'up': [[] for i in range(floor_num)], 'dn': [[] for i in range(floor_num)]}


class Env:
    def __init__(self, elevator_num=4, floor_num=10, person_num=50, file_path='person_info.npy'):
        self.elevator_num = elevator_num
        self.floor_num = floor_num
        self.person_num = person_num
        self.file_path = file_path
        self.person_info = []
        self.Elevs = [Elev(self.floor_num) for i in range(self.elevator_num)]
        self.direct2serve_dir = {'up': 1, 'dn': -1, 'stop': 0}
        self.mansion = Mansion(self.floor_num)
        self.cur_pidx = 0
        self.time_cnt = 0

    def reset(self, force_generate=False, mode='normal'):
        self.person_info = []
        self.Elevs = [Elev(self.floor_num) for i in range(self.elevator_num)]
        self.mansion = Mansion(self.floor_num)
        self.cur_pidx = 0
        self.generate_or_load_person(force_generate, mode)
        self.time_cnt = self.person_info[0][0]

    def generate_or_load_person(self, force_generate=False, mode='normal'):
        if force_generate or not os.path.exists(self.file_path):
            self.person_info = [[0, 0, 0]]
            if mode == 'up_peak':
                src_dst = {'src_begin': 0, 'src_end': 0, 'dst_begin': 1, 'dst_end': self.floor_num - 1}
            elif mode == 'dn_peak':
                src_dst = {'src_begin': 1, 'src_end': self.floor_num - 1, 'dst_begin': 0, 'dst_end': 0}
            else:
                src_dst = {'src_begin': 0, 'src_end': self.floor_num - 1, 'dst_begin': 0, 'dst_end': self.floor_num - 1}
            for p in range(self.person_num):
                src = random.randint(src_dst['src_begin'], src_dst['src_end'])
                dst = random.randint(src_dst['dst_begin'], src_dst['dst_end'])
                while dst == src:
                    dst = random.randint(0, self.floor_num - 1)
                self.person_info.append([random.randint(1, 2) + self.person_info[-1][0], src, dst])
            self.person_info = self.person_info[1:]
            print(self.person_info)
            np.save(self.file_path, self.person_info)
        else:
            self.person_info = np.load(self.file_path).tolist()
            self.person_num = len(self.person_info)
        # print(self.person_info)

    def get_state(self):
        state = []
        for elev in self.Elevs:
            state = state + [elev.pos, elev.run_dir, elev.serve_dir]
            state = state + elev.hall_call['up'] + elev.hall_call['dn'] + elev.car_call
        state = state + [sum(i) for i in self.mansion.hall_waiting_personid['up']]
        state = state + [sum(i) for i in self.mansion.hall_waiting_personid['dn']]
        return state

    def loading_person(self, elev, cur_time):
        hall_call_direct = ''
        if elev.serve_dir > 0 and elev.hall_call['up'][elev.pos]:
            hall_call_direct = 'up'
        elif elev.serve_dir < 0 and elev.hall_call['dn'][elev.pos]:
            hall_call_direct = 'dn'
        elif elev.serve_dir == 0:
            if elev.hall_call['up'][elev.pos]:
                hall_call_direct = 'up'
            elif elev.hall_call['dn'][elev.pos]:
                hall_call_direct = 'dn'

        direct = 'up' if elev.serve_dir > 0 else 'dn'
        if hall_call_direct != '':
            elev.hall_call[hall_call_direct][elev.pos] = 0
            elev.is_loading = 1
            direct = hall_call_direct

        if elev.is_loading:
            if len(self.mansion.hall_waiting_personid[direct][elev.pos]) == 0:
                elev.is_loading = 0
            else:
                elev.is_loading = 1
                loading_personid = self.mansion.hall_waiting_personid[direct][elev.pos][0]
                # print('load person', loading_personid, self.person_info[loading_personid])
                elev.car_call[self.person_info[loading_personid][2]] = 1
                elev.load_pid_per_floor[self.person_info[loading_personid][2]].append(loading_personid)
                self.person_info[loading_personid].append(cur_time)
                self.mansion.hall_waiting_personid[direct][elev.pos] = self.mansion.hall_waiting_personid[direct][elev.pos][1:]
        # for p in self.mansion.hall_waiting_personid[direct][elev.pos]:
        #     elev.car_call[self.person_info[p][2]] = 1
        #     elev.load_pid_per_floor[self.person_info[p][2]].append(p)
        #     self.person_info[p].append(cur_time)
        # self.mansion.hall_waiting_personid[direct][elev.pos] = []

    def unloading_person(self, elev, cur_time):
        if elev.car_call[elev.pos] or elev.is_unloading:
            elev.car_call[elev.pos] = 0
            if len(elev.load_pid_per_floor[elev.pos]) == 0:
                elev.is_unloading = 0
            else:
                elev.is_unloading = 1
                unloading_personid = elev.load_pid_per_floor[elev.pos][0]
                self.person_info[unloading_personid].append(cur_time)
                elev.load_pid_per_floor[elev.pos] = elev.load_pid_per_floor[elev.pos][1:]
            # for p in elev.load_pid_per_floor[elev.pos]:
            #     self.person_info[p].append(cur_time)
            # elev.load_pid_per_floor[elev.pos] = []

    def running_elev(self, elev, direct):
        if direct == 'up':
            elev.pos += 1
            oppo_direct = 'dn'
            check_begin_pos = elev.pos
            check_end_pos = self.floor_num
        else:
            elev.pos += -1
            oppo_direct = 'up'
            check_begin_pos = 0
            check_end_pos = elev.pos+1
        # print(elev.pos)
        # if elev.pos <= 1 or elev.pos >= 3:
        # self.print_render()
        if elev.car_call[elev.pos] or elev.hall_call[direct][elev.pos]:
            elev.run_dir = 0
        elif elev.hall_call[oppo_direct][elev.pos] and sum(elev.car_call[check_begin_pos:check_end_pos]) + sum(
                elev.hall_call[direct][check_begin_pos:check_end_pos]) == 0:
            elev.run_dir = 0

    def determine_serve_dir(self, elev, direct):
        if direct == 'up':
            if sum(elev.car_call[elev.pos + 1:]) + sum((elev.hall_call['up'][elev.pos:])) + sum(
                    (elev.hall_call['dn'][elev.pos + 1:])) > 0:
                # 接着往上走
                elev.serve_dir = 1
                # elev.run_dir = 1
            elif sum(elev.car_call[:elev.pos]) + sum((elev.hall_call['up'][:elev.pos])) + sum(
                    (elev.hall_call['dn'][:elev.pos + 1])) > 0:
                elev.serve_dir = -1
                # elev.run_dir = -1
            else:
                elev.serve_dir = 0
        else:
            if sum(elev.car_call[:elev.pos]) + sum((elev.hall_call['up'][:elev.pos])) + sum(
                    (elev.hall_call['dn'][:elev.pos + 1])) > 0:
                elev.serve_dir = -1
                # elev.run_dir = -1
            elif sum(elev.car_call[elev.pos + 1:]) + sum((elev.hall_call['up'][elev.pos:])) + sum(
                    (elev.hall_call['dn'][elev.pos + 1:])) > 0:
                elev.serve_dir = 1
                # elev.run_dir = 1
            else:
                elev.serve_dir = 0

    def update_for_interval(self, time):
        for elev_idx, elev in enumerate(self.Elevs):
            self_time = 0
            while self_time < time:
                self_time += 1
                # 运动模拟
                direct = 'up' if elev.serve_dir >= 0 else 'dn'
                if elev.run_dir != 0:
                    self.running_elev(elev, direct)
                else:
                    self.unloading_person(elev, self.time_cnt + self_time)
                    if not elev.is_unloading:
                        self.loading_person(elev, self.time_cnt + self_time)
                        if not elev.is_loading:
                            self.determine_serve_dir(elev, direct)
                            self.loading_person(elev, self.time_cnt + self_time)
                        if not elev.is_loading:
                            elev.run_dir = elev.serve_dir
                # self.print_render(self.time_cnt + self_time, elev_idx)

    def is_allocated(self, floor, direct):
        for elev in self.Elevs:
            if elev.hall_call[direct][floor] or \
                    (self.direct2serve_dir[direct] == elev.serve_dir and elev.pos==floor and elev.is_loading):
                return True
        return False

    def add_current_person_to_mansion(self):
        src, dst = self.person_info[self.cur_pidx][1:3]
        direct = 'up' if src < dst else 'dn'
        self.mansion.hall_waiting_personid[direct][src].append(self.cur_pidx)

    def exec_action_for_current_person(self, elev_idx):
        src, dst = self.person_info[self.cur_pidx][1:3]
        direct = 'up' if src < dst else 'dn'
        # 更新电梯、大楼状态
        if not self.is_allocated(src, direct):
            self.Elevs[elev_idx].hall_call[direct][src] = 1

    def update_until_next_person(self):
        if self.cur_pidx == len(self.person_info)-1:
            # 结束，再更新60s
            self.update_for_interval(60)
            self.cur_pidx += 1
        else:
            # 更新电梯至self.cur_pidx+1的出现时间。
            pass_time = self.person_info[self.cur_pidx + 1][0] - self.person_info[self.cur_pidx][0]
            self.update_for_interval(pass_time)
            self.time_cnt += pass_time
            self.cur_pidx += 1

    def step(self, elev_idx):
        # 1. 每个函数只要做一件事情
        # 2、搞一些测试样例+可视化
        # 3、参考一下华为的，实现一下搜索算法。 复杂度、实现细节：分情况、不同种订单分类处理、类是怎么写；非法车是怎么禁用的；延时决策；优化目标
        # 4、以order为单位，参考一下华为的
        # s,a -> st+1    s,a -> st+1
        # sparse/various singularity
        self.add_current_person_to_mansion()
        self.exec_action_for_current_person(elev_idx)
        self.update_until_next_person()

    def is_end(self):
        return self.cur_pidx == len(self.person_info)

    def get_reward(self):
        if not self.is_end():
            return 0, 0
        total_waiting_time = 0
        total_transmit_time = 0
        for p in self.person_info:
            total_waiting_time += p[3] - p[0]
            total_transmit_time += p[4] - p[0]
        return total_waiting_time / self.person_num, total_transmit_time / self.person_num

    def print_render(self, time=None, updating_elev=-1):
        if time is None:
            time = self.time_cnt
        print('Cur Time: %d, updating elev: %d' % (time, updating_elev))
        for idx, elev in enumerate(self.Elevs):
            print(idx, elev.car_call, elev.hall_call, elev.run_dir, elev.pos, elev.is_loading)
        for i in range(self.floor_num-1, -1, -1):
            print(len(self.mansion.hall_waiting_personid['up'][i]), end='  ')
            for elev in self.Elevs:
                if elev.pos == i:
                    if elev.serve_dir > 0:
                        print('u', end=' ')
                    elif elev.serve_dir < 0:
                        print('d', end=' ')
                    else:
                        print('*', end=' ')
                else:
                    print('-', end=' ')
            print(len(self.mansion.hall_waiting_personid['dn'][i]))


if __name__ == '__main__':
    import os

    # 测试一下随机算法
    elev_env = Env(elevator_num=2, floor_num=5, person_num=20)
    elev_env.reset(False)
    while not elev_env.is_end():
        action = random.randint(0, elev_env.elevator_num-1)
        # elev_env.step(action)
        elev_env.add_current_person_to_mansion()
        print('Choosing %d for the %d person at floor %d going to floor %d'
              % (action, elev_env.cur_pidx, elev_env.person_info[elev_env.cur_pidx][1],
                 elev_env.person_info[elev_env.cur_pidx][2]))
        # elev_env.print_render()
        elev_env.exec_action_for_current_person(action)
        elev_env.update_until_next_person()
        print()
    # print(elev_env.person_info)
    print(elev_env.get_reward())  # (1.65, 5.2)

# s, a -> s_{t+1}
# s, a -> s_{t+1} r

