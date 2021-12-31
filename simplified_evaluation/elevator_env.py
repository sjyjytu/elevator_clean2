import random
import numpy as np

import os
# elevator_num = 4
# floor_num = 10
# person_num = 50
#
# file_path = 'person_info.npy'


class Elev:
    def __init__(self, floor_num=10):
        self.pos = 0
        self.serve_dir = 0
        self.run_dir = 0
        self.up_call = [0 for i in range(floor_num)]
        self.dn_call = [0 for i in range(floor_num)]
        self.car_call = [0 for i in range(floor_num)]
        self.load_pid_per_floor = [[] for i in range(floor_num)]


class Mansion:
    def __init__(self, floor_num=10):
        self.up_waiting_personid = [[] for i in range(floor_num)]
        self.dn_waiting_personid = [[] for i in range(floor_num)]


class Env:
    def __init__(self, elevator_num=4, floor_num=10, person_num=50, file_path='person_info.npy'):
        self.elevator_num = elevator_num
        self.floor_num = floor_num
        self.person_num = person_num
        self.file_path = file_path
        self.person_info = []
        self.Elevs = [Elev() for i in range(self.elevator_num)]
        self.mansion = Mansion()
        self.cur_pidx = 0
        self.time_cnt = 0

    def reset(self):
        self.person_info = []
        self.Elevs = [Elev() for i in range(self.elevator_num)]
        self.mansion = Mansion()
        self.cur_pidx = 0
        self.generate_or_load_person()
        self.time_cnt = self.person_info[0][0]

    def generate_or_load_person(self, force_generate=False):
        if force_generate or not os.path.exists(self.file_path):
            self.person_info = [[2, 1, 5]]
            for p in range(self.person_num - 1):
                src = random.randint(0, self.floor_num - 1)
                dst = random.randint(0, self.floor_num - 1)
                while dst == src:
                    dst = random.randint(0, self.floor_num - 1)
                self.person_info.append([random.randint(1, 10) + self.person_info[-1][0], src, dst])
            np.save(self.file_path, self.person_info)
        else:
            self.person_info = np.load(self.file_path).tolist()
            self.person_num = len(self.person_info)
        # print(self.person_info)

    def get_state(self):
        state = []
        for elev in self.Elevs:
            state = state + [elev.pos, elev.run_dir, elev.serve_dir]
            state = state + elev.up_call + elev.dn_call + elev.car_call
        state = state + [sum(i) for i in self.mansion.up_waiting_personid]
        state = state + [sum(i) for i in self.mansion.dn_waiting_personid]
        return state

    def loading_person(self):
        pass

    def unloading_person(self, elev, cur_time):
        if elev.car_call[elev.pos]:
            elev.car_call[elev.pos] = 0
            for p in elev.load_pid_per_floor[elev.pos]:
                self.person_info[p].append(cur_time)
            elev.load_pid_per_floor[elev.pos] = []

    def update(self, time):
        for elev_idx, elev in enumerate(self.Elevs):
            self_time = time
            while self_time > 0:
                self_time -= 1
                # 运动模拟
                if elev.serve_dir >= 0:
                    if elev.run_dir > 0:
                        elev.pos += 1  # 每秒走一层
                        if elev.car_call[elev.pos] or elev.up_call[elev.pos]:
                            elev.run_dir = 0
                        elif elev.dn_call[elev.pos] and sum(elev.car_call[elev.pos:]) + sum(elev.up_call[elev.pos:]) == 0:
                            elev.run_dir = 0
                    elif elev.run_dir == 0:
                        self.unloading_person(elev, self.time_cnt + time - self_time)

                        # 改变elev的运动状态，
                        if sum(elev.car_call[elev.pos+1:]) + sum((elev.up_call[elev.pos:])) + sum((elev.dn_call[elev.pos+1:])) > 0:
                            # 接着往上走
                            elev.serve_dir = 1
                            elev.run_dir = 1
                        elif sum(elev.car_call[:elev.pos]) + sum((elev.up_call[:elev.pos])) + sum((elev.dn_call[:elev.pos+1])) > 0:
                            elev.serve_dir = -1
                            elev.run_dir = -1
                        else:
                            elev.serve_dir = 0

                        if elev.serve_dir > 0 and elev.up_call[elev.pos]:
                            # 上客
                            elev.up_call[elev.pos] = 0
                            for p in self.mansion.up_waiting_personid[elev.pos]:
                                elev.car_call[self.person_info[p][2]] = 1
                                elev.load_pid_per_floor[self.person_info[p][2]].append(p)
                                self.person_info[p].append(self.time_cnt + time - self_time)
                            self.mansion.up_waiting_personid[elev.pos] = []
                        if elev.serve_dir < 0 and elev.dn_call[elev.pos]:
                            # 上客
                            elev.dn_call[elev.pos] = 0
                            for p in self.mansion.dn_waiting_personid[elev.pos]:
                                elev.car_call[self.person_info[p][2]] = 1
                                elev.load_pid_per_floor[self.person_info[p][2]].append(p)
                                self.person_info[p].append(self.time_cnt + time - self_time)
                            self.mansion.dn_waiting_personid[elev.pos] = []
                elif elev.serve_dir < 0:
                    if elev.run_dir < 0:
                        elev.pos -= 1  # 每秒走一层
                        # print(elev.pos)
                        if elev.car_call[elev.pos] or elev.dn_call[elev.pos]:
                            elev.run_dir = 0
                        elif elev.up_call[elev.pos] and sum(elev.car_call[:elev.pos+1]) + sum(elev.dn_call[:elev.pos+1]) == 0:
                            elev.run_dir = 0
                    elif elev.run_dir == 0:
                        self.unloading_person(elev, self.time_cnt + time - self_time)

                        # 改变elev的运动状态，
                        if sum(elev.car_call[:elev.pos]) + sum((elev.up_call[:elev.pos])) + sum(
                                (elev.dn_call[:elev.pos+1])) > 0:
                            elev.serve_dir = -1
                            elev.run_dir = -1
                        elif sum(elev.car_call[elev.pos:]) + sum((elev.up_call[elev.pos:])) + sum(
                                (elev.dn_call[elev.pos+1:])) > 0:
                            # 接着往上走
                            elev.serve_dir = 1
                            elev.run_dir = 1
                        else:
                            elev.serve_dir = 0

                        if elev.serve_dir > 0 and elev.up_call[elev.pos]:
                            # 上客
                            elev.up_call[elev.pos] = 0
                            for p in self.mansion.up_waiting_personid[elev.pos]:
                                elev.car_call[self.person_info[p][2]] = 1
                                elev.load_pid_per_floor[self.person_info[p][2]].append(p)
                                self.person_info[p].append(self.time_cnt + time - self_time)
                            self.mansion.up_waiting_personid[elev.pos] = []
                        if elev.serve_dir < 0 and elev.dn_call[elev.pos]:
                            # 上客
                            elev.dn_call[elev.pos] = 0
                            for p in self.mansion.dn_waiting_personid[elev.pos]:
                                elev.car_call[self.person_info[p][2]] = 1
                                elev.load_pid_per_floor[self.person_info[p][2]].append(p)
                                self.person_info[p].append(self.time_cnt + time - self_time)
                            self.mansion.dn_waiting_personid[elev.pos] = []

    def step(self, elev_idx):
        # 1. 每个函数只要做一件事情
        # 2、搞一些测试样例+可视化
        # 3、参考一下华为的，实现一下搜索算法。 复杂度、实现细节：分情况、不同种订单分类处理、类是怎么写；非法车是怎么禁用的；延时决策；优化目标
        # 4、以order为单位，参考一下华为的
        # s,a -> st+1    s,a -> st+1
        # sparse/various singularity

        # 为此乘客选择电梯
        pinfo = self.person_info[self.cur_pidx]
        src, dst = pinfo[1:3]
        is_up = src < dst
        # 更新电梯、大楼状态
        if is_up:
            # 防止抢电梯
            not_allocate = True
            for elev in self.Elevs:
                if elev.up_call[src]:
                    not_allocate = False
            if not_allocate:
                self.Elevs[elev_idx].up_call[src] = 1
            self.mansion.up_waiting_personid[src].append(self.cur_pidx)
        else:
            # 防止抢电梯
            not_allocate = True
            for elev in self.Elevs:
                if elev.dn_call[src]:
                    not_allocate = False
            if not_allocate:
                self.Elevs[elev_idx].dn_call[src] = 1
            self.mansion.dn_waiting_personid[src].append(self.cur_pidx)

        # 执行动作并更新到下一乘客到来前的时间节点
        if self.cur_pidx == len(self.person_info)-1:
            self.update(60)
            # 统计所有人的awt，att
            total_waiting_time = 0
            total_transmit_time = 0
            for p in self.person_info:
                total_waiting_time += p[3] - p[0]
                total_transmit_time += p[4] - p[0]
            self.cur_pidx += 1
            done = True
            return done, total_waiting_time / self.person_num, total_transmit_time / self.person_num
        else:
            next_person_time = self.person_info[self.cur_pidx + 1][0]
            pass_time = next_person_time - pinfo[0]
            # 更新一下各电梯经过过去的时间之后的状态。
            # 更新电梯至self.cur_pidx+1的出现时间。
            self.update(pass_time)
            self.time_cnt += pass_time
            self.cur_pidx += 1
            done = False
            return done, 0, 0

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


if __name__ == '__main__':
    # 测试一下随机算法
    elev_env = Env(elevator_num=2, floor_num=5, person_num=20)
    elev_env.reset()
    while not elev_env.is_end():
        action = random.randint(0, elev_env.elevator_num-1)
        elev_env.step(action)
        print('Choosing %d for the %d person' % (action, elev_env.cur_pidx))
    print(elev_env.get_reward())  # (1.65, 5.2)



