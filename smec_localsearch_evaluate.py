import numpy as np
from copy import deepcopy
import time
import socket


class TimeCnt:
    def __init__(self):
        self.cnt = 0

    def step(self):
        self.cnt += 1

    def reset(self):
        self.cnt = 0

    def get_time(self, dt):
        return self.cnt * dt


class ElevInfo:
    def __init__(self, floor_height):
        self._hall_dn_call = []
        self._hall_up_call = []
        self._car_call = []

        self.door_state = 0
        self.floor_height = floor_height

        self._current_position = 0
        self._current_velocity = 0
        self._is_overloaded = False
        self._service_direction = 0

    def update(self, elev_info):
        self._hall_dn_call = deepcopy(elev_info[0])
        self._hall_up_call = deepcopy(elev_info[1])
        self._car_call = deepcopy(elev_info[2])

        self.door_state = elev_info[3]

        self._current_position = elev_info[4]
        self._current_velocity = elev_info[5]
        self._is_overloaded = elev_info[6]
        self._service_direction = elev_info[7]


class Mansion:
    def __init__(self, floor_num, elev_num, floor_height):
        self._elevators = [ElevInfo(floor_height) for i in range(elev_num)]
        self.cur_time = 0
        self.init_time = -1
        self.hallcall_last_serve_time = [0 for i in range(floor_num * 2)]
        self.hallcall_last_left_person = [0 for i in range(floor_num * 2)]
        self.floor_height = floor_height

    def update(self, info):
        self.cur_time = info[0]
        if self.init_time == -1:
            self.init_time = self.cur_time
        for idx, elev_info in enumerate(info[1]):
            self._elevators[idx].update(elev_info)
        for floor in info[2]:
            self.hallcall_last_serve_time[floor] = self.cur_time
            # TODO: 应该要c++那边传来这次接了多少人，这个是可以知道的。
            self.hallcall_last_left_person[floor] = max(0, self.hallcall_last_left_person[floor]-10)


    def print_info(self):
        for idx, elev_info in enumerate(self._elevators):
            print(idx, self._elevators[idx].__dict__, file=log_file)


class LocalSearch:
    def __init__(self, elev_num=4, floor_num=16, floor_height=3, mode=0):
        self.current_time = 0
        self.dt = 0.1
        self.mansion = Mansion(floor_num, elev_num, floor_height)
        self.modes = ['lunchpeak', 'notpeak', 'uppeak', 'dnpeak']
        self.mode = self.modes[mode]
        self.elev_num = elev_num
        self.floor_num = floor_num
        self.floor_height = floor_height

        self.candidate_schemes = []
        self.best_scheme = {'dispatch': [-1 for _ in range(self.floor_num * 2)], 'score': -10000}
        self.time_cnt = TimeCnt()

        self.upcall_weights = np.load('weights/weights_%s_upcall.npy' % self.mode)
        self.dncall_weights = np.load('weights/weights_%s_dncall.npy' % self.mode)
        self.carcall_weights = np.load('weights/weights_%s_carcall.npy' % self.mode)
        self.weight_t = 0

        # 记录每个电梯的分派对应时间，每次决策之后都要重置更新，可以加速
        self.elevator_dispatch_time_table = [{} for i in range(self.elev_num)]

        # 用于计算累积人数、权重
        self.updn_last_serve_time = []

    # 根据delta floor计算需要时间。一个更真实的运动估计。
    def df2time(self, df, elev=None):
        if df == 0:
            if elev.door_state == 1 or elev.door_state == 4:
                return 5.5
            elif elev.door_state == 2:
                return 2.9
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
        return hallcalls

    # hallcall是[0]是up，[1]是dn，都是0到16。
    def hallcall2key(self, hallcall):
        key = 0
        for call in hallcall[0]:
            key += 2 ** call
        for call in hallcall[1]:
            key += 2 ** (call + self.floor_num)
        return key

    def add_hallcall_to_elev(self, hallcall, elev):
        floor_num = self.floor_num
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
        first_stop_pos = first_stop_flr * self.floor_height
        # 如果第一落点是纯hallcall且满载了，那么这个电梯不能选
        if (state == 2 or state == 4) and elev._is_overloaded:
            return 10000
        cur_spd = elev._current_velocity
        df = abs(elev._current_position - first_stop_pos) / self.floor_height
        df = int(round(df))
        consume_time = max(self.df2time(df, elev) - cur_spd, 0)  # 观察经验公式
        return consume_time

    # 由楼层和状态（carcall、upcall、dncall）得到call的权重，可以由时间变化。
    def floor_state2weight(self, floor, state):
        # 1 car 2 up 4 dn
        weight = 0
        if state % 2 == 1:
            weight += self.carcall_weights[self.weight_t, floor]
        if state // 2 % 2 == 1:
            weight += self.upcall_weights[self.weight_t, floor] * self.updn_delta_time[floor] / 60 \
                      + self.mansion.hallcall_last_left_person[floor]
        if state // 4 % 2 == 1:
            weight += self.dncall_weights[self.weight_t, floor] * self.updn_delta_time[floor + self.floor_num] / 60 \
                      + self.mansion.hallcall_last_left_person[floor + self.floor_num]
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
    def get_elev_route(self, srv_dir, stp_flr, cur_pos, car_call, hall_up_dn_call):
        route = []
        # 正常来说，carcall只在r1
        f_1 = (srv_dir + 1) * self.floor_num // 2  # f when 1, 0 when -1
        f_m1 = (-srv_dir + 1) * self.floor_num // 2  # f when -1, 0 when 1
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
                if route == [] or (route[0][0] * self.floor_height - cur_pos) * srv_dir < 0:
                    route.insert(0, (stp_flr, 8))
        return route

    # 不用模拟，用两个delta floor的距离来近似计算，计算公式由实验得出。
    def estimate_elev_route_loss(self, elev, hall_up_call, hall_dn_call):
        copy_elev = deepcopy(elev)
        copy_elev._hall_up_call = hall_up_call
        copy_elev._hall_dn_call = hall_dn_call

        cur_pos = copy_elev._current_position
        cur_spd = copy_elev._current_velocity
        srv_dir = copy_elev._service_direction
        car_call = copy_elev._car_call
        hall_up_dn_call = [copy_elev._hall_up_call, copy_elev._hall_dn_call]
        stp_flr = self.cal_stop_floor(cur_pos, cur_spd, 0.557, 3.0)

        # 如果电梯之前是空闲的，可能分配了hallcall之后srv_dir也是0没来得及更新，先运行一个dt给他更新一下。
        # 电梯之前是空闲的，根据hallcall手动算一下srv dir?
        # TODO: Elevate上可能不会出现这个情况，再说吧。
        if srv_dir == 0:
            # print(f'srv dir is 0, hallcall:{hall_up_dn_call}')
            if hall_up_dn_call[0] + hall_up_dn_call[1] == []:
                return 0
            else:
                if hall_up_dn_call[0] != []:
                    call = hall_up_dn_call[0][0]
                    delta_dis = call * copy_elev.floor_height - cur_pos
                    if delta_dis < 0:
                        srv_dir = -1
                    else:
                        srv_dir = 1
                elif hall_up_dn_call[1] != []:
                    call = hall_up_dn_call[1][0]
                    delta_dis = call * copy_elev.floor_height - cur_pos
                    if delta_dis > 0:
                        srv_dir = 1
                    else:
                        srv_dir = -1
            # print(f'new srv dir is {srv_dir}')

        #         copy_elev.run_elevator()
        #         srv_dir = copy_elev._service_direction
        #         # print(hall_up_dn_call, cur_flr, cur_spd, srv_dir)

        route = self.get_elev_route(srv_dir, stp_flr, cur_pos, car_call, hall_up_dn_call)

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
        loss += accumulate_time * self.floor_state2weight(first_stop_flr, route[0][1])

        # 其他段路可以直接用实验公式计算。
        last_flr = first_stop_flr
        for stop_flr in route[1:]:
            df = abs(stop_flr[0] - last_flr)
            consume_time = self.df2time(df, copy_elev)
            accumulate_time += consume_time
            loss += accumulate_time * self.floor_state2weight(stop_flr[0], stop_flr[1])
            last_flr = stop_flr[0]
        return loss

    # 评估一种分配方案：
    # 求所有电梯loss的和。
    def evaluate_dispatch_faster(self, dispatch):
        total_loss = 0
        hallcalls = self.dispatch2hallcalls(dispatch)
        for idx in range(self.elev_num):
            elev_dispatch_key = self.hallcall2key(hallcalls[idx])
            if elev_dispatch_key in self.elevator_dispatch_time_table[idx].keys():
                loss = self.elevator_dispatch_time_table[idx][elev_dispatch_key]
            else:
                loss = self.estimate_elev_route_loss(self.mansion._elevators[idx], hallcalls[idx][0], hallcalls[idx][1])
                self.elevator_dispatch_time_table[idx][elev_dispatch_key] = loss
            total_loss += loss
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
            if score > best_score:
                best_score = score
                best_idx = elev_idx
                best_dispatch = new_dispatch
        return best_idx, best_score, best_dispatch

    def get_cur_dispatch(self):
        to_serve_calls = []
        dispatch = [-1 for _ in range(self.floor_num * 2)]  # -1表示没有hallcall
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
                print(f'Find better dispatch: {dispatch}, score: {score}', file=log_file)

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
            print(f'olddis: {self.print_dispatch(cur_dispatch)} score: {cur_score}', file=log_file)
            for sc in to_serve_calls:
                # 找出替换cur_dispatch[sc]的最小loss的电梯，用这个方法会比之前少很多search
                # print('exploit: ', cur_dispatch, sc)
                sc_other_best_idx, sc_other_best_score, sc_other_best_dispatch = self.get_elev_greedy(sc, cur_dispatch,
                                                                                                      cur_dispatch[sc])
                search_num += self.elev_num - 1
                if sc_other_best_score > best_scheme['score']:
                    better_schemes.append({'dispatch': sc_other_best_dispatch, 'score': sc_other_best_score})
                    best_scheme = {'dispatch': sc_other_best_dispatch,
                                   'score': sc_other_best_score}
                    print(f'better: {self.print_dispatch(sc_other_best_dispatch)}, score: {sc_other_best_score}', file=log_file)

        et = time.time()
        print(f'to serve: {len(to_serve_calls)}, search scheme: {scheme_num}, {search_num} times, {et - st:.2f}', file=log_file)
        self.best_scheme = best_scheme

    def exploit(self, to_serve_calls):
        self.exploit_nearest(to_serve_calls)

    def update_mansion(self, mansion, info):
        # self.current_time += self.dt
        # info[0] = self.current_time  # 传过来的
        mansion.update(info)

    # 返回的是一个所有未完成hallcall的一个分配，可能会改变颠覆之前的分配。
    # add_hallcalls 为0-32
    def get_action(self, add_hallcalls, info):
        self.update_mansion(self.mansion, info)
        cur_time = self.mansion.cur_time - self.mansion.init_time
        # print(cur_time)
        print(f'current time: {cur_time:.2f}', file=log_file)
        self.mansion.print_info()
        self.weight_t = int(cur_time // 60)
        if self.weight_t >= 60:
            self.weight_t = 59
        self.elevator_dispatch_time_table = [{} for i in range(self.elev_num)]
        self.updn_delta_time = [self.mansion.cur_time - last_time for last_time in self.mansion.hallcall_last_serve_time]
        # 先针对当前新加的hallcalls贪心，这会改变mansion的东西，每次都实时传？没事，改就改吧？
        cur_dispatch, to_serve_calls = self.get_cur_dispatch()
        print('add hallcall:', add_hallcalls)
        print('old srv: ', to_serve_calls)
        for ah in add_hallcalls:
            choose_elev_idx, score, cur_dispatch = self.get_elev_greedy(ah, cur_dispatch)
            self.add_hallcall_to_elev(ah, self.mansion._elevators[choose_elev_idx])

        # 把当前的贪心的分派记作最优方案
        cur_dispatch, to_serve_calls = self.get_cur_dispatch()
        print('new srv: ', to_serve_calls)
        self.best_scheme['dispatch'] = cur_dispatch
        self.best_scheme['score'] = self.evaluate_dispatch(cur_dispatch)

        # 选择合适算法基于当前情况寻找更优解。
        self.exploit(to_serve_calls)
        print(f'to serve call: {to_serve_calls}', file=log_file)
        print(f'best dispatch: {self.print_dispatch(self.best_scheme["dispatch"])}', file=log_file)

        return self.best_scheme['dispatch']


def handler(solver):
    # get data from file
    curr_time, elevates, request_combined = [], [], []
    with open('E:\\tmpdata\\data.txt', 'r') as fp:

        # current time
        curr_time = list(map(float, fp.readline().split()))[0]
        for i in range(solver.elev_num):
            down_call, up_call, car_call, door_status, current_position, current_velocity, is_overloaded, direction = [], [], [], [], [], [], [], []

            # down_call
            floor = list(map(int, fp.readline().split()))
            down_call = deepcopy(floor)

            # up_call
            floor = list(map(int, fp.readline().split()))
            up_call = deepcopy(floor)

            # car_call
            floor = list(map(int, fp.readline().split()))
            car_call = deepcopy(floor)

            # door_status
            door_status = list(map(int, fp.readline().split()))[0]

            # current_position
            current_position = list(map(float, fp.readline().split()))[0]

            # current_velocity
            current_velocity = list(map(float, fp.readline().split()))[0]

            # is_overloaded
            is_overloaded = list(map(int, fp.readline().split()))[0]

            # direction
            direction = list(map(int, fp.readline().split()))[0]

            elevate = [down_call, up_call, car_call, door_status, current_position, current_velocity, is_overloaded,
                       direction]
            elevates.append(deepcopy(elevate))

        # up_request
        up_request = list(map(int, fp.readline().split()))
        print('up req: ', up_request)

        # down_request
        down_request = list(map(int, fp.readline().split()))
        print('dn req: ', down_request)

        # request_combined
        request_combined = up_request
        for req in down_request:
            request_combined.append(req + solver.floor_num)

        # up_finish
        up_finished = list(map(int, fp.readline().split()))

        # down_finish
        down_finished = list(map(int, fp.readline().split()))

        # finished_combined
        finished_combined = up_finished
        for finish in down_finished:
            finished_combined.append(finish + solver.floor_num)

        fp.close()

    # get action
    action = solver.get_action(request_combined, [curr_time, elevates, finished_combined])

    # write to file
    with open('E:\\tmpdata\\action.txt', 'w') as fp:
        for i in range(solver.floor_num * 2):
            str_tmp = str(action[i]) + ' '
            fp.write(str_tmp)
        fp.close()


if __name__ == '__main__':
    # solver = LocalSearch(elev_num=4, floor_num=16)
    log_file = open('./logs/12-20_4.log', 'a')
    print('+'*100, file=log_file)
    solver = LocalSearch(elev_num=4, floor_num=16, floor_height=3)

    # socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 2077))
    server.listen(0)
    # try:
    while True:
        print('waiting for instruction...', file=log_file)
        conn, address = server.accept()

        # receive
        recv_str = conn.recv(30)[0:5]
        recv_str = recv_str.decode("ascii")

        # handle
        handler(solver)

        # send
        conn.send(bytes("finished", encoding="ascii"))
        conn.close()
    # except:
    #     log_file.close()
