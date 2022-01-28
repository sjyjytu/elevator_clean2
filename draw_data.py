import os
import numpy as np
from csv import reader, writer
import matplotlib.pyplot as plt
import numpy as np

dt = 60
t = 3600
iter_num = t // dt
floor_num = 16


def process(data_dir='./down_data'):
    data_x = []
    data_y = []
    max_value = 0.0
    files = os.listdir(data_dir)
    for file in files:
        data_file = os.path.join(data_dir, file)
        start_data = [[0 for _ in range(floor_num)] for i in range(iter_num)]
        end_data = [[0 for _ in range(floor_num)] for i in range(iter_num)]
        with open(data_file, 'rt', encoding='utf8', errors='ignore') as f:
            r = reader(f)
            row = None
            for i in range(274):
                row = next(r)
            while True:
                time, start_level, end_level = row[0], int(row[1].replace('Level ', '')) - 1, int(
                    row[2].replace('Level ', '')) - 1
                next_person_time = time.split(':')
                nt = int(next_person_time[-2]) * 60 + float(next_person_time[-1])
                # print(nt, int(nt)//dt)
                start_data[int(nt) // dt][start_level] += 1
                end_data[int(nt) // dt][end_level] += 1
                row = next(r)
                if len(row) < 14 or row[0] == '':
                    break

        # print(start_data)
        tmp_max = np.max(start_data)
        if tmp_max > max_value:
            max_value = tmp_max
        data_x.append(start_data)
    # return np.asarray(data_x) / max_value, np.asarray(data_y) / max_value
    return np.asarray(data_x), np.asarray(data_y)


def draw_one():
    figure_dir = 'file_figures'
    # dataX, _ = process(data_dir='./train_data/up_peak_normal')
    dataX, _ = process(data_dir='train_data/new/lunchpeak')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.figure()
    # plt.title('31天数据中以第1层为出发层的平均人数与时间的关系')
    floor = 8
    plt.title(f'各天数据中以第{floor}层为出发层的人数与时间的关系')
    # plotX = dataX[:,:,0]
    plotX = dataX[:, 0:, floor - 1]
    x = range(0, 0 + plotX.shape[1])
    for i in range(0, plotX.shape[0], 1):
        plt.plot(x, plotX[i], label=f'第{i}天', linestyle=':')
    plotX_mean = np.mean(plotX, axis=0)
    plt.plot(x, plotX_mean, label='平均值', color='red')
    plt.legend(ncol=4)
    plt.xlabel('时间（分钟）')
    plt.ylabel('出现人数')
    # plt.savefig(f'{figure_dir}/30-40分钟中{floor}层出现人数.jpg')
    plt.savefig(f'{figure_dir}/无高峰{floor}层平均出现人数.jpg')
    plt.show()


def process2(data_dir='./down_data'):
    data_x = []
    max_value = 0.0
    files = os.listdir(data_dir)
    for file in files:
        data_file = os.path.join(data_dir, file)
        if not data_file.endswith('.csv'):
            continue
        f_t_s_d = [[[0 for k in range(floor_num)] for j in range(floor_num)] for i in range(iter_num)]
        with open(data_file, 'rt', encoding='utf8', errors='ignore') as f:
            r = reader(f)
            row = None
            while True:
                if row != None and row != [] and row[0] == 'PASSENGER LIST':
                    break
                row = next(r)
            for i in range(10):
                row = next(r)
            while True:
                time, start_level, end_level = row[0], int(row[1].replace('Level ', '')) - 1, int(
                    row[2].replace('Level ', '')) - 1
                next_person_time = time.split(':')
                nt = int(next_person_time[-2]) * 60 + float(next_person_time[-1])
                # print(nt, int(nt)//dt)
                f_t_s_d[int(nt) // dt][start_level][end_level] += 1
                row = next(r)
                if len(row) < 14 or row[0] == '':
                    break

        # print(start_data)
        tmp_max = np.max(f_t_s_d)
        if tmp_max > max_value:
            max_value = tmp_max
        data_x.append(f_t_s_d)
    # return np.asarray(data_x) / max_value, np.asarray(data_y) / max_value
    return np.asarray(data_x)


def draw_whole():
    import pandas as pd

    figure_dir = 'file_figures'
    data_dirs = ['dnpeak', 'notpeak', 'lunchpeak', 'uppeak']
    data_idx = 2
    dataX = process2(data_dir='./train_data/new/%s' % data_dirs[data_idx])
    # print(dataX)

    from mpl_toolkits.mplot3d import Axes3D
    _x = np.arange(0, 16)
    _y = np.arange(0, 16)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    fileid = 0
    # dataX = dataX[fileid]  # 指定文件
    dataX = np.average(dataX, axis=0)  # 均值
    print(dataX)
    # z = np.sum(z, axis=0)  # 查看总人数
    for t in range(dataX.shape[0]):
        z = dataX[t]
        z = z.flatten()
        # print(z)
        bottom = np.zeros_like(z)
        width = depth = 1  # x,y方向的宽厚
        fig = plt.figure(figsize=(10, 5))  # 画布宽长比例
        ax1 = fig.add_subplot(111, projection='3d')
        # ax2 = fig.add_subplot(122, projection='3d')
        ax1.set_title(f'{data_dirs[data_idx]} t={t}')
        # ax2.set_title("colored")
        ax1.bar3d(x, y, bottom, width, depth, z, shade=True)  # x，y为数组
        ax1.set_xlabel('end')
        ax1.set_ylabel('start')
        ax1.set_zlabel('person num')
        plt.show()


def print_hallcall_along_time(data_dir_prefix='./train_data/new/', data_idx=2, fileid=0):
    data_dirs = ['dnpeak', 'notpeak', 'lunchpeak', 'uppeak']
    data_dir = data_dir_prefix + data_dirs[data_idx]
    dataX = process2(data_dir=data_dir)
    # fileid = 0
    dataX = dataX[fileid]  # 指定文件
    # dataX = np.average(dataX, axis=0)  # 均值
    # z = np.sum(z, axis=0)  # 查看总人数
    upcalls = [[0 for i in range(16)] for t in range(60)]
    dncalls = [[0 for i in range(16)] for t in range(60)]
    carcalls = [[0 for i in range(16)] for t in range(60)]
    print(dataX.shape)
    for t in range(60):
        for sf in range(16):
            for ef in range(16):
                if sf == ef:
                    continue
                if sf < ef:
                    # 上行
                    upcalls[t][sf] += dataX[t, sf, ef]
                else:
                    dncalls[t][sf] += dataX[t, sf, ef]
                carcalls[t][ef] += dataX[t, sf, ef]

        upcalls[t] = [round(h, 2) for h in upcalls[t]]
        dncalls[t] = [round(h, 2) for h in dncalls[t]]
        carcalls[t] = [round(h, 2) for h in carcalls[t]]
    np.save('weights_%s_upcall.npy' % data_dirs[data_idx], upcalls)
    np.save('weights_%s_dncall.npy' % data_dirs[data_idx], dncalls)
    np.save('weights_%s_carcall.npy' % data_dirs[data_idx], carcalls)
    return upcalls, dncalls, carcalls


def print_flow_map(data_dir_prefix='./train_data/new/', data_idx=2, fileid=0):
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    data_dirs = ['dnpeak', 'notpeak', 'lunchpeak', 'uppeak']
    data_dir = data_dir_prefix + data_dirs[data_idx]
    dataX = process2(data_dir=data_dir)
    dataX = dataX[fileid]  # 指定文件
    # dataX = np.average(dataX, axis=0)  # 均值
    # z = np.sum(z, axis=0)  # 查看总人数
    data = np.sum(dataX, axis=0)
    start_sum = np.sum(data, axis=1, keepdims=True)
    # end_sum = np.sum(data, axis=0, keepdims=True)
    print(data)

    new_data = np.concatenate([data, start_sum], axis=1)

    print('已知出发层，预测到达层的概率：')
    print(data / start_sum)

    # print(data)

    # 应该上行和下行分开算？
    separate_post_prob = np.zeros_like(data, dtype=np.float32)
    for src in range(separate_post_prob.shape[0]):
        dn = data[src][:src]
        dn_sum = np.sum(dn)
        up = data[src][src:]
        up_sum = np.sum(up)
        for dst in range(separate_post_prob.shape[1]):
            if dst < src:
                if dn_sum != 0:
                    prob = float(data[src][dst]) / dn_sum
                    separate_post_prob[src][dst] = prob
            else:
                if up_sum != 0:
                    prob = float(data[src][dst]) / up_sum
                    separate_post_prob[src][dst] = prob
    print('已知出发层，预测到达层的概率（上下行分开）：')
    print(separate_post_prob)
    np.save('smec_ml2/separate_post_prob.npy', separate_post_prob)


    end_sum = np.sum(new_data, axis=0, keepdims=True)
    new_data = np.concatenate([new_data, end_sum], axis=0)

    # print(new_data)
    print('出发层到到达层的独立概率：')
    print(new_data/60)
    np.save('smec_ml2/independent_flow_map.npy', data / 60)

    # print(data)
    # print(start_sum)
    # print(end_sum)


if __name__ == '__main__':

    draw_whole()
# draw_one()
# for i in range(4):
#     upcalls, dncalls, carcalls = print_hallcall_along_time(i)
#     print(upcalls)
#     print(dncalls)
#     print(carcalls)
# print_hallcall_along_time()

# res = np.load('16floor_weights.npy')
# print(res)

# print_flow_map()