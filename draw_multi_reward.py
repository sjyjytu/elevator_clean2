#
# # exp_names = ['feb08_down_random6-12_attention', 'feb08_lunch_random0-6_attention']
# # exp_names = ['feb08_lunch_random0-6_attention', 'feb08_down_random6-12_attention']
# # exp_names = ['feb02_down_random6-12', 'feb02_down_random6-12']
# exp_names = ['feb08_down_random6-12_attention', 'feb08_down_random6-12_attention']
# legend_names = ['Average sum time', 'Energy consumption']
# # legend_names = exp_names
import matplotlib.pyplot as plt
import os
import numpy as np
#
# # REWARD_STEP = 128
# # TEST_STEP = 3840
# REWARD_STEP = 2048
# TEST_STEP = 102400
#
# # REWARD_STEP = 1024
# # TEST_STEP = 51200
#
# figure_dir = f'train_figures/{"-".join(exp_names)}'
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['figure.figsize'] = (9.0, 6.0) # 设置figure_size尺寸
#
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# legend_size = 18
# xylabel_size = 20
# xystick_size = 18
#
# if not os.path.exists(figure_dir):
#     os.makedirs(figure_dir)
#
# mean_interval = 20
#
# rewards = []
# value_losses = []
# action_losses = []
# disentropy_losses = []
# test_accs = []
# energies = []
# for exp_name in exp_names:
#     with open(f'train_log/{exp_name}.log', 'r') as f:
#         with open('draw_tmp_log', 'w') as wf:
#             l_idx = 0
#             for l in f:
#                 l_idx += 1
#                 if l_idx > 0:
#                 # if l_idx > 6315:
#                     wf.write(l)
#
#     with open('draw_tmp_log', 'r') as f:
#         epoch = 0
#         reward = []
#         value_loss = []
#         action_loss = []
#         disentropy_loss = []
#         energy = []
#         test_acc = []
#         interval_i = 0
#         for l in f:
#             l = l.split()
#             # print(l)
#             if 'Mean' in l:
#                 # print(l)
#                 reward.append(float(l[-1][:-1]))
#
#             if 'loss:' in l:
#                 value_loss.append(float(l[8][:-1]))  # value loss
#                 action_loss.append(float(l[11][:-1]))  # action loss
#                 disentropy_loss.append(float(l[14][:-1]))  # disentropy loss
#
#             if 'Curr' in l:
#                 test_acc.append(float(l[6][:-1]))
#                 energy.append(float(l[-1]))
#
#         rewards.append(reward)
#         value_losses.append(value_loss)
#         action_losses.append(action_loss)
#         disentropy_losses.append(disentropy_loss)
#         test_accs.append(test_acc)
#         energies.append(energy)
#
# COLORS = ['orangered', 'forestgreen',  'purple', 'dodgerblue',  'magenta', 'salmon','lavender', 'turquoise','tan','lime', 'teal',  'lightblue',
#           'darkgreen',   'gold',   'darkblue', 'purple','brown','orange', ]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# i = 0
# y1 = test_accs[i]
# # test_acc = test_acc[:300]
# print(len(y1))
# y1m = [np.mean(y1[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(y1))]
# xs = np.arange(0, len(y1)*TEST_STEP, TEST_STEP)
# i = 0
# ax1.plot(xs, y1, color=COLORS[i], alpha=0.3)
# l1 = ax1.plot(xs, y1m, label=legend_names[i], color=COLORS[i])
#
# ax2 = ax1.twinx()  # this is the important function
# i = 1
# y2 = energies[i]
# # test_acc = test_acc[:300]
# print(len(y2))
# y2m = [np.mean(y2[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(y2))]
# xs = np.arange(0, len(y2)*TEST_STEP, TEST_STEP)
# i = 1
# ax2.plot(xs, y2, color=COLORS[i], alpha=0.3)
# l2 = ax2.plot(xs, y2m, label=legend_names[i], color=COLORS[i])
#
# # legend
# lns = l1+l2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0, fontsize=legend_size)
#
# # label
# ax1.grid()
# ax1.set_xlabel('Time step', fontsize=xylabel_size)
# ax1.set_ylabel('Average sum time', fontsize=xylabel_size)
# ax2.set_ylabel('Energy consumption', fontsize=xylabel_size)
# # ax2.set_ylim(0, 5)
# # ax1.set_ylim(80,100)
# ax1.tick_params(labelsize=xystick_size)
# ax2.tick_params(labelsize=xystick_size)
#
# ax1.xaxis.get_offset_text().set(size=xystick_size)
# ax2.yaxis.get_offset_text().set(size=xystick_size)
# # plt.savefig(f'{figure_dir}/ast_ec.pdf')
# plt.show()
# plt.close()
#
# # 合并
# plt.figure()
# y3 = np.array(y1) + 5e-4 * np.array(y2)
# xs = np.arange(0, len(y3))
# y3m = [np.mean(y3[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(y3))]
# i = 2
# plt.plot(xs, y3, color=COLORS[i], alpha=0.3)
# plt.plot(xs, y3m, label='reward', color=COLORS[i])
# plt.grid(True)
# plt.xlabel('Time step', fontsize=xylabel_size)
# plt.ylabel('Reward', fontsize=xylabel_size)
# plt.tick_params(labelsize=xystick_size)
# plt.show()
# plt.close()
#
mean_interval = 50
COLORS = ['orangered', 'forestgreen',  'purple', 'dodgerblue',  'magenta', 'salmon','lavender', 'turquoise','tan','lime', 'teal',  'lightblue',
          'darkgreen',   'gold',   'darkblue', 'purple','brown','orange', ]


# file = f'experiment_results/sfm2-{mode}.log'
# with open(file, 'r') as f:
#     for l in f:
#         # print(l)
#         if l.startswith('Reward list: '):
#             rs = l.strip('\n')[len('Reward list: ['):-1]
#             rs = rs.split(',')
#             rs = [float(r.strip(' ')) for r in rs]
#             print(len(rs))
#             break
# rsm= [np.mean(rs[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(rs))]
# i = 0
# plt.plot(rs, color=COLORS[i], alpha=0.3)
# plt.plot(rsm, label='sfm', color=COLORS[i])
#
# file = 'experiment_results/rllift2-feb08_lunch_random0-6_attention.log'
# with open(file, 'r') as f:
#     for l in f:
#         # print(l)
#         if l.startswith('Reward list: '):
#             rs = l.strip('\n')[len('Reward list: ['):-1]
#             rs = rs.split(',')
#             rs = [float(r.strip(' ')) for r in rs]
#             print(len(rs))
#             break
# # plt.plot(rs, color='g')
# rsm= [np.mean(rs[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(rs))]
# i = 1
# plt.plot(rs, color=COLORS[i], alpha=0.3)
# plt.plot(rsm, label='rllift', color=COLORS[i])
#
# file = 'experiment_results/eta-lunchpeak.log'
# with open(file, 'r') as f:
#     for l in f:
#         # print(l)
#         if l.startswith('Reward list: '):
#             rs = l.strip('\n')[len('Reward list: ['):-1]
#             rs = rs.split(',')
#             rs = [float(r.strip(' ')) for r in rs]
#             print(len(rs))
#             break
# # plt.plot(rs, color='g')
# rsm= [np.mean(rs[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(rs))]
# i = 2
# plt.plot(rs, color=COLORS[i], alpha=0.3)
# plt.plot(rsm, label='rllift', color=COLORS[i])

mode = 'lunchpeak'
# methods = ['nearest', 'rr', 'longest_first', 'eta', 'sfm2', 'rllift2']
methods = ['dqn', 'egc', 'rllift2']
# methods = ['dqn', 'egc', 'eta', 'sfm2', 'rllift2']
# methods = ['dqn', 'rllift2', 'egc', 'eta']
for mode in ['lunchpeak', 'uppeak', 'dnpeak', 'notpeak']:
# for mode in ['uppeak', 'dnpeak']:
    plt.figure()
    plt.title(mode)
    print(mode)
    for i, method in enumerate(methods):

        file = f'experiment_results/rewards/{method}-{mode}.log'
        rs = []
        with open(file, 'r') as f:
            for l in f:
                # print(l)
                flag = False
                if l.startswith('Reward list: '):
                    rs = l.strip('\n')[len('Reward list: ['):-1].split(',')
                    rs = [float(r.strip(' ')) for r in rs]

                    break
                elif l.startswith('[Reward List]: '):
                    rs_ = l.strip('\n')[len('[Reward List]: ['):-1].split(',')
                    rs_ = [float(r.strip(' ')) for r in rs_]
                    # print(method, mode, len(rs_))
                    if method == 'dqn' or method == 'egc':
                        if mode == 'uppeak':
                            print('here',len(rs_))
                            rs_ = rs_[3600:]
                        if mode == 'dnpeak':
                            print('here2', len(rs_))
                            rs_ = rs_[720:]
                    print(method, len(rs_))
                    rs += rs_
        print(method, len(rs))
        # rsm= [np.sum(rs[0:i + 1]) for i in range(len(rs))]
        # rsm= [np.mean(rs[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(rs))]
        rsm = []
        # rs = rs[:19310]
        # step = int(len(rs)//1000)
        # step = 30
        # for j in range(step, len(rs), step):
        #     rsm.append(sum(rs[j-step:j])/step)
            # rsm.append(sum(rs[j-step:j])/step-0.05*(2-i))
        plt.plot(rs, color=COLORS[i], alpha=0.9, label=method)
        # plt.plot(np.array(rsm), label=method, color=COLORS[i], alpha=0.9)
        # plt.plot(np.exp(np.array(rsm)*10), label=method, color=COLORS[i])

    plt.legend()
    plt.show()
    plt.close()
    # 30*60*2=