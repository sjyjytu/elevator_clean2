# exp_name = 'nov17'
# exp_name = 'nov18_2'
# exp_name = 'nov18_2_3257'
# exp_name = 'nov18_2_7813'
# exp_name = 'nov19_0'
# exp_name = 'nov19_30-45'
# exp_name = 'nov20_30-40'
# exp_name = 'nov25_dnpeak'
# exp_name = 'nov25_uppeak'
# exp_name = 'nov25_lanuchpeak'
# exp_name = 'nov25_nopeak'
# exp_name = 'dec03_uppeak10-20_31days'
# exp_name = 'dec03_uppeak10-20'
# exp_name = 'dec03_uppeak10-20_openmask'
# exp_name = 'dec03_uppeak10-20_31days'
# exp_name = 'dec03_uppeak10-20_31days_openmask'
# exp_name = 'dec06_dnpeak00-60_31days'
# exp_name = 'dec06_lanuchpeak00-10_31days'
# exp_name = 'dec06_lanuchpeak00-10_31days_openmask'
# exp_name = 'dec06_lanuchpeak00-60_31days_openmask'
# exp_name = 'dec07_lanuchpeak00-10_31days_openmask'
# exp_name = 'jan07_simplev2'
# exp_name = 'jan07_simplev2_nomask'
# exp_names = ['jan07_simplev2', 'jan07_simplev2_nomask']
# exp_name = 'feb08_down_random6-12_attention'
# exp_name = 'feb08_lunch_random0-6_attention'
# exp_name = 'feb08_notpeak_random0-6_attention'
# exp_name = 'feb08_notpeak_random0-6'
# exp_names = ['feb08_notpeak_random0-6_attention', 'feb08_notpeak_random0-6']
exp_names = ['jan14_lunch_random0-6', 'jan14_lunch_random0-6_nomask']
# exp_names = ['feb08_down_random6-12_attention', 'feb08_lunch_random0-6_attention']
# exp_names = ['feb08_down_random6-12_attention']
# exp_names = ['feb08_lunch_random0-6_attention']
legend_names = ['RLLift', 'RLLift without soft rules']
# legend_names = exp_names
import matplotlib.pyplot as plt
import os
import numpy as np

# REWARD_STEP = 128
# TEST_STEP = 3840
# REWARD_STEP = 2048
# TEST_STEP = 102400

REWARD_STEP = 128
TEST_STEP = 51200

figure_dir = f'train_figures/{"-".join(exp_names)}'
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.figsize'] = (9.0, 6.0) # 设置figure_size尺寸

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

legend_size = 18
xylabel_size = 20
xystick_size = 18

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

mean_interval = 20

rewards = []
value_losses = []
action_losses = []
disentropy_losses = []
test_accs = []
energies = []
for exp_name in exp_names:
    with open(f'train_log/{exp_name}.log', 'r') as f:
        with open('draw_tmp_log', 'w') as wf:
            l_idx = 0
            for l in f:
                l_idx += 1
                if l_idx > 0:
                # if l_idx > 6315:
                    wf.write(l)

    with open('draw_tmp_log', 'r') as f:
        epoch = 0
        reward = []
        value_loss = []
        action_loss = []
        disentropy_loss = []
        energy = []
        test_acc = []
        interval_i = 0
        for l in f:
            l = l.split()
            # print(l)
            if 'Mean' in l:
                # print(l)
                reward.append(float(l[-1][:-1]))

            if 'loss:' in l:
                value_loss.append(float(l[8][:-1]))  # value loss
                action_loss.append(float(l[11][:-1]))  # action loss
                disentropy_loss.append(float(l[14][:-1]))  # disentropy loss

            if 'Curr' in l:
                test_acc.append(float(l[6][:-1]))
                energy.append(float(l[-1]))

        rewards.append(reward)
        value_losses.append(value_loss)
        action_losses.append(action_loss)
        disentropy_losses.append(disentropy_loss)
        test_accs.append(test_acc)
        energies.append(energy)

COLORS = ['orangered', 'forestgreen',  'purple', 'dodgerblue',  'magenta', 'salmon','lavender', 'turquoise','tan','lime', 'teal',  'lightblue',
          'darkgreen',   'gold',   'darkblue', 'purple','brown','orange', ]

# test time
plt.figure()
# plt.title('evaluation time vs step')
# plt.title('评测时间与迭代次数的关系')
for i, test_acc in enumerate(test_accs):
    # plt.plot(test_acc, label=exp_names[i], color=COLORS[i])
    # if i != 1:
        new_test_acc= [np.mean(test_acc[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(test_acc))]
        xs = np.arange(0, len(test_acc)*TEST_STEP, TEST_STEP)
        plt.plot(xs, test_acc, color=COLORS[i], alpha=0.3)
        plt.plot(xs, new_test_acc, label=legend_names[i], color=COLORS[i])
plt.grid(True)
if len(test_accs) > 1:
    plt.legend(fontsize=legend_size)
plt.xlabel('Time step', fontsize=xylabel_size)
plt.ylabel('Average total time', fontsize=xylabel_size)
plt.tick_params(labelsize=xystick_size)
ax = plt.gca()
ax.xaxis.get_offset_text().set(size=xystick_size)
ax.yaxis.get_offset_text().set(size=xystick_size)

# plt.xlabel('迭代次数')
# plt.ylabel('评测时间')

plt.savefig(f'{figure_dir}/test_time.pdf')
plt.show()
plt.close()

# energy
plt.figure()
# plt.title('energy consumption vs step')
# plt.title('评测时间与迭代次数的关系')
for i, test_acc in enumerate(energies):
    # plt.plot(test_acc, label=exp_names[i], color=COLORS[i])
    # if i != 1:
        new_test_acc= [np.mean(test_acc[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(test_acc))]
        xs = np.arange(0, len(test_acc)*TEST_STEP, TEST_STEP)
        plt.plot(xs, test_acc, color=COLORS[i], alpha=0.3)
        plt.plot(xs, new_test_acc, label=legend_names[i], color=COLORS[i])
plt.grid(True)
if len(energies) > 1:
    plt.legend(fontsize=legend_size)
plt.xlabel('Time step', fontsize=xylabel_size)
plt.ylabel('Energy consumption', fontsize=xylabel_size)
plt.tick_params(labelsize=xystick_size)
ax = plt.gca()
ax.xaxis.get_offset_text().set(size=xystick_size)
ax.yaxis.get_offset_text().set(size=xystick_size)
# plt.xlabel('迭代次数')
# plt.ylabel('评测时间')

plt.savefig(f'{figure_dir}/energy.pdf')
plt.show()
plt.close()

# loss
plt.figure()
# plt.title('value loss vs step')
# plt.title('损失与迭代次数的关系')
a = np.arange(20000)
a = 10 * np.exp(-a / 40000) + 1
for i, loss in enumerate(value_losses):
    # loss = np.array(loss[50:20050])
    loss = np.array(loss[50:20050]) * a
    # new_loss = [np.mean(loss[max(i - (mean_interval - 1), 0):i+1]) for i in range(len(loss))]
    new_loss = [np.mean(loss[i:i+mean_interval]) for i in range(len(loss))]
    xs = np.arange(0, len(loss) * REWARD_STEP, REWARD_STEP)
    plt.plot(xs, loss, color=COLORS[i], alpha=0.3)
    plt.plot(xs, new_loss, label=legend_names[i], color=COLORS[i])
    # plt.plot(test_accs[:length], color='blue')
plt.grid(True)
plt.legend(fontsize=legend_size)
plt.xlabel('Time step', fontsize=xylabel_size)
plt.ylabel('Value loss', fontsize=xylabel_size)
plt.tick_params(labelsize=xystick_size)
ax = plt.gca()
ax.xaxis.get_offset_text().set(size=xystick_size)
ax.yaxis.get_offset_text().set(size=xystick_size)
# plt.xlabel('迭代次数')
# plt.ylabel('评测时间')

plt.savefig(f'{figure_dir}/mean_value_loss.pdf')
plt.show()
plt.close()

# # loss
# plt.figure()
# plt.title('action loss vs step')
# # plt.title('损失与迭代次数的关系')
# for i, loss in enumerate(action_losses):
#     # loss = loss[50:4050]
#     new_loss = [np.mean(loss[max(i - (mean_interval - 1), 0):i+1]) for i in range(len(loss))]
#     xs = np.arange(0, len(loss) * REWARD_STEP, REWARD_STEP)
#     plt.plot(xs, loss, color=COLORS[i], alpha=0.3)
#     plt.plot(xs, new_loss, label=legend_names[i], color=COLORS[i])
#     # plt.plot(test_accs[:length], color='blue')
# plt.grid(True)
# plt.legend(fontsize=legend_size)
# plt.xlabel('step', fontsize=xylabel_size)
# plt.ylabel('loss', fontsize=xylabel_size)
# plt.tick_params(labelsize=xystick_size)
# ax = plt.gca()
# ax.xaxis.get_offset_text().set(size=xystick_size)
# ax.yaxis.get_offset_text().set(size=xystick_size)
# # plt.xlabel('迭代次数')
# # plt.ylabel('评测时间')
#
# plt.savefig(f'{figure_dir}/mean_value_loss.pdf')
# plt.show()
# plt.close()
#
# # loss
# plt.figure()
# plt.title('disentropy loss vs step')
# # plt.title('损失与迭代次数的关系')
# for i, loss in enumerate(disentropy_losses):
#     # loss = loss[50:4050]
#     new_loss = [np.mean(loss[max(i - (mean_interval - 1), 0):i+1]) for i in range(len(loss))]
#     xs = np.arange(0, len(loss) * REWARD_STEP, REWARD_STEP)
#     plt.plot(xs, loss, color=COLORS[i], alpha=0.3)
#     plt.plot(xs, new_loss, label=legend_names[i], color=COLORS[i])
#     # plt.plot(test_accs[:length], color='blue')
# plt.grid(True)
# plt.legend(fontsize=legend_size)
# plt.xlabel('step', fontsize=xylabel_size)
# plt.ylabel('loss', fontsize=xylabel_size)
# plt.tick_params(labelsize=xystick_size)
# ax = plt.gca()
# ax.xaxis.get_offset_text().set(size=xystick_size)
# ax.yaxis.get_offset_text().set(size=xystick_size)
# # plt.xlabel('迭代次数')
# # plt.ylabel('评测时间')
#
# plt.savefig(f'{figure_dir}/mean_value_loss.pdf')
# plt.show()
# plt.close()
#
# # reward
# plt.figure()
# plt.title('reward vs step')
# # plt.title('奖励与迭代次数的关系')
# for i, reward in enumerate(rewards):
#     # reward = reward[:1000]
#     new_reward = [np.mean(reward[i:i+50]) for i in range(len(reward)-50)]
#     xs = np.arange(0, len(reward) * REWARD_STEP, REWARD_STEP)
#     plt.plot(xs, reward, color=COLORS[i], alpha=0.3)
#     xs = np.arange(0, len(new_reward) * REWARD_STEP, REWARD_STEP)
#     plt.plot(xs, new_reward, label=legend_names[i], color=COLORS[i])
# plt.grid(True)
# plt.legend(fontsize=legend_size)
# plt.xlabel('step', fontsize=xylabel_size)
# plt.ylabel('reward', fontsize=xylabel_size)
# plt.tick_params(labelsize=xystick_size)
# ax = plt.gca()
# ax.xaxis.get_offset_text().set(size=xystick_size)
# ax.yaxis.get_offset_text().set(size=xystick_size)
# # plt.xlabel('迭代次数')
# # plt.ylabel('奖励')
#
# plt.savefig(f'{figure_dir}/mean_reward.pdf')
# plt.show()
# plt.close()
#
#
#
#
#
