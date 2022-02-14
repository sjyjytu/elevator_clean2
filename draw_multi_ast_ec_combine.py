def read_log(exp_name):
    with open(f'train_log/{exp_name}.log', 'r') as f:
        with open('draw_tmp_log', 'w') as wf:
            l_idx = 0
            for l in f:
                l_idx += 1
                if l_idx > 0:
                # if l_idx > 6315:
                    wf.write(l)
    reward = []
    value_loss = []
    action_loss = []
    disentropy_loss = []
    energy = []
    test_acc = []
    with open('draw_tmp_log', 'r') as f:
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
    return value_loss, action_loss, disentropy_loss, energy, test_acc

# exp_names = ['feb08_down_random6-12_attention', 'feb08_lunch_random0-6_attention']
# exp_names = ['feb08_lunch_random0-6_attention', 'feb08_down_random6-12_attention']
# exp_names = ['feb02_down_random6-12', 'feb02_down_random6-12']
exp_names = ['jan30_lunch_random0-6_debug', 'feb08_down_random6-12_attention']
legend_names = ['Value loss', 'Reward']
# legend_names = exp_names
import matplotlib.pyplot as plt
import os
import numpy as np

# REWARD_STEP = 128
# TEST_STEP = 3840

# figure_dir = f'train_figures/{"-".join(exp_names)}'
figure_dir = f'train_figures/final'
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

COLORS = ['orangered', 'forestgreen',  'purple', 'dodgerblue',  'magenta', 'salmon','lavender', 'turquoise','tan','lime', 'teal',  'lightblue',
          'darkgreen',   'gold',   'darkblue', 'purple','brown','orange', ]

# value_loss, action_loss, dist_loss, _, _ = read_log('jan14_lunch_random0-6')
value_loss, action_loss, dist_loss, _, _ = read_log('jan30_lunch_random0-6_debug')
value_loss = value_loss[50:25050]
action_loss = action_loss[50:25050]
dist_loss = dist_loss[50:25050]
_, _, _, energy, test_acc = read_log('feb08_down_random6-12_attention')
r = -0.01*(np.array(test_acc) + 5e-4 * np.array(energy))
r = r[:500]

fig = plt.figure()
ax1 = fig.add_subplot(111)

i = 0
y1 = value_loss
# y1 = action_loss
# y1 = dist_loss
# y1 = np.array(value_loss) + np.array(action_loss) + 5 * np.array(dist_loss)
# test_acc = test_acc[:300]
print(len(y1))
a = np.arange(len(y1))
a = np.exp(-a / 4000) + 1
a[20000:] = a[19999]
y1 = y1 * a
y1m = [np.mean(y1[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(y1))]
TEST_STEP = 2048
xs = np.arange(0, len(y1)*TEST_STEP, TEST_STEP)
i = 0
ax1.plot(xs, y1, color=COLORS[i], alpha=0.3)
l1 = ax1.plot(xs, y1m, label='Value loss', color=COLORS[i])
l1 = ax1.plot(xs, y1m, label=legend_names[i], color=COLORS[i])

ax2 = ax1.twinx()  # this is the important function
i = 1
y2 = r
# test_acc = test_acc[:300]
print(len(y2))
y2m = [np.mean(y2[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(y2))]
TEST_STEP = 102400
xs = np.arange(0, len(y2)*TEST_STEP, TEST_STEP)
i = 1
ax2.plot(xs, y2, color=COLORS[i], alpha=0.3)
l2 = ax2.plot(xs, y2m, label=legend_names[i], color=COLORS[i])

# legend
lns = l1+l2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0, fontsize=legend_size)

# label
ax1.grid()
ax1.set_xlabel('Time step', fontsize=xylabel_size)
ax1.set_ylabel('Value loss', fontsize=xylabel_size)
ax2.set_ylabel('Reward', fontsize=xylabel_size)
# ax2.set_ylim(0, 5)
# ax1.set_ylim(80,100)
ax1.tick_params(labelsize=xystick_size)
ax2.tick_params(labelsize=xystick_size)

ax1.xaxis.get_offset_text().set(size=xystick_size)
ax2.yaxis.get_offset_text().set(size=xystick_size)
plt.savefig(f'{figure_dir}/loss_reward.pdf')
plt.show()
# plt.close()

