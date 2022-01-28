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
exp_name = 'jan07_simplev2'
# exp_name = 'jan07_simplev2_nomask'
exp_names = ['jan07_simplev2', 'jan07_simplev2_nomask']
legend_names = ['RLLift', 'RLLift_norule']

import matplotlib.pyplot as plt
import os
import numpy as np

figure_dir = f'train_figures/{"-".join(exp_names)}'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

mean_interval = 20

rewards = []
losses = []
test_accs = []
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
        loss = []
        test_acc = []
        interval_i = 0
        for l in f:
            l = l.split()
            # print(l)
            if 'Mean' in l:
                # print(l)
                reward.append(float(l[-1][:-1]))

            if 'loss:' in l:
                loss.append(float(l[8][:-1]))  # value loss
                # losses[-1].append(float(l[11][:-1]))  # action loss

            if 'Curr' in l:
                test_acc.append(float(l[-1]))

        rewards.append(reward)
        losses.append(loss)
        test_accs.append(test_acc)

COLORS = ['orangered', 'forestgreen',  'purple', 'dodgerblue',  'magenta', 'salmon','lavender', 'turquoise','tan','lime', 'teal',  'lightblue',
          'darkgreen',   'gold',   'darkblue', 'purple','brown','orange', ]

# test time
plt.figure()
plt.title('evaluation time vs iteration')
# plt.title('评测时间与迭代次数的关系')
for i, test_acc in enumerate(test_accs):
    # plt.plot(test_acc, label=exp_names[i], color=COLORS[i])
    # if i != 1:
        new_test_acc= [np.mean(test_acc[max(i - (mean_interval - 1), 0):i + 1]) for i in range(len(test_acc))]
        plt.plot(test_acc, color=COLORS[i], alpha=0.3)
        plt.plot(new_test_acc, label=legend_names[i], color=COLORS[i])
plt.grid(True)
plt.legend()
plt.xlabel('iteration')
plt.ylabel('evaluation time')
# plt.xlabel('迭代次数')
# plt.ylabel('评测时间')

plt.savefig(f'{figure_dir}/test_time.pdf')
plt.show()
plt.close()

# loss
plt.figure()
plt.title('loss vs iteration')
# plt.title('损失与迭代次数的关系')
for i, loss in enumerate(losses):
    # loss = loss[50:4050]
    new_loss = [np.mean(loss[max(i - (mean_interval - 1), 0):i+1]) for i in range(len(loss))]
    plt.plot(loss, color=COLORS[i], alpha=0.3)
    plt.plot(new_loss, label=legend_names[i], color=COLORS[i])
    # plt.plot(test_accs[:length], color='blue')
plt.grid(True)
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
# plt.xlabel('迭代次数')
# plt.ylabel('评测时间')

plt.savefig(f'{figure_dir}/mean_loss.pdf')
plt.show()
plt.close()

# reward
plt.figure()
plt.title('reward vs iteration')
# plt.title('奖励与迭代次数的关系')
for i, reward in enumerate(rewards):
    # reward = reward[:1000]
    new_reward = [np.mean(reward[i:i+50]) for i in range(len(reward)-50)]
    plt.plot(reward, color=COLORS[i], alpha=0.3)
    plt.plot(new_reward, label=legend_names[i], color=COLORS[i])
    # plt.plot(test_accs[:length], color='blue')
plt.grid(True)
plt.legend()
plt.xlabel('iteration')
plt.ylabel('reward')
# plt.xlabel('迭代次数')
# plt.ylabel('奖励')

plt.savefig(f'{figure_dir}/mean_reward.pdf')
plt.show()
plt.close()





