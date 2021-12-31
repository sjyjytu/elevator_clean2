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
exp_name = 'dec07_lanuchpeak00-10_31days_openmask'
mean_interval = 20

# copy log
# with open('1105_log', 'r') as f:
# with open('screenlog.0', 'r') as f:
# with open('screenlog_before_nov17.0', 'r') as f:
with open(f'train_log/{exp_name}.log', 'r') as f:
    with open('draw_tmp_log', 'w') as wf:
        l_idx = 0
        for l in f:
            l_idx += 1
            if l_idx > 0:
            # if l_idx > 6315:
                wf.write(l)


import matplotlib.pyplot as plt
import os
import numpy as np
figure_dir = f'train_figures/{exp_name}'
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
with open('draw_tmp_log', 'r') as f:
# with open('screenlog.0', 'r') as f:
    epoch = 0
    rewards = [[]]
    losses = [[]]
    test_accs = []
    interval_i = 0
    for l in f:
        l = l.split()
        # print(l)
        if 'Mean' in l:
            # print(l)
            rewards[-1].append(float(l[-1][:-1]))

        if 'loss:' in l:
            losses[-1].append(float(l[8][:-1]))  # value loss
            # losses[-1].append(float(l[11][:-1]))  # action loss

            # interval_i += 1
            # if interval_i % 10 == 0:
            #     rewards.append([])
            #     losses.append([])

        if 'reset' in l:
            rewards.append([])
            losses.append([])

        if 'Curr' in l:
            test_accs.append(float(l[-1]))

    # plt.plot(rewards, '.r', )
    print(len(rewards))
    # begin_idx = 0
    # interval = 10
    # for begin_idx in range(interval):
    # plt.figure()
    # plt.title('day %d reward vs time'%begin_idx)
    # for i in range(begin_idx, len(rewards), interval):
    #     plt.plot(rewards[i], label=str(i))
    # plt.grid(True)
    # plt.legend(ncol=4)
    # plt.savefig(f'{figure_dir}/reward_{begin_idx}.jpg')
    # plt.show()
    # plt.close()
    #
    # plt.figure()
    # plt.title('day %d loss vs time'%begin_idx)
    # for i in range(begin_idx, len(losses), interval):
    #     plt.plot(losses[i], label=str(i))
    # plt.legend(ncol=4)
    # plt.grid(True)
    # plt.savefig(f'{figure_dir}/loss_{begin_idx}.jpg')
    # plt.show()
    # plt.close()

    mean_test_accs = []
    # length = 130
    # for i in range(length):
    for i in range(len(test_accs)):
        mean_test_accs.append(np.mean(test_accs[max(0,i-5):i+1]))

    plt.figure()
    # plt.title('evaluation time vs iteration')
    plt.title('评测时间与迭代次数的关系')
    plt.plot(test_accs, color='blue')
    # plt.plot(test_accs[:length], color='blue')
    plt.plot(mean_test_accs, color='red')
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel('evaluation time')
    plt.xlabel('迭代次数')
    plt.ylabel('评测时间')
    plt.savefig(f'{figure_dir}/test_time.jpg')
    plt.show()
    plt.close()

# mean_rewards = []
# mean_losses = []
# for i in range(len(rewards)-1):
#     mean_rewards.append(np.mean(rewards[i]))
#     mean_losses.append(np.mean(losses[i]))
mean_rewards = rewards[0]
mean_losses = losses[0]

total_iter = len(mean_rewards)
begin_idx = 0
interval = 1
plt.figure()
# plt.title('mean loss vs iteration')
plt.title('损失与迭代次数的关系')
# for i in range((total_iter+interval-1) // interval):
#     plt.plot(mean_losses[i*interval:(i+1)*interval], label=str(i))
mean_mean_losses = []
# mean_losses = mean_losses[-500:]
# draw mean reward mean
for i in range(len(mean_losses)):
    mean_mean_losses.append(np.mean(mean_losses[max(i-(mean_interval-1), 0):i+1]))

plt.plot(mean_losses[0:], color='blue')
plt.plot(mean_mean_losses, color='red')
plt.legend()
plt.grid(True)
# plt.xlabel('iteration')
# plt.ylabel('loss')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.savefig(f'{figure_dir}/mean_loss.jpg')
# plt.xticks(np.arange(0, total_iter,22))
plt.show()
plt.close()

plt.figure()
# plt.title('mean reward vs iteration')
plt.title('奖励与迭代次数的关系')
# for i in range((total_iter+interval-1) // interval):
#     plt.plot(mean_rewards[i*interval:(i+1)*interval], label=str(i))
mean_mean_rewards = []
# draw mean reward mean
for i in range(len(mean_rewards)):
    mean_mean_rewards.append(np.mean(mean_rewards[max(i-(mean_interval-1), 0):i+1]))
    # mean_mean_rewards.append(np.mean(mean_rewards[0:2+i]))

plt.plot(mean_rewards[0:], color='blue')
plt.plot(mean_mean_rewards, color='red')
plt.legend()
plt.grid(True)
# plt.xlabel('iteration')
# plt.ylabel('reward')
plt.xlabel('迭代次数')
plt.ylabel('奖励')
plt.savefig(f'{figure_dir}/mean_reward.jpg')
# plt.yticks(np.arange(-0.3, -0.15, 0.01))
# plt.xticks(np.arange(0, total_iter,22))
plt.show()
plt.close()

# #
# import numpy as np
#
# a = np.array([1,2,3])
# b = np.array([3,4,5])
# print(a+b)


