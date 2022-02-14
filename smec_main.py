from collections import deque
from smec_rl_components.smec_rollout_storage import SmecRolloutStorage
from smec_liftsim.smec_rl_env import *

from smec_rl_components.smec_evaluator import evaluate, evaluate_shortest_first, evaluate_smec, evaluate_general
from smec_rl_components.smec_reward import concate_list
from pytorch_rl.a2c_ppo_acktr import algo, utils


def main():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--render', type=bool, default=False, help="True if test, False if train (default: False)")
    parser.add_argument('--lift_num', type=int, default=4, help='number of elevators ')
    parser.add_argument('--num-envs', type=int, default=8, help='number of environments')
    parser.add_argument("--exp-name", type=str, default='sept02', help='experiment name')

    # rl algorithms
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=30, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--reset-interval', type=int, default=2000, help='number of environments')
    parser.add_argument('--num-env-steps', type=int, default=10e6, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-graph', action='store_true', default=False, help='disables gnn in training')
    parser.add_argument('--no-mask', action='store_true', default=False, help='disables action mask in training')
    parser.add_argument('--use-attention', action='store_true', default=False, help='use attention mask for mask')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, help='use a linear schedule on the learning rate')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate the pretrained model.')
    parser.add_argument('--forbid_uncalled', action='store_true', default=False, help='forbid uncalled operations')
    parser.add_argument('--use-rules', action='store_true', default=False, help='use rules to evaluate')
    parser.add_argument('--evaluate-method', type=str, default='rl', help='the method to evaluate. random shortest smec and rl.')
    parser.add_argument('--real-data', action='store_true', default=False, help='use the real data to evaluate')
    parser.add_argument('--react-delay', type=int, default=1, help='whether to do the same action in a few steps.')
    parser.add_argument('--test-num', type=int, default=10, help='num of test while training.')
    parser.add_argument('--use-advice', action='store_true', default=False, help='use the advice choice for elevator.')
    parser.add_argument('--valid-action-mask', action='store_true', default=False, help='use valid action mask to train.')
    parser.add_argument('--special-reward', action='store_true', default=False, help='use reward from another article.')
    parser.add_argument('--data-dir', type=str, default=None, help='use the files in a data dir to train.')
    parser.add_argument('--model-path', type=str, default=None, help='the load model path')
    parser.add_argument('--device', type=str, default='cpu', help='the device')
    parser.add_argument('--dos', type=str, default='', help='data of section')
    args = parser.parse_args()

    if args.use_attention:
        from smec_rl_components.smec_policy2 import SmecPolicy
    else:
        from smec_rl_components.smec_policy import SmecPolicy

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.graph = not args.no_graph
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'
    print('args.lift_num : ', args.lift_num)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = 'cpu'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.join('runs/', args.exp_name)
    utils.cleanup_log_dir(log_dir)
    torch.set_num_threads(1)

    if not args.evaluate:
        # build env
        eval_env = None
        test_env = make_env(seed=0, forbid_uncalled=args.forbid_uncalled, use_graph=args.graph, gamma=args.gamma,
                            # real_data=args.real_data, use_advice=args.use_advice)()
                            real_data=args.real_data, use_advice=args.use_advice, data_dir=args.data_dir, dos=args.dos)()
        env_num, elevator_num, floor_num = args.num_envs, test_env.elevator_num, test_env.floor_num
        envs = [make_env(seed=i+1000, forbid_uncalled=args.forbid_uncalled, use_graph=args.graph, gamma=args.gamma,
                         real_data=args.real_data, use_advice=args.use_advice, special_reward=args.special_reward, data_dir=args.data_dir, file_begin_idx=i, dos=args.dos) for i in range(env_num)]
        envs = AsyncVectorEnv(env_fns=envs)

        # build model
        # actor_critic = SmecPolicy(elevator_num, floor_num, open_mask=not args.no_mask, use_advice=args.use_advice)
        actor_critic = SmecPolicy(elevator_num, floor_num, open_mask=not args.no_mask, use_advice=args.use_advice, device=device)
        actor_critic.to(device)
        # actor_critic = torch.load(os.path.join(log_dir, args.exp_name + ".pt"), map_location=device)[0]
        if args.model_path:
            print(f'loading model from {args.model_path}')
            actor_critic = torch.load(args.model_path, map_location=device)[0]


        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                                   alpha=args.alpha, max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            if args.valid_action_mask:
                from smec_rl_components.smec_multia_ppo_mask import PPO
            else:
                from smec_rl_components.smec_multia_ppo import PPO
            agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                        args.entropy_coef, lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
        else:
            raise RuntimeError("Not supported")

        rollouts = SmecRolloutStorage(args.num_steps, env_num, test_env.elevator_num, test_env.floor_num, use_advice=args.use_advice)
        state = envs.reset()
        for k in rollouts.obs[0]:
            rollouts.obs[0][k].copy_(torch.tensor(state[k]))
        rollouts.to(device)

        episode_rewards = deque(maxlen=100)
        episode_waiting_t = deque(maxlen=100)
        episode_energy = deque(maxlen=100)
        start = time.time()
        num_updates = int(args.num_env_steps) // args.num_steps // env_num
        best_score = 100000000
        # need_reset = False

        save_log_by_hand = True
        log_file = open('train_log/%s.log' % args.exp_name, 'a')
        for j in range(num_updates):
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(agent.optimizer, j, num_updates,
                                             agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, rule = actor_critic.act(rollouts.obs[step])

                # for debug:
                rule = j * torch.ones_like(rule)

                # Obser reward and next obs
                action, action_log_prob, value = action.squeeze(0), action_log_prob.squeeze(0), value.squeeze(0)
                # step the same action for a few step.
                for rd in range(args.react_delay):
                    if env_num != 1:
                        actions = torch.cat((action.cpu(), rule.cpu()), dim=1)
                    else:
                        # for debug or num_env == 1:
                        actions = torch.cat((action.unsqueeze(0).cpu(), rule.cpu()), dim=1)
                    obs, reward, done, info = envs.step(actions)
                    # print(f'step {step} done:{done}')
                    # print("reward: ", reward[0][0])

                # if sum(done) > 0:
                #     need_reset = True

                # If done then clean the history of observations.
                masks = [0.0 if ele else 1.0 for ele in done]
                for rew in concate_list(reward.tolist()):
                    if rew - 0 > 1e-9 or 0 - rew > 1e-9:
                        episode_rewards.append(rew)
                for inf in info:
                    if inf['waiting_time']:
                        episode_waiting_t += inf['waiting_time']
                    if inf['total_energy']:
                        episode_energy += [inf['total_energy']]
                reward, masks = torch.tensor(reward), torch.tensor(masks)
                rollouts.insert(obs, action, action_log_prob, value, reward, masks)
                rollouts.to(device)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value.squeeze(), args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            # if j > 1:
            #     print('debug here')
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_steps * args.num_envs
                end = time.time()
                print(f"[train] Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))};"
                      f" training waiting time: {np.mean(episode_waiting_t):.1f} training energy: {np.mean(episode_energy):.1f}.")
                # print(episode_waiting_t)
                print(f"[train] Max reward {np.max(episode_rewards):.3f}, min reward: {np.min(episode_rewards):.3f}, Mean reward: {np.mean(episode_rewards):.3f}.")
                print(f"[train] Best val waiting time: {best_score:.1f}, Value loss: {value_loss:.5f}, action loss: {action_loss:.5f}, dist entropy: {dist_entropy:.5f}.")

                if save_log_by_hand:
                    print(
                        f"[train] Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))};"
                        f" training waiting time: {np.mean(episode_waiting_t):.1f} training energy: {np.mean(episode_energy):.1f}.", file=log_file, flush=True)
                    print(
                        f"[train] Max reward {np.max(episode_rewards):.3f}, min reward: {np.min(episode_rewards):.3f}, Mean reward: {np.mean(episode_rewards):.3f}.", file=log_file, flush=True)
                    print(
                        f"[train] Best val waiting time: {best_score:.1f}, Value loss: {value_loss:.5f}, action loss: {action_loss:.5f}, dist entropy: {dist_entropy:.5f}.", file=log_file, flush=True)

            if args.eval_interval is not None and (j+1) % args.eval_interval == 0:
                if eval_env is None:
                    eval_env = make_env(seed=0, render=False, forbid_uncalled=args.forbid_uncalled, gamma=args.gamma,
                                        real_data=args.real_data, use_advice=args.use_advice, data_dir=args.data_dir, dos=args.dos)()
                res = 0
                total_energy = 0
                for i in range(args.test_num):
                    evaluate_args = {'actor_critic': actor_critic}
                    r, e = evaluate_general(eval_env, device, "rl", evaluate_args)
                    res += r
                    total_energy += e
                res /= args.test_num
                total_energy /= args.test_num
                print(f"[Evaluation] Curr mean val waiting time: {res:.1f}, mean energy: {total_energy:.1f}")
                if save_log_by_hand:
                    print(f"[Evaluation] Curr mean val waiting time: {res:.1f}, mean energy: {total_energy:.1f}", file=log_file, flush=True)
                # res = evaluate(actor_critic, eval_env, device)
                # print(f"[train] Curr val waiting time: {res:.1f}")
                if res < best_score:
                    best_score = res
                    torch.save([actor_critic], os.path.join(log_dir, args.exp_name + ".pt"))

            # if j % 50 == 0:
            #     torch.save([actor_critic], os.path.join(log_dir, args.exp_name + ".pt"))

            # # if j % args.reset_interval == 0:
            # if need_reset:
            #     need_reset = False
            #     print(f"-------------- [{j}] reset the env --------------")
            #     print(f"-------------- [{j}] reset the env --------------", file=log_file, flush=True)
            #     # reset the environment
            #     state = envs.reset()
            #     actor_critic.reset()
            #     rollouts.reset()
            #     for k in rollouts.obs[0]:
            #         rollouts.obs[0][k].copy_(torch.tensor(state[k]))
            #     rollouts.to(device)

        log_file.close()
    else:
        file = open(f'experiment_results/ablation/rllift2-{args.exp_name}.log', 'a')
        print('-' * 50, file=file)
        evaluate_args = {'use_rules': args.use_rules}
        if args.evaluate_method == 'rl':
            model_path = os.path.join(log_dir, args.exp_name + ".pt")
            actor_critic = torch.load(model_path, map_location=device)[0]
            # print(actor_critic.AttentionFactor)
            # actor_critic = SmecPolicy(4, 16, open_mask=True, use_advice=False)  # 测一下训练是否无效
            # actor_critic.open_mask = False
            evaluate_args['actor_critic'] = actor_critic
            print('mask: ', actor_critic.open_mask)

        test_env = make_env(seed=args.seed, render=args.render, use_graph=args.graph, gamma=args.gamma,
                            real_data=args.real_data, use_advice=args.use_advice, data_dir=args.data_dir, file_begin_idx=17, dos=args.dos)()

        test_num = 20
        avg_awt, avg_att, avg_energy = evaluate_general(test_env, device, args.evaluate_method, evaluate_args,
                                                        file=file, test_num=test_num)
        print(f'average awt: {avg_awt:.2f}, average att: {avg_att:.2f}, average ast: {avg_awt + avg_att:.2f},'
              f' average energy: {avg_energy:.0f}')
        print(f'average awt: {avg_awt:.2f}, average att: {avg_att:.2f}, average ast: {avg_awt + avg_att:.2f},'
              f' average energy: {avg_energy:.0f}', file=file)

        file.close()


if __name__ == '__main__':
    # --shortest_first --evaluate --real-data --use-advice --render True
    # --exp_name sept12 --evaluate --real-data --use-advice
    main()
