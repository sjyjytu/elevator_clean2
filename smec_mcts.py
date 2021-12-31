# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(env):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(env.elevator_num)
    return zip(list(range(env.elevator_num)), action_probs)


def policy_value_fn(env):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(env.elevator_num)/env.elevator_num
    return zip(list(range(env.elevator_num)), action_probs)


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=5000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            env.step(action)

        action_probs = self._policy(env)
        # print(f'play out: {env.is_end()}')
        if not env.is_end():
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(env)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def _evaluate_rollout(self, env, limit=7):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for i in range(limit):
            if env.is_end():
                break
            action_probs = rollout_policy_fn(env)
            max_action = max(action_probs, key=itemgetter(1))[0]
            env.step(max_action)
        # else:
            # If no break from the loop, issue a warning.
            # print("WARNING: rollout reached move limit")
        awt, att = env.get_reward()
        return -0.1 * (awt + 0.5 * att)
        # return -0.1 * (0 * awt + att)
        # return 0

    def get_elev(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        # for k in self._root._children.keys():
        #     print(k, self._root._children[k]._n_visits)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSSolver(object):
    """AI Solver based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def reset_solver(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env):
        if not env.is_end():
            elev = self.mcts.get_elev(env)
            self.mcts.update_with_move(-1)  # 有个问题就是同样的state下，执行一个action得到的下一状态是不确定的，所以不能直接由(s,a)得到s_t+1
            return elev
        else:
            print("WARNING: the board is full")


import numpy as np
from smec_liftsim.generator_proxy import set_seed
from smec_liftsim.generator_proxy import PersonGenerator
from smec_liftsim.fixed_data_generator import FixedDataGenerator
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.mansion_manager import MansionManager
from smec_liftsim.utils import ElevatorHallCall
import configparser
import os
from smec_rl_components.smec_graph_build import *
from smec_liftsim.smec_constants import *


class SmecEnv():
    def __init__(self, data_file='train_data/new/uppeak/UpPeakFlow1_elvx.csv', config_file=None, render=True, seed=None, forbid_uncalled=False,
                 use_graph=True, real_data=True, use_advice=False, special_reward=False, data_dir=None, file_begin_idx=None):
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__) + '/smec_liftsim/rl_config2.ini')
        file_name = config_file
        self.forbid_uncalled = forbid_uncalled
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy.'
        dos = '10:00-20:00'
        # dos = ''
        if dos == '':
            st = 0
        else:
            ts = dos.split('-')[0].split(':')
            st = int(ts[0]) * 60 + int(ts[1])

        if not real_data:
            # Create a different person generator
            gtype = config['PersonGenerator']['PersonGeneratorType']
            person_generator = PersonGenerator(gtype)
            person_generator.configure(config['PersonGenerator'])
        else:
            person_generator = FixedDataGenerator(data_file=data_file, data_dir=data_dir, file_begin_idx=file_begin_idx, data_of_section=dos)
        self._config = MansionConfig(
            dt=time_step,
            number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            floor_height=float(config['MansionInfo']['FloorHeight']),
            maximum_acceleration=float(config['MansionInfo']['Acceleration']),
            maximum_speed=float(config['MansionInfo']['RateSpeed']),
            person_entering_time=float(config['MansionInfo']['PersonEnterTime']),
            door_opening_time=float(config['MansionInfo']['DoorOpeningTime']),
            door_closing_time=float(config['MansionInfo']['DoorClosingTime']),
            keep_door_open_lag=float(config['MansionInfo']['DoorKeepOpenLagTime']),
            door_well_time2=float(config['MansionInfo']['DwellTime2']),
            maximum_parallel_entering_exiting_number=int(config['MansionInfo']['ParallelEnterNum']),
            rated_load=int(config['MansionInfo']['RateLoad']),
            start_time=st
        )

        self.mansion = MansionManager(int(config['MansionInfo']['ElevatorNumber']), person_generator, self._config,
                                      config['MansionInfo']['Name'])
        self.use_graph = use_graph
        self.viewer = None
        self.open_render = render
        # if render:
        #     from smec_liftsim.rendering import Render
        #     self.viewer = Render(self.mansion)
        self.elevator_num = self.mansion.attribute.ElevatorNumber
        self.floor_num = int(config['MansionInfo']['NumberOfFloors'])

        if seed is not None:
            self.seed(seed)
        self.seed_c = seed

        self.graph_node_num = (self.elevator_num + self.floor_num) * 2
        self.gb = GraphBuilder(self.elevator_num, self.floor_num, 'cpu')
        self.empty_adj_matrix = self.gb.get_zero_adj_matrix()
        self.cur_adj_matrix = self.empty_adj_matrix.clone()
        self.empty_node_feature = self.gb.get_zero_node_feature()
        self.cur_node_feature = self.empty_node_feature.clone()
        assert self.use_graph

        self.use_advice = use_advice
        self.special_reward = special_reward
        candidate_num = self.elevator_num
        if use_advice:
            candidate_num += 1

        self.evaluate_info = {'valid_up_action': 0,
                              'advice_up_action': 0,
                              'valid_dn_action': 0,
                              'advice_dn_action': 0}

    @staticmethod
    def get_filter_by_list(list_len, query):
        cur_elv_mask = torch.tensor([0.0 for _ in range(list_len)])
        for elev in query:
            cur_elv_mask[elev] = 1.0
        return cur_elv_mask

    def get_action_mask(self, device):
        # M JY: add advice choice
        candidate_num = self.elevator_num + 1 if self.use_advice else self.elevator_num

        # get a list of action candidates by rules given pre-defined floors.
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        floor2elv_masks = []
        # handle up floors
        for idx in range(self.floor_num):
            if idx not in unallocated_up:
                cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            else:
                uncalled_elevators = self.mansion.get_uncalled_elevators()
                conv_elevators = self.mansion.get_convenience_elevators(up_or_down=True, floor_id=idx)
                if len(conv_elevators) > 0:  # convenient elevators exist
                    cur_elv_mask = self.get_filter_by_list(candidate_num, conv_elevators)
                    if self.use_advice:
                        cur_elv_mask[-1] = 1.0
                else:
                    cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            floor2elv_masks.append(cur_elv_mask)

        # handle down floors
        for idx in range(self.floor_num):
            if idx not in unallocated_dn:
                cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            else:
                uncalled_elevators = self.mansion.get_uncalled_elevators()
                conv_elevators = self.mansion.get_convenience_elevators(up_or_down=False, floor_id=idx)
                if len(conv_elevators) > 0:  # convenient elevators exist
                    cur_elv_mask = self.get_filter_by_list(candidate_num, conv_elevators)
                    if self.use_advice:
                        cur_elv_mask[-1] = 1.0
                # elif len(uncalled_elevators) > 0:  # non-called elevators exist
                #     cur_elv_mask = self.get_filter_by_list(self.elevator_num, uncalled_elevators)
                else:
                    cur_elv_mask = torch.tensor([1.0 for _ in range(candidate_num)])
            floor2elv_masks.append(cur_elv_mask)

        elevator_mask = torch.stack(floor2elv_masks).to(device)
        return elevator_mask

    def get_time(self):
        raw_time = self._config.raw_time
        cur_day = raw_time // (24 * 3600)
        cur_time = raw_time % (24 * 3600)
        return [cur_day, int(cur_time // 3600 + 7), int(cur_time % 3600 // 60), int(cur_time % 60)]

    def step_dt(self, action, verbose=False):
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        is_valid = 0
        for up_floor in unallocated_up:
            cur_elev = action
            all_elv_up_fs[cur_elev].append(up_floor)
            is_valid += 1
        for dn_floor in unallocated_dn:
            cur_elev = action
            all_elv_down_fs[cur_elev].append(dn_floor)
            is_valid += 1
        if verbose and is_valid:
            print(f'Choosing {action} for up:{unallocated_up}, dn:{unallocated_dn}, valid:{is_valid}')
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute)
        self.mansion.generate_person()
        new_obs = self.get_smec_state()
        reward = 0
        done = self.mansion.person_generator.done
        info = {}
        return new_obs, reward, done, info

    def step_dp(self, action):
        unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
        all_elv_up_fs, all_elv_down_fs = [[] for _ in range(self.elevator_num)], [[] for _ in range(self.elevator_num)]
        is_valid = 0
        for up_floor in unallocated_up:
            cur_elev = action
            all_elv_up_fs[cur_elev].append(up_floor)
        for dn_floor in unallocated_dn:
            cur_elev = action
            all_elv_down_fs[cur_elev].append(dn_floor)
        action_to_execute = []
        for idx in range(self.elevator_num):
            action_to_execute.append(ElevatorHallCall(all_elv_up_fs[idx], all_elv_down_fs[idx]))
        next_call_come = False
        while not next_call_come and not self.is_end():
            calling_wt, arrive_wt, loaded_num, enter_num, no_io_masks, awt = self.mansion.run_mansion(action_to_execute, use_rules=False)
            self.mansion.generate_person()
            unallocated_up, unallocated_dn = self.mansion.get_unallocated_floors()
            action_to_execute = [ElevatorHallCall([], []) for _ in range(self.elevator_num)]
            next_call_come = unallocated_up != [] or unallocated_dn != []
        new_obs = self.get_smec_state()
        reward = 0
        done = self.mansion.person_generator.done
        info = {}
        return new_obs, reward, done, info

    def step(self, action):
        return self.step_dp(action)

    def get_floor2elevator_dis(self, device):
        floor2elevator_dis = []
        for call_floor in range(self.floor_num):  # up calls
            cur_distance = []
            for elev in self.mansion._elevators:
                elevator_floor = elev._sync_floor
                # try by JY
                if call_floor == elevator_floor and \
                        (elev._run_state == ELEVATOR_STOP_DOOR_CLOSING or elev._run_state == ELEVATOR_RUN):
                    if elev._service_direction == 1:
                        elevator_floor += 0.01
                    elif elev._service_direction == -1:
                        elevator_floor -= 0.01
                going_up = elev._service_direction == 1  # going up
                if going_up and call_floor >= elevator_floor:
                    distance = call_floor - elevator_floor  # directly move up
                elif going_up and call_floor < elevator_floor:
                    distance = (self.floor_num - elevator_floor) + self.floor_num + call_floor  # move up + move to bottom + move to call
                else:
                    distance = elevator_floor + call_floor  # down to bottom and move up
                cur_distance.append(distance / self.floor_num)  # normalize
            floor2elevator_dis.append(cur_distance)

        for call_floor in range(self.floor_num):  # down calls
            cur_distance = []
            for elev in self.mansion._elevators:
                elevator_floor = elev._sync_floor
                # try by JY
                if call_floor == elevator_floor and \
                        (elev._run_state == ELEVATOR_STOP_DOOR_CLOSING or elev._run_state == ELEVATOR_RUN):
                    if elev._service_direction == 1:
                        elevator_floor += 0.01
                    elif elev._service_direction == -1:
                        elevator_floor -= 0.01
                going_down = elev._service_direction != 1  # going down
                if going_down and call_floor <= elevator_floor:
                    distance = elevator_floor - call_floor  # directly move down
                elif going_down and call_floor > elevator_floor:
                    distance = elevator_floor + self.floor_num + (
                                self.floor_num - call_floor)  # move down + move to top + move to call
                else:
                    distance = (self.floor_num - elevator_floor) + (self.floor_num - call_floor)  # to top and move down
                cur_distance.append(distance / self.floor_num)  # normalize
            floor2elevator_dis.append(cur_distance)
        floor2elevator_dis = torch.tensor(floor2elevator_dis).to(device)
        return floor2elevator_dis

    def get_smec_state(self):
        up_wait, down_wait, loading, location, up_call, down_call, load_up, load_down = self.mansion.get_rl_state(
            encode=True)
        up_wait, down_wait, loading, location = torch.tensor(up_wait), torch.tensor(down_wait), torch.tensor(
            loading), torch.tensor(location)
        legal_masks = self.get_action_mask(up_wait.device)
        self.cur_adj_matrix = self.gb.update_adj_matrix(self.cur_adj_matrix, up_call, down_call)
        self.cur_node_feature = self.gb.update_node_feature(self.cur_node_feature, up_wait, down_wait, load_up,
                                                            load_down, location)
        distances = self.get_floor2elevator_dis(up_wait.device)
        valid_action_mask = self.mansion.get_unallocated_floors_mask()
        valid_action_mask = torch.tensor(valid_action_mask).to(up_wait.device)
        ms = {'adj_m': self.cur_adj_matrix, 'node_feature_m': self.cur_node_feature, 'legal_masks': legal_masks,
              'distances': distances, 'valid_action_mask': valid_action_mask}
        return ms

    def seed(self, seed=None):
        set_seed(seed)

    def reset(self):
        self.mansion.reset_env()
        self.cur_node_feature = self.empty_node_feature.clone()
        self.cur_adj_matrix = self.empty_adj_matrix.clone()
        state = self.get_smec_state()
        if self.seed_c:
            self.seed_c += 100
            self.seed(self.seed_c)

        # self.data_idx = 0
        # self.next_generate_person = self.real_dataset[self.data_idx]
        return state

    def is_end(self):
        return self.mansion.is_done

    def render(self, **kwargs):
        self.viewer.view()

    def close(self):
        pass

    @property
    def attribute(self):
        return self.mansion.attribute

    @property
    def state(self):
        return self.mansion.state

    @property
    def statistics(self):
        return self.mansion.get_statistics()

    @property
    def log_debug(self):
        return self._config.log_notice

    @property
    def log_notice(self):
        return self._config.log_notice

    @property
    def log_warning(self):
        return self._config.log_warning

    @property
    def log_fatal(self):
        return self._config.log_fatal

    def get_reward(self):
        waiting_time = []
        transmit_time = []
        for k in self.mansion.person_info.keys():
            info = self.mansion.person_info[k]
            try:
                waiting_time.append(info[2])
                transmit_time.append(info[4])
            except:
                continue
        return np.mean(waiting_time), np.mean(transmit_time)


if __name__ == '__main__':
    # 这个方法的缺点就是环境（人流）必须是确定的、固定的、已知的，可以重复的模拟的，像棋盘一样。
    # data_file = 'train_data/up_peak_normal/4Ele16FloorUpPeakFlow28_elvx.csv'
    # data_file = 'train_data/up_peak_normal/4Ele16FloorUpPeakFlow28_elvx.csv'
    elev_env = SmecEnv()
    elev_env.reset()
    print(elev_env.mansion.person_generator.data.data)
    solver = MCTSSolver(n_playout=500)
    while not elev_env.is_end():

        # 问题是现在action是多维的，按理来说同一dt内也不会有多个楼层需要分配，但是可以重分配的话，就有了。所以这还是个组合优化的问题？但也可以先不考虑重分配，假设每次都是当前最优，就是贪心。
        unallocated_up, unallocated_dn = elev_env.mansion.get_unallocated_floors()
        if unallocated_dn or unallocated_up:
            # action = solver.get_action(elev_env)

            # 最短
            floor2elevators = elev_env.get_floor2elevator_dis('cpu').cpu().numpy()
            floor2elevators = np.argmin(floor2elevators, axis=1)
            action = floor2elevators[0]
            print(f'Choosing {action} for up:{unallocated_up}, dn:{unallocated_dn}, time:{elev_env._config.raw_time}, cur reward:{elev_env.get_reward()}')
        else:
            action = -1
        elev_env.step(action)

    # print(elev_env.person_info)
    print(elev_env.get_reward())  # (0.35, 3.8)




