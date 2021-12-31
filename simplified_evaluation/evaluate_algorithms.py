from simplified_evaluation.elevator_env_refactor import Env
from simplified_evaluation.my_mcts import MCTSSolver
import random

test_time = 10
elev_env = Env(elevator_num=4, floor_num=10, person_num=20)

# 测试一下随机算法
aawt, aatt = 0, 0
for t in range(test_time):
    elev_env.reset()
    while not elev_env.is_end():
        action = random.randint(0, elev_env.elevator_num-1)
        elev_env.step(action)
    awt, att = elev_env.get_reward()
    print('Test Random', t, awt, att)
    aawt += awt
    aatt += att
aawt /= test_time
aatt /= test_time
print(f'Random average awt: {aawt:.2f}, average att:{aatt:.2f}')
print()

# evaluate mcts
solver = MCTSSolver(n_playout=500)
aawt, aatt = 0, 0
for t in range(test_time):
    elev_env.reset()
    solver.reset_solver()
    while not elev_env.is_end():
        action = solver.get_action(elev_env)
        elev_env.step(action)
    awt, att = elev_env.get_reward()
    print('Test MCTS', t, awt, att)
    aawt += awt
    aatt += att
aawt /= test_time
aatt /= test_time
print(f'MCTS average awt: {aawt:.2f}, average att:{aatt:.2f}')
print()

