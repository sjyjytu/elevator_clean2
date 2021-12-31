"""

for test smec single elevator dispatch

Authors: XuZhongxing
Date:    2019/11/04
"""

from original_env import LiftSim
from utils import ElevatorHallCall


MAX_ITERATION = 10000
fail_flag = False
stop_count = 10


def smec_run_elevator(mansion_env, iteration):
    i = 0

    while i < iteration:
        i += 1
        mansion_env.render()
        state = mansion_env.state
        # print("upcall and down calls are", state.RequiringUpwardFloors, state.RequiringDownwardFloors)
        up_floors, down_floors = state.RequiringUpwardFloors, state.RequiringDownwardFloors
        hallcall = ElevatorHallCall(up_floors, down_floors)  # this is dispatch task single elevator
        state = mansion_env.step(hallcall)
        #print(state.RequiringUpwardFloors, state.RequiringDownwardFloors)
        #print(state)


if __name__ == "__main__":
    env = LiftSim()
    smec_run_elevator(env, MAX_ITERATION)
