import argparse
import sys
import os
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except IndexError:
    print("Cannot add the common path {}".format(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from arguments import add_arguments
from ilqr.obstacles import Obstacle
import numpy as np
from ilqr.constraints import Constraints
from ilqr.iLQR import iLQR
from ilqr.Point import Point


argparser = argparse.ArgumentParser(description='CARLA CILQR')
add_arguments(argparser)
args = argparser.parse_args()



dt = 0.1
sim_time = 100
map_lengthx = 50
map_lengthy = 6
lane1 = 5
lane2 = 0
lane3 = -5
num_vehicles = 2

## Car Parameters
car_dims = np.array([4, 2])

max_speed = 180/3.6
wheelbase = 2.94
steer_min = -1.0
steer_max = 1.0
accel_min = -5.5
accel_max = 3.0
desired_y = 2.5
NPC_max_acc = 0.75


#   X_0 = np.array([ego_state[0][0], ego_state[0][1], ego_state[1][0], ego_state[2][2]])
# start state: x, y, v, yawrate
start_state = np.array([0, 2.5, 15, 0.5])
def get_ego_states():
    ego_states = np.array([[start_state[0], start_state[1],     0],
                           [start_state[2], 0,                  0],
                           [0,              0,     start_state[3]],
                           [0,              0,                  0],
                           [0,              0,                  0]])
    return ego_states

# 障碍物的 trajectory
NPC_vel = 10.0
NPC_start = np.array([[15, 2.5, NPC_vel, 0]])
NPC_traj = NPC_start.T

for i in range(0, args.horizon):
    NPC_start[0][0] += NPC_vel*args.timestep
    NPC_traj = np.hstack((NPC_traj, NPC_start.T))


car_dims = np.array([4, 2])
# 设置粗糙的轨迹

plan_ilqr = []
for i in range(0, 60):
    plan_ilqr.append(np.array([i, desired_y]))

# boundary
bound1 = -0.5
bound2 = 5.0
polyline1 = [Point(0, bound1), Point(60, bound1)]
polyline2 = [Point(0, bound2), Point(60, bound2)]
polylines = [polyline1, polyline2]
navigation_agent = iLQR(args, car_dims, polylines)
navigation_agent.set_global_plan(np.array(plan_ilqr))
traj, ref_traj, U = navigation_agent.run_step(get_ego_states(), NPC_traj)


print(traj)
