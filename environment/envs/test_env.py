#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt

# import rospy
from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param, fntsmc_pos, fntsmc_att
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

'''Parameter list of the quadrotor'''
DT = 0.01
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 10
uav_param.pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [-5, 5]])
'''Parameter list of the quadrotor'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param()
att_ctrl_param.k1 = np.array([25, 25, 40])
att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
att_ctrl_param.alpha = np.array([2.5, 2.5, 2.5])
att_ctrl_param.beta = np.array([0.99, 0.99, 0.99])
att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
att_ctrl_param.dim = 3
att_ctrl_param.dt = DT
att_ctrl_param.ctrl0 = np.array([0., 0., 0.])
att_ctrl_param.saturation = np.array([0.3, 0.3, 0.3])
'''Parameter list of the attitude controller'''

'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param()
pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''


def test_uav_hover_outer_loop():
    from environment.envs.RL.uav_hover_outer_loop_att import uav_hover_outer_loop
    # rospy.init_node(name='env_test', anonymous=False)

    env = uav_hover_outer_loop(uav_param, pos_ctrl_param, att_ctrl_param, target0=np.array([-1, 3, 2]))
    env.msg_print_flag = True
    num = 0
    while num < 1:
        env.reset()
        r = 0
        while not env.is_terminal:
            action = env.pos_control(env.pos_ref, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
            env.step_update(action=np.array(action))
            r += env.reward
        print(r)
        num += 1
    env.collector.plot_pos()
    env.collector.plot_throttle()
    env.collector.plot_att()
    plt.show()


if __name__ == '__main__':
    test_uav_hover_outer_loop()
