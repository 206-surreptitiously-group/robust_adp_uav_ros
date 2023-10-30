#!/usr/bin/python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import deg2rad

from environment.envs.UAV.ref_cmd import ref_inner, generate_uncertainty
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
    from environment.envs.RL.uav_hover_outer_loop import uav_hover_outer_loop
    # rospy.init_node(name='env_test', anonymous=False)

    env = uav_hover_outer_loop(uav_param, pos_ctrl_param, att_ctrl_param, target0=np.array([-1, 3, 2]))
    env.msg_print_flag = True
    num = 0
    while num < 1:
        env.reset()
        r = 0
        while not env.is_terminal:
            action = env.pos_control(env.pos_ref, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
            action = env.pos_ctrl.control
            # env.dis = generate_uncertainty(time=env.time, is_ideal=False)  # 生成干扰信号
            env.step_update(action=np.array(action))
            r += env.reward
        print(r)
        num += 1
        # pd.DataFrame(np.hstack((env.collector.t, env.collector.state)),
        #              columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']) \
        #     .to_csv('../../datasave/data/fntsmc_disturbances.csv', sep=',', index=False)
    env.collector.plot_pos()
    # env.collector.plot_throttle()
    # env.collector.plot_att()
    plt.show()


def test_uav_hover_inner_loop():
    from environment.envs.RL.uav_hover_inner_loop import uav_hover_inner_loop
    # rospy.init_node(name='env_test', anonymous=False)

    env = uav_hover_inner_loop(uav_param, pos_ctrl_param, att_ctrl_param, target0=np.array([-1, 3, 2]))
    env.msg_print_flag = True
    num = 0
    phi_d = phi_d_old = 0.
    theta_d = theta_d_old = 0.
    dot_phi_d = (phi_d - phi_d_old) / uav_param.dt
    dot_theta_d = (theta_d - theta_d_old) / uav_param.dt
    while num < 1:
        env.reset()
        r = 0
        while not env.is_terminal:
            # print(action)
            env.step_update(action=np.array([0.01, 0, 0]))
            r += env.reward
        print(r)
        num += 1
    env.collector.plot_pos()
    env.collector.plot_throttle()
    env.collector.plot_att()
    plt.show()


def test_uav_inner_loop():
    from environment.envs.RL.uav_inner_loop import uav_inner_loop
    # rospy.init_node(name='env_test', anonymous=False)
    uav_param.att_zone = np.atleast_2d([[-deg2rad(90), deg2rad(90)], [-deg2rad(90), deg2rad(90)], [deg2rad(-120), deg2rad(120)]])
    ref_amplitude = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
    ref_period = np.array([4, 4, 4])
    ref_bias_a = np.array([0, 0, 0])
    ref_bias_phase = np.array([0., np.pi / 2, np.pi / 3])
    env = uav_inner_loop(uav_param, att_ctrl_param, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
    env.msg_print_flag = True
    num = 0

    while num < 3:
        env.reset_random()
        env.draw_att_init_image()
        r = 0
        while not env.is_terminal:
            action = env.att_control(ref=env.ref, dot_ref=env.dot_ref, dot2_ref=np.zeros(3))
            # print(action)
            env.step_update(action=np.array(action))
            r += env.reward
            env.att_image = env.att_image_copy.copy()
            env.draw_att(env.ref)
            env.show_att_image(iswait=False)
        print(r)
        num += 1
    env.collector.plot_pqr()
    env.collector.plot_att()
    env.collector.plot_torque()
    plt.show()


if __name__ == '__main__':
    # test_uav_hover_outer_loop()
    test_uav_inner_loop()
