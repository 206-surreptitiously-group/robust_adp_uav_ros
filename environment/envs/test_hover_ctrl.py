#!/usr/bin/python3

import datetime
import os, sys
import matplotlib.pyplot as plt
import rospy

from UAV.uav_visualization import UAV_Visualization
from UAV.FNTSMC import fntsmc_param
from UAV.ref_cmd import *
from UAV.uav import uav_param
from UAV.uav_hover_ctrl import uav_hover_ctrl

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from common.common_func import *

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
uav_param.time_max = 20
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
'''Parameter list of the position controller'''

if __name__ == '__main__':
    rospy.init_node(name='test_hover_ctrl', anonymous=False)
    quad_vis = UAV_Visualization()

    rate = rospy.Rate(1 / DT)

    '''1. Define a controller'''
    hover_ctrl = uav_hover_ctrl(uav_param, att_ctrl_param, pos_ctrl_param)

    '''2. Define parameters for signal generator'''
    ref = np.array([5, 5, 5, deg2rad(0)])

    phi_d = phi_d_old = 0.
    theta_d = theta_d_old = 0.
    dot_phi_d = (phi_d - phi_d_old) / hover_ctrl.dt
    dot_theta_d = (theta_d - theta_d_old) / hover_ctrl.dt
    throttle = hover_ctrl.m * hover_ctrl.g

    '''3. Control'''
    while (hover_ctrl.time < hover_ctrl.time_max) and (not rospy.is_shutdown()):
        if hover_ctrl.n % 1000 == 0:
            print('time: ', hover_ctrl.n * hover_ctrl.dt)

        '''3.1 generate '''
        uncertainty = generate_uncertainty(time=hover_ctrl.time, is_ideal=True)
        obs = np.zeros(3)

        '''3.2 outer-loop control'''
        phi_d_old = phi_d
        theta_d_old = theta_d
        phi_d, theta_d, throttle = hover_ctrl.pos_control(ref[0:3], obs)
        dot_phi_d = (phi_d - phi_d_old) / hover_ctrl.dt
        dot_theta_d = (theta_d - theta_d_old) / hover_ctrl.dt

        '''3.3 inner-loop control'''
        rho_d = np.array([phi_d, theta_d, ref[3]])
        dot_rho_d = np.array([dot_phi_d, dot_theta_d, 0])
        torque = hover_ctrl.att_control(rho_d, dot_rho_d, np.zeros(3), att_only=False)

        '''3.4 update state'''
        action_4_uav = np.array([throttle, torque[0], torque[1], torque[2]])
        hover_ctrl.update(action=action_4_uav, dis=uncertainty)

        '''3.3. publish'''
        quad_vis.render(uav_pos=hover_ctrl.uav_pos(),
                        uav_pos_ref=hover_ctrl.pos_ref,
                        uav_att=hover_ctrl.uav_att(),
                        uav_att_ref=hover_ctrl.att_ref,
                        d=4 * hover_ctrl.d)      # to make it clearer, we increase size ten times

        rate.sleep()
    print('Finish...')
    SAVE = True
    if SAVE:
        new_path = (os.path.dirname(os.path.abspath(__file__)) +
                    '/../../datasave/' +
                    datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/')
        os.mkdir(new_path)
        hover_ctrl.collector.package2file(path=new_path)
    hover_ctrl.collector.plot_att()
    hover_ctrl.collector.plot_torque()
    hover_ctrl.collector.plot_pos()
    hover_ctrl.collector.plot_vel()
    hover_ctrl.collector.plot_throttle()
    hover_ctrl.collector.plot_outer_obs()
    plt.show()
