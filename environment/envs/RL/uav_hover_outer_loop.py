import math
import os
import sys

import numpy as np

from algorithm.rl_base.rl_base import rl_base
from environment.envs.UAV.FNTSMC import fntsmc_att, fntsmc_param
from environment.envs.UAV.collector import data_collector
from environment.envs.UAV.ref_cmd import *
from environment.envs.UAV.uav import UAV, uav_param
# from environment.envs.UAV.uav_visualization import UAV_Visualization

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../'))


class uav_hover_outer_loop(rl_base, UAV):
    def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param, target: np.ndarray):
        rl_base.__init__(self)
        super(uav_hover_outer_loop, self).__init__(UAV_param)
        self.param = UAV_param
        self.att_ctrl = fntsmc_att(att_ctrl_param)
        self.name = 'uav_hover_outer_loop'

        self.collector = data_collector(round(self.time_max / self.dt))

        self.pos_ref = target
        self.dot_pos_ref = np.zeros(3)
        self.att_ref = np.zeros(3)
        self.att_ref_old = np.zeros(3)
        self.dot_att_ref = (self.att_ref - self.att_ref_old) / self.dt

        self.obs = np.zeros(3)

        self.error = self.uav_pos() - self.pos_ref
        self.static_gain = 1.0

        self.pos_min = np.array([-5, -5, -5])
        self.pos_max = np.array([5, 5, 5])
        self.vel_min = np.array([-3, -3, -3])
        self.vel_max = np.array([3, 3, 3])
        self.u_min = -10
        self.u_max = 10

        '''rl_base'''
        self.state_dim = 3 + 3  # ex ey ez vx vy vz
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.static_gain, self.static_gain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]

        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 3  # ux uy uz
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.action_range = [[self.u_min, self.u_max] for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞
        '''rl_base'''

        '''Rviz visualization'''
        # self.uav_vis = UAV_Visualization()
        '''Rviz visualization'''

    def state_norm(self) -> np.ndarray:
        """
        RL状态归一化
        """
        norm_error = (2 * self.error - self.pos_max - self.pos_min) / (self.pos_max - self.pos_min) * self.static_gain
        norm_vel = (2 * self.uav_vel() - self.vel_max - self.vel_min) / (self.vel_max - self.vel_min) * self.static_gain
        norm_state = np.concatenate((norm_error, norm_vel))

        return norm_state

    def get_reward(self):
        """
        计算奖励时用归一化后的状态和动作，才能保证系数设置有效
        """
        Qx, Qv, R = 1, 0, 0
        r1 = - np.linalg.norm(self.next_state[:3]) ** 2 * Qx
        r2 = - np.linalg.norm(self.next_state[3:]) ** 2 * Qv

        norm_action = (np.array(self.current_action) * 2 - self.u_max - self.u_min) / (self.u_max - self.u_min)
        r3 = - np.linalg.norm(norm_action) ** 2 * R

        r4 = 0
        if self.terminal_flag == 1:
            r4 = - 2000

        self.reward = r1 + r2 + r3 + r4

    def is_Terminal(self):
        if self.time > self.time_max:
            self.terminal_flag = 2
            return True

        if any(np.greater(self.uav_pos(), self.pos_max)) or any(np.less(self.uav_pos(), self.pos_min)):
            self.terminal_flag = 1
            return True

        self.terminal_flag = 0
        self.is_terminal = False
        return False

    def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, att_only: bool = False):
        """
        @param ref:         参考信号
        @param dot_ref:     参考信号一阶导数
        @param dot2_ref:    参考信号二阶导数 (仅在姿态控制模式有效)
        @param att_only:    为 True 时，dot2_ref 正常输入
                            为 True 时，dot2_ref 为 0
        @return:            Tx Ty Tz
        """
        self.att_ref = ref
        self.dot_att_ref = dot_ref
        if not att_only:
            dot2_ref = np.zeros(3)

        e = self.rho1() - ref
        de = self.dot_rho1() - dot_ref
        sec_order_att_dy = self.second_order_att_dynamics()
        ctrl_mat = self.att_control_matrix()
        self.att_ctrl.control_update(sec_order_att_dy, ctrl_mat, e, de, dot2_ref)
        return self.att_ctrl.control

    def uo_2_ref_angle_throttle(self):
        # print('fuck', uf)
        ux, uy, uz = self.current_action
        uf = (uz + self.g) * self.m / (np.cos(self.phi) * np.cos(self.theta))
        asin_phi_d = min(max((ux * np.sin(self.psi) - uy * np.cos(self.psi)) * self.m / uf, -1), 1)
        phi_d = np.arcsin(asin_phi_d)
        asin_theta_d = min(max((ux * np.cos(self.psi) + uy * np.sin(self.psi)) * self.m / (uf * np.cos(phi_d)), -1), 1)
        theta_d = np.arcsin(asin_theta_d)
        # print(phi_d * 180 / np.pi, theta_d * 180 / np.pi)
        return phi_d, theta_d, uf

    def update(self, action: np.ndarray, dis: np.ndarray):
        """
        @param dis:     uncertainty
        @param action:  三轴加速度指令 ux uy uz
        @return:
        """
        self.current_action = action.copy()
        self.current_state = self.state_norm()

        phi_d, theta_d, uf = self.uo_2_ref_angle_throttle()

        self.att_ref_old = self.att_ref
        self.att_ref = np.array([phi_d, theta_d, 0.0])  # 偏航角手动设置为0
        self.dot_att_ref = (self.att_ref - self.att_ref_old) / self.dt

        torque = self.att_control(self.att_ref, self.dot_att_ref, np.zeros(3), att_only=False)  # 内环fntsmc控制
        a = np.concatenate(([uf], torque))  # 真实控制量

        data_block = {'time': self.time,  # simulation time
                      'control': a,  # actual control command
                      'ref_angle': self.att_ref,  # reference angle
                      'ref_pos': self.pos_ref,
                      'ref_vel': self.dot_pos_ref,
                      'd_out': dis / self.m,
                      'd_out_obs': self.obs,
                      'state': self.uav_state_call_back()}  # quadrotor state
        self.collector.record(data_block)
        self.rk44(action=a, dis=dis, n=1, att_only=False)
        self.vx, self.vy, self.vz = np.clip(self.uav_vel(), self.vel_min, self.vel_max)
        self.error = self.uav_pos() - self.pos_ref

        self.is_terminal = self.is_Terminal()
        self.next_state = self.state_norm()

        self.get_reward()

    def reset(self):
        self.set_state(np.concatenate((self.param.pos0, self.param.vel0, self.param.angle0, self.param.pqr0)))
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩

        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        # self.uav_vis.render(uav_pos=self.uav_pos(),
        #                     uav_pos_ref=self.pos_ref,
        #                     uav_att=self.uav_att(),
        #                     uav_att_ref=self.att_ref,
        #                     d=4 * self.d)
        self.collector = data_collector(round(self.time_max / self.dt))
        self.reward = 0.0
        self.is_terminal = False
