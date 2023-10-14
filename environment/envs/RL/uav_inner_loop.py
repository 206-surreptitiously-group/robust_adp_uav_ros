import math
import os
import sys
from numpy import deg2rad

from algorithm.rl_base.rl_base import rl_base
from environment.envs.UAV.FNTSMC import fntsmc_att, fntsmc_param
from environment.envs.UAV.collector import data_collector
from environment.envs.UAV.ref_cmd import *
from environment.envs.UAV.uav import UAV, uav_param
from environment.envs.UAV.uav_att_ctrl import uav_att_ctrl

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../'))


class uav_inner_loop(rl_base, uav_att_ctrl):
    def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param):
        rl_base.__init__(self)
        uav_att_ctrl.__init__(self, UAV_param, att_ctrl_param)

        self.uav_param = UAV_param
        self.name = 'uav_inner_loop'

        self.collector = data_collector(round(self.time_max / self.dt))
        self.ref = np.zeros(3)
        self.dot_ref = np.zeros(3)
        self.error = self.uav_att() - self.ref
        self.dot_error = self.dot_rho1() - self.dot_ref

        '''state action limitation'''
        self.static_gain = 1.0

        self.e_att_max = np.array([deg2rad(45), deg2rad(45), deg2rad(120)])
        self.e_att_min = -np.array([deg2rad(45), deg2rad(45), deg2rad(120)])
        self.e_dot_att_max = np.array([deg2rad(45), deg2rad(45), deg2rad(120)])
        self.e_dot_att_min = -np.array([deg2rad(45), deg2rad(45), deg2rad(120)])
        self.torque_min = -0.3
        self.torque_max = 0.3
        '''state action limitation'''

        '''rl_base'''
        self.state_dim = 6  # ephi etheta epsi edot_phi edot_theta edot_psi
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.static_gain, self.static_gain] for _ in range(self.state_dim)]
        self.is_state_continuous = [True for _ in range(self.state_dim)]

        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 3  # Tx Ty Tz
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.action_range = [[self.torque_min, self.torque_max],
                             [self.torque_min, self.torque_max],
                             [self.torque_min, self.torque_max]]
        self.is_action_continuous = [True for _ in range(self.action_dim)]

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞
        '''rl_base'''

    def state_norm(self) -> np.ndarray:
        """
        RL状态归一化
        """
        norm_error = self.error / (self.e_att_max - self.e_att_min) * self.static_gain
        norm_dot_error = self.dot_error / (self.e_dot_att_min - self.e_dot_att_max) * self.static_gain
        norm_state = np.concatenate((norm_error, norm_dot_error))

        return norm_state

    def get_reward(self, param=None):
        """
        计算奖励
        """
        Qx, Qv, R = 5, 0.1, 0.05
        r1 = - np.linalg.norm(self.error) ** 2 * Qx
        r2 = - np.linalg.norm(self.dot_error) ** 2 * Qv
        # norm_action = (np.array(self.current_action) * 2 - self.u_max - self.u_min) / (self.u_max - self.u_min)
        r3 = - np.linalg.norm(self.current_action) ** 2 * R

        r4 = 0
        # 如果因为越界终止，则给剩余时间可能取得的最大惩罚
        if self.is_att_out():
            r4 = - (self.time_max - self.time) / self.dt * (Qx * np.linalg.norm(self.error) ** 2
                                                            + Qv * np.linalg.norm(self.dot_error) ** 2
                                                            + R * np.linalg.norm(self.current_action) ** 2)
        self.reward = r1 + r2 + r3 + r4

    def is_Terminal(self, param=None):
        self.is_terminal, self.terminal_flag = self.is_episode_Terminal()

    def step_update(self, action: np.ndarray):
        """
        @param action:  三轴转矩指令 Tx Ty Tz
        @return:
        """
        self.current_action = action.copy()
        self.current_state = self.state_norm()

        # 内环由RL给出
        self.torque = action.copy()

        self.update(action=self.torque)
        self.error = self.uav_att() - self.ref
        self.dot_error = self.dot_rho1() - self.dot_ref

        self.is_Terminal()
        self.next_state = self.state_norm()

        self.get_reward()

    def reset(self):
        self.m = self.param.m
        self.g = self.param.g
        self.J = self.param.J
        self.d = self.param.d
        self.CT = self.param.CT
        self.CM = self.param.CM
        self.J0 = self.param.J0
        self.kr = self.param.kr
        self.kt = self.param.kt

        self.x = self.param.pos0[0]
        self.y = self.param.pos0[1]
        self.z = self.param.pos0[2]
        self.vx = self.param.vel0[0]
        self.vy = self.param.vel0[1]
        self.vz = self.param.vel0[2]
        self.phi = self.param.angle0[0]
        self.theta = self.param.angle0[1]
        self.psi = self.param.angle0[2]
        self.p = self.param.pqr0[0]
        self.q = self.param.pqr0[1]
        self.r = self.param.pqr0[2]

        self.dt = self.param.dt
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.time_max = self.param.time_max

        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩
        self.terminal_flag = 0

        self.pos_zone = self.param.pos_zone
        self.att_zone = self.param.att_zone
        self.x_min = self.pos_zone[0][0]
        self.x_max = self.pos_zone[0][1]
        self.y_min = self.pos_zone[1][0]
        self.y_max = self.pos_zone[1][1]
        self.z_min = self.pos_zone[2][0]
        self.z_max = self.pos_zone[2][1]

        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()

        self.ref = np.zeros(3)
        self.dot_ref = np.zeros(3)
        self.error = self.uav_att() - self.ref
        self.dot_error = self.dot_rho1() - self.dot_ref
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.collector.reset(round(self.time_max / self.dt))
        self.reward = 0.0
        self.is_terminal = False

    def reset_random(self):
        """
        随机生成目标姿态角和初始姿态角
        """
        self.reset()
        self.ref = self.generate_random_att(offset=deg2rad(1))  # 随机目标姿态
        # self.phi, self.theta, self.psi = self.generate_random_att(offset=deg2rad(1))  # 随机初始姿态

    def generate_random_att(self, offset: float):
        """
        在姿态角范围内随即选择一个姿态，offset防止姿态越界
        """
        return np.random.uniform(low=self.att_zone[:, 0] + offset, high=self.att_zone[:, 1] - offset)
