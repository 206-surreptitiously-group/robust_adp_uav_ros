import numpy as np
from uav import UAV, uav_param
from collector import data_collector
from FNTSMC import fntsmc_att, fntsmc_pos, fntsmc_param
from ref_cmd import *


class uav_hover_ctrl(UAV):
    def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param, pos_ctrl_param: fntsmc_param):
        super(uav_hover_ctrl, self).__init__(UAV_param)
        self.att_ctrl = fntsmc_att(att_ctrl_param)
        self.pos_ctrl = fntsmc_pos(pos_ctrl_param)

        self.collector = data_collector(round(self.time_max / self.dt))

        self.pos_ref = np.zeros(3)
        self.dot_pos_ref = np.zeros(3)
        self.att_ref = np.zeros(3)
        self.dot_att_ref = np.zeros(3)

        self.obs = np.zeros(3)

    def pos_control(self, ref: np.ndarray, obs: np.ndarray):
        """
        @param ref:         x_d y_d z_d
        @param obs:         observer
        @return:            ref_phi ref_theta throttle
        """
        self.pos_ref = ref
        self.obs = obs
        e = self.eta() - ref
        de = self.dot_eta() - self.dot_pos_ref
        self.pos_ctrl.control_update(self.kt, self.m, self.uav_vel(), e, de, np.zeros(3), obs)
        phi_d, theta_d, uf = self.uo_2_ref_angle_throttle()
        return phi_d, theta_d, uf

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
        self.dot_att_ref = ref
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
        ux = self.pos_ctrl.control[0]
        uy = self.pos_ctrl.control[1]
        uz = self.pos_ctrl.control[2]
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
        @param action:  油门 + 三个力矩
        @return:
        """
        data_block = {'time': self.time,  # simulation time
                      'control': action,  # actual control command
                      'ref_angle': self.att_ref,  # reference angle
                      'ref_pos': self.pos_ref,
                      'ref_vel': self.dot_pos_ref,
                      'd_out': dis / self.m,
                      'd_out_obs': self.obs,
                      'state': self.uav_state_call_back()}  # quadrotor state
        self.collector.record(data_block)
        self.rk44(action=action, dis=dis, n=1, att_only=False)
