from common.common_func import *
from environment.Color import Color
import cv2 as cv


class uav_param:
    def __init__(self):
        self.m: float = 0.8  # 无人机质量
        self.g: float = 9.8  # 重力加速度
        self.J: np.ndarray = np.array([4.212e-3, 4.212e-3, 8.255e-3])  # 转动惯量
        self.d: float = 0.12  # 机臂长度 'X'构型
        self.CT: float = 2.168e-6  # 螺旋桨升力系数
        self.CM: float = 2.136e-8  # 螺旋桨力矩系数
        self.J0: float = 1.01e-5  # 电机和螺旋桨的转动惯量
        self.kr: float = 1e-3  # 旋转阻尼系数
        self.kt: float = 1e-3  # 平移阻尼系数
        self.pos0: np.ndarray = np.array([0, 0, 0])
        self.vel0: np.ndarray = np.array([0, 0, 0])
        self.angle0: np.ndarray = np.array([0, 0, 0])
        self.pqr0: np.ndarray = np.array([0, 0, 0])
        self.dt = 0.01
        self.time_max = 30  # 每回合最大时间
        self.pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [0, 3]])  # 定义飞行区域，不可以出界
        self.att_zone = np.atleast_2d([[deg2rad(-45), deg2rad(45)], [deg2rad(-45), deg2rad(45)], [deg2rad(-120), deg2rad(120)]])


class UAV:
    def __init__(self, param: uav_param):
        self.param = param
        self.m = param.m
        self.g = param.g
        self.J = param.J
        self.d = param.d
        self.CT = param.CT
        self.CM = param.CM
        self.J0 = param.J0
        self.kr = param.kr
        self.kt = param.kt

        self.x = param.pos0[0]
        self.y = param.pos0[1]
        self.z = param.pos0[2]
        self.vx = param.vel0[0]
        self.vy = param.vel0[1]
        self.vz = param.vel0[2]
        self.phi = param.angle0[0]
        self.theta = param.angle0[1]
        self.psi = param.angle0[2]
        self.p = param.pqr0[0]
        self.q = param.pqr0[1]
        self.r = param.pqr0[2]

        self.dt = param.dt
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.time_max = param.time_max

        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩
        self.terminal_flag = 0

        self.pos_zone = param.pos_zone
        self.att_zone = param.att_zone
        self.x_min = self.pos_zone[0][0]
        self.x_max = self.pos_zone[0][1]
        self.y_min = self.pos_zone[1][0]
        self.y_max = self.pos_zone[1][1]
        self.z_min = self.pos_zone[2][0]
        self.z_max = self.pos_zone[2][1]

        '''opencv visualization'''
        self.width = 1200
        self.height = 400
        self.x_offset = 40
        self.y_offset = 40
        self.offset = 20
        self.wp = (self.width - 2 * self.x_offset - 4 * self.offset) / 3
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.pmx_p1 = self.wp / dx
        self.pmy_p1 = (self.height - 2 * self.y_offset) / dy
        self.pmx_p2 = self.wp / dy
        self.pmy_p2 = (self.height - 2 * self.y_offset) / dz
        self.pmx_p3 = self.wp / dz
        self.pmy_p3 = (self.height - 2 * self.y_offset) / dx
        '''opencv visualization'''

    def dis2pixel(self, coord, flag: str, offset):
        if flag == 'xoy':
            x = self.x_offset + (coord[0] - self.x_min) * self.pmx_p1
            y = self.height - self.y_offset - (coord[1] - self.y_min) * self.pmy_p1
            return int(x + offset[0]), int(y + offset[1])
        if flag == 'yoz':
            y = self.x_offset + (coord[1] - self.y_min) * self.pmx_p2
            z = self.height - self.y_offset - (coord[2] - self.z_min) * self.pmy_p2
            return int(y + offset[0]), int(z + offset[1])
        if flag == 'zox':
            z = self.x_offset + (coord[2] - self.z_min) * self.pmx_p3
            x = self.height - self.y_offset - (coord[0] - self.x_min) * self.pmy_p3
            return int(z + offset[0]), int(x + offset[1])
        return offset[0], offset[1]

    def dis2pixel_trajectory_numpy2d(self, traj: np.ndarray, flag: str, offset: list) -> np.ndarray:
        """
        @param traj:        无人机轨迹，N * 3
        @param flag:        xoy yoz zox
        @param offset:      偏移
        @return:
        """
        if flag == 'xoy':
            x = self.x_offset + (traj[:, 0] - self.x_min) * self.pmx_p1 + offset[0]
            y = self.height - self.y_offset - (traj[:, 1] - self.y_min) * self.pmy_p1 + offset[1]
            return np.vstack((x, y)).T
        if flag == 'yoz':
            y = self.x_offset + (traj[:, 1] - self.y_min) * self.pmx_p2 + offset[0]
            z = self.height - self.y_offset - (traj[:, 2] - self.z_min) * self.pmy_p2 + offset[1]
            return np.vstack((y, z)).T
        if flag == 'zox':
            z = self.x_offset + (traj[:, 2] - self.z_min) * self.pmx_p3 + offset[0]
            x = self.height - self.y_offset - (traj[:, 0] - self.x_min) * self.pmy_p3 + offset[1]
            return np.vstack((z, x)).T
        return np.array([])

    def draw_boundary_xoy(self):
        cv.rectangle(self.image,
                     self.dis2pixel([self.x_min, self.y_min, 0], 'xoy', [0, 0]),
                     self.dis2pixel([self.x_max, self.y_max, 0], 'xoy', [0, 0]),
                     Color().Black, 2)

    def draw_boundary_yoz(self):
        pt1 = self.dis2pixel([0, self.y_min, self.z_min],
                             'yoz',
                             [self.wp + 2 * self.offset, 0])
        pt2 = self.dis2pixel([0, self.y_max, self.z_max],
                             'yoz',
                             [self.wp + 2 * self.offset, 0])
        cv.rectangle(self.image, pt1, pt2, Color().Black, 2)

    def draw_boundary_zox(self):
        cv.rectangle(self.image,
                     self.dis2pixel([self.x_min, 0, self.z_min],
                                    'zox',
                                    [2 * self.wp + 4 * self.offset, 0]),
                     self.dis2pixel([self.x_max, 0, self.z_max],
                                    'zox',
                                    [2 * self.wp + 4 * self.offset, 0]),
                     Color().Black, 2)

    def draw_boundary(self):
        self.draw_boundary_xoy()
        self.draw_boundary_yoz()
        self.draw_boundary_zox()

    def draw_label(self):
        pts = [self.dis2pixel([(self.x_min + self.x_max) / 2, self.y_min, 0], 'xoy', [-5, -5]),
               self.dis2pixel([self.x_min, (self.y_min + self.y_max) / 2, 0], 'xoy', [5, 0]),
               self.dis2pixel([0, (self.y_min + self.y_max) / 2, self.z_min], 'yoz', [self.wp + 2 * self.offset - 5, -5]),
               self.dis2pixel([0, self.y_min, (self.z_min + self.z_max) / 2], 'yoz', [self.wp + 2 * self.offset + 5, 0]),
               self.dis2pixel([self.x_min, 0, (self.z_min + self.z_max) / 2], 'zox', [2 * self.wp + 4 * self.offset - 5, -5]),
               self.dis2pixel([(self.x_min + self.x_max) / 2, 0, self.z_min], 'zox', [2 * self.wp + 4 * self.offset + 5, 0]),
               (int(self.width / 2 - 55), 20)]
        labels = ['X', 'Y', 'Y', 'Z', 'Z', 'X', 'Projection']
        for _l, _pt in zip(labels, pts):
            cv.putText(self.image, _l, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

    def draw_region_grid(self, xNum: int, yNum: int, zNum: int):
        _dx = (self.x_max - self.x_min) / xNum
        _dy = (self.y_max - self.y_min) / yNum
        _dz = (self.z_max - self.z_min) / zNum

        '''X'''
        for i in range(yNum - 1):
            pt1 = self.dis2pixel([self.x_min, self.y_min + (i + 1) * _dy, 0.], 'xoy', [0, 0])
            pt2 = self.dis2pixel([self.x_max, self.y_min + (i + 1) * _dy, 0.], 'xoy', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        for i in range(zNum - 1):
            pt1 = self.dis2pixel([self.x_min, 0., self.z_min + (i + 1) * _dz], 'zox', [2 * self.wp + 4 * self.offset, 0])
            pt2 = self.dis2pixel([self.x_max, 0., self.z_min + (i + 1) * _dz], 'zox', [2 * self.wp + 4 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)

        '''Y'''
        for i in range(xNum - 1):
            pt1 = self.dis2pixel([self.x_min + (i + 1) * _dx, self.y_min, 0.], 'xoy', [0, 0])
            pt2 = self.dis2pixel([self.x_min + (i + 1) * _dx, self.y_max, 0.], 'xoy', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        for i in range(zNum - 1):
            pt1 = self.dis2pixel([0., self.y_min, self.z_min + (i + 1) * _dz], 'yoz', [self.wp + 2 * self.offset, 0])
            pt2 = self.dis2pixel([0., self.y_max, self.z_min + (i + 1) * _dz], 'yoz', [self.wp + 2 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)

        '''Z'''
        for i in range(yNum - 1):
            pt1 = self.dis2pixel([0., self.y_min + (i + 1) * _dy, self.z_min], 'yoz', [self.wp + 2 * self.offset, 0])
            pt2 = self.dis2pixel([0., self.y_min + (i + 1) * _dy, self.z_max], 'yoz', [self.wp + 2 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        for i in range(xNum - 1):
            pt1 = self.dis2pixel([self.x_min + (i + 1) * _dx, 0., self.z_min], 'zox', [2 * self.wp + 4 * self.offset, 0])
            pt2 = self.dis2pixel([self.x_min + (i + 1) * _dx, 0., self.z_max], 'zox', [2 * self.wp + 4 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)

        self.draw_axis(xNum, yNum, zNum)

    def draw_axis(self, xNum: int, yNum: int, zNum: int):
        _dx = (self.x_max - self.x_min) / xNum
        _dy = (self.y_max - self.y_min) / yNum
        _dz = (self.z_max - self.z_min) / zNum

        _x = np.linspace(self.x_min, self.x_max, xNum + 1)
        _y = np.linspace(self.y_min, self.y_max, yNum + 1)
        _z = np.linspace(self.z_min, self.z_max, zNum + 1)

        for __x in _x:
            if np.fabs(round(__x, 2) - int(__x)) < 0.01:
                _s = str(int(__x))
            else:
                _s = str(round(__x, 2))
            _pt = self.dis2pixel([__x, self.y_min, 0], 'xoy', [-20 if __x < 0 else -7, 20])
            _pt2 = self.dis2pixel([__x, 0., self.z_min],
                                  'zox',
                                  [2 * self.wp + 4 * self.offset - 30 if __x < 0 else 2 * self.wp + 4 * self.offset - 15, 5])
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
            cv.putText(self.image, _s, _pt2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

        for __y in _y:
            if np.fabs(round(__y, 2) - int(__y)) < 0.01:
                _s = str(int(__y))
            else:
                _s = str(round(__y, 2))
            _pt = self.dis2pixel([self.x_min, __y, 0], 'xoy', [-30 if __y < 0 else -15, 7])
            _pt2 = self.dis2pixel([0., __y, self.z_min],
                                  'yoz',
                                  [self.wp + 2 * self.offset - 15 if __y < 0 else self.wp + 2 * self.offset - 5, 20])
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
            cv.putText(self.image, _s, _pt2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

        for __z in _z:
            if np.fabs(round(__z, 2) - int(__z)) < 0.01:  # 是整数
                _s = str(int(__z))
                _pt = self.dis2pixel([0., self.y_min, __z], 'yoz', [self.wp + 2 * self.offset - 20, 7])
                _pt2 = self.dis2pixel([self.x_min, 0., __z], 'zox', [2 * self.wp + 4 * self.offset - 10, 20])
            else:
                _s = str(round(__z, 2))
                _pt = self.dis2pixel([0., self.y_min, __z], 'yoz', [self.wp + 2 * self.offset - 30, 7])
                _pt2 = self.dis2pixel([self.x_min, 0., __z], 'zox', [2 * self.wp + 4 * self.offset - 15, 20])
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
            cv.putText(self.image, _s, _pt2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

    def draw_3d_points_projection(self, points: np.ndarray, colors: list):
        """
        @param colors:
        @param colors:
        @param points:
        @return:
        """
        '''XOY'''
        xy = self.dis2pixel_trajectory_numpy2d(points, 'xoy', [0, 0])
        _l = xy.shape[0]    # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(xy[i][0])), int(round(xy[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)

        '''YOZ'''
        yz = self.dis2pixel_trajectory_numpy2d(points, 'yoz', [self.wp + 2 * self.offset, 0])
        _l = yz.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(yz[i][0])), int(round(yz[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)

        '''ZOX'''
        zx = self.dis2pixel_trajectory_numpy2d(points, 'zox', [2 * self.wp + 4 * self.offset, 0])
        _l = zx.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(zx[i][0])), int(round(zx[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)

    def draw_3d_trajectory_projection(self, trajectory: np.ndarray):
        """
        @param trajectory:
        @return:
        """
        '''XOY'''
        xy = self.dis2pixel_trajectory_numpy2d(trajectory, 'xoy', [0, 0])
        _l = xy.shape[0]    # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(xy[i][0])), int(round(xy[i][1])))
            pt2 = (int(round(xy[i + 1][0])), int(round(xy[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)

        '''YOZ'''
        yz = self.dis2pixel_trajectory_numpy2d(trajectory, 'yoz', [self.wp + 2 * self.offset, 0])
        _l = yz.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(yz[i][0])), int(round(yz[i][1])))
            pt2 = (int(round(yz[i + 1][0])), int(round(yz[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)

        '''ZOX'''
        zx = self.dis2pixel_trajectory_numpy2d(trajectory, 'zox', [2 * self.wp + 4 * self.offset, 0])
        _l = zx.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(zx[i][0])), int(round(zx[i][1])))
            pt2 = (int(round(zx[i + 1][0])), int(round(zx[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)

    def draw_error(self, pos: np.ndarray, ref: np.ndarray):
        """
        @param pos:
        @param ref:
        @return:
        """
        e = pos - ref
        _str = '[%.2f, %.2f, %.2f]' % (e[0], e[1], e[2])
        cv.putText(self.image, _str, (self.x_offset, self.y_offset - 5), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 2)

    def draw_init_image(self):
        self.draw_boundary()
        self.draw_label()
        self.draw_region_grid(6, 6, 6)
        self.draw_axis(6, 6, 6)
        self.image_copy = self.image.copy()

    def show_image(self, iswait: bool = False):
        if iswait:
            cv.imshow('Projection', self.image)
            cv.waitKey(0)
        else:
            cv.imshow('Projection', self.image)
            cv.waitKey(1)

    def ode(self, xx: np.ndarray, dis: np.ndarray):
        """
        @param xx:      state of the uav
        @param dis:     disturbances
        @return:        dot_xx
        """
        [_x, _y, _z, _vx, _vy, _vz, _phi, _theta, _psi, _p, _q, _r] = xx[0:12]
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        self.J0 = 0.  # 不考虑陀螺力矩，用于分析观测器的效果
        dp = (-self.kr * _p - _q * _r * (self.J[2] - self.J[1]) + self.torque[0]) / self.J[0]
        dq = (-self.kr * _q - _p * _r * (self.J[0] - self.J[2]) + self.torque[1]) / self.J[1]
        dr = (-self.kr * _r - _p * _q * (self.J[1] - self.J[0]) + self.torque[2]) / self.J[2]
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''

        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''
        _R_pqr2diner = np.array([[1, np.tan(_theta) * np.sin(_phi), np.tan(_theta) * np.cos(_phi)],
                                 [0, np.cos(_phi), -np.sin(_phi)],
                                 [0, np.sin(_phi) / np.cos(_theta), np.cos(_phi) / np.cos(_theta)]])
        [dphi, dtheta, dpsi] = np.dot(_R_pqr2diner, [_p, _q, _r]).tolist()
        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''

        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        [dx, dy, dz] = [_vx, _vy, _vz]
        dvx = (self.throttle * (np.cos(_psi) * np.sin(_theta) * np.cos(_phi) + np.sin(_psi) * np.sin(_phi))
               - self.kt * _vx + dis[0]) / self.m
        dvy = (self.throttle * (np.sin(_psi) * np.sin(_theta) * np.cos(_phi) - np.cos(_psi) * np.sin(_phi))
               - self.kt * _vy + dis[1]) / self.m
        dvz = -self.g + (self.throttle * np.cos(_phi) * np.cos(_theta)
                         - self.kt * _vz + dis[2]) / self.m
        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''

        return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

    def rk44(self, action: np.ndarray, dis: np.ndarray, n: int = 10, att_only: bool = False):
        self.throttle = action[0]
        self.torque = action[1: 4]
        h = self.dt / n  # RK-44 解算步长
        cc = 0
        xx = self.uav_state_call_back()
        while cc < n:
            K1 = h * self.ode(xx, dis)
            K2 = h * self.ode(xx + K1 / 2, dis)
            K3 = h * self.ode(xx + K2 / 2, dis)
            K4 = h * self.ode(xx + K3, dis)
            xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            cc += 1
        if att_only:
            xx[0:6] = np.zeros(6)[:]
        self.set_state(xx)
        self.time += self.dt
        if self.psi > np.pi:  # 如果角度超过 180 度
            self.psi -= 2 * np.pi
        if self.psi < -np.pi:  # 如果角度小于 -180 度
            self.psi += 2 * np.pi
        self.n += 1  # 拍数 +1

    def uav_state_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def uav_pos_vel_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz])

    def uav_att_pqr_call_back(self):
        return np.array([self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def uav_pos(self):
        return np.array([self.x, self.y, self.z])

    def uav_vel(self):
        return np.array([self.vx, self.vy, self.vz])

    def uav_att(self):
        return np.array([self.phi, self.theta, self.psi])

    def uav_pqr(self):
        return np.array([self.p, self.q, self.r])

    def set_state(self, xx: np.ndarray):
        [self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r] = xx[:]

    def is_pos_out(self) -> bool:
        _flag = False
        if (self.x < self.x_min) or (self.x > self.x_max):
            print('XOUT!!!!!')
            _flag = True
        if (self.x < self.y_min) or (self.x > self.y_max):
            print('YOUT!!!!!')
            _flag = True
        if (self.x < self.z_min) or (self.x > self.z_max):
            print('ZOUT!!!!!')
            _flag = True
        return _flag

    def is_att_out(self) -> bool:
        _flag = False
        if (self.phi < self.att_zone[0][0]) or (self.phi > self.att_zone[0][1]):
            print('Phi OUT!!!!!')
            _flag = True
        if (self.theta < self.att_zone[1][0]) or (self.theta > self.att_zone[1][1]):
            print('Theta OUT!!!!!')
            _flag = True
        if (self.psi < self.att_zone[2][0]) or (self.psi > self.att_zone[2][1]):
            print('Yaw OUT!!!!!')
            _flag = True
        return _flag

    def is_episode_Terminal(self) -> tuple:
        _terminal = False
        if self.time > self.time_max - self.dt / 2:
            print('Time out...')
            self.terminal_flag = 1
            _terminal = True
        if self.is_pos_out():
            print('Position out...')
            self.terminal_flag = 2
            _terminal = True
        if self.is_att_out():
            print('Attitude out...')
            self.terminal_flag = 3
            _terminal = True
        return _terminal, self.terminal_flag

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

    def reset_with_param(self, new_param: uav_param):
        self.param = new_param
        self.reset()

    def f1(self) -> np.ndarray:
        """
        :brief:  [1  sin(phi)tan(theta)      cos(phi)tan(theta)]
                 [0       cos(phi)               -sin(phi)     ]
                 [0  sin(phi)/cos(theta)   -cos(phi)/cos(theta)]
        :return: f1(rho_1)
        """
        _f1_rho1 = np.zeros((3, 3)).astype(float)
        _f1_rho1[0][0] = 1.
        _f1_rho1[0][1] = np.sin(self.phi) * np.tan(self.theta)
        _f1_rho1[0][2] = np.cos(self.phi) * np.tan(self.theta)
        _f1_rho1[1][1] = np.cos(self.phi)
        _f1_rho1[1][2] = -np.sin(self.phi)
        _f1_rho1[2][1] = np.sin(self.phi) / np.cos(self.theta)
        _f1_rho1[2][2] = np.cos(self.phi) / np.cos(self.theta)
        return _f1_rho1

    def f2(self) -> np.ndarray:
        """
        :brief:  [(kr * p + qr * (Iyy - Izz)) / Ixx]
                 [(kr * q + pr * (Izz - Ixx)) / Iyy]
                 [(kr * r + pq * (Ixx - Iyy)) / Izz]
        :return: f2(rho_2)
        """
        _f2_rho2 = np.array([0, 0, 0]).astype(float)
        _f2_rho2[0] = (self.kr * self.p + self.q * self.r * (self.J[1] - self.J[2])) / self.J[0]
        _f2_rho2[1] = (self.kr * self.q + self.p * self.r * (self.J[2] - self.J[0])) / self.J[1]
        _f2_rho2[2] = (self.kr * self.r + self.p * self.q * (self.J[0] - self.J[1])) / self.J[2]
        return _f2_rho2

    def h(self) -> np.ndarray:
        """
        :brief:  [        0             1/Jxx    0       0 ]
                 [        0               0    1/Jyy     0 ]
                 [        0               0      0    1/Jzz]
        :return: h(rho_1)
        """
        _g = np.zeros((3, 3)).astype(float)
        _g[0][0] = 1 / self.J[0]
        _g[1][1] = 1 / self.J[1]
        _g[2][2] = 1 / self.J[2]
        return _g

    def rho1(self) -> np.ndarray:
        return np.array([self.phi, self.theta, self.psi])

    def rho2(self) -> np.ndarray:
        return np.array([self.p, self.q, self.r])

    def dot_rho1(self) -> np.ndarray:
        return np.dot(self.f1(), self.rho2())

    def dot_rho2(self) -> np.ndarray:
        return self.f2() + np.dot(self.h(), self.torque)

    def F(self) -> np.ndarray:
        dot_rho1 = self.dot_rho1()  # dphi dtheta dpsi
        _dot_f1_rho1 = np.zeros((3, 3)).astype(float)
        _dot_f1_rho1[0][1] = dot_rho1[0] * np.tan(self.theta) * np.cos(self.phi) + dot_rho1[1] * np.sin(self.phi) / np.cos(self.theta) ** 2
        _dot_f1_rho1[0][2] = -dot_rho1[0] * np.tan(self.theta) * np.sin(self.phi) + dot_rho1[1] * np.cos(self.phi) / np.cos(self.theta) ** 2

        _dot_f1_rho1[1][1] = -dot_rho1[0] * np.sin(self.phi)
        _dot_f1_rho1[1][2] = -dot_rho1[0] * np.cos(self.phi)

        _temp1 = dot_rho1[0] * np.cos(self.phi) * np.cos(self.theta) + dot_rho1[1] * np.sin(self.phi) * np.sin(self.theta)
        _dot_f1_rho1[2][1] = _temp1 / np.cos(self.theta) ** 2

        _temp2 = -dot_rho1[0] * np.sin(self.phi) * np.cos(self.theta) + dot_rho1[1] * np.cos(self.phi) * np.sin(self.theta)
        _dot_f1_rho1[2][2] = _temp2 / np.cos(self.theta) ** 2
        return _dot_f1_rho1

    def second_order_att_dynamics(self) -> np.ndarray:
        return np.dot(self.F(), self.rho2()) + np.dot(self.f1(), self.f2())

    def att_control_matrix(self) -> np.ndarray:
        return np.dot(self.f1(), self.h())

    def eta(self):
        return np.array([self.x, self.y, self.z])

    def dot_eta(self):
        return np.array([self.vx, self.vy, self.vz])
