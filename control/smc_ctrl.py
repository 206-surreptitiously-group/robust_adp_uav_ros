import numpy as np
from environment.UAV.uav_control import SMCControl
import matplotlib.pyplot as plt
import pandas as pd


IS_IDEAL = False


if __name__ == '__main__':
    ctrl = SMCControl()      # 直接初始化完成

    '''一些中间变量初始化'''
    # TODO 全都按照 0 时刻的初始值去初始化

    dot_inner_u = np.array([0, 0, 0, 0]).astype(float)
    uncertainty = ctrl.generate_uncertainty()
    fake_rhod, fake_dot_rhod, fake_dotdot_rhod, fake_dotdotdot_rhod = ctrl.fake_inner_cmd_generator()  # 参考信号肯定是已知的，这不用说，因为这是认为定义好的

    e_I = ctrl.rho1() - fake_rhod
    de_I = ctrl.dot_rho1() - fake_dot_rhod
    de_I_old = de_I.copy()
    '''一些中间变量初始化'''

    '''数据存储'''
    N = int(ctrl.time_max / ctrl.dt)
    save_t = np.zeros((N, 1)).astype(float)
    save_inner_control = np.zeros((N, 4)).astype(float)
    save_ref_angle = np.zeros((N, 3)).astype(float)
    save_ref_pos = np.zeros((N, 3)).astype(float)
    real_delta_inner = np.zeros((N, 4)).astype(float)
    delta_inner_obs = np.zeros((N, 4)).astype(float)
    save_state = np.zeros((N, 12)).astype(float)
    '''数据存储'''

    while ctrl.time < ctrl.time_max:
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''
        uncertainty = ctrl.generate_uncertainty()
        fake_rhod, fake_dot_rhod, fake_dotdot_rhod, fake_dotdotdot_rhod = ctrl.fake_inner_cmd_generator()

        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''

        save_t[ctrl.n][0] = ctrl.time
        _un_inner = np.array([uncertainty[2] / ctrl.m, uncertainty[3], uncertainty[4], uncertainty[5]])
        _equ_delta = np.dot(ctrl.f1_rho1(), _un_inner) - fake_dotdot_rhod
        real_delta_inner[ctrl.n][:] = _equ_delta[:]
        save_inner_control[ctrl.n][:] = ctrl.inner_control[:]
        save_ref_pos[ctrl.n][:] = np.array([ctrl.x, ctrl.y, fake_rhod[0]])
        save_ref_angle[ctrl.n][:] = fake_rhod[1: 4]
        save_state[ctrl.n][:] = ctrl.uav_state_call_back()

        '''2. 计算 tk 时刻误差信号'''
        de_I_old = de_I.copy()                  # 这个时候 de_I 是上一时刻的
        e_I = ctrl.rho1() - fake_rhod
        de_I = ctrl.dot_rho1() - fake_dot_rhod  # 这个时候 de_I 是新时刻的
        # 先观测一下
        # if ctrl.time > 3.49:
        #     print('嗨嗨嗨')
        delta_obs, dot_delta_obs = ctrl.inner_obs.observe(ctrl.dot_f1_rho1(),
                                                          ctrl.rho2(),
                                                          ctrl.f1_rho1(),
                                                          ctrl.f2_rho2(),
                                                          ctrl.g_rho1(),
                                                          ctrl.inner_control,
                                                          de_I_old,
                                                          de_I)
        delta_inner_obs[ctrl.n][:] = delta_obs[:]
        if IS_IDEAL:
            dde_I = np.dot(ctrl.dot_f1_rho1(), ctrl.rho2()) + \
                    np.dot(ctrl.f1_rho1(), ctrl.f2_rho2() + np.dot(ctrl.g_rho1(), ctrl.inner_control)) - \
                    fake_dotdot_rhod
        else:
            dde_I = np.dot(ctrl.dot_f1_rho1(), ctrl.rho2()) + \
                    np.dot(ctrl.f1_rho1(), ctrl.f2_rho2() + np.dot(ctrl.g_rho1(), ctrl.inner_control)) + \
                    delta_obs
        '''2. 计算 tk 时刻误差信号'''

        '''3. 计算切换函数与滑膜'''
        ctrl.sI = ctrl.CI * e_I + de_I
        ctrl.dsI = ctrl.CI * de_I + dde_I
        ctrl.sigmaI = ctrl.dsI + ctrl.LambdaI * ctrl.sI
        '''3. 计算切换函数与滑膜'''

        '''4. 计算控制量导数'''
        if IS_IDEAL:
            du1 = ctrl.LambdaI * ctrl.CI * de_I + \
                  (ctrl.CI + ctrl.LambdaI) * (np.dot(ctrl.dot_f1_rho1(), ctrl.rho2()) + np.dot(ctrl.f1_rho1(), ctrl.f2_rho2()) - fake_dotdot_rhod) + \
                  ctrl.dot_Frho2_f1f2() - fake_dotdotdot_rhod
        else:
            du1 = ctrl.LambdaI * ctrl.CI * de_I + \
                  (ctrl.CI + ctrl.LambdaI) * (np.dot(ctrl.dot_f1_rho1(), ctrl.rho2()) + np.dot(ctrl.f1_rho1(), ctrl.f2_rho2())) + \
                  ctrl.dot_Frho2_f1f2()

        du2 = np.dot((ctrl.CI + ctrl.LambdaI) * np.dot(ctrl.f1_rho1(), ctrl.g_rho1()) + ctrl.dot_f1g(), ctrl.inner_control)
        if IS_IDEAL:
            V_dis = np.array([0., 0., 0., 0.])
        else:
            V_dis = (ctrl.CI + ctrl.LambdaI) * delta_obs # + dot_delta_obs
        du3 = (np.fabs(V_dis) + ctrl.K0_I) * np.sign(ctrl.sigmaI)
        du = -np.dot(np.linalg.inv(np.dot(ctrl.f1_rho1(), ctrl.g_rho1())), du1 + du2 + du3)

        '''4. 计算控制量导数'''

        '''5. 微分方程解算'''
        # 解算之后，所有状态更新，时间更新，并且 f1g 和 Frho2 + f1f2 会刷新，从rk44中出来之后，这俩变量可以被直接求导，直到下一次计算
        ctrl.rk44(action=ctrl.inner_control, dis=uncertainty, n=1)
        '''5. 微分方程解算'''

        '''6. 计算新的控制量'''
        ctrl.inner_control += du * ctrl.dt
        # print('======== START ========')
        # print('time:', ctrl.time)
        # print('Observer delta_obs:', delta_obs)
        # print('Observer dot_delta_obs:', dot_delta_obs)
        # print('du:', np.linalg.norm(du1), np.linalg.norm(du2), np.linalg.norm(du3))
        # print('control:', ctrl.inner_control)
        # print('\n\n')
        '''6. 计算新的控制量'''

    csv_uav_state = np.hstack((save_t, save_state))
    csv_ref = np.hstack((save_t, save_ref_pos, save_ref_angle))
    csv_control = np.hstack((save_t, save_inner_control))
    csv_observe = np.hstack((save_t, real_delta_inner, delta_inner_obs))

    path = '../environment/UAV/data/'
    pd1 = pd.DataFrame(csv_uav_state, columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r'])
    pd1.to_csv(path + 'uav_state.csv', sep=',', index=False)

    pd2 = pd.DataFrame(csv_ref, columns=['time', 'ref_x', 'ref_y', 'ref_z', 'ref_phi', 'ref_theta', 'ref_psi'])
    pd2.to_csv(path + 'ref_cmd.csv', sep=',', index=False)

    pd3 = pd.DataFrame(csv_control, columns=['time', 'throttle', 'torque_x', 'torque_y', 'torque_z'])
    pd3.to_csv(path + 'control.csv', sep=',', index=False)

    pd4 = pd.DataFrame(csv_observe, columns=['time', 'in_01', 'in_02', 'in_03', 'in_04', 'in_01_obs','in_02_obs','in_03_obs', 'in_04_obs'])
    pd4.to_csv(path + 'observe.csv', sep=',', index=False)
