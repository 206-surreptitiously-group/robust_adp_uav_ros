import numpy as np
from environment.UAV.uav_control import SMCControl
import matplotlib.pyplot as plt


IS_IDEAL = False


if __name__ == '__main__':
    ctrl = SMCControl()      # 直接初始化完成

    '''一些中间变量初始化'''
    # TODO 全都按照 0 时刻的初始值去初始化

    uncertainty = ctrl.generate_uncertainty()
    fake_rhod, fake_dot_rhod, fake_dotdot_rhod, fake_dotdotdot_rhod = ctrl.fake_inner_cmd_generator()  # 参考信号肯定是已知的，这不用说，因为这是认为定义好的

    e_I = ctrl.rho1() - fake_rhod
    de_I = ctrl.dot_rho1() - fake_dot_rhod
    de_I_old = de_I.copy()
    '''一些中间变量初始化'''

    '''数据存储'''
    save_t = []
    save_inner_control = []
    save_pos_error = []
    save_angle_error = []
    '''数据存储'''

    while ctrl.time < ctrl.time_max:
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''
        uncertainty = ctrl.generate_uncertainty()
        fake_rhod, fake_dot_rhod, fake_dotdot_rhod, fake_dotdotdot_rhod = ctrl.fake_inner_cmd_generator()  # 参考信号肯定是已知的，这不用说，因为这是人为定义好的
        '''1. 计算 tk 时刻参考信号 和 生成不确定性'''

        save_t.append(ctrl.time)
        save_inner_control.append(ctrl.inner_control.copy())
        save_pos_error.append(np.array([ctrl.x, ctrl.y, ctrl.z]) - np.array([ctrl.x, ctrl.y, fake_rhod[0]]))
        save_angle_error.append(np.array([ctrl.phi, ctrl.theta, ctrl.psi]) - fake_rhod[1: 4])

        '''2. 计算 tk 时刻误差信号'''
        de_I_old = de_I.copy()                  # 这个时候 de_I 是上一时刻的
        e_I = ctrl.rho1() - fake_rhod
        de_I = ctrl.dot_rho1() - fake_dot_rhod  # 这个时候 de_I 是新时刻的
        # 先观测一下
        delta_obs, dot_delta_obs = ctrl.inner_obs.observe(ctrl.dot_f1_rho1(),
                                                          ctrl.rho2(),
                                                          ctrl.f1_rho1(),
                                                          ctrl.f2_rho2(),
                                                          ctrl.g_rho1(),
                                                          ctrl.inner_control,
                                                          de_I_old,
                                                          de_I)
        '''2. 计算 tk 时刻误差信号'''

        '''3. 计算切换函数与滑膜'''
        ctrl.sI = ctrl.CI * e_I + de_I
        '''3. 计算切换函数与滑膜'''

        '''4. 计算控制量导数'''
        if IS_IDEAL:
            u1 = ctrl.CI * de_I + np.dot(ctrl.dot_f1_rho1(), ctrl.rho2()) + np.dot(ctrl.f1_rho1(), ctrl.f2_rho2()) - fake_dotdot_rhod
            u2 = ctrl.K0_I * np.sign(ctrl.sI)
        else:
            u1 = ctrl.CI * de_I + np.dot(ctrl.dot_f1_rho1(), ctrl.rho2()) + np.dot(ctrl.f1_rho1(), ctrl.f2_rho2()) + delta_obs
            u2 = (np.fabs(delta_obs) + ctrl.K0_I) * np.sign(ctrl.sI)

        u = -np.dot(np.linalg.inv(np.dot(ctrl.f1_rho1(), ctrl.g_rho1())), u1 + u2)
        '''4. 计算控制量导数'''

        '''5. 微分方程解算'''
        # 解算之后，所有状态更新，时间更新，并且 f1g 和 Frho2 + f1f2 会刷新，从rk44中出来之后，这俩变量可以被直接求导，直到下一次计算
        ctrl.rk44(action=ctrl.inner_control, dis=uncertainty, n=1)
        '''5. 微分方程解算'''

        '''6. 计算新的控制量'''
        ctrl.inner_control  = u.copy()
        '''6. 计算新的控制量'''

    save_t = np.array(save_t)
    save_inner_control = np.array(save_inner_control)
    save_pos_error = np.array(save_pos_error)
    save_angle_error = np.array(save_angle_error) * 180 / np.pi

    print('     time:     ', save_t.shape)
    print('inner control: ', save_inner_control.shape)
    print('  pos error:   ', save_pos_error.shape)
    print(' angle error:  ', save_angle_error.shape)

    plt.figure(0)
    plt.plot(save_t, save_pos_error[:, 2], 'red')
    plt.xlabel('time(s)')
    plt.ylabel('error(m)')
    plt.title('Z error')

    plt.figure(1)
    plt.plot(save_t, save_angle_error[:, 0], 'red')
    plt.xlabel('time(s)')
    plt.ylabel('$\phi$   error(m)')
    plt.title('roll  $\phi$  error')

    plt.figure(2)
    plt.plot(save_t, save_angle_error[:, 1], 'red')
    plt.xlabel('time(s)')
    plt.ylabel('$\Theta$   error(m)')
    plt.title('pitch  $\Theta$  error')

    plt.figure(3)
    plt.plot(save_t, save_angle_error[:, 2], 'red')
    plt.xlabel('time(s)')
    plt.ylabel('$\psi$   error(m)')
    plt.title('yaw  $\psi$  error')

    plt.figure(4)
    plt.plot(save_t, save_inner_control[:, 0], 'red')   # 油门
    plt.xlabel('time(s)')
    plt.ylabel('throttle(N)')
    plt.title('throttle')

    plt.show()
