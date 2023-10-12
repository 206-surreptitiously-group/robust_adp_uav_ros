#!/usr/bin/python3
import datetime
import os
import sys
import time

import torch
from matplotlib import pyplot as plt

from environment.Color import Color

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# import rospy
from environment.envs.RL.uav_hover_outer_loop import uav_hover_outer_loop as env
from environment.envs.UAV.ref_cmd import generate_uncertainty
from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from algorithm.actor_critic.Twin_Delayed_DDPG import Twin_Delayed_DDPG as TD3
from common.common_cls import *

optPath = '../datasave/network/'
optPath = 'temp/'
show_per = 50
timestep = 0
ENV = 'TD3-uav-hover-outer-loop'


class Critic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        state_value = self.fc1(state)  # forward
        state_value = self.batch_norm1(state_value)  # batch normalization
        state_value = func.relu(state_value)  # relu

        state_value = self.fc2(state_value)
        state_value = self.batch_norm2(state_value)

        action_value = func.relu(self.action_value(_action))
        state_action_value = func.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, name, chkpt_dir):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # 输入 -> 第一个隐藏层
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # 第一个隐藏层 -> 第二个隐藏层
        self.batch_norm2 = nn.LayerNorm(64)

        self.mu = nn.Linear(64, self.action_dim)  # 第二个隐藏层 -> 输出层

        self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.batch_norm1(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = func.relu(x)

        x = torch.tanh(self.mu(x))  # bound the output to [-1, 1]

        return x

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


def fullFillReplayMemory_with_Optimal(randomEnv: bool,
                                      fullFillRatio: float,
                                      is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    # agent.load_models(optPath)
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            # _action_from_actor = agent.choose_action(env.current_state, is_optimal=False)
            # _action = agent.action_linear_trans(_action_from_actor)
            aa = env.pos_control(env.pos_ref, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
            _action = np.clip(env.pos_ctrl.control, env.u_min, env.u_max)
            env.step_update(np.array(_action))
            # env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                agent.memory.store_transition(np.array(env.current_state),
                                              np.array(env.current_action),
                                              np.array(env.reward),
                                              np.array(env.next_state),
                                              1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :param randomEnv:       init env randomly
    :param fullFillRatio:   the ratio
    :return:                None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = agent.choose_action(env.current_state, False)
            _action = agent.action_linear_trans(_action_from_actor)
            env.step_update(np.array(_action))
            # env.show_dynamic_image(isWait=False)
            # if env.reward > 0:
            agent.memory.store_transition(np.array(env.current_state),
                                          np.array(env.current_action),
                                          np.array(env.reward),
                                          np.array(env.next_state),
                                          1 if env.is_terminal else 0)


if __name__ == '__main__':
    # rospy.init_node(name='PPO_uav_hover_outer_loop', anonymous=False)

    log_dir = '../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)
    TRAIN = True
    RETRAIN = False
    TEST = not TRAIN
    is_storage_only_success = False

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
    uav_param.pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [0, 5]])
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

    env = env(uav_param, pos_ctrl_param, att_ctrl_param, target0=np.array([-1, 3, 2]))
    env.msg_print_flag = False  # 别疯狂打印出界了
    # rate = rospy.Rate(1 / env.dt)

    if TRAIN:
        actor = Actor(1e-4, env.state_dim, env.action_dim, 'Actor', simulation_path)
        target_actor = Actor(1e-4, env.state_dim, env.action_dim, 'TargetActor', simulation_path)
        critic1 = Critic(1e-3, env.state_dim, env.action_dim, 'Critic1', simulation_path)
        target_critic1 = Critic(1e-3, env.state_dim, env.action_dim, 'TargetCritic1', simulation_path)
        critic2 = Critic(1e-3, env.state_dim, env.action_dim, 'Critic2', simulation_path)
        target_critic2 = Critic(1e-3, env.state_dim, env.action_dim, 'TargetCritic2', simulation_path)
        agent = TD3(env=env,
                    gamma=0.99,
                    noise_clip=1 / 2, noise_policy=1 / 4, policy_delay=3,
                    critic1_soft_update=1e-2,
                    critic2_soft_update=1e-2,
                    actor_soft_update=1e-2,
                    memory_capacity=20000,  # 20000
                    batch_size=512,  # 1024
                    actor=actor,
                    target_actor=target_actor,
                    critic1=critic1,
                    target_critic1=target_critic1,
                    critic2=critic2,
                    target_critic2=target_critic2,
                    path=simulation_path)
        agent.TD3_info()
        # cv.waitKey(0)
        MAX_EPISODE = 1200
        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=False,
                                              fullFillRatio=0.9,
                                              is_only_success=False)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            # agent.actor.initialization()
            # agent.target_actor.initialization()
            # agent.critic1.initialization()
            # agent.target_critic1.initialization()
            # agent.critic2.initialization()
            # agent.target_critic2.initialization()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=False, fullFillRatio=0.8)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        while agent.episode <= MAX_EPISODE:
            env.reset()
            if agent.episode % show_per == 0:
                env.draw_init_image()
            # env.reset_random()
            sumr = 0
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                if random.uniform(0, 1) < 0.8:  # 有一定探索概率完全随机探索
                    # print('...random...')
                    action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 3)  # 剩下的是神经网络加噪声
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.step_update(np.array(action))  # 环境更新的action需要是物理的action
                agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                if agent.episode % show_per == 0:
                    env.image = env.image_copy.copy()
                    env.draw_3d_points_projection(np.atleast_2d([env.uav_pos()]), [Color().Red])
                    env.draw_3d_points_projection(np.atleast_2d([env.pos_ref]), [Color().Green])
                    env.draw_error(env.uav_pos(), env.pos_ref[0:3])
                    env.show_image(False)
                    time.sleep(0.01)
                sumr = sumr + env.reward
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    # if env.reward > 0:
                    agent.memory.store_transition(np.array(env.current_state),
                                                  np.array(env.current_action),
                                                  np.array(env.reward),
                                                  np.array(env.next_state),
                                                  1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)
            '''跳出循环代表回合结束'''
            if is_storage_only_success:
                if sumr >= -10000:
                    print('Update Replay Memory......')
                    agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
            '''跳出循环代表回合结束'''
            print(
                '=========START=========',
                'Episode:', agent.episode,
                'Cumulative reward:', round(sumr, 3),
                '==========END=========\n')
            agent.episode += 1
            if agent.episode % 10 == 0:
                agent.save_models()

    if TEST:
        print('TESTing...')
        optPath = '../datasave/network/TD3-SecondOrderIntegration/parameters/'
        agent = TD3(env=env, target_actor=Actor(1e-4, env.state_dim, env.action_dim, 'TargetActor', simulation_path))
        agent.load_target_actor_optimal(path=optPath, file='TargetActor_ddpg')
        simulation_num = 10
        error = []
        terminal_list = []
        for i in range(simulation_num):
            print('==========START==========')
            print('episode = ', i)
            env.reset()
            env.draw_init_image()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                action_from_actor = agent.evaluate(env.current_state)
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.step_update(np.array(action))
                env.image = env.image_copy.copy()
                env.draw_3d_points_projection(np.atleast_2d([env.uav_pos()]), [Color().Red])
                env.draw_3d_points_projection(np.atleast_2d([env.pos_ref]), [Color().Green])
                env.draw_error(env.uav_pos(), env.pos_ref[0:3])
                env.show_image(False)
            print('===========END===========')
        env.collector.plot_pos()
        env.collector.plot_vel()
        env.collector.plot_throttle()
        plt.show()
