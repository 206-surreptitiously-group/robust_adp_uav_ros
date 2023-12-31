#!/usr/bin/python3
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import deg2rad

from environment.Color import Color

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# import rospy
from environment.envs.RL.uav_inner_loop import uav_inner_loop as env
from environment.envs.UAV.ref_cmd import generate_uncertainty
from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *

optPath = '../../datasave/network'
show_per = 1
timestep = 0
ENV = 'PPO-uav-inner-loop'


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# setup_seed(2162)

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
# 只有姿态时范围可以给大点方便训练
uav_param.att_zone = np.atleast_2d(
    [[-deg2rad(90), deg2rad(90)], [-deg2rad(90), deg2rad(90)], [deg2rad(-120), deg2rad(120)]])
'''Parameter list of the quadrotor'''


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
        super(PPOActorCritic, self).__init__()
        self.checkpoint_file = chkpt_dir + name + '_ppo'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
        self.state_dim = _state_dim
        self.action_dim = _action_dim
        self.action_std_init = _action_std_init
        self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor_reset_orthogonal()
        self.critic_reset_orthogonal()
        self.device = 'cpu'
        self.to(self.device)

    def actor_reset_orthogonal(self):
        nn.init.orthogonal_(self.actor[0].weight, gain=1.0)
        nn.init.constant_(self.actor[0].bias, val=1e-3)
        nn.init.orthogonal_(self.actor[2].weight, gain=1.0)
        nn.init.constant_(self.actor[2].bias, val=1e-3)
        nn.init.orthogonal_(self.actor[4].weight, gain=0.01)
        nn.init.constant_(self.actor[4].bias, val=1e-3)

    def critic_reset_orthogonal(self):
        nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
        nn.init.constant_(self.critic[0].bias, val=1e-3)
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
        nn.init.constant_(self.critic[2].bias, val=1e-3)
        nn.init.orthogonal_(self.critic[4].weight, gain=1.0)
        nn.init.constant_(self.critic[4].bias, val=1e-3)

    def set_action_std(self, new_action_std):
        """手动设置动作方差"""
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, s):
        """选取动作"""
        action_mean = self.actor(s)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        _a = dist.sample()
        action_logprob = dist.log_prob(_a)
        state_val = self.critic(s)

        return _a.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, s, a):
        """评估状态动作价值"""
        action_mean = self.actor(s)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # 一维动作单独处理
        if self.action_dim == 1:
            a = a.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(a)
        dist_entropy = dist.entropy()
        state_values = self.critic(s)

        return action_logprobs, state_values, dist_entropy

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


if __name__ == '__main__':
    # rospy.init_node(name='PPO_uav_hover_outer_loop', anonymous=False)

    log_dir = '../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)
    TRAIN = True
    RETRAIN = False
    TEST = not TRAIN

    env = env(uav_param, fntsmc_param(),
              ref_amplitude=np.array([np.pi / 3, np.pi / 3, np.pi / 2]),
              ref_period=np.array([4, 4, 4]),
              ref_bias_a=np.array([0, 0, 0]),
              ref_bias_phase=np.array([0., np.pi / 2, np.pi / 3]))

    env.msg_print_flag = False  # 别疯狂打印出界了
    reward_norm = Normalization(dim=1, update=True)
    # rate = rospy.Rate(1 / env.dt)

    if TRAIN:
        action_std_init = 0.8
        '''重新加载Policy网络结构，这是必须的操作'''
        policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulation_path)
        policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulation_path)
        agent = PPO(env=env,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    gamma=0.99,
                    K_epochs=30,
                    eps_clip=0.2,
                    action_std_init=action_std_init,
                    buffer_size=int(env.time_max / env.dt * 4),
                    policy=policy,
                    policy_old=policy_old,
                    path=simulation_path)
        if RETRAIN:
            agent.policy.load_state_dict(torch.load('Policy_PPO12160000'))
            agent.policy_old.load_state_dict(torch.load('Policy_PPO12160000'))
            '''如果修改了奖励函数，则原来的critic网络已经不起作用了，需要重新初始化'''
            agent.policy.critic_reset_orthogonal()
            agent.policy_old.critic_reset_orthogonal()
        agent.PPO_info()

        max_training_timestep = int(env.time_max / env.dt) * 40000
        action_std_decay_freq = int(env.time_max / env.dt) * 2000
        action_std_decay_rate = 0.05
        min_action_std = 0.1

        sumr = 0
        start_eps = 0
        train_num = 0
        test_num = 0
        test_reward = []
        index = 0
        while timestep <= max_training_timestep:
            # env.reset()
            env.reset_random()
            sumr = 0.
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)
                action = agent.action_linear_trans(action_from_actor.detach().cpu().numpy().flatten())
                uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
                env.step_update(action)  # 环境更新的动作必须是实际物理动作
                sumr += env.reward
                '''存数'''
                agent.buffer.append(s=env.current_state,
                                    a=action_from_actor,
                                    log_prob=a_log_prob.numpy(),
                                    r=reward_norm(env.reward),
                                    sv=s_value.numpy(),
                                    done=1.0 if env.is_terminal else 0.0,
                                    index=index)
                index += 1
                timestep += 1
                '''学习'''
                if timestep % agent.buffer.batch_size == 0:
                    print('========= Training =========')
                    print('Episode: {}'.format(agent.episode))
                    print('Num of learning: {}'.format(train_num))
                    agent.learn()
                    train_num += 1
                    start_eps = agent.episode
                    index = 0
                    if train_num % 20 == 0 and train_num > 0:
                        print('========= Testing =========')
                        n = 1
                        average_test_r = 0
                        for i in range(n):
                            # env.reset()
                            env.reset_random()
                            env.draw_att_init_image()
                            while not env.is_terminal:
                                env.current_state = env.next_state.copy()
                                action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)
                                action = agent.action_linear_trans(action_from_actor.detach().cpu().numpy().flatten())
                                uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
                                env.step_update(action)  # 环境更新的动作必须是实际物理动作
                                average_test_r += env.reward
                                env.att_image = env.att_image_copy.copy()
                                env.draw_att(env.ref)
                                env.show_att_image(iswait=False)
                        test_num += 1
                        average_test_r = round(average_test_r / n, 3)
                        test_reward.append(average_test_r)
                        print('   Evaluating %.0f | Reward: %.2f ' % (test_num, average_test_r))
                        temp = simulation_path + 'test_num' + '_' + str(test_num - 1) + '_save/'
                        os.mkdir(temp)
                        pd.DataFrame({'reward': test_reward}).to_csv(simulation_path + 'retrain_reward.csv')
                        time.sleep(0.01)
                        agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
                if timestep % action_std_decay_freq == 0:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)
            if agent.episode % 5 == 0:
                print('Episode: ', agent.episode, ' Reward: ', sumr)
            agent.episode += 1
    else:
        action_std_init = 0.8
        policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulation_path)
        policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulation_path)
        agent = PPO(env=env,
                    actor_lr=3e-4,
                    critic_lr=1e-3,
                    gamma=0.99,
                    K_epochs=20,
                    eps_clip=0.2,
                    action_std_init=action_std_init,
                    buffer_size=int(env.time_max / env.dt * 2),
                    policy=policy,
                    policy_old=policy_old,
                    path=simulation_path)
        # agent.policy.load_state_dict(torch.load('../datasave/network/'))
        agent.policy.load_state_dict(torch.load('Policy_PPO12160000'))
        test_num = 1
        r = 0
        ux, uy, uz = [], [], []
        for _ in range(test_num):
            # env.reset()
            env.reset_random()
            env.init_image()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                _action_from_actor = agent.evaluate(env.current_state)
                _action = agent.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将actor输出动作转换到实际动作范围
                uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
                env.step_update(_action)  # 环境更新的动作必须是实际物理动作
                r += env.reward
                env.draw_image(isWait=False)
                ux.append(_action[0])
                uy.append(_action[1])
                uz.append(_action[2])
                # print(_action)
                # env.uav_vis.render(uav_pos=env.uav_pos(),
                #                    uav_pos_ref=env.pos_ref,
                #                    uav_att=env.uav_att(),
                #                    uav_att_ref=env.att_ref,
                #                    d=4 * env.d)  # to make it clearer, we increase the size 4 times
                # rate.sleep()
            print(r)
            env.collector.plot_att()
            plt.plot(ux, label='ux')
            plt.plot(uy, label='uy')
            plt.plot(uz, label='uz')
            plt.legend()
            plt.show()
