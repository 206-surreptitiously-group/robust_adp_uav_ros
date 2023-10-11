#!/usr/bin/python3
import datetime
import os
import sys
import time

from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# import rospy
from environment.envs.RL.uav_hover_outer_loop import uav_hover_outer_loop as env
from environment.envs.UAV.ref_cmd import generate_uncertainty
from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *

optPath = '../../datasave/network'
show_per = 1
timestep = 0
ENV = 'PPO-uav-hover-outer-loop'


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2162)

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
uav_param.time_max = 5
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
        self.device = 'cpu'
        self.to(self.device)

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
        cov_mat = torch.diag_embed(self.action_var).to(self.device)
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

    env = env(uav_param, fntsmc_param(), att_ctrl_param, target0=np.array([-1, 3, 2]))
    env.msg_print_flag = False  # 别疯狂打印出界了
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
            agent.policy.load_state_dict(torch.load('Policy'))
            agent.policy_old.load_state_dict(torch.load('Policy'))
        agent.PPO_info()

        max_training_timestep = int(env.time_max / env.dt) * 10000
        action_std_decay_freq = int(5e6)
        action_std_decay_rate = 0.05
        min_action_std = 0.1

        sumr = 0
        start_eps = 0
        train_num = 0
        test_num = 0
        index = 0
        while timestep <= max_training_timestep:
            env.reset()
            sumr = 0.
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)
                # action_from_actor = action_from_actor.numpy()     # YYF
                action = agent.action_linear_trans(action_from_actor.detach().cpu().numpy().flatten())  # YYF 改过
                uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
                env.step_update(action)  # 环境更新的动作必须是实际物理动作
                sumr += env.reward
                '''存数'''
                agent.buffer.append(s=env.current_state,
                                    a=action_from_actor,
                                    log_prob=a_log_prob.numpy(),
                                    r=env.reward,
                                    sv=s_value.numpy(),
                                    done=1.0 if env.is_terminal else 0.0,
                                    index=index)
                index += 1
                timestep += 1
                '''学习'''
                if timestep % agent.buffer.batch_size == 0:
                    print('========= LEARN =========')
                    print('Episode: {}'.format(agent.episode))
                    print('Num of learning: {}'.format(train_num))
                    agent.learn()
                    # average_train_r = round(sumr / (agent.episode + 1 - start_eps), 3)
                    # print('Average reward:', average_train_r)
                    train_num += 1
                    start_eps = agent.episode
                    # sumr = 0
                    index = 0
                    if train_num % 10 == 0 and train_num > 0:
                        average_test_r = agent.agent_evaluate(1)  # YYF
                        test_num += 1
                        print('check point save')
                        temp = simulation_path + 'episode' + '_' + str(agent.episode) + '_save/'
                        os.mkdir(temp)
                        time.sleep(0.01)
                        agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
                    print('========= LEARN =========')
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
        agent.policy.load_state_dict(torch.load('Policy_PPO4940000'))
        test_num = 1
        for _ in range(test_num):
            env.reset()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                _action_from_actor = agent.evaluate(env.current_state)
                _action = agent.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将actor输出动作转换到实际动作范围
                uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
                env.step_update(_action)  # 环境更新的动作必须是实际物理动作
                # env.uav_vis.render(uav_pos=env.uav_pos(),
                #                    uav_pos_ref=env.pos_ref,
                #                    uav_att=env.uav_att(),
                #                    uav_att_ref=env.att_ref,
                #                    d=4 * env.d)  # to make it clearer, we increase the size 4 times
                # rate.sleep()
                # env.image = env.image_copy.copy()
                # env.draw_3d_points_projection(np.atleast_2d([env.uav_pos()]), [Color().Red])
                # env.draw_error(env.uav_pos(), env.pos_ref[0:3])
                # env.show_image(False)
            env.collector.plot_pos()
            env.collector.plot_vel()
            env.collector.plot_att()
            env.collector.plot_throttle()
            plt.show()
