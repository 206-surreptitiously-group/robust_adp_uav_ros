#!/usr/bin/python3
import datetime
import os
import sys
import matplotlib.pyplot as plt
from numpy import deg2rad

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# import rospy
from environment.envs.RL.uav_tracking_outer_loop import uav_tracking_outer_loop as env
from environment.envs.UAV.ref_cmd import generate_uncertainty
from environment.envs.UAV.uav import uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from algorithm.policy_base.Distributed_PPO import Distributed_PPO as DPPO
from algorithm.policy_base.Distributed_PPO import Worker
from common.common_cls import *
import torch.multiprocessing as mp

optPath = '../datasave/network/'
show_per = 1
timestep = 0
ENV = 'DPPO-uav-tracking-outer-loop'


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# setup_seed(5443)
os.environ["OMP_NUM_THREADS"] = "1"

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
uav_param.pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [0, 5]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-45), deg2rad(45)], [deg2rad(-45), deg2rad(45)], [deg2rad(-120), deg2rad(120)]])
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
            nn.Linear(64, 1),
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

    TRAIN = False
    RETRAIN = True
    TEST = not TRAIN

    env = env(uav_param, fntsmc_param(), att_ctrl_param, np.ones(3), np.ones(3), np.ones(3), np.ones(3))
    env.msg_print_flag = False  # 别疯狂打印出界了
    # rate = rospy.Rate(1 / env.dt)

    if TRAIN:
        '''1. 启动多进程'''
        mp.set_start_method('spawn', force=True)
        '''2. 定义DPPO参数'''
        process_num = 10
        actor_lr = 1e-5 / min(process_num, 5)
        critic_lr = 1e-4 / min(process_num, 5)
        action_std = 0.9
        k_epo_init = 100
        agent = DPPO(env=env, actor_lr=actor_lr, critic_lr=critic_lr, num_of_pro=process_num, path=simulation_path)
        '''3. 重新加载全局网络和优化器，这是必须的操作，考虑到不同学习环境设计不同的网络结构，训练前要重写PPOActorCritic'''
        agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'GlobalPolicy',
                                             simulation_path)
        agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'EvalPolicy',
                                           simulation_path)
        if RETRAIN:
            agent.global_policy.load_state_dict(torch.load('Policy_PPO106'))
            '''如果修改了奖励函数，则原来的critic网络已经不起作用了，需要重新初始化'''
            agent.global_policy.critic_reset_orthogonal()
        agent.global_policy.share_memory()
        agent.optimizer = SharedAdam([
            {'params': agent.global_policy.actor.parameters(), 'lr': actor_lr},
            {'params': agent.global_policy.critic.parameters(), 'lr': critic_lr}
        ])
        '''4. 添加进程'''
        ppo_msg = {'gamma': 0.99, 'k_epo': int(k_epo_init / process_num * 1.5), 'eps_c': 0.2, 'a_std': action_std,
                   'device': 'cpu', 'loss': nn.MSELoss()}
        for i in range(agent.num_of_pro):
            worker = Worker(g_pi=agent.global_policy,
                            l_pi=PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'LocalPolicy',
                                                simulation_path),
                            g_opt=agent.optimizer,
                            g_train_n=agent.global_training_num,
                            _index=i,
                            _name='worker' + str(i),
                            _env=env,
                            _queue=agent.queue,
                            _lock=agent.lock,
                            _ppo_msg=ppo_msg)
            agent.add_worker(worker)
        agent.DPPO_info()
        '''5. 原神(多进程学习)启动'''
        """
        多个学习进程和一个评估进程，学习进程在结束后会释放标志，评估进程收集到所有学习进程标志时结束评估，
        评估结束时，评估程序会跳出while死循环，整个程序结束，结果在评估过程中自动存储在simulation_path中。
        """
        agent.start_multi_process()
    else:
        agent = DPPO(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=0, path=simulation_path)
        agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, 0.1,
                                             'GlobalPolicy_ppo', simulation_path)
        agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, 0.1,
                                           'EvalPolicy_ppo', simulation_path)
        # 加载模型参数文件
        agent.load_models(optPath + 'DPPO_uav_tracking_outer_loop/second-order-information/retrain1')
        # agent.load_models('Policy_PPO86')
        agent.eval_policy.load_state_dict(agent.global_policy.state_dict())
        env.msg_print_flag = True
        test_num = 10
        aver_r = 0
        for _ in range(test_num):
            r = 0
            env.reset_random()
            env.init_image()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                action_from_actor = agent.evaluate(env.current_state).numpy()
                action = agent.action_linear_trans(action_from_actor.flatten())  # 将actor输出动作转换到实际动作范围
                uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
                env.step_update(action)  # 环境更新的动作必须是实际物理动作
                r += env.reward
                env.draw_image(isWait=False)
                # env.uav_vis.render(uav_pos=env.uav_pos(),
                #                    uav_pos_ref=env.pos_ref,
                #                    uav_att=env.uav_att(),
                #                    uav_att_ref=env.att_ref,
                #                    d=4 * env.d)  # to make it clearer, we increase the size 4 times
                # rate.sleep()
            aver_r += r
            print(r)
            env.collector.plot_pos()
            env.collector.plot_vel()
            env.collector.plot_att()
            env.collector.plot_pqr()
            plt.show()
        print(aver_r / test_num)

