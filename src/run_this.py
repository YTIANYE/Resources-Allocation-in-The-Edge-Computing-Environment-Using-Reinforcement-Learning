from env import Env
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt
import os
import time

#####################  hyper parameters  ####################
CHECK_EPISODE = 4
LEARNING_MAX_EPISODE = 10       # 最大reward 10个episode不变后，结束
MAX_EP_STEPS = 3000     # 每个episode中最大的steps
# TEXT_RENDER = False
TEXT_RENDER = True  # 控制是否打印过程
SCREEN_RENDER = True  # 控制是否图像形式显示
CHANGE = False      # 记录max_rewards是否发生改变
SLEEP_TIME = 0.1    # 每个episode需要进程挂起 0.1s 的时间


#####################  function  ####################
# add randomness to action selection for exploration
def exploration(a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a


###############################  training  ####################################

if __name__ == "__main__":

    """
    实验变量
    """
    env = Env()
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location = env.get_inf()
    ddpg = DDPG(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)

    # control exploration    Exploration: trying new things that might enable the agent to make better decisions in the future
    r_var = 1   # resource 的 var    max_reward变化，var不断乘以0.9999
    b_var = 1   # bandwidth 的 var   max_reward变化，var不断乘以0.9999
 
    ep_reward = []      # 记录每个episode的reward,最后用于画图
    r_v, b_v = [], []   # 记录每个episode的r_var，b_var的变化，最后用于画图
    var_reward = []     # 记录所有episode的reward的和

    max_rewards = 0
    episode = 0
    var_counter = 0     # 帮助计数，不超过最大学习的episode数
    epoch_inf = []      # 记录每个episode的打印信息 Episode Reward r_var b_var


    """
    训练学习过程，并记录结果
    """
    # 最大reward 10个episode不变后，结束
    while var_counter < LEARNING_MAX_EPISODE:       # LEARNING_MAX_EPISODE = 10
        # initialize
        s = env.reset()
        ep_reward.append(0)
        if SCREEN_RENDER:
            env.initial_screen_demo()       # 初始化屏幕

        # 每个episode中执行 MAX_EP_STEPS = 3000 个step
        for j in range(MAX_EP_STEPS):       # j 每个episode中的step计数器，即记录当前是第几个step
            time.sleep(SLEEP_TIME)      # time sleep() 函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。
            # render
            if SCREEN_RENDER:
                env.screen_demo()       # 绘制节点服务器分布情况
            if TEXT_RENDER and j % 30 == 0:     # 每个MAX_EP_STEPS打印100次
                env.text_render()       # 打印RBO、user、edge、reward

            # DDPG
            # choose action according to state
            a = ddpg.choose_action(s)  # a = [R B O]
            # add randomness to action selection for exploration
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # store the transition parameter
            s_, r = env.ddpg_step_forward(a, r_dim, b_dim)
            ddpg.store_transition(s, a, r / 10, s_)     # s a r s'
            # learn
            if ddpg.pointer == ddpg.memory_capacity:
                print("start learning")
            if ddpg.pointer > ddpg.memory_capacity:
                ddpg.learn()
                if CHANGE:      # max_rewards发生改变，更新r_var， b_var
                    r_var *= .99999
                    b_var *= .99999
            # replace the state
            s = s_
            # sum up the reward
            ep_reward[episode] += r
            # in the end of the episode
            if j == MAX_EP_STEPS - 1:       # 最后一个episode
                var_reward.append(ep_reward[episode])
                r_v.append(r_var)
                b_v.append(b_var)
                print("episode =", episode, "   j =", j)
                print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], '###  r_var: %.2f ' % r_var, 'b_var: %.2f ' % b_var, )
                string = 'Episode:%3d' % episode + ' Reward: %5d' % ep_reward[episode] + '###  r_var: %.2f ' % r_var + 'b_var: %.2f ' % b_var
                epoch_inf.append(string)
                # variation change  ，大于4个episode并且后4个的平均reward >= max_rewards 时
                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:        # CHECK_EPISODE = 4 # mean()函数功能：求取均值
                    CHANGE = True       # 记录max_rewards发生改变
                    var_counter = 0     # 一旦max_rewards发生改变，从新开始学习
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])      # 记录最新的max_rewards
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1

        # end the episode
        if SCREEN_RENDER:
            env.canvas.tk.destroy()     # 每执行完一个 episode 之后消除 节点服务器变化情况的屏幕显示
        episode += 1


    """
    产生实验结果：实验信息记录文档、reward变化曲线、variance变化曲线
    """

    # make directory    新建输出文件夹，文件夹名包含：user数目、edge数目、限制limit、位置
    dir_name = 'output/' + 'ddpg_' + str(r_dim) + 'u' + str(int(o_dim / r_dim)) + 'e' + str(limit) + 'l' + location
    if os.path.isdir(dir_name):     # 用于判断对象是否为一个目录
        os.rmdir(dir_name)      # os.rmdir() 方法用于删除指定路径的目录
    os.makedirs(dir_name)

    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i + 1 for i in range(episode)], ep_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards.png')
    # plot the variance
    fig_variance = plt.figure()
    plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    plt.xlabel("episode")
    plt.ylabel("variance")
    fig_variance.savefig(dir_name + '/variance.png')

    # write the record
    f = open(dir_name + '/record.txt', 'a')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    for i in range(episode):
        f.write(epoch_inf[i] + '\n')
    # mean
    print("the mean of the rewards in the last", LEARNING_MAX_EPISODE, " epochs:",
          str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards:" + str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # standard deviation 标准差
    print("the standard deviation of the rewards:", str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the rewards:" + str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the rewards:",
          str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the rewards:" + str(
        max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    f.close()
