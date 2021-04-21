import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time

#####################  hyper parameters  ####################
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
BATCH_SIZE = 32
# OUTPUT_GRAPH = False
OUTPUT_GRAPH = True


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, s_dim, r_dim, b_dim, o_dim, r_bound, b_bound):
        self.memory_capacity = 10000
        # dimension
        self.s_dim = s_dim      # 140
        self.a_dim = r_dim + b_dim + o_dim      # action存储大小，120个 []
        self.r_dim = r_dim      # 10
        self.b_dim = b_dim      # 10
        self.o_dim = o_dim      # 100
        # self.a_bound
        self.r_bound = r_bound  # maximum resource  最大资源
        self.b_bound = b_bound  # maximum bandwidth 最大带宽
        # S, S_, R  输入
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')     # dtype：数据类型。shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）name：名称
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        # memory
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=np.float32)  # s_dim + a_dim + r + s_dim = 401 ，memory_capacity = 10000行，401列
        self.pointer = 0
        # session
        self.sess = tf.Session()        # Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.

        # define the input and output
        self.a = self._build_a(self.S, )        # 输入：a
        q = self._build_c(self.S, self.a, )     # 输出： Q(s,a)

        # replaced target parameters with the trainning  parameters for every step
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')       # tf.get_collection 该函数的作用是从一个collection中取出全部变量，形成列表，key参数中输入的是collection的名称。
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        # TODO 是否考虑num_updates 以改进模型
        # soft replacement  TAU = 0.01  decay 衰减因子
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # tf.train.ExponentialMovingAverage是指数加权平均的求法，具体的公式是 total=a*total+(1-a)*next,

        # 在tf.variable_scope()中，你也可以指定custom_getter作为一个参数，这样你可以一次传入多个变量。
        def ema_getter(getter, name, *args, **kwargs):
            #TODO 取平均是为什么
            return ema.average(getter(name, *args, **kwargs))

        # update the weight for every step
        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation     # ema_op = ema.apply([w]) 的时候，如果 w 是 Variable， 那么将会用 w 的初始值初始化 ema 中关于 w 的 ema_value，所以 emaVal0=1.0。如果 w 是 Tensor的话，将会用 0.0 初始化。
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)       # reuse=True 表示获取变量

        # Actor learn()
        a_loss = - tf.reduce_mean(q)  # maximize the q
        # 更新 a_params
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)      # LR_A = 0.0001  # learning rate for actor # minimize 功能：通过更新 var_list 添加操作以最大限度地最小化 loss。

        # Critic learn()
        with tf.control_dependencies(target_update):  # soft replacement happened at here       # control_dependencies(control_inputs)返回一个控制依赖的上下文管理器，使用with关键字可以让在这个上下文环境中的操作都在control_inputs 执行。
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            # 更新 c_params
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)        # LR_C = 0.0002  # learning rate for critic

        # 运行更新所有参数
        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # 输入： a
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False        # trainable 表示变量是否需要被训练，如果需要被训练，将加入到tf.GraphKeys.TRAINABLE_VARIABLES集合中，TensorFlow将计算其梯度的变量
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):      # tf.name_scope()、tf.variable_scope()会在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
            n_l = 50
            net = tf.layers.dense(s, n_l, activation=tf.nn.relu, name='l1', trainable=trainable)
            # resource ( 0 - r_bound)
            # inputs：输入该网络层的数据， units：输出的维度大小，改变inputs的最后一维， activation：激活函数，即神经网络的非线性变化， trainable=True:表明该层的参数是否参与训练。如果为真则变量加入到图集合中
            layer_r0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.layers.dense(layer_r0, n_l, activation=tf.nn.relu, name='r_1', trainable=trainable)
            layer_r2 = tf.layers.dense(layer_r1, n_l, activation=tf.nn.relu, name='r_2', trainable=trainable)
            layer_r3 = tf.layers.dense(layer_r2, n_l, activation=tf.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.layers.dense(layer_r3, self.r_dim, activation=tf.nn.relu, name='r_4', trainable=trainable)

            # bandwidth ( 0 - b_bound)
            layer_b0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='b_0', trainable=trainable)
            layer_b1 = tf.layers.dense(layer_b0, n_l, activation=tf.nn.relu, name='b_1', trainable=trainable)
            layer_b2 = tf.layers.dense(layer_b1, n_l, activation=tf.nn.relu, name='b_2', trainable=trainable)
            layer_b3 = tf.layers.dense(layer_b2, n_l, activation=tf.nn.relu, name='b_3', trainable=trainable)
            layer_b4 = tf.layers.dense(layer_b3, self.b_dim, activation=tf.nn.relu, name='b_4', trainable=trainable)

            # offloading (probability: 0 - 1)
            # layer     用于建立网络，10个用户，每个用户4个层，每个层命名：user_id+所在层数 00 01 02 03 11 ...
            layer = [["layer" + str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # name      记录所有layer层的名字 layer00
            name = [["layer" + str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # user      记录所有用户经过网络后的输出 ，初始化为 user0 ...
            user = ["user" + str(user_id) for user_id in range(self.r_dim)]
            # softmax   记录所有softmax的名字，softmax0 ...
            softmax = ["softmax" + str(user_id) for user_id in range(self.r_dim)]
            for user_id in range(self.r_dim):
                layer[user_id][0] = tf.layers.dense(net, n_l, activation=tf.nn.relu, name=name[user_id][0], trainable=trainable)
                layer[user_id][1] = tf.layers.dense(layer[user_id][0], n_l, activation=tf.nn.relu, name=name[user_id][1], trainable=trainable)
                layer[user_id][2] = tf.layers.dense(layer[user_id][1], n_l, activation=tf.nn.relu, name=name[user_id][2], trainable=trainable)
                #TODO: (self.o_dim / self.r_dim)的意义？
                layer[user_id][3] = tf.layers.dense(layer[user_id][2], (self.o_dim / self.r_dim), activation=tf.nn.relu, name=name[user_id][3], trainable=trainable)
                user[user_id] = tf.nn.softmax(layer[user_id][3], name=softmax[user_id])

            # concate
            a = tf.concat([layer_r4, layer_b4], 1)      # 用来拼接张量的函数 axis=1     代表在第1个维度拼接：行不变，列相加
            for user_id in range(self.r_dim):
                a = tf.concat([a, user[user_id]], 1)
            return a

    # 输出：Q(s,a)
    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # Q value (0 - inf)
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            # 常量初始化器        # shape=[self.s_dim, n_l]  [行， 列]
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l], trainable=trainable)      # state权重
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l], trainable=trainable)      # action权重
            b1 = tf.get_variable('b1', [1, n_l], trainable=trainable)       # 偏差 bia
            # TODO： 为什么都是过4层网络？
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(net_1, n_l, activation=tf.nn.relu, trainable=trainable)
            net_3 = tf.layers.dense(net_2, n_l, activation=tf.nn.relu, trainable=trainable)
            net_4 = tf.layers.dense(net_3, n_l, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net_4, 1, activation=tf.nn.relu, trainable=trainable)  # Q(s,a)
