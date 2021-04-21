import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import render

#####################  hyper parameters  ####################
LOCATION = "KAIST"
USER_NUM = 10
EDGE_NUM = 10
LIMIT = 4  # 每个边缘服务器最多可以提供4种任务处理服务
MAX_EP_STEPS = 3000
TXT_NUM = 92
r_bound = 1e9 * 0.063
b_bound = 1e9


#####################  function  ####################
def trans_rate(user_loc, edge_loc):
    B = 2e6
    P = 0.25
    d = np.sqrt(np.sum(np.square(user_loc[0] - edge_loc))) + 0.01
    h = 4.11 * math.pow(3e8 / (4 * math.pi * 915e6 * d), 2)
    N = 1e-10
    return B * math.log2(1 + P * h / N)


# 初始化edge之间的带宽记录表
def BandwidthTable(edge_num):
    BandwidthTable = np.zeros((edge_num, edge_num))
    for i in range(0, edge_num):
        for j in range(i + 1, edge_num):
            BandwidthTable[i][j] = 1e9
    return BandwidthTable


# 转化成一维表
def two_to_one(two_table):
    one_table = two_table.flatten()
    return one_table


# 状态包括edge可用资源、edge之间连接的带宽、需要卸载的用户、用户的位置（x,y)
def generate_state(two_table, U, E, x_min, y_min):
    # initial
    one_table = two_to_one(two_table)
    S = np.zeros((len(E) + one_table.size + len(U) + len(U) * 2))  # 140=10+100+10+20
    # transform
    count = 0
    # available resource of each edge server
    for edge in E:
        S[count] = edge.capability / (r_bound * 10)  # 0.1   平均每个edge的承载能力0.1   #这里或许可以改进，因为实际上不一定每个edge平均能力
        count += 1
    # available bandwidth of each connection
    for i in range(len(one_table)):
        S[count] = one_table[i] / (b_bound * 10)  # 0.1   每个 edge之间带宽 0.1
        count += 1
    # offloading of each user       已经/需要 卸载的用户
    for user in U:
        S[count] = user.req.edge_id / 100  # 为什么除以100？     以小于1的形式存放
        count += 1
    # location of the user      以小于1的形式存放
    # TODO: 为什么以小于1的形式存放？（为了屏幕显示吗）为什么加上 min?
    for user in U:
        S[count] = (user.loc[0][0] + abs(x_min)) / 1e5  # 位置的x坐标
        S[count + 1] = (user.loc[0][1] + abs(y_min)) / 1e5  # 位置的y坐标
        count += 2
    return S


def generate_action(R, B, O):
    # resource
    a = np.zeros(USER_NUM + USER_NUM + EDGE_NUM * USER_NUM)
    a[:USER_NUM] = R / r_bound
    # bandwidth
    a[USER_NUM:USER_NUM + USER_NUM] = B / b_bound
    # offload
    base = USER_NUM + USER_NUM
    for user_id in range(USER_NUM):
        a[base + int(O[user_id])] = 1
        base += EDGE_NUM
    return a


# 返回data中最小的x ,y
def get_minimum():
    cal = np.zeros((1, 2))  # [[0. 0.]] 一维2列
    for data_num in range(TXT_NUM):  # TXT_NUM = 92 个数据文件
        data_name = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_name + ".txt"
        file_path = "../data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        print(f)
        f1 = f.readlines()
        # get line_num
        line_num = 0
        for line in f1:
            line_num += 1
        # collect the data from the .txt
        data = np.zeros((line_num, 2))
        index = 0
        for line in f1:
            # 数据txt文件每行第一列表示index，无具体实意
            data[index][0] = line.split()[1]  # x
            data[index][1] = line.split()[2]  # y
            index += 1
        # put data into the cal     cla[0] = [0., 0.]
        cal = np.vstack((cal, data))  # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    return min(cal[:, 0]), min(cal[:, 1])  # 返回data中最小的x ,y


# 计算edge_num 个 group中data.txt每一列数据（x,y）的平均值
def proper_edge_loc(edge_num):
    # initial the e_l
    e_l = np.zeros((edge_num, 2))
    # calculate the mean of the data
    group_num = math.floor(TXT_NUM / edge_num)  # 向下取整
    edge_id = 0
    for base in range(0, group_num * edge_num, group_num):
        for data_num in range(base, base + group_num):
            data_name = str("%03d" % (data_num + 1))  # plus zero
            file_name = LOCATION + "_30sec_" + data_name + ".txt"
            file_path = "../data/" + LOCATION + "/" + file_name
            f = open(file_path, "r")
            f1 = f.readlines()
            # get line_num and initial data
            line_num = 0
            for line in f1:
                line_num += 1
            data = np.zeros((line_num, 2))
            # collect the data from the .txt
            index = 0
            for line in f1:
                data[index][0] = line.split()[1]  # x
                data[index][1] = line.split()[2]  # y
                index += 1
            # stack the collected data
            if data_num % group_num == 0:
                cal = data
            else:
                cal = np.vstack((cal, data))  # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
        e_l[edge_id] = np.mean(cal, axis=0)  # axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）。还可以这么理解，axis是几，那就表明哪一维度被压缩成1。
        edge_id += 1
    return e_l


#############################UE###########################
# 采样的user_num个user数据data
class UE:
    def __init__(self, user_id, data_num):
        self.user_id = user_id  # number of the user
        self.loc = np.zeros((1, 2))  # [[0.0, 0.0]]
        self.num_step = 0  # the number of step

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_num + ".txt"
        file_path = "../data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0  # 记录txt文件数据的条数
        for line in f1:
            data += 1
        # TODO *30是什么意思？ 大概是STEP = 30步，记录每一步的数据？
        self.num_step = data * 30
        self.mob = np.zeros((self.num_step, 2))  # 数据移动的记录

        # write data to self.mob
        now_sec = 0
        for line in f1:
            for sec in range(30):
                # self.mob中每个(x, y)连续存储30次
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30
        self.loc[0] = self.mob[0]  # 当前所处位置是初始位置

    # 初始化一个user到edge的请求实例
    def generate_request(self, edge_id):
        self.req = Request(self.user_id, edge_id)       # 创建request实例，初始化   # __init__函数里不包含self.req

    def request_update(self):
        # default request.state == 5 means disconnection ,6 means migration
        if self.req.state == 5:
            self.req.timer += 1
        else:
            self.req.timer = 0
            if self.req.state == 0:
                self.req.state = 1
                self.req.u2e_size = self.req.tasktype.req_u2e_size
                self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
            elif self.req.state == 1:
                if self.req.u2e_size > 0:
                    self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
                else:
                    self.req.state = 2
                    self.req.process_size = self.req.tasktype.process_loading
                    self.req.process_size -= self.req.resource
            elif self.req.state == 2:
                if self.req.process_size > 0:
                    self.req.process_size -= self.req.resource
                else:
                    self.req.state = 3
                    self.req.e2u_size = self.req.tasktype.req_e2u_size
                    self.req.e2u_size -= 10000  # value is small,so simplify
            else:
                if self.req.e2u_size > 0:
                    self.req.e2u_size -= 10000  # B*math.log(1+SINR(self.user.loc, self.offloading_serv.loc), 2)/(8*time_scale)
                else:
                    self.req.state = 4

    def mobility_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]  # x

        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf


# 单个user对edge的请求
class Request:
    def __init__(self, user_id, edge_id):
        # id
        self.user_id = user_id  # 发出请求的用户id
        self.edge_id = edge_id  # 请求的edge id
        self.edge_loc = 0
        # state
        self.state = 5  # 5: not connect
        self.pre_state = 5
        # transmission size
        self.u2e_size = 0
        self.process_size = 0
        self.e2u_size = 0
        # edge state
        self.resource = 0
        self.mig_size = 0
        # tasktype
        self.tasktype = TaskType()
        self.last_offlaoding = 0
        # timer
        self.timer = 0


# 任务类型，即各项限制条件，参数， 包含 传播 迁移
class TaskType:
    def __init__(self):
        ##Objection detection: VOC SSD300
        # transmission
        self.req_u2e_size = 300 * 300 * 3 * 1
        self.process_loading = 300 * 300 * 3 * 4  # 卸载大小
        self.req_e2u_size = 4 * 4 + 20 * 4
        # migration
        self.migration_size = 2e9  # 迁移大小

    # 返回任务信息
    def task_inf(self):
        return "req_u2e_size:" + str(self.req_u2e_size) + "\nprocess_loading:" + str(
            self.process_loading) + "\nreq_e2u_size:" + str(self.req_e2u_size)


#############################EdgeServer###################

class EdgeServer:

    def __init__(self, edge_id, loc):
        self.edge_id = edge_id  # edge server number
        self.loc = loc      # 位置
        self.capability = 1e9 * 0.063  # 所有edge server 的总承载能力
        self.user_group = []
        self.limit = LIMIT
        self.connection_num = 0

    def maintain_request(self, R, U):
        for user in U:
            # the number of the connection user
            self.connection_num = 0
            for user_id in self.user_group:
                if U[user_id].req.state != 6:
                    self.connection_num += 1
            # maintain the request
            if user.req.edge_id == self.edge_id and self.capability - R[user.user_id] > 0:
                # maintain the preliminary connection
                if user.req.user_id not in self.user_group and self.connection_num + 1 <= self.limit:
                    # first time : do not belong to any edge(user_group)
                    self.user_group.append(user.user_id)  # add to the user_group
                    user.req.state = 0  # prepare to connect
                    # notify the request
                    user.req.edge_id = self.edge_id
                    user.req.edge_loc = self.loc

                # dispatch the resource
                user.req.resource = R[user.user_id]
                self.capability -= R[user.user_id]

    def migration_update(self, O, B, table, U, E):

        # maintain the the migration
        for user_id in self.user_group:
            # prepare to migration
            if U[user_id].req.edge_id != O[user_id]:
                # initial
                ini_edge = int(U[user_id].req.edge_id)
                target_edge = int(O[user_id])
                if table[ini_edge][target_edge] - B[user_id] >= 0:
                    # on the way to migration, but offloading to another edge computer(step 1)
                    if U[user_id].req.state == 6 and target_edge != U[user_id].req.last_offlaoding:
                        # reduce the bandwidth
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        # print("user", U[user_id].req.user_id, ":migration step 1")
                    # first try to migration(step 1)
                    elif U[user_id].req.state != 6:
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        # store the pre state
                        U[user_id].req.pre_state = U[user_id].req.state
                        # on the way to migration, disconnect to the old edge
                        U[user_id].req.state = 6
                        # print("user", U[user_id].req.user_id, ":migration step 1")
                    elif U[user_id].req.state == 6 and target_edge == U[user_id].req.last_offlaoding:
                        # keep migration(step 2)
                        if U[user_id].req.mig_size > 0:
                            # reduce the bandwidth
                            table[ini_edge][target_edge] -= B[user_id]
                            U[user_id].req.mig_size -= B[user_id]
                            # print("user", U[user_id].req.user_id, ":migration step 2")
                        # end the migration(step 3)
                        else:
                            # the number of the connection user
                            target_connection_num = 0
                            for target_user_id in E[target_edge].user_group:
                                if U[target_user_id].req.state != 6:
                                    target_connection_num += 1
                            # print("user", U[user_id].req.user_id, ":migration step 3")
                            # change to another edge
                            if E[target_edge].capability - U[user_id].req.resource >= 0 and target_connection_num + 1 <= \
                                    E[target_edge].limit:
                                # register in the new edge
                                E[target_edge].capability -= U[user_id].req.resource
                                E[target_edge].user_group.append(user_id)
                                self.user_group.remove(user_id)
                                # update the request
                                # id
                                U[user_id].req.edge_id = E[target_edge].edge_id
                                U[user_id].req.edge_loc = E[target_edge].loc
                                # release the pre-state, continue to transmission process
                                U[user_id].req.state = U[user_id].req.pre_state
                                # print("user", U[user_id].req.user_id, ":migration finish")
            # store pre_offloading
            U[user_id].req.last_offlaoding = int(O[user_id])

        return table

    # release the all resource
    def release(self):
        self.capability = 1e9 * 0.063


#############################Policy#######################
# 优先策略：10*10 数组  每一行代表一个user在各个edge服务器的优先顺序，数字越小，离得越近，优先权越高
class priority_policy():
    # 生成优先策略
    def generate_priority(self, U, E, priority):
        for user in U:
            # get a list of the offloading priority     卸载优先权
            dist = np.zeros(EDGE_NUM)
            for edge in E:
                dist[edge.edge_id] = np.sqrt(np.sum(np.square(user.loc[0] - edge.loc)))     # 求向量的距离公式：  np.sqrt(np.sum(np.square(vector1 - vector2))) # (x, y) - (x, y)
            dist_sort = np.sort(dist)       # 默认由小到大，所以距离越近，优先权越高
            for index in range(EDGE_NUM):
                # ind = np.argwhere(dist == dist_sort[index])[0]     # [[9]] np.array类型
                priority[user.user_id][index] = np.argwhere(dist == dist_sort[index])[0]        # np.argwhere返回非0的数组元组的索引，其中a是要索引数组的条件。
        return priority     # 10*10 数组

    def indicate_edge(self, O, U, priority):
        edge_limit = np.ones((EDGE_NUM)) * LIMIT    # [4,4,4,4,4,4,4,4,4,4] 表示每个edge目前可以挂在的user数，为0的时候表示不能加入user了     # np.ones 返回一个全1的n维数组
        for user in U:
            for index in range(EDGE_NUM):
                # TODO: 为什么这么做？
                # 10*10的数组每一行第一列的数加入O[user.user_id]，若这个数加入了四次，看第二列的数
                if edge_limit[int(priority[user.user_id][index])] - 1 >= 0:
                    edge_limit[int(priority[user.user_id][index])] -= 1
                    O[user.user_id] = priority[user.user_id][index]
                    break
        return O

    def resource_update(self, R, E, U):
        for edge in E:
            # count the number of the connection user
            connect_num = 0
            for user_id in edge.user_group:
                if U[user_id].req.state != 5 and U[user_id].req.state != 6:
                    connect_num += 1
            # dispatch the resource to the connection user
            for user_id in edge.user_group:
                # no need to provide resource to the disconnecting users
                if U[user_id].req.state == 5 or U[user_id].req.state == 6:
                    R[user_id] = 0
                # provide resource to connecting users
                else:
                    R[user_id] = edge.capability / (connect_num + 2)  # reserve the resource to those want to migration
        return R

    def bandwidth_update(self, O, table, B, U, E):
        for user in U:
            share_number = 1
            ini_edge = int(user.req.edge_id)
            target_edge = int(O[user.req.user_id])
            # no need to migrate
            if ini_edge == target_edge:
                B[user.req.user_id] = 0
            # provide bandwidth to migrate
            else:
                # share bandwidth with user from migration edge
                for user_id in E[target_edge].user_group:
                    if O[user_id] == ini_edge:
                        share_number += 1
                # share bandwidth with the user from the original edge to migration edge
                for ini_user_id in E[ini_edge].user_group:
                    if ini_user_id != user.req.user_id and O[ini_user_id] == target_edge:
                        share_number += 1
                # allocate the bandwidth
                B[user.req.user_id] = table[min(ini_edge, target_edge)][max(ini_edge, target_edge)] / (share_number + 2)

        return B


#############################Env###########################

# DDPG 所需的环境
class Env:
    def __init__(self):
        self.step = 30
        self.time = 0
        self.edge_num = EDGE_NUM  # the number of servers
        self.user_num = USER_NUM  # the number of users
        # define environment object
        self.reward_all = []
        self.U = []  # 记录采样的user_num个user
        self.fin_req_count = 0
        self.prev_count = 0
        self.rewards = 0
        self.R = np.zeros((self.user_num))
        self.O = np.zeros((self.user_num))
        self.B = np.zeros((self.user_num))
        self.table = BandwidthTable(self.edge_num)  # edge之间带宽的记录表
        self.priority = np.zeros((self.user_num, self.edge_num))
        self.E = []  # EdgeServer
        self.x_min, self.y_min = get_minimum()  # data表中最小的x, y

        self.e_l = 0
        self.model = 0

    # 返回环境的信息 s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, LIMIT, LOCATION
    def get_inf(self):
        # s_dim     状态state的存储大小
        self.reset()
        s = generate_state(self.table, self.U, self.E, self.x_min, self.y_min)  # 状态包括edge可用资源、edge之间连接的带宽、需要卸载的用户、用户的位置（x,y)
        s_dim = s.size

        # a_dim     行动action的存储大小
        r_dim = len(self.U)  # 所有user的资源所占存储空间大小
        b_dim = len(self.U)  # 所有带宽所占存储空间的大小
        o_dim = self.edge_num * len(self.U)     # 100， 大概是连接数？

        # maximum resource  最大资源
        r_bound = self.E[0].capability  # 6300 0000.0

        # maximum bandwidth 最大带宽
        b_bound = self.table[0][1]  # 10 0000 0000.0
        b_bound = b_bound.astype(np.float32)

        # task size 任务数
        task = TaskType()
        task_inf = task.task_inf()  # 任务信息

        return s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, LIMIT, LOCATION

    # 返回初始状态
    def reset(self):
        # reset time
        self.time = 0
        # reward
        self.reward_all = []
        # user，采样user_num个user
        self.U = []     # 记录采样的user_num个user，每个是UE的实例，user的id由0到1
        self.fin_req_count = 0
        self.prev_count = 0
        data_num = random.sample(list(range(TXT_NUM)), self.user_num)  # data集中随机选取user_num个txt文件，记录文件序号
        for i in range(self.user_num):
            new_user = UE(i, data_num[i])  # new_user 记录采样出来的data用户数据，主要包含移动位置变化数据
            self.U.append(new_user)  # 记录采样的user_num个user数据，包含移动位置变化数据
        # Resource
        self.R = np.zeros((self.user_num))
        # Offlaoding
        self.O = np.zeros((self.user_num))  # len = 10
        # bandwidth
        self.B = np.zeros((self.user_num))
        # bandwidth table
        self.table = BandwidthTable(self.edge_num)
        # server
        self.E = []     # edge_server的基本信息
        e_l = proper_edge_loc(self.edge_num)  # 计算edge_num 个 group中data.txt每一列数据（x,y）的平均值，作为edge的位置
        for i in range(self.edge_num):
            new_e = EdgeServer(i, e_l[i, :])
            self.E.append(new_e)
            """
            print("edge", new_e.edge_id, "'s loc:\n", new_e.loc)
        print("========================================================")
        """
        # model
        self.model = priority_policy()

        # initialize the request
        self.priority = self.model.generate_priority(self.U, self.E, self.priority)     # # 优先策略：10*10 数组  每一行代表一个user在各个edge服务器的优先顺序，数字越小，离得越近，优先权越高
        self.O = self.model.indicate_edge(self.O, self.U, self.priority)        # index表示user_id，value表示edge_id
        for user in self.U:
            user.generate_request(self.O[user.user_id])     # 时间上传入了edge_id
        return generate_state(self.table, self.U, self.E, self.x_min, self.y_min)

    # 前向训练forward
    def ddpg_step_forward(self, a, r_dim, b_dim):
        # release the bandwidth
        self.table = BandwidthTable(self.edge_num)
        # release the resource
        for edge in self.E:
            edge.release()

        # update the policy every second
        # resource update
        self.R = a[:r_dim]
        # bandwidth update
        self.B = a[r_dim:r_dim + b_dim]
        # offloading update
        base = r_dim + b_dim
        for user_id in range(self.user_num):
            prob_weights = a[base:base + self.edge_num]
            # print("user", user_id, ":", prob_weights)
            action = np.random.choice(range(len(prob_weights)),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
            base += self.edge_num
            self.O[user_id] = action

        # request update
        for user in self.U:
            # update the state of the request
            user.request_update()
            if user.req.timer >= 5:
                user.generate_request(self.O[user.user_id])  # offload according to the priority
            # it has already finished the request
            if user.req.state == 4:
                # rewards
                self.fin_req_count += 1
                user.req.state = 5  # request turn to "disconnect"
                self.E[int(user.req.edge_id)].user_group.remove(user.req.user_id)
                user.generate_request(self.O[user.user_id])  # offload according to the priority

        # edge update
        for edge in self.E:
            edge.maintain_request(self.R, self.U)
            self.table = edge.migration_update(self.O, self.B, self.table, self.U, self.E)

        # rewards
        self.rewards = self.fin_req_count - self.prev_count
        self.prev_count = self.fin_req_count

        # every user start to move
        if self.time % self.step == 0:
            for user in self.U:
                user.mobility_update(self.time)

        # update time
        self.time += 1

        # return s_, r
        return generate_state(self.table, self.U, self.E, self.x_min, self.y_min), self.rewards

    # 更新时屏幕打印信息，包括 R B O,user，edge，reward
    def text_render(self):
        # 打印 R B O
        print("R:", self.R)
        print("B:", self.B)
        """
        base = USER_NUM +USER_NUM
        for user in range(len(self.U)):
            print("user", user, " offload probabilty:", a[base:base + self.edge_num])
            base += self.edge_num
        """
        print("O:", self.O)
        # 打印每个user的loc、request state、edge serve
        for user in self.U:
            print("user", user.user_id, "'s loc:\n", user.loc)
            print("request state:", user.req.state)
            print("edge serve:", user.req.edge_id)
        # 打印每个edge的user_group, 即每个边缘服务器正在为哪些用户服务
        for edge in self.E:
            print("edge", edge.edge_id, "user_group:", edge.user_group)
        # 打印reward
        print("reward:", self.rewards)
        print("=====================update==============================")

    # 初始化屏幕
    def initial_screen_demo(self):
        self.canvas = render.Demo(self.E, self.U, self.O, MAX_EP_STEPS)

    #   屏幕画图 移动节点 服务器 分布变化
    def screen_demo(self):
        self.canvas.draw(self.E, self.U, self.O)
