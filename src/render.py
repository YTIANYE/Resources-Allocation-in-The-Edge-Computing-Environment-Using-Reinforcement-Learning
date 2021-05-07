from tkinter import *
import random
import numpy as np
import math

from src.env import LOCATION


# 随机设定edge的颜色
def dispatch_color(edge_color , E):
    for egde_id in range(len(E)):
        color = '#' + str("%03d" % random.randint(0, 255))[2:] + str("%03d" % random.randint(0, 255))[2:] + str("%03d" %random.randint(0, 255))[2:]
        edge_color.append(color)
    return edge_color

# 返回所有user移动轨迹中x, y 的最大最小值
def get_info(U, MAX_EP_STEPS):
    x_min, x_Max, y_min, y_Max = np.inf, -np.inf, np.inf, -np.inf       # inf 正无穷大的浮点表示,常用于数值比较当中的初始值
    # x axis
    for user in U:
       if(max(user.mob[:, 0]) > x_Max):
           x_Max = max(user.mob[:, 0])
       if(min(user.mob[:, 0]) < x_min):
           x_min = min(user.mob[:, 0])
    # y axis
    for user in U:
        if (max(user.mob[:MAX_EP_STEPS, 1]) > y_Max):
            y_Max = max(user.mob[:MAX_EP_STEPS, 1])
        if (min(user.mob[:MAX_EP_STEPS, 1]) < y_min):
            y_min = min(user.mob[:MAX_EP_STEPS, 1])
    return x_min, x_Max, y_min, y_Max       # 返回所有user移动轨迹中x, y 的最大最小值

#####################  hyper parameters  ####################
MAX_SCREEN_SIZE = 1000      # 窗口最大宽度
EDGE_SIZE = 20              # Edge 大小
USER_SIZE = 10              # user 大小

#####################  User  ####################
# user的圆形表示
class oval_User:
    def __init__(self, canvas, color, user_id):
        self.user_id = user_id
        self.canvas = canvas
        self.id = canvas.create_oval(500, 500, 500 + USER_SIZE, 500 + USER_SIZE, fill=color)        # 初始化在中心点的位置

    def draw(self, vector, edge_color, user):
        info = self.canvas.coords(self.id)      # [500.0, 500.0, 510.0, 510.0]
        self.canvas.delete(self.id)     # 删掉所有args参数里面指定的tag和id所标识的项目
        # connect       request task is handled by the edge server with the same color and is in state 1 ~ state 4
        if user.req.state != 5 and user.req.state != 6:         # state 1 : start to offload a task to the edge server, state 2 : request task is on the way to the edge server (2.7 * 1e4 byte), state 3 : request task is proccessed (1.08 * 1e6 byte), state 4 : request task is on the way back to the mobile user (96 byte)
            self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill=edge_color)
        # not connected
        else:
            # disconnection     state 5 : disconnect (default)
            if user.req.state == 5:     # Red : request task is in state 5
                self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill="red")
            # migration     state 6 : request task is migrated to another edge server
            elif user.req.state == 6:       # Green : request task is in state 6
                self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill="green")
        # move the user     更改位置
        self.canvas.move(self.id, vector[0][0], vector[0][1])

#####################  Edge  ####################
# edge的圆形表示
class oval_Edge:
    def __init__(self, canvas, color, edge_id):
        self.edge_id = edge_id
        self.canvas = canvas
        # 每个edge初始位置[500, 500, 520, 520] 由于窗口长宽1000，所以初始化的左上角为窗口中心点[500, 500]
        self.id = canvas.create_oval(500, 500, 500 + EDGE_SIZE, 500 + EDGE_SIZE, fill=color)        # 画一个圆，坐标两双，为圆的边界矩形左上角和底部右下角

    # 画出edge位置
    def draw(self, vector):
        self.canvas.move(self.id, vector[0][0], vector[0][1])       # 移动到（x, y），没有原始位置，edge不需要移动，所以就是在这里画出来

#####################  convas  ####################
class Demo:
    def __init__(self, E, U, O, MAX_EP_STEPS):
        # create canvas 创建画布
        self.x_min, self.x_Max, self.y_min, self.y_Max = get_info(U, MAX_EP_STEPS)      # 返回所有user移动轨迹中x, y 的最大最小值
        self.tk = Tk()      # 实例化一个Tk 用于容纳整个GUI程序
        self.tk.title("Simulation: Resource Allocation in Egde Computing Environment")
        self.tk.resizable(0, 0)     # 不允许调整窗口大小
        self.tk.wm_attributes("-topmost", 1)        # 1 置顶 0 不置顶
        # self.canvas = Canvas(self.tk, width=MAX_SCREEN_SIZE, height=1000, bd=0, highlightthickness=0, bg='black')       # master: 按钮的父容器，bd边框宽度，单位像素，默认为 2 像素，highlightthickness边框宽度
        self.canvas = Canvas(self.tk, width=MAX_SCREEN_SIZE, height=1000, bd=0, highlightthickness=0, bg='white')
        self.canvas.pack()      # 将小部件放置到主窗口中
        self.tk.update()        # 可以接收用户改变程序进程
        # TODO x_rate y_rate 为什么用MAX_SCREEN_SIZE除以更大的range？ 答：选择最大的，为了更好的进行距离转换，按照比例，实际距离转换为窗口中的距离
        x_range = self.x_Max - self.x_min
        y_range = self.y_Max - self.y_min
        # x_rate x的最小单位     y_rate y的最小单位
        self.rate = x_range/y_range
        if self.rate > 1:
            self.x_rate = (MAX_SCREEN_SIZE / x_range)       # 实际距离与窗口中像素距离之间换算的比例尺
            self.y_rate = (MAX_SCREEN_SIZE / y_range) * (1/self.rate)
        else:
            self.x_rate = (MAX_SCREEN_SIZE / x_range) * (self.rate)
            self.y_rate = (MAX_SCREEN_SIZE / y_range)

        self.edge_color = []
        self.edge_color = dispatch_color(self.edge_color, E)        # 随机设定edge的颜色
        self.oval_U, self.oval_E = [], []       # 存储画布中的user edge
        # initialize the object
        for edge_id in range(len(E)):
            self.oval_E.append(oval_Edge(self.canvas, self.edge_color[edge_id], edge_id))
        for user_id in range(len(U)):       # user的颜色根据O[]中和edge的关系确定与edge相同的颜色
            self.oval_U.append(oval_User(self.canvas, self.edge_color[int(O[user_id])], user_id))

    def draw(self, E, U, O):
        # edge
        # TODO 这里的位置计算可能有问题，因为画出来的图所有的点总是聚集在屏幕的一小部分
        edge_vector = np.zeros((1, 2))
        for edge in E:
            # TODO 位置为什么这么算？
            # 乘以x_rate y_rate是为了把实际距离换算为窗口中的像素距离，       减去的是窗口中心点坐标[500, 500]
            # 比如: edge.loc[0] = -297.31984486193113， self.x_min = -4068.363564179178， self.x_rate = 0.12756728744880452， (edge.loc[0] - self.x_min) * self.x_rate = 481.0618181241522， self.canvas.coords(self.oval_E[edge.edge_id].id) = [500.0, 500.0, 520.0, 520.0]
            edge_vector[0][0] = (edge.loc[0] - self.x_min) * self.x_rate - self.canvas.coords(self.oval_E[edge.edge_id].id)[0]      # self.canvas.coords() 返回一个坐标，[0]表示 x
            # j = self.canvas.coords(self.oval_E[edge.edge_id].id)[1]     # 500.0
            edge_vector[0][1] = (edge.loc[1] - self.y_min) * self.y_rate - self.canvas.coords(self.oval_E[edge.edge_id].id)[1]      # coords(ID)          返回对象的位置的两个坐标（4个数字元组）
            self.oval_E[edge.edge_id].draw(edge_vector)
        # user
        user_vector = np.zeros((1, 2))
        for user in U:
            user_vector[0][0] = (user.loc[0][0] - self.x_min) * self.x_rate - self.canvas.coords(self.oval_U[user.user_id].id)[0]
            user_vector[0][1] = (user.loc[0][1] - self.y_min) * self.y_rate - self.canvas.coords(self.oval_U[user.user_id].id)[1]
            self.oval_U[user.user_id].draw(user_vector, self.edge_color[int(O[user.user_id])], user)        # user的颜色根据O[]中和edge的关系确定与edge相同的颜色
        # 快速刷新屏幕
        self.tk.update_idletasks()      # update_idletasks()导致要处理的事件的某些子集update()。
        self.tk.update()        # 使用update命令使应用程序“更新”
        # 执行完上述命令画布出现 分布情况

#####################  Outer parameter  ####################
class UE():
    def __init__(self, user_id, data_num):
        self.user_id = user_id  # number of the user
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_num + ".txt"
        file_path = LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0
        for line in f1:
            data += 1
        self.num_step = data * 30
        self.mob = np.zeros((self.num_step, 2))

        # write data to self.mob
        now_sec = 0
        for line in f1:
            for sec in range(30):
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30

    def mobility_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]   # x

        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf

class EdgeServer():
    def __init__(self, edge_id, loc):
        self.edge_id = edge_id  # edge server number
        self.loc = loc


