import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 周向参数
R = 0.01  # 管道半径 [m] (与plane_2D.py保持一致)
L = 2 * np.pi * R  # 周长 = 2πR
N_points = 1000
x = np.linspace(0, L, N_points)

# 时间参数
v_rde = 1400  # m/s
T_travel = L / v_rde  # 每圈传播时间（秒）

n_frames = 100
t_vals = np.linspace(0, T_travel, n_frames)

# 爆轰信号模型：陡升 + 指数衰减
def detonation_wave(x, center, A=1.0, width=0.001):
    dx = (x - center + L) % L  # 周期处理
    signal = np.zeros_like(x)
    rise_region = dx < 1e-4
    decay_region = ~rise_region
    signal[rise_region] = A
    signal[decay_region] = A * np.exp(- (dx[decay_region] - 1e-4) / width)
    return signal

# 生成动画帧
frames = []
for t in t_vals:
    center = -(t / T_travel) * L  # 爆轰波中心位置随时间移动
    p = detonation_wave(x, center)
    frames.append(p)

# 绘图与动画
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, L * 1000)  # 单位换成 mm
ax.set_ylim(0, 1.2)
ax.set_xlabel("Distance x (mm)")
ax.set_ylabel("Pressure (a.u.)")
ax.set_title("Propagation mode in xy surface")

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(x * 1000, frames[i])
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=n_frames, interval=50, blit=True)

plt.show()
