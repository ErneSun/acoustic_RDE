import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 参数设置
R = 0.01  # 管道半径 [m] (与plane_2D.py保持一致)
L = 2 * np.pi * R  # 周长 = 2πR
H = 0.01  # 管道高度 [m]
diameter = 2 * R  # 管道直径 [m]
v_rde = 1400  # 爆轰波速度 [m/s]
wave_height = 0.002  # 爆轰波高度 [m]
wave_pressure = 10.0  # 爆轰波最大压强 [Pa] (注意：10 Pa，不是10 atm)
tilt_angle = 10  # 前倾角度 [度]
tilt_rad = np.radians(tilt_angle)  # 转换为弧度
x_N_points = 1000  # 网格点数
y_N_points = 500
n_frames = 100  # 动画帧数

# 网格定义
x = np.linspace(0, L, x_N_points)
y = np.linspace(0, H, y_N_points)
X, Y = np.meshgrid(x, y)

# 时间参数
T_travel = L / v_rde  # 每圈传播时间
t_vals = np.linspace(0, T_travel, n_frames)

# 计算每一层高度的压力分布（阶跃+反高斯结构）
def pressure_field(X, y, t, wave_pressure, tilt_rad):
    # 计算波头的位置，考虑倾斜角度
    wave_position = (v_rde * t) + y * np.tan(tilt_rad)  # 波头在圆周的位置

    # 计算与波前的距离
    distance = X - wave_position

    # 模拟压力场：波前（阶跃）和波尾（反高斯）
    pressure = np.zeros_like(X)+ 1

    # 处理阶跃部分（波前）
    pressure[(distance > 0) & (distance < 1e-4)] = wave_pressure
    # 处理反高斯部分（波尾）
    pressure[distance <= 0] = wave_pressure * np.exp(
        - (distance[distance <= 0]) ** 2 / (2 * (0.001) ** 2))

    return pressure

# 动画帧生成
frames = []
for t in t_vals:
    # 逐层遍历y轴（高度）
    pressure_layer = np.zeros_like(X) + 1
    for i in range(len(y)):
        if y[i] < 0.002:
            pressure_layer[i, :] = pressure_field(X[i, :], y[i], t, wave_pressure, tilt_rad)
    frames.append(pressure_layer)

# 绘图与动画
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(frames[0], origin='lower', extent=[0, L * 1000, 0, H * 1000], cmap='viridis', animated=True)
fig.colorbar(cax, ax=ax, label='Pressure (a.u.)')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title('Pressure Cloud of Moving Detonation Wave')

def animate(i):
    cax.set_data(frames[i])
    return cax,

ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)
plt.show()
