import numpy as np
import os
import sys
import time

# 物理参数
rho0 = 1.225        # 空气密度 [kg/m³]
c0   = 343.0        # 声速 [m/s]
nu   = 1.5e-5       # 运动黏度 [m²/s]

# 几何与网格
R    = 0.01         # 管道半径 [m]
dr   = 0.0002       # 网格间距 [m]
x = np.arange(-R, R+dr, dr)
y = np.arange(-R, R+dr, dr)
nx, ny = len(x), len(y)
X, Y = np.meshgrid(x, y)
mask = (X**2 + Y**2) <= R**2  # 管道内部掩蔽

# 时间参数
dt    = 1e-9        # 时间步长 [s]
t_end = 1e-3        # 结束时间 [s]
nt    = int(t_end / dt)

# 场变量初始化
p = np.zeros((ny, nx))
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

# 声源参数
wave_pressure = 10.0      # 爆轰波压强幅度 [Pa]
inner_r  = R - 0.002      # 环带内半径 [m]
outer_r  = R              # 环带外半径 [m]
omega    = 1400.0 / R     # 角速度 [rad/s]
sigma    = 0.001          # 衰减尺度 [m]

# 输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def source_term(X, Y, t):
    theta = np.arctan2(Y, X) % (2*np.pi)
    r     = np.sqrt(X**2 + Y**2)
    theta0 = (omega * t) % (2*np.pi)
    dtheta = (theta - theta0 + np.pi) % (2*np.pi) - np.pi
    mask_ring = (r>=inner_r)&(r<=outer_r)&(dtheta>=0)
    S = np.zeros_like(X)
    S[mask_ring] = wave_pressure * np.exp(- (dtheta[mask_ring]*R)**2/(2*sigma**2))
    return S


# 差分算子
def laplacian(f):
    return (np.roll(f,1,0)-2*f+np.roll(f,-1,0))/dr**2 + (np.roll(f,1,1)-2*f+np.roll(f,-1,1))/dr**2

def divergence(u, v):
    du_dx = (np.roll(u,-1,1)-np.roll(u,1,1))/(2*dr)
    dv_dy = (np.roll(v,-1,0)-np.roll(v,1,0))/(2*dr)
    return du_dx + dv_dy

def gradient(p):
    dp_dx = (np.roll(p,-1,1)-np.roll(p,1,1))/(2*dr)
    dp_dy = (np.roll(p,-1,0)-np.roll(p,1,0))/(2*dr)
    return dp_dx, dp_dy


# 主循环
start = time.time()
for n in range(nt):
    t = n * dt

    # 1. 计算源项
    S = source_term(X, Y, t)
    # 进度打印
    print(f"[{n}/{nt}] t={t:.3e}s – 1. 计算声源项完成", file=sys.stderr)

    # 2. 更新声压 p
    div_uv = divergence(u, v)
    dp = -rho0 * c0**2 * div_uv + S
    p += dt * dp
    print(f"[{n}/{nt}] – 2. 声压 p 更新完成", file=sys.stderr)

    # 3. 更新速度 u,v
    dp_dx, dp_dy = gradient(p)
    u += dt * (-dp_dx/rho0 + nu * laplacian(u))
    v += dt * (-dp_dy/rho0 + nu * laplacian(v))
    print(f"[{n}/{nt}] – 3. 声速 u,v 更新完成", file=sys.stderr)

    # 4. 边界条件：掩蔽外部 & 壁面反射
    u *= mask
    v *= mask
    p *= mask
    print(f"[{n}/{nt}] – 4. 边界条件应用完成", file=sys.stderr)

    # 5. 周期性输出 VTK
    if n % 100000 == 0:
        idx = n // 100000
        fname = f"{output_dir}/acoustic_{idx:04d}.vtk"
        with open(fname, 'w') as f:
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Acoustic field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"ORIGIN {x[0]} {y[0]} 0\n")
            f.write(f"SPACING {dr} {dr} 1\n")
            f.write(f"POINT_DATA {nx*ny}\n")
            f.write("SCALARS pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in p.flatten():
                f.write(f"{val}\n")
        print(f"[{n}/{nt}] – 5. 数据已保存到 {fname}", file=sys.stderr)

end = time.time()
print(f"Simulation complete in {end-start:.1f}s", file=sys.stderr)
