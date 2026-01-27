import numpy as np
import os
import sys
import time

# 物理参数
rho0 = 1.225        # 空气密度 [kg/m³]
c0   = 343.0        # 声速 [m/s]
nu   = 1.5e-5       # 运动黏度 [m²/s]
# 注意：nu用于速度方程中的黏性项 nu*laplacian(u)
# 对于线性声学方程，这可以视为：
# 1) 人工数值耗散（用于稳定性），或
# 2) 声学吸收（但通常声学吸收系数不同）
# 当前实现主要用于数值稳定性，实际物理耗散很小

# 几何与网格
R      = 0.01         # 外半径 [m] (与mesh.py保持一致)
r_core = 0.002        # 内半径 [m] (与mesh.py保持一致，避免中心奇点)
dr     = 0.0002       # 网格间距 [m]
x = np.arange(-R, R+dr, dr)
y = np.arange(-R, R+dr, dr)
nx, ny = len(x), len(y)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
# 环形区域掩蔽：内边界 <= r <= 外边界
mask = (r >= r_core) & (r <= R)

# 边界识别：内壁和外壁边界点
# 使用容差判断边界点（考虑网格间距）
tolerance = 1.5 * dr  # 边界容差
inner_wall = mask & (np.abs(r - r_core) < tolerance)
outer_wall = mask & (np.abs(r - R) < tolerance)
wall_mask = inner_wall | outer_wall

# 计算单位法向量（径向方向）
# 对于外壁：法向量向外 (X/r, Y/r)，指向外部
# 对于内壁：法向量向内 (-X/r, -Y/r)，指向圆心
# 为了避免除零，使用r > 0的条件
r_safe = np.where(r > 1e-10, r, 1e-10)  # 避免除零
n_x = X / r_safe  # 径向单位向量x分量（默认向外）
n_y = Y / r_safe  # 径向单位向量y分量（默认向外）
# 内壁法向量需要反向（指向圆心）
n_x[inner_wall] = -n_x[inner_wall]
n_y[inner_wall] = -n_y[inner_wall]
# 确保法向量归一化（虽然理论上已经是单位向量，但数值误差可能导致不归一）
n_mag = np.sqrt(n_x**2 + n_y**2)
n_x = np.where(n_mag > 1e-10, n_x / n_mag, 0)
n_y = np.where(n_mag > 1e-10, n_y / n_mag, 0)

# 时间参数
dt    = 1e-9        # 时间步长 [s]
t_end = 1e-3        # 结束时间 [s]
nt    = int(t_end / dt)

# CFL稳定性条件检查
# CFL = c0 * dt / dr，对于2D问题，CFL应该 < 1/sqrt(2) ≈ 0.707
CFL = c0 * dt / dr
CFL_max = 0.5  # 保守的CFL限制
if CFL > CFL_max:
    print(f"警告: CFL数 = {CFL:.4f} > {CFL_max:.2f}，可能导致数值不稳定！", file=sys.stderr)
    print(f"建议: 减小时间步长 dt < {CFL_max * dr / c0:.2e} s", file=sys.stderr)
else:
    print(f"CFL数 = {CFL:.4f} < {CFL_max:.2f}，稳定性条件满足", file=sys.stderr)

# 场变量初始化
p = np.zeros((ny, nx))
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

# 声源参数
wave_pressure = 10.0      # 爆轰波压强幅度 [Pa]
v_rde = 1400.0            # 爆轰波传播速度 [m/s]
inner_r  = r_core         # 环带内半径 [m] (与几何保持一致)
outer_r  = R              # 环带外半径 [m] (与几何保持一致)
omega    = v_rde / R      # 角速度 [rad/s] = v_rde / R
sigma    = 0.001          # 衰减尺度 [m]
# 特征时间：爆轰波传播一圈的时间
tau = 2 * np.pi * R / v_rde  # [s]，用于源项单位转换

# 输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def source_term(X, Y, t, valid_mask):
    """
    计算声源项
    
    物理意义：质量源项或能量源项，单位应该是 [Pa/s] 或 [kg/(m³·s)]
    对于声学方程 dp/dt = -rho0*c0^2*div(u) + S, S的单位应该是 [Pa/s]
    
    参数:
        X, Y: 网格坐标
        t: 时间
        valid_mask: 有效区域掩码（只在有效区域内计算）
    """
    # 只在有效区域内计算，提高效率
    r = np.sqrt(X**2 + Y**2)
    valid_ring = valid_mask & (r >= inner_r) & (r <= outer_r)
    
    if not np.any(valid_ring):
        return np.zeros_like(X)
    
    theta = np.arctan2(Y, X) % (2*np.pi)
    theta0 = (omega * t) % (2*np.pi)
    dtheta = (theta - theta0 + np.pi) % (2*np.pi) - np.pi
    
    # 源项在环形区域内，且只在波前方向
    mask_ring = valid_ring & (dtheta >= 0)
    
    S = np.zeros_like(X)
    # 源项物理模型：
    # 源项S的单位必须是[Pa/s]（压力变化率）
    # wave_pressure是压力幅度[Pa]，需要除以特征时间转换为压力变化率
    # 使用特征时间tau = 2*pi*R/v_rde（爆轰波传播一圈的时间）
    # 这样S = wave_pressure / tau 的单位是[Pa/s]，符合物理要求
    # 源项在波前方向（dtheta >= 0）使用高斯衰减模型
    S[mask_ring] = (wave_pressure / tau * 
                    np.exp(- (dtheta[mask_ring]*R)**2/(2*sigma**2)))
    return S


# 差分算子（修正边界处理，避免周期性边界）
def laplacian(f):
    """
    计算拉普拉斯算子，在边界处使用单侧差分
    """
    # 初始化结果
    laplacian_f = np.zeros_like(f)
    
    # 内部点使用中心差分
    # x方向二阶导数
    f_xx = np.zeros_like(f)
    f_yy = np.zeros_like(f)
    
    # 对于有效区域内的点，使用中心差分
    # 检查邻居是否在有效区域内
    valid = mask.copy()
    
    # x方向二阶导数
    f_left = np.roll(f, 1, axis=1)
    f_right = np.roll(f, -1, axis=1)
    # 只在邻居也在有效区域内时使用中心差分
    valid_left = np.roll(valid, 1, axis=1) & valid
    valid_right = np.roll(valid, -1, axis=1) & valid
    
    # 中心差分（当两侧都有有效点时）
    center_valid = valid_left & valid_right
    f_xx[center_valid] = (f_right[center_valid] - 2*f[center_valid] + f_left[center_valid]) / dr**2
    
    # 单侧差分（边界处）
    # 右边界：使用后向差分
    right_boundary = valid & ~valid_right & valid_left
    f_xx[right_boundary] = (f[right_boundary] - 2*f_left[right_boundary] + np.roll(f_left, 1, axis=1)[right_boundary]) / dr**2
    
    # 左边界：使用前向差分
    left_boundary = valid & ~valid_left & valid_right
    f_xx[left_boundary] = (f_right[left_boundary] - 2*f[left_boundary] + np.roll(f_right, -1, axis=1)[left_boundary]) / dr**2
    
    # y方向二阶导数（类似处理）
    f_up = np.roll(f, 1, axis=0)
    f_down = np.roll(f, -1, axis=0)
    valid_up = np.roll(valid, 1, axis=0) & valid
    valid_down = np.roll(valid, -1, axis=0) & valid
    
    center_valid_y = valid_up & valid_down
    f_yy[center_valid_y] = (f_down[center_valid_y] - 2*f[center_valid_y] + f_up[center_valid_y]) / dr**2
    
    right_boundary_y = valid & ~valid_down & valid_up
    f_yy[right_boundary_y] = (f[right_boundary_y] - 2*f_up[right_boundary_y] + np.roll(f_up, 1, axis=0)[right_boundary_y]) / dr**2
    
    left_boundary_y = valid & ~valid_up & valid_down
    f_yy[left_boundary_y] = (f_down[left_boundary_y] - 2*f[left_boundary_y] + np.roll(f_down, -1, axis=0)[left_boundary_y]) / dr**2
    
    laplacian_f = f_xx + f_yy
    laplacian_f[~valid] = 0  # 无效区域置零
    return laplacian_f

def divergence(u, v):
    """
    计算速度散度，在边界处使用单侧差分
    """
    # 初始化
    div = np.zeros_like(u)
    valid = mask.copy()
    
    # du/dx
    u_left = np.roll(u, 1, axis=1)
    u_right = np.roll(u, -1, axis=1)
    valid_left = np.roll(valid, 1, axis=1) & valid
    valid_right = np.roll(valid, -1, axis=1) & valid
    
    du_dx = np.zeros_like(u)
    # 中心差分
    center_valid = valid_left & valid_right
    du_dx[center_valid] = (u_right[center_valid] - u_left[center_valid]) / (2*dr)
    # 右边界：后向差分
    right_boundary = valid & ~valid_right & valid_left
    du_dx[right_boundary] = (u[right_boundary] - u_left[right_boundary]) / dr
    # 左边界：前向差分
    left_boundary = valid & ~valid_left & valid_right
    du_dx[left_boundary] = (u_right[left_boundary] - u[left_boundary]) / dr
    
    # dv/dy
    v_up = np.roll(v, 1, axis=0)
    v_down = np.roll(v, -1, axis=0)
    valid_up = np.roll(valid, 1, axis=0) & valid
    valid_down = np.roll(valid, -1, axis=0) & valid
    
    dv_dy = np.zeros_like(v)
    center_valid_y = valid_up & valid_down
    dv_dy[center_valid_y] = (v_down[center_valid_y] - v_up[center_valid_y]) / (2*dr)
    right_boundary_y = valid & ~valid_down & valid_up
    dv_dy[right_boundary_y] = (v[right_boundary_y] - v_up[right_boundary_y]) / dr
    left_boundary_y = valid & ~valid_up & valid_down
    dv_dy[left_boundary_y] = (v_down[left_boundary_y] - v[left_boundary_y]) / dr
    
    div = du_dx + dv_dy
    div[~valid] = 0
    return div

def gradient(p):
    """
    计算压力梯度，在边界处使用单侧差分
    """
    valid = mask.copy()
    
    # dp/dx
    p_left = np.roll(p, 1, axis=1)
    p_right = np.roll(p, -1, axis=1)
    valid_left = np.roll(valid, 1, axis=1) & valid
    valid_right = np.roll(valid, -1, axis=1) & valid
    
    dp_dx = np.zeros_like(p)
    center_valid = valid_left & valid_right
    dp_dx[center_valid] = (p_right[center_valid] - p_left[center_valid]) / (2*dr)
    right_boundary = valid & ~valid_right & valid_left
    dp_dx[right_boundary] = (p[right_boundary] - p_left[right_boundary]) / dr
    left_boundary = valid & ~valid_left & valid_right
    dp_dx[left_boundary] = (p_right[left_boundary] - p[left_boundary]) / dr
    
    # dp/dy
    p_up = np.roll(p, 1, axis=0)
    p_down = np.roll(p, -1, axis=0)
    valid_up = np.roll(valid, 1, axis=0) & valid
    valid_down = np.roll(valid, -1, axis=0) & valid
    
    dp_dy = np.zeros_like(p)
    center_valid_y = valid_up & valid_down
    dp_dy[center_valid_y] = (p_down[center_valid_y] - p_up[center_valid_y]) / (2*dr)
    right_boundary_y = valid & ~valid_down & valid_up
    dp_dy[right_boundary_y] = (p[right_boundary_y] - p_up[right_boundary_y]) / dr
    left_boundary_y = valid & ~valid_up & valid_down
    dp_dy[left_boundary_y] = (p_down[left_boundary_y] - p[left_boundary_y]) / dr
    
    dp_dx[~valid] = 0
    dp_dy[~valid] = 0
    return dp_dx, dp_dy


def apply_wall_boundary_conditions(u, v, p, wall_mask, n_x, n_y):
    """
    应用硬壁边界条件：壁面处法向速度为零
    
    物理原理：
    - 硬壁边界条件要求：u·n = 0（法向速度为零）
    - 对于压力，在硬壁处法向压力梯度应该为零（或使用镜像点方法）
    
    参数:
        u, v: 速度分量
        p: 压力
        wall_mask: 壁面点掩码
        n_x, n_y: 法向量分量
    """
    u_new = u.copy()
    v_new = v.copy()
    p_new = p.copy()
    
    # 计算法向速度
    u_normal = u * n_x + v * n_y
    
    # 在壁面处，法向速度必须为零
    # 修正速度：从总速度中减去法向分量
    u_new[wall_mask] = u[wall_mask] - u_normal[wall_mask] * n_x[wall_mask]
    v_new[wall_mask] = v[wall_mask] - u_normal[wall_mask] * n_y[wall_mask]
    
    # 对于压力，使用镜像点方法：壁面处的压力应该等于壁面内最近点的压力
    # 或者使用法向压力梯度为零的条件
    # 这里使用简单的处理：保持压力连续（不强制压力梯度为零，让数值方法自然处理）
    # 如果需要更精确，可以使用镜像点方法
    
    return u_new, v_new, p_new


def apply_numerical_filter(f, filter_strength=0.01):
    """
    应用数值滤波，抑制高频振荡
    
    使用简单的空间平均滤波，只在有效区域内应用
    
    参数:
        f: 场变量
        filter_strength: 滤波强度 (0-1)
    """
    if filter_strength <= 0:
        return f
    
    f_filtered = f.copy()
    valid = mask.copy()
    
    # 简单的空间平均滤波（只在有效区域内）
    # 对每个有效点，与其有效邻居进行加权平均
    f_avg = np.zeros_like(f)
    count = np.zeros_like(f, dtype=int)
    
    # 检查四个方向的邻居
    neighbors = [
        (np.roll(f, 1, axis=0), np.roll(valid, 1, axis=0)),  # 上
        (np.roll(f, -1, axis=0), np.roll(valid, -1, axis=0)),  # 下
        (np.roll(f, 1, axis=1), np.roll(valid, 1, axis=1)),  # 左
        (np.roll(f, -1, axis=1), np.roll(valid, -1, axis=1)),  # 右
    ]
    
    f_avg[valid] = f[valid]
    count[valid] = 1
    
    for f_neighbor, valid_neighbor in neighbors:
        neighbor_valid = valid & valid_neighbor
        f_avg[neighbor_valid] += f_neighbor[neighbor_valid]
        count[neighbor_valid] += 1
    
    # 计算平均值
    avg_mask = count > 0
    f_avg[avg_mask] /= count[avg_mask]
    
    # 应用滤波：f_new = (1-alpha)*f + alpha*f_avg
    f_filtered[valid] = (1 - filter_strength) * f[valid] + filter_strength * f_avg[valid]
    
    return f_filtered


def rk4_step(p, u, v, t, dt, mask, wall_mask, n_x, n_y):
    """
    使用四阶Runge-Kutta方法进行时间积分
    
    参数:
        p, u, v: 当前状态
        t: 当前时间
        dt: 时间步长
        mask: 有效区域掩码
        wall_mask: 壁面掩码
        n_x, n_y: 法向量分量
    
    返回:
        p_new, u_new, v_new: 新状态
    """
    def rhs(p_state, u_state, v_state, t_state):
        """计算右端项"""
        # 计算源项
        S = source_term(X, Y, t_state, mask)
        
        # 计算散度和梯度
        div_uv = divergence(u_state, v_state)
        dp_dx, dp_dy = gradient(p_state)
        
        # 压力方程: dp/dt = -rho0*c0^2*div(u) + S
        dp_dt = -rho0 * c0**2 * div_uv + S
        
        # 速度方程: du/dt = -grad(p)/rho0 + nu*laplacian(u)
        # 注意：nu*laplacian(u)项主要用于数值稳定性（人工耗散）
        # 对于线性声学，通常不考虑黏性，但这里保留用于抑制数值振荡
        # 实际物理耗散很小（nu很小），主要起数值稳定作用
        du_dt = -dp_dx / rho0 + nu * laplacian(u_state)
        dv_dt = -dp_dy / rho0 + nu * laplacian(v_state)
        
        return dp_dt, du_dt, dv_dt
    
    # RK4的四个阶段
    k1_p, k1_u, k1_v = rhs(p, u, v, t)
    k2_p, k2_u, k2_v = rhs(p + 0.5*dt*k1_p, u + 0.5*dt*k1_u, v + 0.5*dt*k1_v, t + 0.5*dt)
    k3_p, k3_u, k3_v = rhs(p + 0.5*dt*k2_p, u + 0.5*dt*k2_u, v + 0.5*dt*k2_v, t + 0.5*dt)
    k4_p, k4_u, k4_v = rhs(p + dt*k3_p, u + dt*k3_u, v + dt*k3_v, t + dt)
    
    # 更新
    p_new = p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    u_new = u + (dt/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
    v_new = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    # 应用边界条件
    u_new *= mask
    v_new *= mask
    p_new *= mask
    u_new, v_new, p_new = apply_wall_boundary_conditions(u_new, v_new, p_new, wall_mask, n_x, n_y)
    u_new[~mask] = 0
    v_new[~mask] = 0
    p_new[~mask] = 0
    
    return p_new, u_new, v_new


def write_vtk_output(fname, p, u, v, mask, X, Y, x, y, dr, nx, ny):
    """
    输出VTK格式文件，包含压力场和速度场
    
    参数:
        fname: 输出文件名
        p, u, v: 压力场和速度分量
        mask: 有效区域掩码
        X, Y: 网格坐标
        x, y: 坐标数组
        dr: 网格间距
        nx, ny: 网格尺寸
    """
    with open(fname, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("RDE Acoustic Field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"ORIGIN {x[0]} {y[0]} 0\n")
        f.write(f"SPACING {dr} {dr} 1\n")
        f.write(f"POINT_DATA {nx*ny}\n")
        
        # 输出有效区域掩码（用于可视化时过滤）
        f.write("SCALARS valid_mask int 1\n")
        f.write("LOOKUP_TABLE default\n")
        # 使用numpy的savetxt提高效率
        np.savetxt(f, mask.flatten().astype(int), fmt='%d')
        
        # 输出压力场（标量）
        f.write("SCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        # 在无效区域输出NaN，便于可视化时识别
        p_output = p.copy()
        p_output[~mask] = float('nan')
        # 使用numpy的savetxt，但需要处理NaN
        p_flat = p_output.flatten()
        # 将NaN替换为字符串"nan"，然后写入
        p_str = np.where(np.isnan(p_flat), 'nan', p_flat)
        for val in p_str:
            if isinstance(val, str):
                f.write("nan\n")
            else:
                f.write(f"{val:.6e}\n")
        
        # 输出速度场（向量）
        f.write("VECTORS velocity float\n")
        # 在无效区域输出(0,0,0)
        u_output = u.copy()
        v_output = v.copy()
        u_output[~mask] = 0.0
        v_output[~mask] = 0.0
        # 使用更高效的写入方式
        for i in range(ny):
            for j in range(nx):
                f.write(f"{u_output[i,j]:.6e} {v_output[i,j]:.6e} 0.0\n")
        
        # 输出速度大小（标量，便于可视化）
        velocity_magnitude = np.sqrt(u**2 + v**2)
        velocity_magnitude[~mask] = float('nan')
        f.write("SCALARS velocity_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        v_mag_flat = velocity_magnitude.flatten()
        v_mag_str = np.where(np.isnan(v_mag_flat), 'nan', v_mag_flat)
        for val in v_mag_str:
            if isinstance(val, str):
                f.write("nan\n")
            else:
                f.write(f"{val:.6e}\n")


# 数值滤波参数
filter_strength = 0.01  # 滤波强度（0-1），0表示不滤波
apply_filter = True  # 是否应用数值滤波

# 输出和打印频率
output_freq = 100000  # 每N步输出一次
print_freq = max(1, nt // 100)  # 打印进度频率（总共打印约100次）

# 主循环
start = time.time()
print(f"开始模拟: nt={nt}, dt={dt:.2e}s, CFL={CFL:.4f}", file=sys.stderr)
print(f"输出频率: 每{output_freq}步输出一次", file=sys.stderr)
print(f"打印频率: 每{print_freq}步打印一次", file=sys.stderr)

for n in range(nt):
    t = n * dt

    # 使用RK4方法进行时间积分
    p, u, v = rk4_step(p, u, v, t, dt, mask, wall_mask, n_x, n_y)
    
    # 应用数值滤波（抑制高频振荡）
    if apply_filter and n % 10 == 0:  # 每10步滤波一次，减少计算开销
        p = apply_numerical_filter(p, filter_strength)
        u = apply_numerical_filter(u, filter_strength)
        v = apply_numerical_filter(v, filter_strength)

    # 进度打印（降低频率）
    if n % print_freq == 0 or n == nt - 1:
        progress = 100.0 * n / nt
        elapsed = time.time() - start
        if n > 0:
            eta = elapsed / n * (nt - n)
            print(f"[{n:8d}/{nt}] ({progress:5.1f}%) t={t:.3e}s, 已用时={elapsed:.1f}s, 预计剩余={eta:.1f}s", 
                  file=sys.stderr)

    # 周期性输出 VTK
    if n % output_freq == 0 or n == nt - 1:
        idx = n // output_freq
        fname = f"{output_dir}/acoustic_{idx:04d}.vtk"
        write_vtk_output(fname, p, u, v, mask, X, Y, x, y, dr, nx, ny)
        
        # 计算并输出统计信息
        p_valid = p[mask]
        u_valid = u[mask]
        v_valid = v[mask]
        v_mag = np.sqrt(u_valid**2 + v_valid**2)
        
        print(f"[{n}/{nt}] 数据已保存到 {fname}", file=sys.stderr)
        print(f"  压力统计: min={np.min(p_valid):.3e}, max={np.max(p_valid):.3e}, "
              f"mean={np.mean(p_valid):.3e} Pa", file=sys.stderr)
        print(f"  速度统计: |v|_min={np.min(v_mag):.3e}, |v|_max={np.max(v_mag):.3e}, "
              f"|v|_mean={np.mean(v_mag):.3e} m/s", file=sys.stderr)

end = time.time()
print(f"Simulation complete in {end-start:.1f}s", file=sys.stderr)
