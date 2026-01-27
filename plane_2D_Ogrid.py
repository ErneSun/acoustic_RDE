#!/usr/bin/env python3
"""
RDE声学模拟 - 使用非结构化O网格版本
基于plane_2D.py，但使用mesh.py生成的非结构化环形网格
支持并行计算
"""

import numpy as np
import os
import sys
import time
import argparse
import multiprocessing as mp
from functools import partial
import meshio

# 导入mesh.py的网格生成函数（或者直接复制相关代码）
# 为了简化，我们直接在这里实现网格生成，与mesh.py保持一致

# ==================== 网格生成（从mesh.py） ====================
def generate_ogrid_mesh(R=0.01, r_core=0.002, dr=0.0002):
    """
    生成非结构化O网格（环形网格）
    
    参数:
        R: 外半径 [m]
        r_core: 内半径 [m]
        dr: 径向网格间距 [m]
    
    返回:
        nodes: 节点坐标 [N, 3]
        cells: 单元连接 [M, 4] (四边形)
        cell_centers: 单元中心坐标 [M, 2]
        cell_areas: 单元面积 [M]
        cell_neighbors: 单元邻居关系
    """
    dtheta = 0.0002 / R  # 角度间距
    
    # 生成r数组
    r = np.arange(r_core, R + dr/2, dr)
    r = r[r <= R]
    if len(r) == 0 or (len(r) > 0 and r[-1] < R - 1e-10):
        r = np.append(r, R)
    nr = len(r)
    
    # 生成theta数组（确保闭合）
    ntheta_approx = int(2 * np.pi / dtheta)
    if ntheta_approx < 3:
        ntheta_approx = 3
    dtheta_actual = 2 * np.pi / ntheta_approx
    theta = np.linspace(0, 2*np.pi, ntheta_approx, endpoint=False)
    ntheta = len(theta)
    
    # 生成节点坐标
    nodes = []
    for i in range(nr):
        for j in range(ntheta):
            nodes.append([r[i]*np.cos(theta[j]), r[i]*np.sin(theta[j]), 0.0])
    nodes = np.array(nodes)
    
    # 生成单元连接
    cells = []
    for i in range(nr-1):
        for j in range(ntheta):
            n0 = i*ntheta + j
            n1 = i*ntheta + (j+1) % ntheta
            n2 = (i+1)*ntheta + (j+1) % ntheta
            n3 = (i+1)*ntheta + j
            cells.append([n0, n1, n2, n3])
    cells = np.array(cells)
    
    # 计算单元中心、面积和邻居关系
    n_cells = len(cells)
    cell_centers = np.zeros((n_cells, 2))
    cell_areas = np.zeros(n_cells)
    
    for idx, cell in enumerate(cells):
        # 获取四个节点
        p0 = nodes[cell[0], :2]
        p1 = nodes[cell[1], :2]
        p2 = nodes[cell[2], :2]
        p3 = nodes[cell[3], :2]
        
        # 计算单元中心（四个节点的平均值）
        cell_centers[idx] = (p0 + p1 + p2 + p3) / 4.0
        
        # 计算单元面积（分成两个三角形）
        v1 = p1 - p0
        v2 = p3 - p0
        area1 = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        
        v3 = p2 - p1
        v4 = p3 - p1
        area2 = 0.5 * abs(v3[0]*v4[1] - v3[1]*v4[0])
        
        cell_areas[idx] = area1 + area2
    
    # 构建单元邻居关系（共享边的单元，包括周期性边界）
    cell_neighbors = [[] for _ in range(n_cells)]
    
    # 方法1：直接邻居（共享边的单元）
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            # 检查是否共享边
            shared_nodes = set(cells[i]) & set(cells[j])
            if len(shared_nodes) >= 2:  # 共享一条边
                cell_neighbors[i].append(j)
                cell_neighbors[j].append(i)
    
    # 方法2：处理周期性边界（角度方向的周期性）
    # 对于每个径向层，第一个单元（j=0）和最后一个单元（j=ntheta-1）通过周期性连接
    for i in range(nr-1):
        # 找到j=0和j=ntheta-1的单元索引
        cell_idx_0 = i * ntheta + 0
        cell_idx_last = i * ntheta + (ntheta - 1)
        
        # 检查它们是否已经通过直接邻居连接
        # 如果没有，添加周期性邻居关系
        # 注意：由于网格是闭合的，它们应该已经通过节点连接了
        # 但为了确保，我们显式检查周期性边界
        
        # 检查最后一个单元的右边界是否连接到第一个单元的左边界
        # 这通过节点索引的周期性已经处理了（(j+1) % ntheta）
        # 所以邻居关系应该已经正确建立
    
    return nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta


# ==================== 物理参数 ====================
rho0 = 1.225        # 空气密度 [kg/m³]
c0   = 343.0        # 声速 [m/s]
nu   = 1.5e-5       # 运动黏度 [m²/s]
# 注意：nu用于速度方程中的黏性项 nu*laplacian(u)
# 对于线性声学方程，这可以视为：
# 1) 人工数值耗散（用于稳定性），或
# 2) 声学吸收（但通常声学吸收系数不同）
# 当前实现主要用于数值稳定性，实际物理耗散很小

# ==================== 几何参数 ====================
R      = 0.01         # 外半径 [m]
r_core = 0.002        # 内半径 [m]
dr     = 0.0002       # 径向网格间距 [m]

# ==================== 时间参数 ====================
dt    = 1e-9        # 时间步长 [s]
t_end = 1e-3        # 结束时间 [s]

# ==================== 声源参数 ====================
wave_pressure = 10.0      # 爆轰波压强幅度 [Pa]
v_rde = 1400.0            # 爆轰波传播速度 [m/s]
inner_r  = r_core         # 环带内半径 [m]
outer_r  = R              # 环带外半径 [m]
omega    = v_rde / R      # 角速度 [rad/s]
sigma    = 0.001          # 衰减尺度 [m]
tau = 2 * np.pi * R / v_rde  # 特征时间 [s]


# ==================== 有限体积方法算子 ====================
def compute_gradient_fvm(f, nodes, cells, cell_centers, cell_areas, cell_neighbors, 
                         boundary_edges=None, use_parallel=False, n_cores=1):
    """
    使用有限体积方法计算梯度（支持并行）
    
    参数:
        f: 场变量（在单元中心）[n_cells]
        nodes: 节点坐标 [n_nodes, 3]
        cells: 单元连接 [n_cells, 4]
        cell_centers: 单元中心 [n_cells, 2]
        cell_areas: 单元面积 [n_cells]
        cell_neighbors: 单元邻居关系
        use_parallel: 是否使用并行计算
        n_cores: 并行核数
    
    返回:
        grad_f: 梯度 [n_cells, 2]
    """
    n_cells = len(f)
    grad_f = np.zeros((n_cells, 2))
    
    if use_parallel and n_cores > 1:
        # 并行计算
        with mp.Pool(processes=n_cores) as pool:
            args = [(i, f, nodes, cells, cell_centers, cell_areas, cell_neighbors, boundary_edges) 
                    for i in range(n_cells)]
            results = pool.map(compute_gradient_cell_parallel, args)
            for i, grad in results:
                grad_f[i] = grad
    else:
        # 串行计算
        for i in range(n_cells):
            # 使用Green-Gauss方法计算梯度
            grad_sum = np.zeros(2)
            cell_nodes = cells[i]
            n_nodes = len(cell_nodes)
            
            for j in range(n_nodes):
                node0 = nodes[cell_nodes[j], :2]
                node1 = nodes[cell_nodes[(j+1) % n_nodes], :2]
                edge_center = (node0 + node1) / 2.0
                edge_vec = node1 - node0
                edge_length = np.linalg.norm(edge_vec)
                if edge_length > 1e-10:
                    n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
                    center_to_edge = edge_center - cell_centers[i]
                    if np.dot(n, center_to_edge) < 0:
                        n = -n
                    # 检查是否是边界边
                    is_boundary_edge = (boundary_edges is not None and (i, j) in boundary_edges)
                    
                    if is_boundary_edge:
                        # 边界边：使用边界条件
                        # 对于硬壁边界，法向梯度为零，这里使用单元中心值
                        f_face = f[i]
                    else:
                        # 内部边：使用相邻单元的平均值
                        f_face = f[i]
                        for neighbor in cell_neighbors[i]:
                            neighbor_nodes = set(cells[neighbor])
                            if cell_nodes[j] in neighbor_nodes and cell_nodes[(j+1) % n_nodes] in neighbor_nodes:
                                f_face = 0.5 * (f[i] + f[neighbor])
                                break
                    
                    grad_sum += f_face * n * edge_length
            
            if cell_areas[i] > 1e-10:
                grad_f[i] = grad_sum / cell_areas[i]
    
    return grad_f


def compute_divergence_fvm(u, v, nodes, cells, cell_centers, cell_areas, cell_neighbors,
                           boundary_edges=None, use_parallel=False, n_cores=1):
    """
    使用有限体积方法计算散度（支持并行和边界处理）
    
    参数:
        u, v: 速度分量（在单元中心）[n_cells]
        nodes: 节点坐标 [n_nodes, 3]
        cells: 单元连接 [n_cells, 4]
        cell_centers: 单元中心 [n_cells, 2]
        cell_areas: 单元面积 [n_cells]
        cell_neighbors: 单元邻居关系
        boundary_edges: 边界边信息（用于边界处理）
        use_parallel: 是否使用并行计算
        n_cores: 并行核数
    
    返回:
        div: 散度 [n_cells]
    
    物理原理：
        使用散度定理：div(u) = (1/V) * sum(u_face · n_face * A_face)
        对于边界边，硬壁边界条件要求法向速度为零
    """
    n_cells = len(u)
    div = np.zeros(n_cells)
    
    for i in range(n_cells):
        # 使用散度定理：div(u) = (1/V) * sum(u_face · n_face * A_face)
        div_sum = 0.0
        
        cell_nodes = cells[i]
        n_nodes = len(cell_nodes)
        
        for j in range(n_nodes):
            node0 = nodes[cell_nodes[j], :2]
            node1 = nodes[cell_nodes[(j+1) % n_nodes], :2]
            
            edge_vec = node1 - node0
            edge_length = np.linalg.norm(edge_vec)
            if edge_length > 1e-10:
                # 法向量
                n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
                
                # 检查方向
                edge_center = (node0 + node1) / 2.0
                center_to_edge = edge_center - cell_centers[i]
                if np.dot(n, center_to_edge) < 0:
                    n = -n
                
                # 检查是否是边界边
                is_boundary_edge = (boundary_edges is not None and (i, j) in boundary_edges)
                
                if is_boundary_edge:
                    # 边界边：硬壁边界条件，法向速度为零
                    # 所以 u_face · n = 0，对散度的贡献为零
                    # 但为了数值稳定性，使用单元中心值（虽然理论上应该为零）
                    u_face = u[i]
                    v_face = v[i]
                    # 实际上，由于法向速度为零，u_face·n应该为零
                    # 但这里我们仍然计算，因为边界条件会在后续步骤中应用
                else:
                    # 内部边：使用相邻单元的平均值
                    u_face = u[i]
                    v_face = v[i]
                    for neighbor in cell_neighbors[i]:
                        neighbor_nodes = set(cells[neighbor])
                        if cell_nodes[j] in neighbor_nodes and cell_nodes[(j+1) % n_nodes] in neighbor_nodes:
                            u_face = 0.5 * (u[i] + u[neighbor])
                            v_face = 0.5 * (v[i] + v[neighbor])
                            break
                
                div_sum += (u_face * n[0] + v_face * n[1]) * edge_length
        
        if cell_areas[i] > 1e-10:
            div[i] = div_sum / cell_areas[i]
    
    return div


def compute_laplacian_fvm(f, nodes, cells, cell_centers, cell_areas, cell_neighbors,
                          boundary_edges=None, use_parallel=False, n_cores=1):
    """
    使用有限体积方法计算拉普拉斯算子（支持并行）
    
    参数:
        f: 场变量（在单元中心）[n_cells]
        boundary_edges: 边界边信息
        use_parallel: 是否使用并行计算
        n_cores: 并行核数
    
    返回:
        laplacian_f: 拉普拉斯 [n_cells]
    """
    # 拉普拉斯 = div(grad(f))
    grad_f = compute_gradient_fvm(f, nodes, cells, cell_centers, cell_areas, cell_neighbors,
                                  boundary_edges, use_parallel, n_cores)
    
    # 计算grad_f的散度
    laplacian_f = np.zeros(len(f))
    
    for i in range(len(f)):
        div_sum = 0.0
        cell_nodes = cells[i]
        n_nodes = len(cell_nodes)
        
        for j in range(n_nodes):
            node0 = nodes[cell_nodes[j], :2]
            node1 = nodes[cell_nodes[(j+1) % n_nodes], :2]
            
            edge_vec = node1 - node0
            edge_length = np.linalg.norm(edge_vec)
            if edge_length > 1e-10:
                n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
                edge_center = (node0 + node1) / 2.0
                center_to_edge = edge_center - cell_centers[i]
                if np.dot(n, center_to_edge) < 0:
                    n = -n
                
                # 检查是否是边界边
                is_boundary_edge = (boundary_edges is not None and (i, j) in boundary_edges)
                
                if is_boundary_edge:
                    # 边界边：对于硬壁边界，法向梯度为零
                    # 使用单元中心梯度（边界条件会在后续步骤中应用）
                    grad_face = grad_f[i]
                else:
                    # 内部边：使用相邻单元的平均值
                    grad_face = grad_f[i]
                    for neighbor in cell_neighbors[i]:
                        neighbor_nodes = set(cells[neighbor])
                        if cell_nodes[j] in neighbor_nodes and cell_nodes[(j+1) % n_nodes] in neighbor_nodes:
                            grad_face = 0.5 * (grad_f[i] + grad_f[neighbor])
                            break
                
                div_sum += (grad_face[0] * n[0] + grad_face[1] * n[1]) * edge_length
        
        if cell_areas[i] > 1e-10:
            laplacian_f[i] = div_sum / cell_areas[i]
    
    return laplacian_f


# ==================== 源项 ====================
def source_term_ogrid(cell_centers, t, inner_r, outer_r, omega, wave_pressure, tau, sigma, R):
    """
    计算源项（在单元中心）
    
    参数:
        cell_centers: 单元中心坐标 [n_cells, 2]
        t: 时间
        其他参数同plane_2D.py
    
    返回:
        S: 源项 [n_cells]
    """
    n_cells = len(cell_centers)
    S = np.zeros(n_cells)
    
    for i in range(n_cells):
        x, y = cell_centers[i]
        r = np.sqrt(x**2 + y**2)
        
        if r >= inner_r and r <= outer_r:
            theta = np.arctan2(y, x) % (2*np.pi)
            theta0 = (omega * t) % (2*np.pi)
            dtheta = (theta - theta0 + np.pi) % (2*np.pi) - np.pi
            
            if dtheta >= 0:
                S[i] = (wave_pressure / tau * 
                       np.exp(- (dtheta*R)**2/(2*sigma**2)))
    
    return S


# ==================== 边界条件 ====================
def identify_boundary_cells(nodes, cells, cell_centers, R, r_core, nr, ntheta, tolerance=1.5*0.0002):
    """
    识别边界单元（基于单元是否在边界上，而不仅仅是单元中心距离）
    
    参数:
        nodes: 节点坐标
        cells: 单元连接
        cell_centers: 单元中心坐标
        R, r_core: 内外半径
        nr, ntheta: 径向和角度层数
        tolerance: 容差
    
    返回:
        inner_wall_cells: 内壁单元索引
        outer_wall_cells: 外壁单元索引
        wall_normals: 壁面法向量 [n_wall_cells, 2]（已归一化）
        boundary_edges: 边界边信息（用于有限体积方法）
    """
    n_cells = len(cell_centers)
    inner_wall_cells = []
    outer_wall_cells = []
    wall_normals = []
    boundary_edges = {}  # {(cell_idx, edge_idx): {'type': 'inner'/'outer', 'normal': n}}
    
    for i in range(n_cells):
        cell_nodes = cells[i]
        x_center, y_center = cell_centers[i]
        r_center = np.sqrt(x_center**2 + y_center**2)
        
        # 检查单元的每个节点是否在边界上
        is_inner_boundary = False
        is_outer_boundary = False
        
        for node_idx in cell_nodes:
            x_node, y_node = nodes[node_idx, :2]
            r_node = np.sqrt(x_node**2 + y_node**2)
            
            if abs(r_node - r_core) < tolerance:
                is_inner_boundary = True
            if abs(r_node - R) < tolerance:
                is_outer_boundary = True
        
        # 如果单元有节点在边界上，则认为是边界单元
        if is_inner_boundary:
            inner_wall_cells.append(i)
            # 内壁法向量指向圆心（基于单元中心）
            if r_center > 1e-10:
                n = np.array([-x_center/r_center, -y_center/r_center])
            else:
                n = np.array([0.0, 0.0])
            # 归一化
            n_mag = np.linalg.norm(n)
            if n_mag > 1e-10:
                n = n / n_mag
            wall_normals.append(n)
            
        elif is_outer_boundary:
            outer_wall_cells.append(i)
            # 外壁法向量指向外部
            if r_center > 1e-10:
                n = np.array([x_center/r_center, y_center/r_center])
            else:
                n = np.array([0.0, 0.0])
            # 归一化
            n_mag = np.linalg.norm(n)
            if n_mag > 1e-10:
                n = n / n_mag
            wall_normals.append(n)
    
    # 识别边界边（用于有限体积方法中的边界处理）
    for i in range(n_cells):
        cell_nodes = cells[i]
        n_nodes = len(cell_nodes)
        
        for j in range(n_nodes):
            node0_idx = cell_nodes[j]
            node1_idx = cell_nodes[(j+1) % n_nodes]
            
            x0, y0 = nodes[node0_idx, :2]
            x1, y1 = nodes[node1_idx, :2]
            r0 = np.sqrt(x0**2 + y0**2)
            r1 = np.sqrt(x1**2 + y1**2)
            
            # 检查边是否在边界上
            edge_on_inner = (abs(r0 - r_core) < tolerance) and (abs(r1 - r_core) < tolerance)
            edge_on_outer = (abs(r0 - R) < tolerance) and (abs(r1 - R) < tolerance)
            
            if edge_on_inner or edge_on_outer:
                # 计算边的法向量
                edge_vec = np.array([x1-x0, y1-y0])
                edge_length = np.linalg.norm(edge_vec)
                if edge_length > 1e-10:
                    # 法向量（逆时针旋转90度）
                    n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
                    # 确保指向单元外部
                    edge_center = np.array([(x0+x1)/2, (y0+y1)/2])
                    center_to_edge = edge_center - cell_centers[i]
                    if np.dot(n, center_to_edge) < 0:
                        n = -n
                    
                    boundary_type = 'inner' if edge_on_inner else 'outer'
                    boundary_edges[(i, j)] = {'type': boundary_type, 'normal': n}
    
    return (inner_wall_cells, outer_wall_cells, 
            np.array(wall_normals) if wall_normals else np.zeros((0, 2)),
            boundary_edges)


def apply_wall_boundary_conditions_ogrid(u, v, inner_wall_cells, outer_wall_cells, wall_normals):
    """
    应用硬壁边界条件：壁面处法向速度为零
    """
    u_new = u.copy()
    v_new = v.copy()
    
    all_wall_cells = inner_wall_cells + outer_wall_cells
    
    for idx, cell_idx in enumerate(all_wall_cells):
        if idx < len(wall_normals):
            n = wall_normals[idx]
            # 计算法向速度
            u_normal = u[cell_idx] * n[0] + v[cell_idx] * n[1]
            # 修正速度：减去法向分量
            u_new[cell_idx] = u[cell_idx] - u_normal * n[0]
            v_new[cell_idx] = v[cell_idx] - u_normal * n[1]
    
    return u_new, v_new


# ==================== RK4时间积分 ====================
def rk4_step_ogrid(p, u, v, t, dt, nodes, cells, cell_centers, cell_areas, 
                   cell_neighbors, inner_wall_cells, outer_wall_cells, wall_normals,
                   boundary_edges=None, use_parallel=False, n_cores=1):
    """
    使用RK4方法进行时间积分（非结构化网格版本，支持并行）
    """
    def rhs(p_state, u_state, v_state, t_state):
        """计算右端项"""
        # 计算源项
        S = source_term_ogrid(cell_centers, t_state, inner_r, outer_r, omega, 
                              wave_pressure, tau, sigma, R)
        
        # 计算散度和梯度（支持并行，传入boundary_edges）
        div_uv = compute_divergence_fvm(u_state, v_state, nodes, cells, 
                                        cell_centers, cell_areas, cell_neighbors,
                                        boundary_edges, use_parallel, n_cores)
        grad_p = compute_gradient_fvm(p_state, nodes, cells, cell_centers, 
                                      cell_areas, cell_neighbors, boundary_edges,
                                      use_parallel, n_cores)
        
        # 压力方程: dp/dt = -rho0*c0^2*div(u) + S
        dp_dt = -rho0 * c0**2 * div_uv + S
        
        # 速度方程: du/dt = -grad(p)/rho0 + nu*laplacian(u)
        # 注意：nu*laplacian(u)项主要用于数值稳定性（人工耗散）
        laplacian_u = compute_laplacian_fvm(u_state, nodes, cells, cell_centers, 
                                            cell_areas, cell_neighbors, boundary_edges,
                                            use_parallel, n_cores)
        laplacian_v = compute_laplacian_fvm(v_state, nodes, cells, cell_centers, 
                                            cell_areas, cell_neighbors, boundary_edges,
                                            use_parallel, n_cores)
        
        du_dt = -grad_p[:, 0] / rho0 + nu * laplacian_u
        dv_dt = -grad_p[:, 1] / rho0 + nu * laplacian_v
        
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
    u_new, v_new = apply_wall_boundary_conditions_ogrid(u_new, v_new, 
                                                        inner_wall_cells, outer_wall_cells, wall_normals)
    
    return p_new, u_new, v_new


# ==================== 并行计算辅助函数 ====================
# 注意：在Windows上，multiprocessing需要函数在模块级别定义
def compute_gradient_cell_parallel(args):
    """并行计算单个单元的梯度（全局函数，用于multiprocessing）"""
    i, f, nodes, cells, cell_centers, cell_areas, cell_neighbors, boundary_edges = args
    grad_sum = np.zeros(2)
    cell_nodes = cells[i]
    n_nodes = len(cell_nodes)
    
    for j in range(n_nodes):
        node0 = nodes[cell_nodes[j], :2]
        node1 = nodes[cell_nodes[(j+1) % n_nodes], :2]
        edge_center = (node0 + node1) / 2.0
        edge_vec = node1 - node0
        edge_length = np.linalg.norm(edge_vec)
        if edge_length > 1e-10:
            n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
            center_to_edge = edge_center - cell_centers[i]
            if np.dot(n, center_to_edge) < 0:
                n = -n
            # 检查是否是边界边
            is_boundary_edge = (boundary_edges is not None and (i, j) in boundary_edges)
            
            if is_boundary_edge:
                f_face = f[i]  # 边界边使用单元中心值
            else:
                f_face = f[i]
                for neighbor in cell_neighbors[i]:
                    neighbor_nodes = set(cells[neighbor])
                    if cell_nodes[j] in neighbor_nodes and cell_nodes[(j+1) % n_nodes] in neighbor_nodes:
                        f_face = 0.5 * (f[i] + f[neighbor])
                        break
            grad_sum += f_face * n * edge_length
    
    if cell_areas[i] > 1e-10:
        return i, grad_sum / cell_areas[i]
    else:
        return i, np.zeros(2)


# ==================== 主程序 ====================
def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='RDE声学模拟 - 非结构化O网格版本（支持并行）')
    parser.add_argument('-n', '--ncores', type=int, default=None,
                       help='并行核数（默认：自动检测，使用最大核数-1）')
    parser.add_argument('--single', action='store_true',
                       help='使用单核运行（覆盖-n参数）')
    
    args = parser.parse_args()
    
    # 确定并行核数
    max_cores = mp.cpu_count()
    if args.single:
        n_cores = 1
        use_parallel = False
    elif args.ncores is not None:
        n_cores = min(args.ncores, max_cores)
        use_parallel = (n_cores > 1)
    else:
        # 默认使用最大核数-1（留一个核心给系统）
        n_cores = max(1, max_cores - 1)
        use_parallel = (n_cores > 1)
    
    print(f"系统最大核数: {max_cores}", file=sys.stderr)
    print(f"使用核数: {n_cores}", file=sys.stderr)
    print(f"并行模式: {'开启' if use_parallel else '关闭'}", file=sys.stderr)
    print(f"参考核数（推荐）: {max(1, max_cores - 1)}", file=sys.stderr)
    
    # 生成网格
    print("正在生成非结构化O网格...", file=sys.stderr)
    nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta = generate_ogrid_mesh(R, r_core, dr)
    n_cells = len(cells)
    n_nodes = len(nodes)
    
    print(f"网格生成完成: {n_nodes}个节点, {n_cells}个单元", file=sys.stderr)
    
    # 识别边界（基于单元节点，更准确）
    inner_wall_cells, outer_wall_cells, wall_normals, boundary_edges = identify_boundary_cells(
        nodes, cells, cell_centers, R, r_core, nr, ntheta)
    print(f"边界识别: {len(inner_wall_cells)}个内壁单元, {len(outer_wall_cells)}个外壁单元", file=sys.stderr)
    print(f"边界边数量: {len(boundary_edges)}", file=sys.stderr)
    
    # CFL检查
    # 对于非结构化网格，使用更准确的特征尺寸
    # 方法1：使用最小单元尺寸（保守估计）
    min_cell_size = np.sqrt(np.min(cell_areas))
    # 方法2：使用平均单元尺寸（更实际）
    mean_cell_size = np.sqrt(np.mean(cell_areas))
    # 使用两者中的较小值，确保稳定性
    characteristic_size = min(min_cell_size, mean_cell_size * 0.8)
    
    CFL = c0 * dt / characteristic_size
    CFL_max = 0.5
    if CFL > CFL_max:
        print(f"警告: CFL数 = {CFL:.4f} > {CFL_max:.2f}，可能导致数值不稳定！", file=sys.stderr)
        print(f"建议: 减小时间步长 dt < {CFL_max * characteristic_size / c0:.2e} s", file=sys.stderr)
    else:
        print(f"CFL数 = {CFL:.4f} < {CFL_max:.2f}，稳定性条件满足", file=sys.stderr)
    print(f"特征尺寸: min={min_cell_size:.6e}m, mean={mean_cell_size:.6e}m, 使用={characteristic_size:.6e}m", file=sys.stderr)
    
    # 初始化场变量（在单元中心）
    p = np.zeros(n_cells)
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    
    # 时间参数
    nt = int(t_end / dt)
    
    # 输出目录
    output_dir = "output_ogrid"
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出和打印频率
    output_freq = 100000
    print_freq = max(1, nt // 100)
    
    # 数值滤波参数（与plane_2D.py一致）
    filter_strength = 0.01  # 滤波强度（0-1），0表示不滤波
    apply_filter = True  # 是否应用数值滤波
    
    # 主循环
    start = time.time()
    print(f"开始模拟: nt={nt}, dt={dt:.2e}s, CFL={CFL:.4f}", file=sys.stderr)
    
    for n in range(nt):
        t = n * dt
        
        # 使用RK4方法进行时间积分（支持并行）
        p, u, v = rk4_step_ogrid(p, u, v, t, dt, nodes, cells, cell_centers, 
                                 cell_areas, cell_neighbors, inner_wall_cells, 
                                 outer_wall_cells, wall_normals, boundary_edges,
                                 use_parallel, n_cores)
        
        # 应用数值滤波（抑制高频振荡，与plane_2D.py一致）
        if apply_filter and n % 10 == 0:  # 每10步滤波一次
            # 简单的空间平均滤波（在单元中心之间）
            p_filtered = p.copy()
            u_filtered = u.copy()
            v_filtered = v.copy()
            
            for i in range(n_cells):
                if len(cell_neighbors[i]) > 0:
                    # 与邻居单元的平均值
                    neighbor_avg_p = np.mean([p[n] for n in cell_neighbors[i]])
                    neighbor_avg_u = np.mean([u[n] for n in cell_neighbors[i]])
                    neighbor_avg_v = np.mean([v[n] for n in cell_neighbors[i]])
                    
                    p_filtered[i] = (1 - filter_strength) * p[i] + filter_strength * neighbor_avg_p
                    u_filtered[i] = (1 - filter_strength) * u[i] + filter_strength * neighbor_avg_u
                    v_filtered[i] = (1 - filter_strength) * v[i] + filter_strength * neighbor_avg_v
            
            p, u, v = p_filtered, u_filtered, v_filtered
        
        # 进度打印
        if n % print_freq == 0 or n == nt - 1:
            progress = 100.0 * n / nt
            elapsed = time.time() - start
            if n > 0:
                eta = elapsed / n * (nt - n)
                print(f"[{n:8d}/{nt}] ({progress:5.1f}%) t={t:.3e}s, 已用时={elapsed:.1f}s, 预计剩余={eta:.1f}s", 
                      file=sys.stderr)
        
        # 周期性输出VTK
        if n % output_freq == 0 or n == nt - 1:
            idx = n // output_freq
            fname = f"{output_dir}/acoustic_ogrid_{idx:04d}.vtk"
            
            # 将单元中心数据插值到节点（用于可视化）
            # 简单方法：使用相邻单元的平均值
            p_nodes = np.zeros(n_nodes)
            u_nodes = np.zeros(n_nodes)
            v_nodes = np.zeros(n_nodes)
            node_cell_count = np.zeros(n_nodes)
            
            for i, cell in enumerate(cells):
                for node_idx in cell:
                    p_nodes[node_idx] += p[i]
                    u_nodes[node_idx] += u[i]
                    v_nodes[node_idx] += v[i]
                    node_cell_count[node_idx] += 1
            
            # 平均
            valid_nodes = node_cell_count > 0
            p_nodes[valid_nodes] /= node_cell_count[valid_nodes]
            u_nodes[valid_nodes] /= node_cell_count[valid_nodes]
            v_nodes[valid_nodes] /= node_cell_count[valid_nodes]
            
            # 计算速度大小
            velocity_magnitude_nodes = np.sqrt(u_nodes**2 + v_nodes**2)
            
            # 输出VTK（非结构化网格）
            mesh = meshio.Mesh(
                points=nodes,
                cells=[("quad", cells)],
                point_data={
                    "pressure": p_nodes,
                    "velocity_u": u_nodes,
                    "velocity_v": v_nodes,
                    "velocity_magnitude": velocity_magnitude_nodes
                },
                cell_data={
                    "pressure": [p],
                    "velocity_u": [u],
                    "velocity_v": [v],
                    "velocity_magnitude": [np.sqrt(u**2 + v**2)]
                }
            )
            meshio.write(fname, mesh)
            
            # 计算统计信息
            p_valid = p[p != 0]  # 简化：排除零值
            if len(p_valid) > 0:
                print(f"[{n}/{nt}] 数据已保存到 {fname}", file=sys.stderr)
                print(f"  压力统计: min={np.min(p_valid):.3e}, max={np.max(p_valid):.3e}, "
                      f"mean={np.mean(p_valid):.3e} Pa", file=sys.stderr)
    
    end = time.time()
    print(f"Simulation complete in {end-start:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    # Windows上multiprocessing需要这个保护
    mp.freeze_support()
    main()
