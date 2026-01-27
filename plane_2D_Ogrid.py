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

# 导入参数控制模块
from control import control, validate_params

# 导入mesh.py的网格生成函数（或者直接复制相关代码）
# 为了简化，我们直接在这里实现网格生成，与mesh.py保持一致

# ==================== 网格生成（从mesh.py） ====================
def generate_ogrid_mesh(R, r_core, dr, dtheta_base):
    """
    生成非结构化O网格（环形网格）
    
    参数:
        R: 外半径 [m]
        r_core: 内半径 [m]
        dr: 径向网格间距 [m]
        dtheta_base: 角度间距基准值 [m]
    
    返回:
        nodes: 节点坐标 [N, 3]
        cells: 单元连接 [M, 4] (四边形)
        cell_centers: 单元中心坐标 [M, 2]
        cell_areas: 单元面积 [M]
        cell_neighbors: 单元邻居关系
    """
    dtheta = dtheta_base / R  # 角度间距
    
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


# ==================== 参数说明 ====================
# 所有参数现在通过 control() 函数设置
# 物理参数、几何参数、时间参数、声源参数等都在 control.py 中定义
# 主程序通过 control() 函数获取这些参数


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
def identify_boundary_cells(nodes, cells, cell_centers, R, r_core, nr, ntheta, tolerance):
    """
    识别边界单元（基于单元是否在边界上，而不仅仅是单元中心距离）
    
    参数:
        nodes: 节点坐标
        cells: 单元连接
        cell_centers: 单元中心坐标
        R, r_core: 内外半径
        nr, ntheta: 径向和角度层数
        tolerance: 容差（用于边界识别）
    
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
                   boundary_edges, use_parallel, n_cores,
                   rho0, c0, nu,
                   inner_r, outer_r, omega, 
                   wave_pressure, tau, sigma, R):
    """
    使用RK4方法进行时间积分（非结构化网格版本，支持并行）
    
    参数:
        p, u, v: 场变量
        t, dt: 时间和时间步长
        nodes, cells, cell_centers, cell_areas, cell_neighbors: 网格信息
        inner_wall_cells, outer_wall_cells, wall_normals: 边界信息
        boundary_edges: 边界边信息
        use_parallel, n_cores: 并行计算参数
        rho0, c0, nu: 物理参数
        inner_r, outer_r, omega, wave_pressure, tau, sigma, R: 声源参数
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


# ==================== 进度条和日志输出辅助函数 ====================
def update_progress_bar(progress, width=50, elapsed_time=None, eta=None):
    """
    更新并显示进度条
    
    参数:
        progress: 进度百分比 (0-100)
        width: 进度条宽度
        elapsed_time: 已用时间（秒）
        eta: 预计剩余时间（秒）
    
    返回:
        progress_bar_str: 进度条字符串
    """
    filled = int(width * progress / 100)
    bar = '█' * filled + '░' * (width - filled)
    progress_str = f"\r进度: |{bar}| {progress:.1f}%"
    
    if elapsed_time is not None:
        progress_str += f" | 已用时: {elapsed_time:.1f}s"
    if eta is not None:
        progress_str += f" | 预计剩余: {eta:.1f}s"
    
    return progress_str


def compute_residuals(p_old, u_old, v_old, p_new, u_new, v_new):
    """
    计算残差（用于监控收敛性）
    
    参数:
        p_old, u_old, v_old: 上一时间步的场变量
        p_new, u_new, v_new: 当前时间步的场变量
    
    返回:
        dict: 包含各场变量残差的字典
    """
    residuals = {
        'pressure': np.linalg.norm(p_new - p_old) if len(p_old) > 0 else 0.0,
        'velocity_u': np.linalg.norm(u_new - u_old) if len(u_old) > 0 else 0.0,
        'velocity_v': np.linalg.norm(v_new - v_old) if len(v_old) > 0 else 0.0,
    }
    residuals['total'] = np.sqrt(residuals['pressure']**2 + 
                                 residuals['velocity_u']**2 + 
                                 residuals['velocity_v']**2)
    return residuals


def write_log_entry(log_file_path, n, t, dt, residuals, p, u, v, 
                   p_min=None, p_max=None, p_mean=None):
    """
    写入日志条目
    
    参数:
        log_file_path: 日志文件路径
        n: 时间步数
        t: 当前时间
        dt: 时间步长
        residuals: 残差字典
        p, u, v: 场变量
        p_min, p_max, p_mean: 压力统计（可选）
    """
    # 计算统计信息（如果未提供）
    if p_min is None:
        p_valid = p[p != 0] if len(p) > 0 else p
        if len(p_valid) > 0:
            p_min = np.min(p_valid)
            p_max = np.max(p_valid)
            p_mean = np.mean(p_valid)
        else:
            p_min = p_max = p_mean = 0.0
    
    # 写入日志（追加模式）
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{n:10d}  {t:.6e}  {dt:.6e}  "
                f"{residuals['pressure']:.6e}  {residuals['velocity_u']:.6e}  "
                f"{residuals['velocity_v']:.6e}  {residuals['total']:.6e}  "
                f"{p_min:.6e}  {p_max:.6e}  {p_mean:.6e}\n")


def initialize_log_file(log_file_path):
    """
    初始化日志文件，写入表头
    
    参数:
        log_file_path: 日志文件路径
    """
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("# RDE声学模拟 - 计算日志\n")
        f.write("# " + "=" * 100 + "\n")
        f.write("# 列说明:\n")
        f.write("#   1. 时间步数 (n)\n")
        f.write("#   2. 当前时间 [s] (t)\n")
        f.write("#   3. 时间步长 [s] (dt)\n")
        f.write("#   4. 压力残差 (residual_p)\n")
        f.write("#   5. 速度u残差 (residual_u)\n")
        f.write("#   6. 速度v残差 (residual_v)\n")
        f.write("#   7. 总残差 (residual_total)\n")
        f.write("#   8. 压力最小值 [Pa] (p_min)\n")
        f.write("#   9. 压力最大值 [Pa] (p_max)\n")
        f.write("#  10. 压力平均值 [Pa] (p_mean)\n")
        f.write("# " + "=" * 100 + "\n")
        f.write(f"{'n':>10}  {'t':>12}  {'dt':>12}  "
                f"{'residual_p':>12}  {'residual_u':>12}  {'residual_v':>12}  "
                f"{'residual_total':>14}  {'p_min':>12}  {'p_max':>12}  {'p_mean':>12}\n")


def write_vtk_output(nodes, cells, p, u, v, output_dir, output_idx):
    """
    将场变量数据写入VTK文件
    
    参数:
        nodes: 节点坐标
        cells: 单元连接
        p, u, v: 场变量（在单元中心）
        output_dir: 输出目录
        output_idx: 输出文件索引
    
    返回:
        fname: 输出文件名
    """
    fname = f"{output_dir}/acoustic_ogrid_{output_idx:04d}.vtk"
    n_nodes = len(nodes)
    n_cells = len(cells)
    
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
    
    return fname


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
    # 从control函数获取所有参数
    try:
        params = control()
    except Exception as e:
        print(f"错误：无法从control模块获取参数", file=sys.stderr)
        print(f"详细错误：{e}", file=sys.stderr)
        print(f"请检查control.py文件是否正确配置", file=sys.stderr)
        sys.exit(1)
    
    # 验证参数完整性和有效性
    is_valid, missing_params, invalid_params = validate_params(params)
    if not is_valid:
        print("=" * 60, file=sys.stderr)
        print("错误：参数验证失败！", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # 报告缺少的参数
        if missing_params:
            print(f"\n缺少的参数数量: {len(missing_params)}", file=sys.stderr)
            print("\n缺少的参数列表:", file=sys.stderr)
            for i, (param, description) in enumerate(missing_params, 1):
                print(f"  {i}. {param} - {description}", file=sys.stderr)
        
        # 报告无效的参数
        if invalid_params:
            print(f"\n无效的参数数量: {len(invalid_params)}", file=sys.stderr)
            print("\n无效的参数列表:", file=sys.stderr)
            for i, (param, description, reason) in enumerate(invalid_params, 1):
                print(f"  {i}. {param} - {description}", file=sys.stderr)
                print(f"     原因: {reason}", file=sys.stderr)
        
        print("\n请检查control.py文件中的control()函数，确保：", file=sys.stderr)
        print("  1. 所有必需参数都已定义", file=sys.stderr)
        print("  2. 所有参数值都是有效的（不为None，数值参数为正数等）", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
    
    # 提取参数（物理参数）
    rho0 = params['rho0']
    c0 = params['c0']
    nu = params['nu']
    
    # 提取参数（几何参数）
    R = params['R']
    r_core = params['r_core']
    dr = params['dr']
    
    # 提取参数（时间参数）
    dt = params['dt']
    t_end = params['t_end']
    
    # 提取参数（声源参数）
    wave_pressure = params['wave_pressure']
    v_rde = params['v_rde']
    inner_r = params['inner_r']
    outer_r = params['outer_r']
    omega = params['omega']
    sigma = params['sigma']
    tau = params['tau']
    
    # 提取参数（数值计算参数）
    filter_strength = params['filter_strength']
    apply_filter = params['apply_filter']
    filter_frequency = params['filter_frequency']
    CFL_max = params['CFL_max']
    boundary_tolerance = params['boundary_tolerance']
    
    # 提取参数（输出参数）
    output_dir = params['output_dir']
    output_time_interval = params['output_time_interval']
    log_time_interval = params['log_time_interval']
    log_file = params['log_file']
    progress_bar_width = params['progress_bar_width']
    
    # 提取参数（网格生成参数）
    dtheta_base = params['dtheta_base']
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='RDE声学模拟 - 非结构化O网格版本（支持并行）')
    parser.add_argument('-n', '--ncores', type=int, default=None,
                       help='并行核数（默认：自动检测，使用最大核数-1）')
    parser.add_argument('--single', action='store_true',
                       help='使用单核运行（覆盖-n参数）')
    
    args = parser.parse_args()
    
    # 确定并行核数（命令行参数优先，否则使用control中的默认值）
    max_cores = mp.cpu_count()
    if args.single:
        n_cores = 1
        use_parallel = False
    elif args.ncores is not None:
        n_cores = min(args.ncores, max_cores)
        use_parallel = (n_cores > 1)
    else:
        # 使用control中的默认值
        if params['n_cores_default'] is not None:
            n_cores = min(params['n_cores_default'], max_cores)
        else:
            # 默认使用最大核数-1（留一个核心给系统）
            n_cores = max(1, max_cores - 1)
        use_parallel = params['use_parallel_default'] and (n_cores > 1)
    
    print(f"系统最大核数: {max_cores}", file=sys.stderr)
    print(f"使用核数: {n_cores}", file=sys.stderr)
    print(f"并行模式: {'开启' if use_parallel else '关闭'}", file=sys.stderr)
    print(f"参考核数（推荐）: {max(1, max_cores - 1)}", file=sys.stderr)
    
    # 生成网格
    print("正在生成非结构化O网格...", file=sys.stderr)
    nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta = generate_ogrid_mesh(
        R, r_core, dr, dtheta_base)
    n_cells = len(cells)
    n_nodes = len(nodes)
    
    print(f"网格生成完成: {n_nodes}个节点, {n_cells}个单元", file=sys.stderr)
    
    # 识别边界（基于单元节点，更准确）
    inner_wall_cells, outer_wall_cells, wall_normals, boundary_edges = identify_boundary_cells(
        nodes, cells, cell_centers, R, r_core, nr, ntheta, boundary_tolerance)
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
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化日志文件
    log_file_path = os.path.join(output_dir, log_file)
    initialize_log_file(log_file_path)
    
    # 基于时间的输出控制
    next_output_time = 0.0  # 下次VTK输出的时间
    next_log_time = 0.0      # 下次日志输出的时间
    output_idx = 0          # VTK文件索引
    
    # 用于残差计算的上一时间步场变量
    p_old = p.copy()
    u_old = u.copy()
    v_old = v.copy()
    
    # 主循环
    start = time.time()
    print(f"开始模拟: nt={nt}, dt={dt:.2e}s, CFL={CFL:.4f}", file=sys.stderr)
    print(f"VTK输出时间间隔: {output_time_interval:.2e}s", file=sys.stderr)
    print(f"日志输出时间间隔: {log_time_interval:.2e}s", file=sys.stderr)
    print(f"日志文件: {log_file_path}", file=sys.stderr)
    print("", file=sys.stderr)  # 空行
    
    for n in range(nt):
        t = n * dt
        
        # 使用RK4方法进行时间积分（支持并行）
        p_new, u_new, v_new = rk4_step_ogrid(p, u, v, t, dt, nodes, cells, cell_centers, 
                                             cell_areas, cell_neighbors, inner_wall_cells, 
                                             outer_wall_cells, wall_normals, boundary_edges,
                                             use_parallel, n_cores,
                                             rho0, c0, nu,
                                             inner_r, outer_r, omega, wave_pressure, tau, sigma, R)
        
        # 应用数值滤波（抑制高频振荡，与plane_2D.py一致）
        if apply_filter and n % filter_frequency == 0:  # 按配置的频率滤波
            # 简单的空间平均滤波（在单元中心之间）
            p_filtered = p_new.copy()
            u_filtered = u_new.copy()
            v_filtered = v_new.copy()
            
            for i in range(n_cells):
                if len(cell_neighbors[i]) > 0:
                    # 与邻居单元的平均值
                    neighbor_avg_p = np.mean([p_new[n] for n in cell_neighbors[i]])
                    neighbor_avg_u = np.mean([u_new[n] for n in cell_neighbors[i]])
                    neighbor_avg_v = np.mean([v_new[n] for n in cell_neighbors[i]])
                    
                    p_filtered[i] = (1 - filter_strength) * p_new[i] + filter_strength * neighbor_avg_p
                    u_filtered[i] = (1 - filter_strength) * u_new[i] + filter_strength * neighbor_avg_u
                    v_filtered[i] = (1 - filter_strength) * v_new[i] + filter_strength * neighbor_avg_v
            
            p_new, u_new, v_new = p_filtered, u_filtered, v_filtered
        
        # 计算残差（用于监控）
        residuals = compute_residuals(p_old, u_old, v_old, p_new, u_new, v_new)
        
        # 更新场变量
        p, u, v = p_new, u_new, v_new
        p_old, u_old, v_old = p.copy(), u.copy(), v.copy()
        
        # 基于时间的日志输出
        if t >= next_log_time or n == nt - 1:
            # 计算压力统计
            p_valid = p[p != 0] if len(p) > 0 else p
            if len(p_valid) > 0:
                p_min = np.min(p_valid)
                p_max = np.max(p_valid)
                p_mean = np.mean(p_valid)
            else:
                p_min = p_max = p_mean = 0.0
            
            # 写入日志
            write_log_entry(log_file_path, n, t, dt, residuals, p, u, v,
                           p_min, p_max, p_mean)
            
            # 更新下次日志输出时间
            next_log_time = ((int(t / log_time_interval) + 1) * log_time_interval)
        
        # 基于时间的VTK输出
        if t >= next_output_time or n == nt - 1:
            fname = write_vtk_output(nodes, cells, p, u, v, output_dir, output_idx)
            
            # 更新下次输出时间
            next_output_time = ((int(t / output_time_interval) + 1) * output_time_interval)
            output_idx += 1
        
        # 进度条显示（每次迭代都更新，但只在stderr输出）
        progress = 100.0 * (n + 1) / nt
        elapsed = time.time() - start
        if n > 0:
            eta = elapsed / (n + 1) * (nt - n - 1)
        else:
            eta = 0.0
        
        # 使用进度条格式
        progress_str = update_progress_bar(progress, progress_bar_width, elapsed, eta)
        print(progress_str, end='', file=sys.stderr, flush=True)
        
        # 在最后一步或特定时间点换行并输出详细信息
        if n == nt - 1 or (n > 0 and n % max(1, nt // 100) == 0):
            print("", file=sys.stderr)  # 换行
            if n == nt - 1:
                print(f"完成: t={t:.6e}s, 总残差={residuals['total']:.6e}", file=sys.stderr)
    
    end = time.time()
    print(f"\n模拟完成！总用时: {end-start:.1f}s", file=sys.stderr)
    print(f"日志文件已保存到: {log_file_path}", file=sys.stderr)


if __name__ == "__main__":
    # Windows上multiprocessing需要这个保护
    mp.freeze_support()
    main()
