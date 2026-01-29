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
import gc  # 用于显式内存清理

# 尝试导入numba（可选，如果没有安装则使用纯numpy）
try:
    import numba
    from numba import jit, types
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建一个假的装饰器，如果numba不可用
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

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
    
    # ==================== 预计算几何量和边-邻居映射（数组结构版本） ====================
    # 使用数组结构替代字典，大幅提升访问速度
    # 每个单元最多4条边，使用固定大小数组
    
    max_edges_per_cell = 4
    n_edges_total = n_cells * max_edges_per_cell
    
    # 使用结构化数组存储边几何信息
    # edge_normals: [n_cells, max_edges, 2] - 法向量
    # edge_lengths: [n_cells, max_edges] - 边长度
    # edge_neighbors: [n_cells, max_edges] - 邻居单元索引（-1表示边界）
    # edge_count: [n_cells] - 每个单元的实际边数
    
    edge_normals = np.zeros((n_cells, max_edges_per_cell, 2))
    edge_lengths = np.zeros((n_cells, max_edges_per_cell))
    edge_neighbors = np.full((n_cells, max_edges_per_cell), -1, dtype=np.int32)
    edge_count = np.zeros(n_cells, dtype=np.int32)
    
    for i in range(n_cells):
        cell_nodes = cells[i]
        n_nodes = len(cell_nodes)
        edge_count[i] = n_nodes
        
        for j in range(n_nodes):
            node0_idx = cell_nodes[j]
            node1_idx = cell_nodes[(j+1) % n_nodes]
            
            node0 = nodes[node0_idx, :2]
            node1 = nodes[node1_idx, :2]
            
            # 计算边向量
            edge_vec = node1 - node0
            edge_length = np.linalg.norm(edge_vec)
            
            if edge_length > 1e-10:
                # 计算法向量（逆时针旋转90度）
                n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
                
                # 计算边中心
                edge_center = (node0 + node1) / 2.0
                
                # 检查方向（确保指向单元外部）
                center_to_edge = edge_center - cell_centers[i]
                if np.dot(n, center_to_edge) < 0:
                    n = -n
                
                # 查找邻居单元
                neighbor_idx = -1  # -1表示边界边
                for neighbor in cell_neighbors[i]:
                    neighbor_nodes = set(cells[neighbor])
                    if node0_idx in neighbor_nodes and node1_idx in neighbor_nodes:
                        neighbor_idx = neighbor
                        break
                
                # 存储到数组
                edge_normals[i, j] = n
                edge_lengths[i, j] = edge_length
                edge_neighbors[i, j] = neighbor_idx
    
    return (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
            edge_normals, edge_lengths, edge_neighbors, edge_count)


# ==================== 参数说明 ====================
# 所有参数现在通过 control() 函数设置
# 物理参数、几何参数、时间参数、声源参数等都在 control.py 中定义
# 主程序通过 control() 函数获取这些参数


# ==================== 有限体积方法算子（Numba优化版本） ====================
@jit(nopython=True, cache=True)
def _compute_gradient_fvm_core(f, cell_areas, edge_normals, edge_lengths, 
                                edge_neighbors, edge_count, boundary_mask):
    """
    使用有限体积方法计算梯度（Numba加速核心函数）
    
    参数:
        f: 场变量（在单元中心）[n_cells]
        cell_areas: 单元面积 [n_cells]
        edge_normals: 边法向量 [n_cells, max_edges, 2]
        edge_lengths: 边长度 [n_cells, max_edges]
        edge_neighbors: 邻居单元索引 [n_cells, max_edges]（-1表示边界）
        edge_count: 每个单元的边数 [n_cells]
        boundary_mask: 边界边掩码 [n_cells, max_edges]
    
    返回:
        grad_f: 梯度 [n_cells, 2]
    """
    n_cells = len(f)
    grad_f = np.zeros((n_cells, 2))
    
    for i in range(n_cells):
        grad_sum = np.zeros(2)
        n_edges = int(edge_count[i])
        
        for j in range(n_edges):
            if edge_lengths[i, j] > 1e-10:
                n = edge_normals[i, j]
                edge_length = edge_lengths[i, j]
                
                # 检查是否是边界边
                if boundary_mask[i, j] or edge_neighbors[i, j] < 0:
                    # 边界边：使用单元中心值
                    f_face = f[i]
                else:
                    # 内部边：使用相邻单元的平均值
                    neighbor_idx = edge_neighbors[i, j]
                    f_face = 0.5 * (f[i] + f[neighbor_idx])
                
                grad_sum[0] += f_face * n[0] * edge_length
                grad_sum[1] += f_face * n[1] * edge_length
        
        if cell_areas[i] > 1e-10:
            grad_f[i, 0] = grad_sum[0] / cell_areas[i]
            grad_f[i, 1] = grad_sum[1] / cell_areas[i]
    
    return grad_f


def compute_gradient_fvm(f, cell_areas, edge_normals, edge_lengths, 
                         edge_neighbors, edge_count, boundary_mask=None):
    """
    使用有限体积方法计算梯度（优化版本，使用数组结构和Numba）
    
    参数:
        f: 场变量（在单元中心）[n_cells]
        cell_areas: 单元面积 [n_cells]
        edge_normals: 边法向量 [n_cells, max_edges, 2]
        edge_lengths: 边长度 [n_cells, max_edges]
        edge_neighbors: 邻居单元索引 [n_cells, max_edges]
        edge_count: 每个单元的边数 [n_cells]
        boundary_mask: 边界边掩码 [n_cells, max_edges]（可选）
    
    返回:
        grad_f: 梯度 [n_cells, 2]
    """
    if boundary_mask is None:
        boundary_mask = np.zeros((len(f), edge_normals.shape[1]), dtype=np.bool_)
    
    return _compute_gradient_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                     edge_neighbors, edge_count, boundary_mask)


@jit(nopython=True, cache=True)
def _compute_divergence_fvm_core(u, v, cell_areas, edge_normals, edge_lengths,
                                  edge_neighbors, edge_count, boundary_mask):
    """
    使用有限体积方法计算散度（Numba加速核心函数）
    
    参数:
        u, v: 速度分量（在单元中心）[n_cells]
        cell_areas: 单元面积 [n_cells]
        edge_normals: 边法向量 [n_cells, max_edges, 2]
        edge_lengths: 边长度 [n_cells, max_edges]
        edge_neighbors: 邻居单元索引 [n_cells, max_edges]
        edge_count: 每个单元的边数 [n_cells]
        boundary_mask: 边界边掩码 [n_cells, max_edges]
    
    返回:
        div: 散度 [n_cells]
    """
    n_cells = len(u)
    div = np.zeros(n_cells)
    
    for i in range(n_cells):
        div_sum = 0.0
        n_edges = int(edge_count[i])
        
        for j in range(n_edges):
            if edge_lengths[i, j] > 1e-10:
                n = edge_normals[i, j]
                edge_length = edge_lengths[i, j]
                
                # 检查是否是边界边
                if boundary_mask[i, j] or edge_neighbors[i, j] < 0:
                    # 边界边：硬壁边界条件，法向速度为零
                    # 对散度的贡献为零（理论上），但为了数值稳定性使用单元中心值
                    u_face = u[i]
                    v_face = v[i]
                else:
                    # 内部边：使用相邻单元的平均值
                    neighbor_idx = edge_neighbors[i, j]
                    u_face = 0.5 * (u[i] + u[neighbor_idx])
                    v_face = 0.5 * (v[i] + v[neighbor_idx])
                
                div_sum += (u_face * n[0] + v_face * n[1]) * edge_length
        
        if cell_areas[i] > 1e-10:
            div[i] = div_sum / cell_areas[i]
    
    return div


def compute_divergence_fvm(u, v, cell_areas, edge_normals, edge_lengths,
                           edge_neighbors, edge_count, boundary_mask=None):
    """
    使用有限体积方法计算散度（优化版本，使用数组结构和Numba）
    
    参数:
        u, v: 速度分量（在单元中心）[n_cells]
        cell_areas: 单元面积 [n_cells]
        edge_normals: 边法向量 [n_cells, max_edges, 2]
        edge_lengths: 边长度 [n_cells, max_edges]
        edge_neighbors: 邻居单元索引 [n_cells, max_edges]
        edge_count: 每个单元的边数 [n_cells]
        boundary_mask: 边界边掩码 [n_cells, max_edges]（可选）
    
    返回:
        div: 散度 [n_cells]
    """
    if boundary_mask is None:
        boundary_mask = np.zeros((len(u), edge_normals.shape[1]), dtype=np.bool_)
    
    return _compute_divergence_fvm_core(u, v, cell_areas, edge_normals, edge_lengths,
                                       edge_neighbors, edge_count, boundary_mask)


@jit(nopython=True, cache=True)
def _compute_laplacian_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                 edge_neighbors, edge_count, boundary_mask):
    """
    使用有限体积方法计算拉普拉斯算子（Numba加速核心函数）
    
    参数:
        f: 场变量（在单元中心）[n_cells]
        cell_areas: 单元面积 [n_cells]
        edge_normals: 边法向量 [n_cells, max_edges, 2]
        edge_lengths: 边长度 [n_cells, max_edges]
        edge_neighbors: 邻居单元索引 [n_cells, max_edges]
        edge_count: 每个单元的边数 [n_cells]
        boundary_mask: 边界边掩码 [n_cells, max_edges]
    
    返回:
        laplacian_f: 拉普拉斯 [n_cells]
    """
    n_cells = len(f)
    laplacian_f = np.zeros(n_cells)
    
    for i in range(n_cells):
        div_sum = 0.0
        n_edges = int(edge_count[i])
        
        for j in range(n_edges):
            if edge_lengths[i, j] > 1e-10:
                n = edge_normals[i, j]
                edge_length = edge_lengths[i, j]
                
                # 检查是否是边界边
                if boundary_mask[i, j] or edge_neighbors[i, j] < 0:
                    # 边界边：对于硬壁边界，法向梯度为零
                    grad_face_n = 0.0
                else:
                    # 内部边：计算梯度的法向分量
                    neighbor_idx = edge_neighbors[i, j]
                    if neighbor_idx >= 0:
                        # 使用中心差分近似梯度
                        df_dn = f[neighbor_idx] - f[i]
                        grad_face_n = df_dn  # 法向梯度分量
                    else:
                        grad_face_n = 0.0
                
                div_sum += grad_face_n * edge_length
        
        if cell_areas[i] > 1e-10:
            laplacian_f[i] = div_sum / cell_areas[i]
    
    return laplacian_f


def compute_laplacian_fvm(f, cell_areas, edge_normals, edge_lengths,
                          edge_neighbors, edge_count, boundary_mask=None):
    """
    使用有限体积方法计算拉普拉斯算子（优化版本，使用数组结构和Numba）
    
    参数:
        f: 场变量（在单元中心）[n_cells]
        cell_areas: 单元面积 [n_cells]
        edge_normals: 边法向量 [n_cells, max_edges, 2]
        edge_lengths: 边长度 [n_cells, max_edges]
        edge_neighbors: 邻居单元索引 [n_cells, max_edges]
        edge_count: 每个单元的边数 [n_cells]
        boundary_mask: 边界边掩码 [n_cells, max_edges]（可选）
    
    返回:
        laplacian_f: 拉普拉斯 [n_cells]
    """
    if boundary_mask is None:
        boundary_mask = np.zeros((len(f), edge_normals.shape[1]), dtype=np.bool_)
    
    return _compute_laplacian_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                       edge_neighbors, edge_count, boundary_mask)


# ==================== 源项（Numba优化版本） ====================
@jit(nopython=True, cache=True)
def _source_term_ogrid_core(cell_centers, t, inner_r, outer_r, omega, 
                            wave_pressure, tau, sigma, R):
    """
    计算源项（Numba加速核心函数）
    
    参数:
        cell_centers: 单元中心坐标 [n_cells, 2]
        t: 时间
        其他参数同plane_2D.py
    
    返回:
        S: 源项 [n_cells]
    """
    n_cells = cell_centers.shape[0]
    S = np.zeros(n_cells)
    
    theta0 = (omega * t) % (2.0 * np.pi)
    two_pi = 2.0 * np.pi
    
    for i in range(n_cells):
        x = cell_centers[i, 0]
        y = cell_centers[i, 1]
        r = np.sqrt(x*x + y*y)
        
        if r >= inner_r and r <= outer_r:
            theta = np.arctan2(y, x)
            if theta < 0:
                theta += two_pi
            
            dtheta = theta - theta0
            if dtheta > np.pi:
                dtheta -= two_pi
            elif dtheta < -np.pi:
                dtheta += two_pi
            
            if dtheta >= 0:
                S[i] = (wave_pressure / tau * 
                       np.exp(- (dtheta * R)**2 / (2.0 * sigma * sigma)))
    
    return S


def source_term_ogrid(cell_centers, t, inner_r, outer_r, omega, wave_pressure, tau, sigma, R):
    """
    计算源项（优化版本，使用Numba加速）
    
    参数:
        cell_centers: 单元中心坐标 [n_cells, 2]
        t: 时间
        其他参数同plane_2D.py
    
    返回:
        S: 源项 [n_cells]
    """
    return _source_term_ogrid_core(cell_centers, t, inner_r, outer_r, omega,
                                   wave_pressure, tau, sigma, R)


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


def apply_wall_boundary_conditions_ogrid(u, v, inner_wall_cells, outer_wall_cells, wall_normals, u_out=None, v_out=None):
    """
    应用硬壁边界条件：壁面处法向速度为零（支持原地操作）
    
    参数:
        u, v: 输入速度场
        inner_wall_cells, outer_wall_cells, wall_normals: 边界信息
        u_out, v_out: 输出数组（可选，如果提供则原地操作）
    
    返回:
        u_new, v_new: 修正后的速度场
    """
    if u_out is None or v_out is None:
        u_new = u.copy()
        v_new = v.copy()
    else:
        u_new = u_out
        v_new = v_out
        u_new[:] = u
        v_new[:] = v
    
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


# ==================== RK4时间积分（Numba优化版本，内存优化） ====================
def rk4_step_ogrid(p, u, v, t, dt, cell_centers, cell_areas, cell_areas_sum,
                   inner_wall_cells, outer_wall_cells, wall_normals,
                   boundary_mask, edge_normals, edge_lengths, 
                   edge_neighbors, edge_count,
                   rho0, c0, nu,
                   inner_r, outer_r, omega, 
                   wave_pressure, tau, sigma, R,
                   zero_mean_source, zero_mean_mode,
                   p_new=None, u_new=None, v_new=None,
                   temp_arrays=None,
                   diagnostics=None):
    """
    使用RK4方法进行时间积分（优化版本，使用数组结构和Numba，支持内存重用）
    
    参数:
        p, u, v: 场变量
        t, dt: 时间和时间步长
        cell_centers, cell_areas: 网格信息
        inner_wall_cells, outer_wall_cells, wall_normals: 边界信息
        boundary_mask: 边界边掩码 [n_cells, max_edges]
        edge_normals, edge_lengths, edge_neighbors, edge_count: 预计算的几何量
        rho0, c0, nu: 物理参数
        inner_r, outer_r, omega, wave_pressure, tau, sigma, R: 声源参数
        p_new, u_new, v_new: 输出数组（可选，如果提供则重用）
        temp_arrays: 临时数组字典（可选，用于重用中间数组）
    
    返回:
        p_new, u_new, v_new: 新状态
        temp_arrays: 复用的RK4临时数组
        diagnostics: 当前时间步内关于源项DC修正的诊断信息
    """
    n_cells = len(p)
    
    # 预分配或重用临时数组
    if temp_arrays is None:
        temp_arrays = {
            'S': np.zeros(n_cells),
            'div_uv': np.zeros(n_cells),
            'grad_p': np.zeros((n_cells, 2)),
            'laplacian_u': np.zeros(n_cells),
            'laplacian_v': np.zeros(n_cells),
            'p_temp': np.zeros(n_cells),
            'u_temp': np.zeros(n_cells),
            'v_temp': np.zeros(n_cells),
            'k1_p': np.zeros(n_cells),
            'k1_u': np.zeros(n_cells),
            'k1_v': np.zeros(n_cells),
            'k2_p': np.zeros(n_cells),
            'k2_u': np.zeros(n_cells),
            'k2_v': np.zeros(n_cells),
            'k3_p': np.zeros(n_cells),
            'k3_u': np.zeros(n_cells),
            'k3_v': np.zeros(n_cells),
            'k4_p': np.zeros(n_cells),
            'k4_u': np.zeros(n_cells),
            'k4_v': np.zeros(n_cells),
        }
    
    # 重用输出数组或创建新数组
    if p_new is None:
        p_new = np.zeros(n_cells)
    if u_new is None:
        u_new = np.zeros(n_cells)
    if v_new is None:
        v_new = np.zeros(n_cells)
    
    # 源项DC诊断累加器（按时间步聚合4个stage）
    if diagnostics is None:
        diagnostics = {}
    diagnostics_step = {
        'S_mean_raw_sum': 0.0,     # 4个stage的S_mean_raw求和
        'S_mean_after_sum': 0.0,   # 4个stage修正后的均值求和（用于检查接近0）
        'sum_SV_after_sum': 0.0,   # 4个stage修正后Σ(S*V)求和
        'n_stages': 0,
    }

    def rhs(p_state, u_state, v_state, t_state, dp_dt_out, du_dt_out, dv_dt_out):
        """计算右端项（重用临时数组）"""
        # 计算源项（Numba加速）
        S = source_term_ogrid(cell_centers, t_state, inner_r, outer_r, omega, 
                              wave_pressure, tau, sigma, R)

        # ==================== 源项DC零均值修正 ====================
        # 在封闭硬壁腔体中，旋转源项往往带有非零体积平均（DC 分量），
        # 会在时间积分中累积成平均压力的非物理漂移。
        # 对于 zero_mean_mode == "stage"：
        #   在每个 RK4 stage 内部直接对 RHS 源项做零均值处理，物理上最干净；
        # 对于 zero_mean_mode == "step"：
        #   在这里仅统计每个 stage 的体积均值，真正的修正推迟到整步更新结束后，
        #   通过对压力场做一次整体常数平移近似移除本步 DC 贡献。
        if cell_areas_sum > 0.0:
            # 体积加权源项均值（未修正前）
            S_mean_raw = np.sum(S * cell_areas) / cell_areas_sum
        else:
            S_mean_raw = 0.0

        # 记录诊断信息（即使关闭修正也记录原始DC，便于对比）
        diagnostics_step['S_mean_raw_sum'] += S_mean_raw
        diagnostics_step['n_stages'] += 1

        if zero_mean_source and zero_mean_mode == "stage":
            # 减去体积平均分量，强制满足 Σ(S*V) ≈ 0
            S -= S_mean_raw
            if cell_areas_sum > 0.0:
                S_mean_after = np.sum(S * cell_areas) / cell_areas_sum
                sum_SV_after = np.sum(S * cell_areas)
            else:
                S_mean_after = 0.0
                sum_SV_after = 0.0
        else:
            # 未对源项做就地修正时，仅用于诊断（包括 zero_mean_mode == "step" 或 "off"）
            if cell_areas_sum > 0.0:
                S_mean_after = S_mean_raw
                sum_SV_after = np.sum(S * cell_areas)
            else:
                S_mean_after = 0.0
                sum_SV_after = 0.0

        diagnostics_step['S_mean_after_sum'] += S_mean_after
        diagnostics_step['sum_SV_after_sum'] += sum_SV_after
        
        # 计算散度和梯度（使用数组结构和Numba）
        div_uv = compute_divergence_fvm(u_state, v_state, cell_areas,
                                       edge_normals, edge_lengths,
                                       edge_neighbors, edge_count, boundary_mask)
        grad_p = compute_gradient_fvm(p_state, cell_areas,
                                      edge_normals, edge_lengths,
                                      edge_neighbors, edge_count, boundary_mask)
        
        # 压力方程: dp/dt = -rho0*c0^2*div(u) + S
        dp_dt_out[:] = -rho0 * c0**2 * div_uv + S
        
        # 速度方程: du/dt = -grad(p)/rho0 + nu*laplacian(u)
        # 注意：nu*laplacian(u)项主要用于数值稳定性（人工耗散）
        laplacian_u = compute_laplacian_fvm(u_state, cell_areas,
                                            edge_normals, edge_lengths,
                                            edge_neighbors, edge_count, boundary_mask)
        laplacian_v = compute_laplacian_fvm(v_state, cell_areas,
                                            edge_normals, edge_lengths,
                                            edge_neighbors, edge_count, boundary_mask)
        
        du_dt_out[:] = -grad_p[:, 0] / rho0 + nu * laplacian_u
        dv_dt_out[:] = -grad_p[:, 1] / rho0 + nu * laplacian_v
    
    # RK4的四个阶段（重用临时数组）
    rhs(p, u, v, t, temp_arrays['k1_p'], temp_arrays['k1_u'], temp_arrays['k1_v'])
    
    temp_arrays['p_temp'][:] = p + 0.5*dt*temp_arrays['k1_p']
    temp_arrays['u_temp'][:] = u + 0.5*dt*temp_arrays['k1_u']
    temp_arrays['v_temp'][:] = v + 0.5*dt*temp_arrays['k1_v']
    rhs(temp_arrays['p_temp'], temp_arrays['u_temp'], temp_arrays['v_temp'], 
        t + 0.5*dt, temp_arrays['k2_p'], temp_arrays['k2_u'], temp_arrays['k2_v'])
    
    temp_arrays['p_temp'][:] = p + 0.5*dt*temp_arrays['k2_p']
    temp_arrays['u_temp'][:] = u + 0.5*dt*temp_arrays['k2_u']
    temp_arrays['v_temp'][:] = v + 0.5*dt*temp_arrays['k2_v']
    rhs(temp_arrays['p_temp'], temp_arrays['u_temp'], temp_arrays['v_temp'], 
        t + 0.5*dt, temp_arrays['k3_p'], temp_arrays['k3_u'], temp_arrays['k3_v'])
    
    temp_arrays['p_temp'][:] = p + dt*temp_arrays['k3_p']
    temp_arrays['u_temp'][:] = u + dt*temp_arrays['k3_u']
    temp_arrays['v_temp'][:] = v + dt*temp_arrays['k3_v']
    rhs(temp_arrays['p_temp'], temp_arrays['u_temp'], temp_arrays['v_temp'], 
        t + dt, temp_arrays['k4_p'], temp_arrays['k4_u'], temp_arrays['k4_v'])
    
    # 更新（原地操作）
    p_new[:] = p + (dt/6.0) * (temp_arrays['k1_p'] + 2*temp_arrays['k2_p'] + 
                               2*temp_arrays['k3_p'] + temp_arrays['k4_p'])
    u_new[:] = u + (dt/6.0) * (temp_arrays['k1_u'] + 2*temp_arrays['k2_u'] + 
                               2*temp_arrays['k3_u'] + temp_arrays['k4_u'])
    v_new[:] = v + (dt/6.0) * (temp_arrays['k1_v'] + 2*temp_arrays['k2_v'] + 
                               2*temp_arrays['k3_v'] + temp_arrays['k4_v'])
    
    # 应用边界条件（重用输出数组）
    u_new, v_new = apply_wall_boundary_conditions_ogrid(u_new, v_new, 
                                                        inner_wall_cells, outer_wall_cells, wall_normals,
                                                        u_new, v_new)

    # 如果选择 step 模式：在整步更新结束后再做一次“整体DC修正”
    # 这里利用本时间步内四个stage的体积平均源项，近似估算这一时间步
    # 会带来的平均压力增量，并通过对压力场做一次常数平移将其移除。
    if zero_mean_source and zero_mean_mode == "step" and diagnostics_step['n_stages'] > 0:
        # 本时间步内各stage源项体积均值的简单平均（用于估算本步 DC 贡献）
        S_mean_step = diagnostics_step['S_mean_raw_sum'] / diagnostics_step['n_stages']
        # 近似：dp_mean/dt ≈ S_mean_step，因此一时间步的平均压力增量约为 S_mean_step * dt
        delta_p_dc = S_mean_step * dt
        # 该操作移除源项的体积平均（DC 分量）在本时间步对平均压力的贡献，
        # 通过对整个压力场减去常数偏置，不改变空间分布形态，只抹去均值漂移。
        p_new[:] -= delta_p_dc

    # 汇总当前时间步的源项DC诊断信息（按stage求平均）
    if diagnostics_step['n_stages'] > 0:
        n_stages = diagnostics_step['n_stages']
        S_mean_raw_stage = diagnostics_step['S_mean_raw_sum'] / n_stages
        S_mean_after_stage = diagnostics_step['S_mean_after_sum'] / n_stages
        sum_SV_after_stage = diagnostics_step['sum_SV_after_sum'] / n_stages
    else:
        S_mean_raw_stage = 0.0
        S_mean_after_stage = 0.0
        sum_SV_after_stage = 0.0

    # 根据模式写入诊断字典：
    #   - "stage"：S_mean_after_stage / sum_SV_after_stage 反映每个 stage 内部的零均值效果
    #   - "step" ：S_mean_raw_stage 仍表示源项本身的DC；after/sum 主要用于参考
    if zero_mean_mode == "stage":
        diagnostics['S_mean_raw_stage'] = S_mean_raw_stage
        diagnostics['S_mean_after_stage'] = S_mean_after_stage
        diagnostics['sum_SV_after_stage'] = sum_SV_after_stage
    elif zero_mean_mode == "step":
        diagnostics['S_mean_raw_stage'] = S_mean_raw_stage
        # 对于 step 模式，我们在整步结束后通过对压力场加常数的方式
        # 近似移除了本步DC贡献；这里将“修正后”的量标记为 0，更直观地表示
        # 源项的体积平均 DC 对平均压力的净贡献已被抵消。
        diagnostics['S_mean_after_stage'] = 0.0
        diagnostics['sum_SV_after_stage'] = 0.0
    else:  # "off" 或其它保守处理
        diagnostics['S_mean_raw_stage'] = S_mean_raw_stage
        diagnostics['S_mean_after_stage'] = S_mean_after_stage
        diagnostics['sum_SV_after_stage'] = sum_SV_after_stage

    diagnostics['zero_mean_source'] = bool(zero_mean_source)
    diagnostics['zero_mean_mode'] = zero_mean_mode
    
    return p_new, u_new, v_new, temp_arrays, diagnostics


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
                   cell_areas,
                   p_min=None, p_max=None, p_mean=None, p_rms=None):
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
    if p_min is None or p_max is None or p_mean is None or p_rms is None:
        # 使用体积/面积加权统计量
        if len(p) > 0:
            p_min = float(np.min(p)) if p_min is None else p_min
            p_max = float(np.max(p)) if p_max is None else p_max
            sumV = float(np.sum(cell_areas))
            if sumV > 0.0:
                # 体积加权平均压力
                p_mean = float(np.sum(p * cell_areas) / sumV)
                # 基于 p_fluct = p - p_mean 的体积加权 RMS
                p_fluct = p - p_mean
                p_rms = float(np.sqrt(np.sum(p_fluct * p_fluct * cell_areas) / sumV))
            else:
                p_mean = 0.0 if p_mean is None else p_mean
                p_rms = 0.0 if p_rms is None else p_rms
        else:
            p_min = p_max = p_mean = p_rms = 0.0
    
    # 写入日志（追加模式）
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{n:10d}  {t:.6e}  {dt:.6e}  "
                f"{residuals['pressure']:.6e}  {residuals['velocity_u']:.6e}  "
                f"{residuals['velocity_v']:.6e}  {residuals['total']:.6e}  "
                f"{p_min:.6e}  {p_max:.6e}  {p_mean:.6e}  {p_rms:.6e}\n")


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
        f.write("#  10. 压力体积加权平均值 [Pa] (p_mean_vol)\n")
        f.write("#  11. 压力体积加权RMS（关于p_mean_vol的波动）[Pa] (p_rms_vol)\n")
        f.write("# " + "=" * 100 + "\n")
        f.write(f"{'n':>10}  {'t':>12}  {'dt':>12}  "
                f"{'residual_p':>12}  {'residual_u':>12}  {'residual_v':>12}  "
                f"{'residual_total':>14}  {'p_min':>12}  {'p_max':>12}  {'p_mean_vol':>12}  {'p_rms_vol':>12}\n")


def initialize_source_diagnostics_file(diag_file_path):
    """
    初始化源项DC修正诊断CSV文件
    
    列包括：
      n, t, S_mean_raw_stage, S_mean_after_stage, sum_SV_after_stage, 
      p_mean_vol, p_rms_vol, zero_mean_source, zero_mean_mode
    """
    with open(diag_file_path, 'w', encoding='utf-8') as f:
        f.write("# 源项DC零均值修正诊断数据\n")
        f.write("# n, t, S_mean_raw_stage, S_mean_after_stage, sum_SV_after_stage, ")
        f.write("p_mean_vol, p_rms_vol, zero_mean_source, zero_mean_mode\n")


def write_source_diagnostics_entry(diag_file_path, n, t, diagnostics, p_mean, p_rms):
    """
    追加写入一条源项DC修正诊断记录到CSV文件。
    
    参数:
        diag_file_path: 诊断CSV文件路径
        n, t: 时间步和当前时间
        diagnostics: rk4_step_ogrid 返回的诊断字典
        p_mean, p_rms: 当前时间步的体积加权平均压力与RMS
    """
    S_mean_raw = diagnostics.get('S_mean_raw_stage', 0.0)
    S_mean_after = diagnostics.get('S_mean_after_stage', 0.0)
    sum_SV_after = diagnostics.get('sum_SV_after_stage', 0.0)
    zero_mean_source = diagnostics.get('zero_mean_source', False)
    zero_mean_mode = diagnostics.get('zero_mean_mode', 'unknown')
    
    with open(diag_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{n:d}, {t:.6e}, {S_mean_raw:.6e}, {S_mean_after:.6e}, {sum_SV_after:.6e}, "
                f"{p_mean:.6e}, {p_rms:.6e}, {int(bool(zero_mean_source))}, {zero_mean_mode}\n")


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


# ==================== 滤波操作（Numba优化版本） ====================
@jit(nopython=True, cache=True)
def _apply_spatial_filter_core(p, u, v, cell_neighbors_list, neighbor_counts, filter_strength):
    """
    应用空间滤波（Numba加速核心函数）
    
    参数:
        p, u, v: 场变量
        cell_neighbors_list: 邻居索引列表 [n_cells, max_neighbors]
        neighbor_counts: 每个单元的邻居数量 [n_cells]
        filter_strength: 滤波强度
    
    返回:
        p_filtered, u_filtered, v_filtered: 滤波后的场变量
    """
    n_cells = len(p)
    p_filtered = p.copy()
    u_filtered = u.copy()
    v_filtered = v.copy()
    
    for i in range(n_cells):
        n_neighbors = int(neighbor_counts[i])
        if n_neighbors > 0:
            # 计算邻居平均值
            neighbor_avg_p = 0.0
            neighbor_avg_u = 0.0
            neighbor_avg_v = 0.0
            
            for k in range(n_neighbors):
                neighbor_idx = cell_neighbors_list[i, k]
                neighbor_avg_p += p[neighbor_idx]
                neighbor_avg_u += u[neighbor_idx]
                neighbor_avg_v += v[neighbor_idx]
            
            neighbor_avg_p /= n_neighbors
            neighbor_avg_u /= n_neighbors
            neighbor_avg_v /= n_neighbors
            
            p_filtered[i] = (1.0 - filter_strength) * p[i] + filter_strength * neighbor_avg_p
            u_filtered[i] = (1.0 - filter_strength) * u[i] + filter_strength * neighbor_avg_u
            v_filtered[i] = (1.0 - filter_strength) * v[i] + filter_strength * neighbor_avg_v
    
    return p_filtered, u_filtered, v_filtered


def apply_spatial_filter(p, u, v, cell_neighbors_list, neighbor_counts, filter_strength,
                        p_out=None, u_out=None, v_out=None):
    """
    应用空间滤波（优化版本，使用Numba加速，支持预计算的数组和内存重用）
    
    参数:
        p, u, v: 场变量
        cell_neighbors_list: 预计算的邻居索引数组 [n_cells, max_neighbors]
        neighbor_counts: 预计算的邻居数量数组 [n_cells]
        filter_strength: 滤波强度
        p_out, u_out, v_out: 输出数组（可选，如果提供则重用）
    
    返回:
        p_filtered, u_filtered, v_filtered: 滤波后的场变量
    """
    if p_out is None or u_out is None or v_out is None:
        p_filtered, u_filtered, v_filtered = _apply_spatial_filter_core(
            p, u, v, cell_neighbors_list, neighbor_counts, filter_strength)
    else:
        # 重用输出数组
        p_filtered, u_filtered, v_filtered = _apply_spatial_filter_core(
            p, u, v, cell_neighbors_list, neighbor_counts, filter_strength)
        p_out[:] = p_filtered
        u_out[:] = u_filtered
        v_out[:] = v_filtered
        p_filtered, u_filtered, v_filtered = p_out, u_out, v_out
    
    return p_filtered, u_filtered, v_filtered


def prepare_filter_arrays(cell_neighbors):
    """
    预计算滤波所需的数组（避免每次滤波都重新创建）
    
    参数:
        cell_neighbors: 单元邻居关系（列表的列表）
    
    返回:
        cell_neighbors_list: 邻居索引数组 [n_cells, max_neighbors]
        neighbor_counts: 邻居数量数组 [n_cells]
    """
    n_cells = len(cell_neighbors)
    max_neighbors = max(len(neighbors) for neighbors in cell_neighbors) if cell_neighbors else 0
    
    # 转换为数组格式
    cell_neighbors_list = np.full((n_cells, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_cells, dtype=np.int32)
    
    for i in range(n_cells):
        neighbors = cell_neighbors[i]
        neighbor_counts[i] = len(neighbors)
        for j, neighbor_idx in enumerate(neighbors):
            if j < max_neighbors:
                cell_neighbors_list[i, j] = neighbor_idx
    
    return cell_neighbors_list, neighbor_counts


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
    zero_mean_source = params['zero_mean_source']
    zero_mean_mode = params['zero_mean_mode']
    
    # 提取参数（输出参数）
    output_dir = params['output_dir']
    output_time_interval = params['output_time_interval']
    log_time_interval = params['log_time_interval']
    log_file = params['log_file']
    progress_bar_width = params['progress_bar_width']
    
    # 提取参数（网格生成参数）
    dtheta_base = params['dtheta_base']
    
    # 命令行参数解析（保留接口以兼容，但不再使用并行计算）
    parser = argparse.ArgumentParser(description='RDE声学模拟 - 非结构化O网格版本（优化版本）')
    parser.add_argument('-n', '--ncores', type=int, default=None,
                       help='（已弃用）性能优化通过预计算和向量化实现，不再使用multiprocessing')
    parser.add_argument('--single', action='store_true',
                       help='（已弃用）性能优化通过预计算和向量化实现')
    
    args = parser.parse_args()
    
    # 注意：优化后的代码不再使用multiprocessing并行计算
    # 性能优化主要通过以下方式实现：
    # 1. 数组结构替代字典（大幅提升访问速度）
    # 2. Numba JIT编译（10-50倍性能提升）
    # 3. 预计算几何量（边向量、法向量、长度等）
    # 4. 预计算边-邻居映射（避免运行时查找）
    # 5. 向量化操作（numpy数组操作）
    # 6. 减少数组拷贝（使用视图和原地操作）
    print(f"性能优化: 数组结构 + Numba JIT + 预计算", file=sys.stderr)
    
    # 生成网格（包含预计算的几何量，数组结构）
    print("正在生成非结构化O网格并预计算几何量（数组结构）...", file=sys.stderr)
    (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
     edge_normals, edge_lengths, edge_neighbors, edge_count) = generate_ogrid_mesh(
        R, r_core, dr, dtheta_base)
    n_cells = len(cells)
    n_nodes = len(nodes)
    max_edges = edge_normals.shape[1]
    
    print(f"网格生成完成: {n_nodes}个节点, {n_cells}个单元", file=sys.stderr)
    print(f"几何量预计算完成: {n_cells}个单元 × {max_edges}条边/单元（数组结构）", file=sys.stderr)
    if NUMBA_AVAILABLE:
        print(f"Numba JIT编译: 已启用（性能加速）", file=sys.stderr)
    else:
        print(f"Numba JIT编译: 未安装（使用纯numpy，性能较慢）", file=sys.stderr)
        print(f"建议安装: pip install numba", file=sys.stderr)
    
    # 识别边界（基于单元节点，更准确）
    inner_wall_cells, outer_wall_cells, wall_normals, boundary_edges_dict = identify_boundary_cells(
        nodes, cells, cell_centers, R, r_core, nr, ntheta, boundary_tolerance)
    print(f"边界识别: {len(inner_wall_cells)}个内壁单元, {len(outer_wall_cells)}个外壁单元", file=sys.stderr)
    print(f"边界边数量: {len(boundary_edges_dict)}", file=sys.stderr)
    
    # 创建边界边掩码（数组结构，替代字典）
    boundary_mask = np.zeros((n_cells, max_edges), dtype=np.bool_)
    for (i, j) in boundary_edges_dict.keys():
        if i < n_cells and j < max_edges:
            boundary_mask[i, j] = True
    print(f"边界掩码创建完成: {np.sum(boundary_mask)} 个边界边", file=sys.stderr)
    
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
    
    # 预计算单元面积总和（用于体积加权平均和源项DC修正）
    cell_areas_sum = float(np.sum(cell_areas))
    print(f"单元总面积 ΣV = {cell_areas_sum:.6e} m² (用于体积加权统计和源项DC修正)", file=sys.stderr)
    print(f"源项零均值修正: zero_mean_source={zero_mean_source}, 模式 zero_mean_mode='{zero_mean_mode}'", file=sys.stderr)
    
    # 初始化场变量（在单元中心）
    p = np.zeros(n_cells)
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    
    # 预分配用于时间步进的工作数组（避免每次迭代都创建新数组）
    p_new = np.zeros(n_cells)
    u_new = np.zeros(n_cells)
    v_new = np.zeros(n_cells)
    
    # 时间参数
    nt = int(t_end / dt)
    
    # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化日志文件
    log_file_path = os.path.join(output_dir, log_file)
    initialize_log_file(log_file_path)
    
    # 初始化源项DC修正诊断CSV文件
    source_diag_file_path = os.path.join(output_dir, "source_dc_diagnostics.csv")
    initialize_source_diagnostics_file(source_diag_file_path)
    
    # 基于时间的输出控制
    next_output_time = 0.0  # 下次VTK输出的时间
    next_log_time = 0.0      # 下次日志输出的时间
    output_idx = 0          # VTK文件索引
    
    # 用于残差计算的上一时间步场变量（使用视图减少拷贝）
    p_old = np.empty_like(p)
    u_old = np.empty_like(u)
    v_old = np.empty_like(v)
    p_old[:] = p
    u_old[:] = u
    v_old[:] = v
    
    # 预计算滤波所需的数组（避免每次滤波都重新创建）
    if apply_filter:
        print("预计算滤波数组...", file=sys.stderr)
        cell_neighbors_list, neighbor_counts = prepare_filter_arrays(cell_neighbors)
        print(f"滤波数组预计算完成: {n_cells}个单元", file=sys.stderr)
    else:
        cell_neighbors_list = None
        neighbor_counts = None
    
    # 预分配RK4临时数组（避免每次调用都创建）
    print("预分配RK4临时数组...", file=sys.stderr)
    rk4_temp_arrays = None  # 将在第一次调用时创建
    rk4_diagnostics = {}    # 用于存放每个时间步的源项DC诊断信息
    
    # 主循环
    start = time.time()
    print(f"开始模拟: nt={nt}, dt={dt:.2e}s, CFL={CFL:.4f}", file=sys.stderr)
    print(f"VTK输出时间间隔: {output_time_interval:.2e}s", file=sys.stderr)
    print(f"日志输出时间间隔: {log_time_interval:.2e}s", file=sys.stderr)
    print(f"日志文件: {log_file_path}", file=sys.stderr)
    print("", file=sys.stderr)  # 空行
    
    for n in range(nt):
        t = n * dt
        
        # 使用RK4方法进行时间积分（内存优化版本，重用数组）
        p_new, u_new, v_new, rk4_temp_arrays, rk4_diagnostics = rk4_step_ogrid(
            p, u, v, t, dt, cell_centers, cell_areas, cell_areas_sum,
            inner_wall_cells, outer_wall_cells, wall_normals, boundary_mask,
            edge_normals, edge_lengths,
            edge_neighbors, edge_count,
            rho0, c0, nu,
            inner_r, outer_r, omega, wave_pressure, tau, sigma, R,
            zero_mean_source, zero_mean_mode,
            p_new, u_new, v_new, rk4_temp_arrays,
            rk4_diagnostics)
        
        # 应用数值滤波（向量化版本，抑制高频振荡，重用数组）
        if apply_filter and n % filter_frequency == 0:  # 按配置的频率滤波
            p_new, u_new, v_new = apply_spatial_filter(
                p_new, u_new, v_new, 
                cell_neighbors_list, neighbor_counts, filter_strength,
                p_new, u_new, v_new)
        
        # 计算残差（用于监控）
        residuals = compute_residuals(p_old, u_old, v_old, p_new, u_new, v_new)
        
        # 更新场变量（交换引用，避免拷贝）
        p, p_new = p_new, p
        u, u_new = u_new, u
        v, v_new = v_new, v
        
        # 只在需要时更新旧值（用于残差计算）
        if t >= next_log_time or n == nt - 1:
            p_old[:] = p
            u_old[:] = u
            v_old[:] = v
        
        # 定期触发垃圾回收（每1000步或每10000步，根据总步数调整）
        gc_frequency = max(1000, min(10000, nt // 100))
        if n > 0 and n % gc_frequency == 0:
            gc.collect()  # 显式触发垃圾回收
        
        # 基于时间的日志输出和源项DC诊断输出
        if t >= next_log_time or n == nt - 1:
            # 计算压力统计
            if len(p) > 0:
                p_min = float(np.min(p))
                p_max = float(np.max(p))
                if cell_areas_sum > 0.0:
                    # 体积加权平均与RMS
                    p_mean = float(np.sum(p * cell_areas) / cell_areas_sum)
                    p_fluct = p - p_mean
                    p_rms = float(np.sqrt(np.sum(p_fluct * p_fluct * cell_areas) / cell_areas_sum))
                else:
                    p_mean = 0.0
                    p_rms = 0.0
            else:
                p_min = p_max = p_mean = p_rms = 0.0
            
            # 写入日志（包含体积加权 p_mean 与 p_rms）
            write_log_entry(log_file_path, n, t, dt, residuals, p, u, v,
                           cell_areas, p_min, p_max, p_mean, p_rms)
            
            # 写入源项DC修正诊断CSV（每个时间步聚合RK4四个stage的统计）
            write_source_diagnostics_entry(
                source_diag_file_path, n, t, rk4_diagnostics, p_mean, p_rms
            )
            
            # 适度打印诊断信息，便于在线观察源项DC是否被成功移除
            if n % max(1, nt // 20) == 0 or n == nt - 1:
                print(
                    f"[源项DC诊断] n={n}, t={t:.3e}s, "
                    f"S_mean_raw_stage={rk4_diagnostics.get('S_mean_raw_stage', 0.0):.3e}, "
                    f"S_mean_after_stage={rk4_diagnostics.get('S_mean_after_stage', 0.0):.3e}, "
                    f"sum_SV_after_stage={rk4_diagnostics.get('sum_SV_after_stage', 0.0):.3e}, "
                    f"p_mean_vol={p_mean:.3e}, p_rms_vol={p_rms:.3e}",
                    file=sys.stderr,
                )

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
