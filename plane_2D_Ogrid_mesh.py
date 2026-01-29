#!/usr/bin/env python3
"""
RDE O网格 - 网格生成
生成非结构化O网格（环形网格）及预计算几何量、边-邻居映射。
支持将网格保存到 output_ogrid/ogrid_mesh.npz，下次参数一致时直接加载以节省时间。
"""

import os
import numpy as np

# 网格文件保存在 output_dir 下的文件名
MESH_FILENAME = "ogrid_mesh.npz"
# 网格参数比对容差（浮点）
_MESH_PARAM_TOL = 1e-12


def get_mesh_path(output_dir):
    """返回网格文件的完整路径。"""
    return os.path.join(output_dir, MESH_FILENAME)


def mesh_params_match(R, r_core, dr, dtheta_base,
                      loaded_R, loaded_r_core, loaded_dr, loaded_dtheta_base,
                      tol=_MESH_PARAM_TOL):
    """判断当前网格参数与文件中保存的参数是否一致（在容差内）。"""
    return (abs(R - loaded_R) <= tol and abs(r_core - loaded_r_core) <= tol
            and abs(dr - loaded_dr) <= tol and abs(dtheta_base - loaded_dtheta_base) <= tol)


def load_ogrid_mesh(output_dir):
    """
    从 output_dir 下的 ogrid_mesh.npz 加载网格。
    返回:
        (params, mesh_tuple) 其中 params = (R, r_core, dr, dtheta_base)，
        mesh_tuple = (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
                      edge_normals, edge_lengths, edge_neighbors, edge_count)
        若文件不存在或读取失败则返回 (None, None)。
    """
    path = get_mesh_path(output_dir)
    if not os.path.isfile(path):
        return None, None
    try:
        data = np.load(path, allow_pickle=False)
        R = float(data["R"])
        r_core = float(data["r_core"])
        dr = float(data["dr"])
        dtheta_base = float(data["dtheta_base"])
        nr = int(data["nr"])
        ntheta = int(data["ntheta"])
        nodes = np.asarray(data["nodes"])
        cells = np.asarray(data["cells"])
        cell_centers = np.asarray(data["cell_centers"])
        cell_areas = np.asarray(data["cell_areas"])
        cn_arr = np.asarray(data["cell_neighbors_arr"], dtype=np.int32)
        edge_normals = np.asarray(data["edge_normals"])
        edge_lengths = np.asarray(data["edge_lengths"])
        edge_neighbors = np.asarray(data["edge_neighbors"], dtype=np.int32)
        edge_count = np.asarray(data["edge_count"], dtype=np.int32)
        n_cells = len(cells)
        cell_neighbors = []
        for i in range(n_cells):
            row = cn_arr[i]
            neighbors = [int(row[j]) for j in range(len(row)) if row[j] >= 0]
            cell_neighbors.append(neighbors)
        params = (R, r_core, dr, dtheta_base)
        mesh = (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
                edge_normals, edge_lengths, edge_neighbors, edge_count)
        return params, mesh
    except Exception:
        return None, None


def save_ogrid_mesh(output_dir, R, r_core, dr, dtheta_base, mesh_tuple):
    """
    将网格写入 output_dir/ogrid_mesh.npz，若目录不存在会创建；若已有文件则覆盖。
    mesh_tuple = (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
                  edge_normals, edge_lengths, edge_neighbors, edge_count)
    返回写入的完整路径。
    """
    (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
     edge_normals, edge_lengths, edge_neighbors, edge_count) = mesh_tuple
    n_cells = len(cells)
    max_n = max(len(x) for x in cell_neighbors) if cell_neighbors else 0
    cn_arr = np.full((n_cells, max_n), -1, dtype=np.int32)
    for i in range(n_cells):
        for j, nb in enumerate(cell_neighbors[i]):
            if j < max_n:
                cn_arr[i, j] = nb
    os.makedirs(output_dir, exist_ok=True)
    path = get_mesh_path(output_dir)
    np.savez_compressed(
        path,
        R=np.float64(R),
        r_core=np.float64(r_core),
        dr=np.float64(dr),
        dtheta_base=np.float64(dtheta_base),
        nr=np.int32(nr),
        ntheta=np.int32(ntheta),
        nodes=nodes,
        cells=cells,
        cell_centers=cell_centers,
        cell_areas=cell_areas,
        cell_neighbors_arr=cn_arr,
        edge_normals=edge_normals,
        edge_lengths=edge_lengths,
        edge_neighbors=edge_neighbors,
        edge_count=edge_count,
    )
    return path


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
        nr, ntheta: 径向/角度层数
        edge_normals, edge_lengths, edge_neighbors, edge_count: 边几何与邻居
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
        p0 = nodes[cell[0], :2]
        p1 = nodes[cell[1], :2]
        p2 = nodes[cell[2], :2]
        p3 = nodes[cell[3], :2]
        cell_centers[idx] = (p0 + p1 + p2 + p3) / 4.0
        v1 = p1 - p0
        v2 = p3 - p0
        area1 = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        v3 = p2 - p1
        v4 = p3 - p1
        area2 = 0.5 * abs(v3[0]*v4[1] - v3[1]*v4[0])
        cell_areas[idx] = area1 + area2

    # 构建单元邻居关系（共享边的单元，包括周期性边界）
    cell_neighbors = [[] for _ in range(n_cells)]
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            shared_nodes = set(cells[i]) & set(cells[j])
            if len(shared_nodes) >= 2:
                cell_neighbors[i].append(j)
                cell_neighbors[j].append(i)

    # 预计算几何量和边-邻居映射（数组结构版本）
    max_edges_per_cell = 4
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
            edge_vec = node1 - node0
            edge_length = np.linalg.norm(edge_vec)

            if edge_length > 1e-10:
                n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
                edge_center = (node0 + node1) / 2.0
                center_to_edge = edge_center - cell_centers[i]
                if np.dot(n, center_to_edge) < 0:
                    n = -n
                neighbor_idx = -1
                for neighbor in cell_neighbors[i]:
                    neighbor_nodes = set(cells[neighbor])
                    if node0_idx in neighbor_nodes and node1_idx in neighbor_nodes:
                        neighbor_idx = neighbor
                        break
                edge_normals[i, j] = n
                edge_lengths[i, j] = edge_length
                edge_neighbors[i, j] = neighbor_idx

    return (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
            edge_normals, edge_lengths, edge_neighbors, edge_count)
