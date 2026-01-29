#!/usr/bin/env python3
"""
RDE O网格 - 边界条件
识别内外壁边界单元与边界边，并应用硬壁边界条件（法向速度为零）。
"""

import numpy as np


def identify_boundary_cells(nodes, cells, cell_centers, R, r_core, nr, ntheta, tolerance):
    """
    识别边界单元（基于单元是否在边界上）及边界边。

    返回:
        inner_wall_cells, outer_wall_cells: 内/外壁单元索引
        wall_normals: 壁面法向量 [n_wall_cells, 2]
        boundary_edges: {(cell_idx, edge_idx): {'type': 'inner'/'outer', 'normal': n}}
    """
    n_cells = len(cell_centers)
    inner_wall_cells = []
    outer_wall_cells = []
    wall_normals = []
    boundary_edges = {}

    for i in range(n_cells):
        cell_nodes = cells[i]
        x_center, y_center = cell_centers[i]
        r_center = np.sqrt(x_center**2 + y_center**2)
        is_inner_boundary = False
        is_outer_boundary = False
        for node_idx in cell_nodes:
            x_node, y_node = nodes[node_idx, :2]
            r_node = np.sqrt(x_node**2 + y_node**2)
            if abs(r_node - r_core) < tolerance:
                is_inner_boundary = True
            if abs(r_node - R) < tolerance:
                is_outer_boundary = True
        if is_inner_boundary:
            inner_wall_cells.append(i)
            if r_center > 1e-10:
                n = np.array([-x_center/r_center, -y_center/r_center])
            else:
                n = np.array([0.0, 0.0])
            n_mag = np.linalg.norm(n)
            if n_mag > 1e-10:
                n = n / n_mag
            wall_normals.append(n)
        elif is_outer_boundary:
            outer_wall_cells.append(i)
            if r_center > 1e-10:
                n = np.array([x_center/r_center, y_center/r_center])
            else:
                n = np.array([0.0, 0.0])
            n_mag = np.linalg.norm(n)
            if n_mag > 1e-10:
                n = n / n_mag
            wall_normals.append(n)

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
            edge_on_inner = (abs(r0 - r_core) < tolerance) and (abs(r1 - r_core) < tolerance)
            edge_on_outer = (abs(r0 - R) < tolerance) and (abs(r1 - R) < tolerance)
            if edge_on_inner or edge_on_outer:
                edge_vec = np.array([x1-x0, y1-y0])
                edge_length = np.linalg.norm(edge_vec)
                if edge_length > 1e-10:
                    n = np.array([-edge_vec[1], edge_vec[0]]) / edge_length
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
    """应用硬壁边界条件：壁面处法向速度为零（支持原地操作）。"""
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
            u_normal = u[cell_idx] * n[0] + v[cell_idx] * n[1]
            u_new[cell_idx] = u[cell_idx] - u_normal * n[0]
            v_new[cell_idx] = v[cell_idx] - u_normal * n[1]
    return u_new, v_new
