#!/usr/bin/env python3
"""
RDE O网格 - 空间滤波
基于邻居平均的空间滤波（Numba 优化版本）。
"""

import numpy as np

try:
    from numba import jit, prange
except ImportError:
    prange = range
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True, parallel=True)
def _apply_spatial_filter_core(p, u, v, cell_neighbors_list, neighbor_counts, filter_strength):
    n_cells = len(p)
    p_filtered = p.copy()
    u_filtered = u.copy()
    v_filtered = v.copy()
    for i in prange(n_cells):
        n_neighbors = int(neighbor_counts[i])
        if n_neighbors > 0:
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
    if p_out is None or u_out is None or v_out is None:
        p_filtered, u_filtered, v_filtered = _apply_spatial_filter_core(
            p, u, v, cell_neighbors_list, neighbor_counts, filter_strength)
    else:
        p_filtered, u_filtered, v_filtered = _apply_spatial_filter_core(
            p, u, v, cell_neighbors_list, neighbor_counts, filter_strength)
        p_out[:] = p_filtered
        u_out[:] = u_filtered
        v_out[:] = v_filtered
        p_filtered, u_filtered, v_filtered = p_out, u_out, v_out
    return p_filtered, u_filtered, v_filtered


def prepare_filter_arrays(cell_neighbors):
    """预计算滤波所需的邻居数组。返回 cell_neighbors_list, neighbor_counts。"""
    n_cells = len(cell_neighbors)
    max_neighbors = max(len(neighbors) for neighbors in cell_neighbors) if cell_neighbors else 0
    cell_neighbors_list = np.full((n_cells, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_cells):
        neighbors = cell_neighbors[i]
        neighbor_counts[i] = len(neighbors)
        for j, neighbor_idx in enumerate(neighbors):
            if j < max_neighbors:
                cell_neighbors_list[i, j] = neighbor_idx
    return cell_neighbors_list, neighbor_counts
