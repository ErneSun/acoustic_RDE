#!/usr/bin/env python3
"""
RDE O网格 - 有限体积方法算子
梯度、散度、拉普拉斯（Numba 优化版本，支持多线程 prange）。
"""

import numpy as np

try:
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    prange = range
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True, parallel=True)
def _compute_gradient_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                edge_neighbors, edge_count, boundary_mask):
    n_cells = len(f)
    grad_f = np.zeros((n_cells, 2))
    for i in prange(n_cells):
        grad_sum = np.zeros(2)
        n_edges = int(edge_count[i])
        for j in range(n_edges):
            if edge_lengths[i, j] > 1e-10:
                n = edge_normals[i, j]
                edge_length = edge_lengths[i, j]
                if boundary_mask[i, j] or edge_neighbors[i, j] < 0:
                    f_face = f[i]
                else:
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
    if boundary_mask is None:
        boundary_mask = np.zeros((len(f), edge_normals.shape[1]), dtype=np.bool_)
    return _compute_gradient_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                       edge_neighbors, edge_count, boundary_mask)


@jit(nopython=True, cache=True, parallel=True)
def _compute_divergence_fvm_core(u, v, cell_areas, edge_normals, edge_lengths,
                                  edge_neighbors, edge_count, boundary_mask):
    n_cells = len(u)
    div = np.zeros(n_cells)
    for i in prange(n_cells):
        div_sum = 0.0
        n_edges = int(edge_count[i])
        for j in range(n_edges):
            if edge_lengths[i, j] > 1e-10:
                n = edge_normals[i, j]
                edge_length = edge_lengths[i, j]
                if boundary_mask[i, j] or edge_neighbors[i, j] < 0:
                    u_face = u[i]
                    v_face = v[i]
                else:
                    neighbor_idx = edge_neighbors[i, j]
                    u_face = 0.5 * (u[i] + u[neighbor_idx])
                    v_face = 0.5 * (v[i] + v[neighbor_idx])
                div_sum += (u_face * n[0] + v_face * n[1]) * edge_length
        if cell_areas[i] > 1e-10:
            div[i] = div_sum / cell_areas[i]
    return div


def compute_divergence_fvm(u, v, cell_areas, edge_normals, edge_lengths,
                           edge_neighbors, edge_count, boundary_mask=None):
    if boundary_mask is None:
        boundary_mask = np.zeros((len(u), edge_normals.shape[1]), dtype=np.bool_)
    return _compute_divergence_fvm_core(u, v, cell_areas, edge_normals, edge_lengths,
                                        edge_neighbors, edge_count, boundary_mask)


@jit(nopython=True, cache=True, parallel=True)
def _compute_laplacian_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                 edge_neighbors, edge_count, boundary_mask):
    n_cells = len(f)
    laplacian_f = np.zeros(n_cells)
    for i in prange(n_cells):
        div_sum = 0.0
        n_edges = int(edge_count[i])
        for j in range(n_edges):
            if edge_lengths[i, j] > 1e-10:
                edge_length = edge_lengths[i, j]
                if boundary_mask[i, j] or edge_neighbors[i, j] < 0:
                    grad_face_n = 0.0
                else:
                    neighbor_idx = edge_neighbors[i, j]
                    if neighbor_idx >= 0:
                        df_dn = f[neighbor_idx] - f[i]
                        grad_face_n = df_dn
                    else:
                        grad_face_n = 0.0
                div_sum += grad_face_n * edge_length
        if cell_areas[i] > 1e-10:
            laplacian_f[i] = div_sum / cell_areas[i]
    return laplacian_f


def compute_laplacian_fvm(f, cell_areas, edge_normals, edge_lengths,
                          edge_neighbors, edge_count, boundary_mask=None):
    if boundary_mask is None:
        boundary_mask = np.zeros((len(f), edge_normals.shape[1]), dtype=np.bool_)
    return _compute_laplacian_fvm_core(f, cell_areas, edge_normals, edge_lengths,
                                       edge_neighbors, edge_count, boundary_mask)
