#!/usr/bin/env python3
"""
RDE O网格 - RK4 时间积分
四阶 Runge-Kutta 时间步进，含源项 DC 零均值修正与边界条件应用。
"""

import numpy as np

from plane_2D_Ogrid_fvm import (
    compute_gradient_fvm,
    compute_divergence_fvm,
    compute_laplacian_fvm,
)
from plane_2D_Ogrid_source import source_term_ogrid
from plane_2D_Ogrid_boundary import apply_wall_boundary_conditions_ogrid


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
    RK4 时间积分（支持内存重用与源项 DC 修正）。
    返回: p_new, u_new, v_new, temp_arrays, diagnostics
    """
    n_cells = len(p)

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

    if p_new is None:
        p_new = np.zeros(n_cells)
    if u_new is None:
        u_new = np.zeros(n_cells)
    if v_new is None:
        v_new = np.zeros(n_cells)

    if diagnostics is None:
        diagnostics = {}
    diagnostics_step = {
        'S_mean_raw_sum': 0.0,
        'S_mean_after_sum': 0.0,
        'sum_SV_after_sum': 0.0,
        'n_stages': 0,
    }

    def rhs(p_state, u_state, v_state, t_state, dp_dt_out, du_dt_out, dv_dt_out):
        S = source_term_ogrid(cell_centers, t_state, inner_r, outer_r, omega,
                              wave_pressure, tau, sigma, R)
        if cell_areas_sum > 0.0:
            S_mean_raw = np.sum(S * cell_areas) / cell_areas_sum
        else:
            S_mean_raw = 0.0
        diagnostics_step['S_mean_raw_sum'] += S_mean_raw
        diagnostics_step['n_stages'] += 1
        if zero_mean_source and zero_mean_mode == "stage":
            S -= S_mean_raw
            if cell_areas_sum > 0.0:
                S_mean_after = np.sum(S * cell_areas) / cell_areas_sum
                sum_SV_after = np.sum(S * cell_areas)
            else:
                S_mean_after = 0.0
                sum_SV_after = 0.0
        else:
            if cell_areas_sum > 0.0:
                S_mean_after = S_mean_raw
                sum_SV_after = np.sum(S * cell_areas)
            else:
                S_mean_after = 0.0
                sum_SV_after = 0.0
        diagnostics_step['S_mean_after_sum'] += S_mean_after
        diagnostics_step['sum_SV_after_sum'] += sum_SV_after

        div_uv = compute_divergence_fvm(u_state, v_state, cell_areas,
                                        edge_normals, edge_lengths,
                                        edge_neighbors, edge_count, boundary_mask)
        grad_p = compute_gradient_fvm(p_state, cell_areas,
                                     edge_normals, edge_lengths,
                                     edge_neighbors, edge_count, boundary_mask)
        dp_dt_out[:] = -rho0 * c0**2 * div_uv + S
        laplacian_u = compute_laplacian_fvm(u_state, cell_areas,
                                             edge_normals, edge_lengths,
                                             edge_neighbors, edge_count, boundary_mask)
        laplacian_v = compute_laplacian_fvm(v_state, cell_areas,
                                             edge_normals, edge_lengths,
                                             edge_neighbors, edge_count, boundary_mask)
        du_dt_out[:] = -grad_p[:, 0] / rho0 + nu * laplacian_u
        dv_dt_out[:] = -grad_p[:, 1] / rho0 + nu * laplacian_v

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

    p_new[:] = p + (dt/6.0) * (temp_arrays['k1_p'] + 2*temp_arrays['k2_p'] +
                               2*temp_arrays['k3_p'] + temp_arrays['k4_p'])
    u_new[:] = u + (dt/6.0) * (temp_arrays['k1_u'] + 2*temp_arrays['k2_u'] +
                               2*temp_arrays['k3_u'] + temp_arrays['k4_u'])
    v_new[:] = v + (dt/6.0) * (temp_arrays['k1_v'] + 2*temp_arrays['k2_v'] +
                               2*temp_arrays['k3_v'] + temp_arrays['k4_v'])

    u_new, v_new = apply_wall_boundary_conditions_ogrid(u_new, v_new,
                                                       inner_wall_cells, outer_wall_cells, wall_normals,
                                                       u_new, v_new)

    if zero_mean_source and zero_mean_mode == "step" and diagnostics_step['n_stages'] > 0:
        S_mean_step = diagnostics_step['S_mean_raw_sum'] / diagnostics_step['n_stages']
        delta_p_dc = S_mean_step * dt
        p_new[:] -= delta_p_dc

    if diagnostics_step['n_stages'] > 0:
        n_stages = diagnostics_step['n_stages']
        S_mean_raw_stage = diagnostics_step['S_mean_raw_sum'] / n_stages
        S_mean_after_stage = diagnostics_step['S_mean_after_sum'] / n_stages
        sum_SV_after_stage = diagnostics_step['sum_SV_after_sum'] / n_stages
    else:
        S_mean_raw_stage = 0.0
        S_mean_after_stage = 0.0
        sum_SV_after_stage = 0.0

    if zero_mean_mode == "stage":
        diagnostics['S_mean_raw_stage'] = S_mean_raw_stage
        diagnostics['S_mean_after_stage'] = S_mean_after_stage
        diagnostics['sum_SV_after_stage'] = sum_SV_after_stage
    elif zero_mean_mode == "step":
        diagnostics['S_mean_raw_stage'] = S_mean_raw_stage
        diagnostics['S_mean_after_stage'] = 0.0
        diagnostics['sum_SV_after_stage'] = 0.0
    else:
        diagnostics['S_mean_raw_stage'] = S_mean_raw_stage
        diagnostics['S_mean_after_stage'] = S_mean_after_stage
        diagnostics['sum_SV_after_stage'] = sum_SV_after_stage
    diagnostics['zero_mean_source'] = bool(zero_mean_source)
    diagnostics['zero_mean_mode'] = zero_mean_mode

    return p_new, u_new, v_new, temp_arrays, diagnostics
