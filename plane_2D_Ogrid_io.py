#!/usr/bin/env python3
"""
RDE O网格 - 输入输出与进度
进度条、残差、日志文件、源项诊断CSV、VTK 输出。
"""

import os
import numpy as np
import meshio


def update_progress_bar(progress, width=50, elapsed_time=None, eta=None):
    filled = int(width * progress / 100)
    bar = '█' * filled + '░' * (width - filled)
    progress_str = f"\r进度: |{bar}| {progress:.1f}%"
    if elapsed_time is not None:
        progress_str += f" | 已用时: {elapsed_time:.1f}s"
    if eta is not None:
        progress_str += f" | 预计剩余: {eta:.1f}s"
    return progress_str


def compute_residuals(p_old, u_old, v_old, p_new, u_new, v_new):
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
    if p_min is None or p_max is None or p_mean is None or p_rms is None:
        if len(p) > 0:
            p_min = float(np.min(p)) if p_min is None else p_min
            p_max = float(np.max(p)) if p_max is None else p_max
            sumV = float(np.sum(cell_areas))
            if sumV > 0.0:
                p_mean = float(np.sum(p * cell_areas) / sumV)
                p_fluct = p - p_mean
                p_rms = float(np.sqrt(np.sum(p_fluct * p_fluct * cell_areas) / sumV))
            else:
                p_mean = 0.0 if p_mean is None else p_mean
                p_rms = 0.0 if p_rms is None else p_rms
        else:
            p_min = p_max = p_mean = p_rms = 0.0
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{n:10d}  {t:.6e}  {dt:.6e}  "
                f"{residuals['pressure']:.6e}  {residuals['velocity_u']:.6e}  "
                f"{residuals['velocity_v']:.6e}  {residuals['total']:.6e}  "
                f"{p_min:.6e}  {p_max:.6e}  {p_mean:.6e}  {p_rms:.6e}\n")


def initialize_log_file(log_file_path):
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
    with open(diag_file_path, 'w', encoding='utf-8') as f:
        f.write("# 源项DC零均值修正诊断数据\n")
        f.write("# n, t, S_mean_raw_stage, S_mean_after_stage, sum_SV_after_stage, ")
        f.write("p_mean_vol, p_rms_vol, zero_mean_source, zero_mean_mode\n")


def write_source_diagnostics_entry(diag_file_path, n, t, diagnostics, p_mean, p_rms):
    S_mean_raw = diagnostics.get('S_mean_raw_stage', 0.0)
    S_mean_after = diagnostics.get('S_mean_after_stage', 0.0)
    sum_SV_after = diagnostics.get('sum_SV_after_stage', 0.0)
    zero_mean_source = diagnostics.get('zero_mean_source', False)
    zero_mean_mode = diagnostics.get('zero_mean_mode', 'unknown')
    with open(diag_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{n:d}, {t:.6e}, {S_mean_raw:.6e}, {S_mean_after:.6e}, {sum_SV_after:.6e}, "
                f"{p_mean:.6e}, {p_rms:.6e}, {int(bool(zero_mean_source))}, {zero_mean_mode}\n")


def write_vtk_output(nodes, cells, p, u, v, output_dir, output_idx):
    fname = f"{output_dir}/acoustic_ogrid_{output_idx:04d}.vtk"
    n_nodes = len(nodes)
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
    valid_nodes = node_cell_count > 0
    p_nodes[valid_nodes] /= node_cell_count[valid_nodes]
    u_nodes[valid_nodes] /= node_cell_count[valid_nodes]
    v_nodes[valid_nodes] /= node_cell_count[valid_nodes]
    velocity_magnitude_nodes = np.sqrt(u_nodes**2 + v_nodes**2)
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
