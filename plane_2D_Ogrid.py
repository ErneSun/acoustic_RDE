#!/usr/bin/env python3
"""
RDE声学模拟 - 使用非结构化O网格版本（聚合模块）
基于 plane_2D_Ogrid_XXX 子模块，统一导出接口并实现 main 入口。
执行顺序：参数加载 → 网格生成 → 网格检查（失败则退出）→ 边界/CFL/初始化 → 时间步进。
"""

import numpy as np
import os
import sys
import time
import argparse
import multiprocessing as mp
import gc

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from control import control, validate_params

# 子模块：按功能拆分，便于扩展
from plane_2D_Ogrid_mesh import (
    generate_ogrid_mesh,
    load_ogrid_mesh,
    save_ogrid_mesh,
    mesh_params_match,
    get_mesh_path,
)
from plane_2D_Ogrid_check import check_ogrid_mesh
from plane_2D_Ogrid_boundary import identify_boundary_cells, apply_wall_boundary_conditions_ogrid
from plane_2D_Ogrid_rk4 import rk4_step_ogrid
from plane_2D_Ogrid_filter import apply_spatial_filter, prepare_filter_arrays
from plane_2D_Ogrid_io import (
    update_progress_bar,
    compute_residuals,
    write_log_entry,
    initialize_log_file,
    initialize_source_diagnostics_file,
    write_source_diagnostics_entry,
    write_vtk_output,
)

# 对外统一导出（兼容 ogrid_preview_gui 及直接 import plane_2D_Ogrid 的代码）
__all__ = [
    "generate_ogrid_mesh",
    "identify_boundary_cells",
    "apply_wall_boundary_conditions_ogrid",
    "rk4_step_ogrid",
    "apply_spatial_filter",
    "prepare_filter_arrays",
    "update_progress_bar",
    "compute_residuals",
    "write_log_entry",
    "initialize_log_file",
    "initialize_source_diagnostics_file",
    "write_source_diagnostics_entry",
    "write_vtk_output",
    "main",
]


def main():
    try:
        params = control()
    except Exception as e:
        print("错误：无法从control模块获取参数", file=sys.stderr)
        print(f"详细错误：{e}", file=sys.stderr)
        print("请检查control.py文件是否正确配置", file=sys.stderr)
        sys.exit(1)

    is_valid, missing_params, invalid_params = validate_params(params)
    if not is_valid:
        print("=" * 60, file=sys.stderr)
        print("错误：参数验证失败！", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        if missing_params:
            print(f"\n缺少的参数数量: {len(missing_params)}", file=sys.stderr)
            print("\n缺少的参数列表:", file=sys.stderr)
            for i, (param, description) in enumerate(missing_params, 1):
                print(f"  {i}. {param} - {description}", file=sys.stderr)
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

    rho0 = params['rho0']
    c0 = params['c0']
    nu = params['nu']
    R = params['R']
    r_core = params['r_core']
    dr = params['dr']
    dt = params['dt']
    t_end = params['t_end']
    wave_pressure = params['wave_pressure']
    inner_r = params['inner_r']
    outer_r = params['outer_r']
    omega = params['omega']
    rotation_clockwise = params['rotation_clockwise']
    omega_eff = -omega if rotation_clockwise else omega
    sigma = params['sigma']
    tau = params['tau']
    filter_strength = params['filter_strength']
    apply_filter = params['apply_filter']
    filter_frequency = params['filter_frequency']
    CFL_max = params['CFL_max']
    boundary_tolerance = params['boundary_tolerance']
    zero_mean_source = params['zero_mean_source']
    zero_mean_mode = params['zero_mean_mode']
    output_dir = params['output_dir']
    output_time_interval = params['output_time_interval']
    log_time_interval = params['log_time_interval']
    log_file = params['log_file']
    progress_bar_width = params['progress_bar_width']
    dtheta_base = params['dtheta_base']
    use_parallel = params['use_parallel']
    parallel_min_cells = params['parallel_min_cells']
    n_cores_param = params['n_cores']

    parser = argparse.ArgumentParser(description='RDE声学模拟 - 非结构化O网格版本（优化版本）')
    parser.add_argument('-n', '--ncores', type=int, default=None, metavar='N',
                        help='指定并行线程数；指定后即启用多核并行，例如: -n 8 或 --ncores 4')
    parser.add_argument('--single', action='store_true',
                        help='强制单核运行（忽略 control.py 中的 use_parallel）')
    args = parser.parse_args()

    print("性能优化: 数组结构 + Numba JIT + 预计算", file=sys.stderr)
    loaded_params, loaded_mesh = load_ogrid_mesh(output_dir)
    if (loaded_params is not None and loaded_mesh is not None
            and mesh_params_match(R, r_core, dr, dtheta_base, *loaded_params)):
        (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
         edge_normals, edge_lengths, edge_neighbors, edge_count) = loaded_mesh
        n_cells = len(cells)
        n_nodes = len(nodes)
        max_edges = edge_normals.shape[1]
        print(f"从文件加载网格: {get_mesh_path(output_dir)}", file=sys.stderr)
        print(f"网格: {n_nodes}个节点, {n_cells}个单元", file=sys.stderr)
    else:
        if loaded_params is not None:
            print("网格参数与已保存文件不一致，正在重新生成...", file=sys.stderr)
        else:
            print("正在生成非结构化O网格并预计算几何量（数组结构）...", file=sys.stderr)
        (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
         edge_normals, edge_lengths, edge_neighbors, edge_count) = generate_ogrid_mesh(
            R, r_core, dr, dtheta_base)
        n_cells = len(cells)
        n_nodes = len(nodes)
        max_edges = edge_normals.shape[1]
        mesh_path = save_ogrid_mesh(output_dir, R, r_core, dr, dtheta_base,
                                    (nodes, cells, cell_centers, cell_areas, cell_neighbors,
                                     nr, ntheta, edge_normals, edge_lengths, edge_neighbors, edge_count))
        print(f"网格生成完成并已保存: {mesh_path}", file=sys.stderr)
    print(f"几何量: {n_cells}个单元 × {max_edges}条边/单元（数组结构）", file=sys.stderr)

    # 网格检查：若错误则在命令行反馈并直接退出
    ok, check_messages = check_ogrid_mesh(
        nodes, cells, cell_centers, cell_areas, nr, ntheta,
        edge_normals, edge_lengths, edge_neighbors, edge_count)
    if not ok:
        print("=" * 60, file=sys.stderr)
        print("网格检查失败，程序退出。", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for msg in check_messages:
            print(msg, file=sys.stderr)
        sys.exit(1)
    for msg in check_messages:
        if msg.strip():
            print(msg, file=sys.stderr)

    if NUMBA_AVAILABLE:
        print("Numba JIT编译: 已启用（性能加速）", file=sys.stderr)
        if args.single:
            enable_parallel = False
            n_threads = 1
        elif args.ncores is not None:
            enable_parallel = True
            n_threads = max(1, min(args.ncores, 256))
        else:
            enable_parallel = use_parallel and n_cells >= parallel_min_cells
            n_threads = (n_cores_param if n_cores_param is not None else mp.cpu_count()) if enable_parallel else 1
            n_threads = max(1, min(n_threads, 256))
        numba.config.NUMBA_NUM_THREADS = n_threads
        if enable_parallel:
            print(f"多核并行: 已启用，线程数 = {n_threads}", file=sys.stderr)
        else:
            print("多核并行: 未启用，单核运行", file=sys.stderr)
            if n_cells > parallel_min_cells:
                print(f"提示: 网格数 ({n_cells}) 超过 {parallel_min_cells}，建议启用并行以加速。"
                      "可在 control.py 中设置 use_parallel=True，或运行时使用 -n N 指定线程数，例如: python plane_2D_Ogrid.py -n 8",
                      file=sys.stderr)
    else:
        print("Numba JIT编译: 未安装（使用纯numpy，性能较慢）", file=sys.stderr)
        print("建议安装: pip install numba", file=sys.stderr)

    inner_wall_cells, outer_wall_cells, wall_normals, boundary_edges_dict = identify_boundary_cells(
        nodes, cells, cell_centers, R, r_core, nr, ntheta, boundary_tolerance)
    print(f"边界识别: {len(inner_wall_cells)}个内壁单元, {len(outer_wall_cells)}个外壁单元", file=sys.stderr)
    print(f"边界边数量: {len(boundary_edges_dict)}", file=sys.stderr)

    boundary_mask = np.zeros((n_cells, max_edges), dtype=np.bool_)
    for (i, j) in boundary_edges_dict.keys():
        if i < n_cells and j < max_edges:
            boundary_mask[i, j] = True
    print(f"边界掩码创建完成: {np.sum(boundary_mask)} 个边界边", file=sys.stderr)

    min_cell_size = np.sqrt(np.min(cell_areas))
    mean_cell_size = np.sqrt(np.mean(cell_areas))
    characteristic_size = min(min_cell_size, mean_cell_size * 0.8)
    CFL = c0 * dt / characteristic_size
    if CFL > CFL_max:
        print(f"警告: CFL数 = {CFL:.4f} > {CFL_max:.2f}，可能导致数值不稳定！", file=sys.stderr)
        print(f"建议: 减小时间步长 dt < {CFL_max * characteristic_size / c0:.2e} s", file=sys.stderr)
    else:
        print(f"CFL数 = {CFL:.4f} < {CFL_max:.2f}，稳定性条件满足", file=sys.stderr)
    print(f"特征尺寸: min={min_cell_size:.6e}m, mean={mean_cell_size:.6e}m, 使用={characteristic_size:.6e}m", file=sys.stderr)

    cell_areas_sum = float(np.sum(cell_areas))
    print(f"单元总面积 ΣV = {cell_areas_sum:.6e} m² (用于体积加权统计和源项DC修正)", file=sys.stderr)
    print(f"源项零均值修正: zero_mean_source={zero_mean_source}, 模式 zero_mean_mode='{zero_mean_mode}'", file=sys.stderr)

    p = np.zeros(n_cells)
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    p_new = np.zeros(n_cells)
    u_new = np.zeros(n_cells)
    v_new = np.zeros(n_cells)
    nt = int(t_end / dt)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_file)
    initialize_log_file(log_file_path)
    source_diag_file_path = os.path.join(output_dir, "source_dc_diagnostics.csv")
    initialize_source_diagnostics_file(source_diag_file_path)
    next_output_time = 0.0
    next_log_time = 0.0
    output_idx = 0
    p_old = np.empty_like(p)
    u_old = np.empty_like(u)
    v_old = np.empty_like(v)
    p_old[:] = p
    u_old[:] = u
    v_old[:] = v

    if apply_filter:
        print("预计算滤波数组...", file=sys.stderr)
        cell_neighbors_list, neighbor_counts = prepare_filter_arrays(cell_neighbors)
        print(f"滤波数组预计算完成: {n_cells}个单元", file=sys.stderr)
    else:
        cell_neighbors_list = None
        neighbor_counts = None

    print("预分配RK4临时数组...", file=sys.stderr)
    rk4_temp_arrays = None
    rk4_diagnostics = {}

    start = time.time()
    print(f"开始模拟: nt={nt}, dt={dt:.2e}s, CFL={CFL:.4f}", file=sys.stderr)
    print(f"VTK输出时间间隔: {output_time_interval:.2e}s", file=sys.stderr)
    print(f"日志输出时间间隔: {log_time_interval:.2e}s", file=sys.stderr)
    print(f"日志文件: {log_file_path}", file=sys.stderr)
    print("", file=sys.stderr)

    for n in range(nt):
        t = n * dt
        p_new, u_new, v_new, rk4_temp_arrays, rk4_diagnostics = rk4_step_ogrid(
            p, u, v, t, dt, cell_centers, cell_areas, cell_areas_sum,
            inner_wall_cells, outer_wall_cells, wall_normals, boundary_mask,
            edge_normals, edge_lengths,
            edge_neighbors, edge_count,
            rho0, c0, nu,
            inner_r, outer_r, omega_eff, wave_pressure, tau, sigma, R,
            zero_mean_source, zero_mean_mode,
            p_new, u_new, v_new, rk4_temp_arrays,
            rk4_diagnostics)

        if apply_filter and n % filter_frequency == 0:
            p_new, u_new, v_new = apply_spatial_filter(
                p_new, u_new, v_new,
                cell_neighbors_list, neighbor_counts, filter_strength,
                p_new, u_new, v_new)

        residuals = compute_residuals(p_old, u_old, v_old, p_new, u_new, v_new)
        p, p_new = p_new, p
        u, u_new = u_new, u
        v, v_new = v_new, v
        if t >= next_log_time or n == nt - 1:
            p_old[:] = p
            u_old[:] = u
            v_old[:] = v

        gc_frequency = max(1000, min(10000, nt // 100))
        if n > 0 and n % gc_frequency == 0:
            gc.collect()

        if t >= next_log_time or n == nt - 1:
            if len(p) > 0:
                p_min = float(np.min(p))
                p_max = float(np.max(p))
                if cell_areas_sum > 0.0:
                    p_mean = float(np.sum(p * cell_areas) / cell_areas_sum)
                    p_fluct = p - p_mean
                    p_rms = float(np.sqrt(np.sum(p_fluct * p_fluct * cell_areas) / cell_areas_sum))
                else:
                    p_mean = 0.0
                    p_rms = 0.0
            else:
                p_min = p_max = p_mean = p_rms = 0.0
            write_log_entry(log_file_path, n, t, dt, residuals, p, u, v,
                           cell_areas, p_min, p_max, p_mean, p_rms)
            write_source_diagnostics_entry(
                source_diag_file_path, n, t, rk4_diagnostics, p_mean, p_rms
            )
            if n % max(1, nt // 20) == 0 or n == nt - 1:
                print(
                    f"[源项DC诊断] n={n}, t={t:.3e}s, "
                    f"S_mean_raw_stage={rk4_diagnostics.get('S_mean_raw_stage', 0.0):.3e}, "
                    f"S_mean_after_stage={rk4_diagnostics.get('S_mean_after_stage', 0.0):.3e}, "
                    f"sum_SV_after_stage={rk4_diagnostics.get('sum_SV_after_stage', 0.0):.3e}, "
                    f"p_mean_vol={p_mean:.3e}, p_rms_vol={p_rms:.3e}",
                    file=sys.stderr,
                )
            next_log_time = ((int(t / log_time_interval) + 1) * log_time_interval)

        if t >= next_output_time or n == nt - 1:
            write_vtk_output(nodes, cells, p, u, v, output_dir, output_idx)
            next_output_time = ((int(t / output_time_interval) + 1) * output_time_interval)
            output_idx += 1

        progress = 100.0 * (n + 1) / nt
        elapsed = time.time() - start
        eta = elapsed / (n + 1) * (nt - n - 1) if n > 0 else 0.0
        progress_str = update_progress_bar(progress, progress_bar_width, elapsed, eta)
        print(progress_str, end='', file=sys.stderr, flush=True)
        if n == nt - 1 or (n > 0 and n % max(1, nt // 100) == 0):
            print("", file=sys.stderr)
            if n == nt - 1:
                print(f"完成: t={t:.6e}s, 总残差={residuals['total']:.6e}", file=sys.stderr)

    end = time.time()
    print(f"\n模拟完成！总用时: {end-start:.1f}s", file=sys.stderr)
    print(f"日志文件已保存到: {log_file_path}", file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()
    main()
