#!/usr/bin/env python3
"""
RDE O网格模拟 - 初始化预览 GUI
运行后先弹窗展示初始化压力分布；用户确认设置正确则继续计算，否则退出。
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tkinter as tk
from tkinter import ttk

# 使用 TkAgg 后端以便在 tkinter 中嵌入
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 从 control 和 plane_2D_Ogrid 获取参数与初始化逻辑
from control import control, validate_params
from plane_2D_Ogrid import (
    generate_ogrid_mesh,
    identify_boundary_cells,
    rk4_step_ogrid,
)


def run_init_and_one_step():
    """
    执行与主程序一致的初始化，并推进一个时间步，得到初始压力分布。
    返回: (nodes, cells, cell_centers, p_init, params) 或 None（失败时）
    """
    try:
        params = control()
    except Exception as e:
        print(f"错误：无法从 control 获取参数: {e}", file=sys.stderr)
        return None

    is_valid, missing_params, invalid_params = validate_params(params)
    if not is_valid:
        print("错误：参数验证失败", file=sys.stderr)
        if missing_params:
            for p, d in missing_params:
                print(f"  缺少: {p} - {d}", file=sys.stderr)
        if invalid_params:
            for p, d, r in invalid_params:
                print(f"  无效: {p} - {d} - {r}", file=sys.stderr)
        return None

    R = params['R']
    r_core = params['r_core']
    dr = params['dr']
    dtheta_base = params['dtheta_base']
    boundary_tolerance = params['boundary_tolerance']
    rho0 = params['rho0']
    c0 = params['c0']
    nu = params['nu']
    dt = params['dt']
    inner_r = params['inner_r']
    outer_r = params['outer_r']
    omega = params['omega']
    wave_pressure = params['wave_pressure']
    tau = params['tau']
    sigma = params['sigma']

    # 生成网格与预计算几何量
    (nodes, cells, cell_centers, cell_areas, cell_neighbors, nr, ntheta,
     edge_normals, edge_lengths, edge_neighbors, edge_count) = generate_ogrid_mesh(
        R, r_core, dr, dtheta_base)
    n_cells = len(cells)
    max_edges = edge_normals.shape[1]

    # 边界识别与掩码
    inner_wall_cells, outer_wall_cells, wall_normals, boundary_edges_dict = identify_boundary_cells(
        nodes, cells, cell_centers, R, r_core, nr, ntheta, boundary_tolerance)
    boundary_mask = np.zeros((n_cells, max_edges), dtype=np.bool_)
    for (i, j) in boundary_edges_dict.keys():
        if i < n_cells and j < max_edges:
            boundary_mask[i, j] = True

    # 初始场与一个时间步
    p = np.zeros(n_cells)
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    p_new = np.zeros(n_cells)
    u_new = np.zeros(n_cells)
    v_new = np.zeros(n_cells)
    t0 = 0.0

    p_new, u_new, v_new, _ = rk4_step_ogrid(
        p, u, v, t0, dt, cell_centers, cell_areas,
        inner_wall_cells, outer_wall_cells, wall_normals,
        boundary_mask, edge_normals, edge_lengths,
        edge_neighbors, edge_count,
        rho0, c0, nu,
        inner_r, outer_r, omega, wave_pressure, tau, sigma, R,
        p_new, u_new, v_new, None)

    return (nodes, cells, cell_centers, p_new, params)


def build_preview_window(nodes, cells, cell_centers, p_init, params):
    """
    构建预览窗口：左侧为压力分布图，右侧为 Continue / Break 按钮。
    返回: (root, user_continue: tk.BooleanVar)
    """
    root = tk.Tk()
    root.title("RDE O网格 - 初始化预览")
    root.geometry("900x600")
    root.minsize(700, 450)

    user_continue = tk.BooleanVar(value=False)

    # 左侧：压力分布图
    left_frame = ttk.Frame(root, padding=5)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    xc = cell_centers[:, 0]
    yc = cell_centers[:, 1]
    p_min, p_max = np.min(p_init), np.max(p_init)
    if p_max <= p_min:
        p_max = p_min + 1e-10
    norm = Normalize(vmin=p_min, vmax=p_max)
    sc = ax.scatter(xc, yc, c=p_init, cmap='RdBu_r', norm=norm, s=8, rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('初始化压力分布 (1 个时间步后) [Pa]')
    plt.colorbar(ScalarMappable(norm=norm, cmap=sc.get_cmap()), ax=ax, label='p [Pa]')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 右侧：按钮
    right_frame = ttk.Frame(root, padding=15)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    ttk.Label(right_frame, text="初始化检查", font=('', 12, 'bold')).pack(pady=(0, 15))

    def on_continue():
        user_continue.set(True)
        root.quit()
        root.destroy()

    def on_break():
        user_continue.set(False)
        root.quit()
        root.destroy()

    btn_continue = ttk.Button(right_frame, text="Continue\n(继续计算)", command=on_continue, width=14)
    btn_continue.pack(pady=10)
    btn_break = ttk.Button(right_frame, text="Break\n(退出程序)", command=on_break, width=14)
    btn_break.pack(pady=10)

    # 点击窗口关闭按钮视为退出
    root.protocol("WM_DELETE_WINDOW", on_break)

    # 简要参数信息
    R, r_core = params['R'], params['r_core']
    detonation_width = params.get('detonation_width', R - params['inner_r'])
    info = f"R={R:.4f} m\nr_core={r_core:.4f} m\n爆轰波宽度≈{detonation_width:.4f} m"
    ttk.Label(right_frame, text=info, justify=tk.LEFT).pack(pady=20, anchor=tk.W)

    return root, user_continue


def main():
    print("正在初始化网格并计算 1 个时间步...", file=sys.stderr)
    result = run_init_and_one_step()
    if result is None:
        sys.exit(1)

    nodes, cells, cell_centers, p_init, params = result
    print("初始化完成，弹出预览窗口。", file=sys.stderr)

    root, user_continue = build_preview_window(nodes, cells, cell_centers, p_init, params)
    root.mainloop()

    if user_continue.get():
        print("用户选择继续，启动主模拟...", file=sys.stderr)
        from plane_2D_Ogrid import main as run_main
        run_main()
    else:
        print("用户选择退出。", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
