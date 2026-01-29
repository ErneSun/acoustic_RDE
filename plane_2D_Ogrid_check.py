#!/usr/bin/env python3
"""
RDE O网格 - 网格检查
对生成后的网格进行一致性检查；若发现错误则在命令行反馈并建议退出。
"""

import numpy as np


def check_ogrid_mesh(nodes, cells, cell_centers, cell_areas, nr, ntheta,
                     edge_normals, edge_lengths, edge_neighbors, edge_count):
    """
    检查O网格的一致性与合理性。

    参数:
        nodes, cells, cell_centers, cell_areas: 网格与几何量
        nr, ntheta: 径向/角度层数
        edge_normals, edge_lengths, edge_neighbors, edge_count: 边几何与邻居

    返回:
        ok: bool，True 表示通过检查
        messages: list[str]，错误或警告信息（错误时 ok=False，调用方应退出）
    """
    messages = []
    n_nodes = len(nodes)
    n_cells = len(cells)

    # 1. 基本尺寸
    if n_nodes <= 0:
        messages.append("[网格检查] 错误: 节点数为 0。")
    if n_cells <= 0:
        messages.append("[网格检查] 错误: 单元数为 0。")
    if nr < 2 or ntheta < 3:
        messages.append(f"[网格检查] 错误: nr={nr}, ntheta={ntheta} 不满足环形网格要求 (nr>=2, ntheta>=3)。")

    # 2. 单元索引越界
    if n_nodes > 0 and n_cells > 0:
        max_idx = np.max(cells)
        min_idx = np.min(cells)
        if min_idx < 0 or max_idx >= n_nodes:
            messages.append(f"[网格检查] 错误: 单元节点索引越界 (节点索引范围 [{min_idx}, {max_idx}]，节点数 {n_nodes})。")

    # 3. 单元面积与退化
    if n_cells > 0:
        if np.any(cell_areas <= 0):
            bad = np.where(cell_areas <= 0)[0]
            messages.append(f"[网格检查] 错误: 存在非正面积单元，数量={len(bad)}，示例索引: {bad[:5].tolist()}。")
        if np.any(~np.isfinite(cell_areas)):
            messages.append("[网格检查] 错误: 存在 NaN/Inf 单元面积。")

    # 4. 边几何
    if edge_normals.shape[0] != n_cells:
        messages.append(f"[网格检查] 错误: edge_normals 行数 ({edge_normals.shape[0]}) 与单元数 ({n_cells}) 不一致。")
    if np.any(edge_count < 0) or np.any(edge_count > 4):
        messages.append("[网格检查] 错误: 存在非法 edge_count（应为 0~4）。")
    if n_cells > 0 and edge_lengths is not None and np.any(~np.isfinite(edge_lengths)):
        messages.append("[网格检查] 错误: 存在 NaN/Inf 边长度。")

    # 5. 邻居索引一致性（有效邻居应在 [0, n_cells-1]，-1 表示边界）
    if n_cells > 0 and edge_neighbors is not None:
        invalid_neighbor = (edge_neighbors >= 0) & (edge_neighbors >= n_cells)
        if np.any(invalid_neighbor):
            messages.append(f"[网格检查] 错误: 存在越界的边邻居索引 (单元数={n_cells})。")

    ok = len([m for m in messages if "错误" in m]) == 0
    return ok, messages
