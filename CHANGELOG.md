# Changelog

本文件记录项目的所有重要变更与功能更新，按时间倒序（最新在上）。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，类型包括：`Added`（新增）、`Changed`（变更）、`Fixed`（修复）、`Deprecated`（弃用）、`Removed`（移除）、`Security`（安全）。

---

## [Unreleased]

（预留：尚未发布的改动可写在此处。）

---

## 2026-01-29

### Added

- **统一入口 `main_ogrid.py`**：推荐入口，执行顺序为先进入 GUI 预览、用户确认后再运行主模拟。
- **O 网格模块化**：将原 `plane_2D_Ogrid.py` 按功能拆分为子模块，便于扩展与维护：
  - `plane_2D_Ogrid_mesh.py`：网格生成，以及网格文件的加载/保存/参数比对（`load_ogrid_mesh`、`save_ogrid_mesh`、`mesh_params_match`、`get_mesh_path`）。
  - `plane_2D_Ogrid_check.py`：网格一致性检查 `check_ogrid_mesh`（节点与单元、面积、边与邻居等），失败时返回错误信息供主程序退出。
  - `plane_2D_Ogrid_fvm.py`：有限体积算子（梯度、散度、拉普拉斯）。
  - `plane_2D_Ogrid_source.py`：旋转爆轰源项。
  - `plane_2D_Ogrid_boundary.py`：边界识别与硬壁边界条件施加。
  - `plane_2D_Ogrid_rk4.py`：RK4 时间积分（含源项 DC 修正）。
  - `plane_2D_Ogrid_filter.py`：空间滤波。
  - `plane_2D_Ogrid_io.py`：进度条、残差、日志、VTK、源项诊断 CSV。
- **网格缓存**：主模拟运行结束后将网格写入 `output_ogrid/ogrid_mesh.npz`；下次运行时若 `R`、`r_core`、`dr`、`dtheta_base` 与文件一致则直接加载，否则重新生成并覆盖该文件，以节省重复生成时间。
- **网格检查**：主程序在获得网格（加载或生成）后立即执行网格检查；若检查失败则在命令行输出所有错误信息并直接退出，不进入时间步进。
- **标准化日志文件 `CHANGELOG.md`**：集中记录每次更新的内容与功能，便于追溯。

### Changed

- **`plane_2D_Ogrid.py`**：改为聚合模块，从上述子模块导入并统一导出接口，保留 `main()`；主流程中集成“先尝试加载网格 → 参数一致则用缓存，否则生成并保存”及“网格检查失败则退出”。
- **预览 GUI（`ogrid_preview_gui.py`）**：使用与主程序相同的网格逻辑（可加载缓存或重新生成）；预览阶段不写入网格文件。点 Continue 后先隐藏窗口并关闭 matplotlib 图形，再延迟约 1.5 秒后启动主模拟，以减少窗口未关闭或退出报错的问题。
- **说明文档**：重写 `README.md` 与 `README_INSTALL.md`，覆盖当前入口、子模块、网格缓存、网格检查、运行方式、故障排除等，并与代码一致。

### Fixed

- 预览窗口在点 Continue 后未及时关闭：通过先 `withdraw()`、关闭图形、再 `quit()`/`destroy()`，并在启动主模拟前 `time.sleep(1.5)`，使窗口关闭完成后再继续执行。
- 退出时出现 Python 错误报告：在销毁 Tk 窗口前显式 `plt.close(fig)` 释放 matplotlib 资源，减轻进程退出时的报错或弹窗。

---

## 2026-01-27（及更早）

### Added

- **参数控制模块 `control.py`**：统一管理所有模拟参数；`control()` 返回参数字典，`validate_params(params)` 校验参数完整性与合理性。
- **基于时间间隔的输出**：VTK 与日志按时间间隔输出（`output_time_interval`、`log_time_interval`），替代按步数输出。
- **进度条**：主模拟运行时在 stderr 显示百分比进度条、已用时间与预计剩余时间。
- **仿真日志**：`output_ogrid/simulation_log.txt` 记录时间步、残差、压力统计、体积加权 `p_mean_vol` 与 `p_rms_vol`。
- **源项 DC 零均值修正与诊断**：参数 `zero_mean_source`、`zero_mean_mode`（`"stage"` / `"step"` / `"off"`）；诊断文件 `output_ogrid/source_dc_diagnostics.csv` 记录每步源项 DC 相关统计。
- **O 网格预览 GUI**：`ogrid_preview_gui.py` 在正式计算前弹窗展示 1 步后的压力分布，用户可选择 Continue 或 Break。
- **物理正确性**：源项单位修正为 [Pa/s]，使用特征时间 `τ = 2πR/v_rde`；运动黏度明确为人工数值耗散；硬壁边界条件（法向速度为零）。
- **网格质量（mesh.py）**：theta 闭合、r 不超界、单元面积与周期性边界检查，并输出统计信息。
- **RK4 时间积分**：四阶 Runge-Kutta；边界处单侧差分；CFL 检查与建议。

### Changed

- **O 网格性能优化**：预计算几何量与边–邻居映射、Numba JIT、向量化、预分配/重用数组；移除 multiprocessing，改用 Numba 多线程（prange）；网格数 ≥ `parallel_min_cells` 且 `use_parallel=True` 时启用多核；支持命令行 `-n N`、`--single`。
- **plane_2D.py 输出**：增加速度场、速度大小、有效区域掩码及统计信息；减少打印频率以提升效率。
- **参数一致性**：几何与半径等参数在多处统一（如 R=0.01m）；可视化脚本注释与单位修正。

---

*后续更新请在本文件顶部「Unreleased」或新日期下追加，并注明类型（Added/Changed/Fixed 等）与简要说明。*
