# RDE 声学模拟代码库 (acoustic_RDE)

## 项目简介

本代码库提供旋转爆轰发动机（Rotating Detonation Engine, RDE）内声学现象模拟的基础实现。采用有限差分与有限体积方法求解线性声学波动方程，用于研究 RDE 中旋转爆轰波产生的声场及其传播规律。

---

## 主要功能

- **声学波动方程求解**：线性声学方程数值求解（压力–速度形式）
- **旋转爆轰波源项**：环带内旋转高斯源项，支持顺时针/逆时针与 DC 零均值修正
- **多种网格**：结构化网格（plane_2D）与非结构化 O 网格（plane_2D_Ogrid）
- **高性能 O 网格**：Numba JIT、预计算几何量、边–邻居映射、可选多线程
- **网格缓存**：O 网格运行后写入 `output_ogrid/ogrid_mesh.npz`；下次参数一致时直接加载，节省时间；参数不一致时重新生成并覆盖
- **网格检查**：主程序启动时对网格做一致性检查，若有错误则在命令行输出并退出
- **可视化输出**：VTK 结果，便于 ParaView 等后处理
- **进度与日志**：进度条、仿真日志、源项 DC 诊断 CSV

---

## 代码与模块说明

### 入口与主程序

| 文件 | 说明 |
|------|------|
| **main_ogrid.py** | **推荐入口**。先弹出 GUI 预览（初始化压力分布），用户点 Continue 后启动主模拟；执行顺序不变。 |
| **plane_2D_Ogrid.py** | O 网格主程序聚合模块。可直接运行以跳过 GUI、仅做主模拟；内含网格加载/生成/保存与网格检查逻辑。 |
| **plane_2D.py** | 结构化网格版本主程序（有限差分）。 |

### O 网格子模块（按功能拆分，便于扩展）

| 文件 | 功能 |
|------|------|
| **plane_2D_Ogrid_mesh.py** | 网格生成 `generate_ogrid_mesh`；网格文件读写 `load_ogrid_mesh`、`save_ogrid_mesh`；参数比对 `mesh_params_match`、`get_mesh_path`。 |
| **plane_2D_Ogrid_check.py** | 网格检查 `check_ogrid_mesh`（节点/单元、面积、边与邻居一致性等）；失败时返回错误信息供主程序退出。 |
| **plane_2D_Ogrid_fvm.py** | 有限体积算子：梯度、散度、拉普拉斯（Numba 可选）。 |
| **plane_2D_Ogrid_source.py** | 旋转爆轰源项 `source_term_ogrid`。 |
| **plane_2D_Ogrid_boundary.py** | 边界识别 `identify_boundary_cells`、硬壁边界施加 `apply_wall_boundary_conditions_ogrid`。 |
| **plane_2D_Ogrid_rk4.py** | RK4 时间积分 `rk4_step_ogrid`（含源项 DC 修正与边界应用）。 |
| **plane_2D_Ogrid_filter.py** | 空间滤波 `apply_spatial_filter`、`prepare_filter_arrays`。 |
| **plane_2D_Ogrid_io.py** | 进度条、残差、日志、源项诊断 CSV、VTK 输出。 |

### 预览与参数

| 文件 | 说明 |
|------|------|
| **ogrid_preview_gui.py** | O 网格初始化预览 GUI：推进 1 步后显示压力分布；Continue 启动主模拟，Break 退出。使用与主程序相同的网格逻辑（可加载缓存或重新生成）。 |
| **control.py** | 统一参数：`control()` 返回参数字典，`validate_params(params)` 做完整性及合理性校验。 |

### 网格与可视化脚本

| 文件 | 说明 |
|------|------|
| **mesh.py** | 独立环形 O 网格生成，输出 VTK（如 `mesh/o_cut_annulus.vtk`）。 |
| **signal_acoustic_1D.py** | 1D 圆周爆轰信号动画。 |
| **signal_acoustic_2D.py** | 2D 压力云图动画。 |

---

## 运行方式

### 推荐：先预览再计算（GUI 入口）

```bash
python main_ogrid.py
```

流程：启动 → 初始化/加载网格 → 弹窗显示 1 步后压力分布 → 用户点 **Continue** 后等待约 1.5 秒关闭窗口 → 运行主模拟；点 **Break** 或关窗则退出。

### 仅主模拟（跳过 GUI）

```bash
python plane_2D_Ogrid.py
python plane_2D_Ogrid.py -n 8      # 指定 8 线程
python plane_2D_Ogrid.py --single # 强制单核
```

### 仅预览（不跑主模拟）

```bash
python ogrid_preview_gui.py
```

### 结构化网格与网格生成

```bash
python plane_2D.py
python mesh.py
```

---

## 网格缓存规则

- **文件位置**：`<output_dir>/ogrid_mesh.npz`，默认即 `output_ogrid/ogrid_mesh.npz`。
- **比对参数**：`R`、`r_core`、`dr`、`dtheta_base`（与文件中保存值在容差内一致则视为同一网格）。
- **行为**：
  - 首次运行或文件不存在：生成网格，主模拟结束后**保存**到该文件。
  - 再次运行且四参数一致：**直接加载**网格，不再生成，节省时间。
  - 再次运行但参数任一不一致：**重新生成**网格并**覆盖**原文件。
- 预览 GUI 使用同一套逻辑（可加载或生成），但**不会写入**网格文件；只有主模拟会写入/覆盖。

---

## 网格检查

主程序在获得网格（加载或生成）后立即执行 `check_ogrid_mesh`：

- 检查节点/单元数、索引越界、面积与边几何、邻居一致性等。
- 若有错误：在**命令行**输出所有检查信息并 **退出**（不进入时间步进）。

---

## 物理模型

### 控制方程

线性声学（压力–速度形式）：

- 压力方程：`dp/dt = -ρ₀ c₀² div(u) + S`
- 速度方程：`du/dt = -grad(p)/ρ₀ + ν lap(u)`

其中 `p` 为声压 [Pa]，`u` 为速度 [m/s]，`ρ₀`、`c₀`、`ν` 为密度、声速、运动黏度，`S` 为源项 [Pa/s]。`ν lap(u)` 主要为数值稳定性（人工耗散）。

### 源项

- 环带内旋转高斯源项；角速度 `ω = v_rde/R`；特征时间 `τ = 2πR/v_rde`。
- 支持顺时针/逆时针（`rotation_clockwise`）。
- 封闭腔下可开启源项 DC 零均值修正：`zero_mean_source` + `zero_mean_mode`（`"stage"` / `"step"` / `"off"`）。

### 边界条件

- 硬壁：壁面法向速度为零（`u·n = 0`）；内/外壁分别识别并施加。

---

## 输出文件（O 网格版本）

- **VTK**：`output_ogrid/acoustic_ogrid_XXXX.vtk`（按 `output_time_interval` 输出）。
- **仿真日志**：`output_ogrid/simulation_log.txt`（时间步、残差、压力统计、体积加权 p_mean_vol、p_rms_vol 等）。
- **源项 DC 诊断**：`output_ogrid/source_dc_diagnostics.csv`（每步源项 DC 相关统计）。
- **网格缓存**：`output_ogrid/ogrid_mesh.npz`（网格及四参数，供下次复用）。

---

## 项目结构

```
acoustic_RDE/
├── README.md                    # 本说明
├── README_INSTALL.md            # 安装与运行细节
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── main_ogrid.py                # 统一入口：先 GUI 预览，再主模拟
├── plane_2D_Ogrid.py            # O 网格主程序（聚合 + 主流程）
├── plane_2D_Ogrid_mesh.py       # 网格生成与缓存读写
├── plane_2D_Ogrid_check.py     # 网格检查
├── plane_2D_Ogrid_fvm.py       # FVM 算子
├── plane_2D_Ogrid_source.py   # 源项
├── plane_2D_Ogrid_boundary.py # 边界
├── plane_2D_Ogrid_rk4.py       # RK4 时间积分
├── plane_2D_Ogrid_filter.py    # 空间滤波
├── plane_2D_Ogrid_io.py        # 日志 / VTK / 进度
│
├── ogrid_preview_gui.py        # O 网格预览 GUI
├── control.py                  # 参数与校验
├── plane_2D.py                 # 结构化网格主程序
├── mesh.py                     # 独立网格生成
├── signal_acoustic_1D.py       # 1D 可视化
├── signal_acoustic_2D.py       # 2D 可视化
│
├── mesh/                       # mesh.py 输出目录
├── output/                     # plane_2D 输出
└── output_ogrid/               # O 网格输出与网格缓存
```

---

## 参数说明（control.py）

所有模拟参数由 `control.py` 的 `control()` 提供，并用 `validate_params(params)` 校验。

- **物理**：`rho0`、`c0`、`nu`
- **几何**：`R`、`r_core`、`dr`、`dtheta_base`
- **时间**：`dt`、`t_end`
- **声源**：`wave_pressure`、`v_rde`、`detonation_width`、`inner_r`、`outer_r`、`omega`、`sigma`、`tau`、`rotation_clockwise`、`zero_mean_source`、`zero_mean_mode`
- **数值**：`filter_strength`、`apply_filter`、`filter_frequency`、`CFL_max`、`boundary_tolerance`
- **输出**：`output_dir`、`output_time_interval`、`log_time_interval`、`log_file`、`progress_bar_width`
- **并行**：`use_parallel`、`parallel_min_cells`、`n_cores`

修改参数请编辑 `control.py` 中 `control()` 的赋值；运行主程序或预览时会自动读取并校验。

---

## 安装与依赖

详见 [README_INSTALL.md](README_INSTALL.md)。简要：

- **核心**：`numpy`、`meshio`
- **推荐**：`numba`（O 网格加速与多线程）
- **可视化/预览**：`matplotlib`；GUI 需 Python 带 Tk（如缺失见安装说明中的 tkinter 说明）

```bash
pip install -r requirements.txt
```

---

## 注意事项

1. **入口**：日常使用建议 `python main_ogrid.py`，先预览再计算；不需要 GUI 时用 `python plane_2D_Ogrid.py`。
2. **网格缓存**：同一组 `R, r_core, dr, dtheta_base` 会复用 `output_ogrid/ogrid_mesh.npz`；改任一参数会重新生成并覆盖。
3. **网格检查**：若检查失败，程序会在命令行打印原因并退出，请根据提示排查。
4. **并行**：`-n N` 指定线程数；`--single` 强制单核；也可在 `control.py` 中设置 `use_parallel`、`parallel_min_cells`、`n_cores`。
5. **CFL**：程序会检查 CFL，超过 `CFL_max` 会给出警告与建议步长。

---

## 引用与作者

若在研究中使用了本代码，可引用：

```
Qi Sun, RDE Acoustic Simulation Code, 2026
College of Mechanics and Engineering Science, Peking University
```

**作者**：Qi Sun  
**邮箱**：ernest-llg@outlook.com  
**单位**：北京大学 工学院 力学与工程科学系

---

## 版权与许可

本项目采用 **MIT License**。详见 [LICENSE](LICENSE)。

---

## 更新日志

### 近期更新

- **入口**：新增 `main_ogrid.py` 统一入口，先 GUI 预览再主模拟。
- **O 网格模块化**：核心拆分为 `plane_2D_Ogrid_*.py`（mesh / check / fvm / source / boundary / rk4 / filter / io），`plane_2D_Ogrid.py` 为聚合与主流程。
- **网格缓存**：运行后在 `output_ogrid/ogrid_mesh.npz` 保存网格；参数一致时直接加载，不一致时重新生成并覆盖。
- **网格检查**：主程序启动时执行网格检查，错误时在命令行输出并退出。
- **预览 GUI**：点 Continue 后延迟约 1.5 秒再启动主模拟，并改进窗口与 matplotlib 的关闭以减轻退出报错。

### 历史要点

- 参数统一由 `control.py` 管理；O 网格性能优化（Numba、预计算、可选多线程）；源项 DC 零均值修正与诊断输出；基于时间的 VTK/日志输出与进度条。
