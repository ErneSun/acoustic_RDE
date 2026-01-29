# RDE 声学模拟项目 - 安装与运行指南

## 快速开始

### 1. 创建虚拟环境（推荐）

```bash
python3 -m venv .venv

# 激活
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import numpy; import meshio; print('核心依赖 OK')"
python -c "import numba; print('numba OK')"   # 可选
python -c "import matplotlib; print('matplotlib OK')"  # 可选
```

---

## 依赖说明

### 核心依赖（主模拟必需）

- **numpy** (>=1.20.0)：数值计算，`control.py`、`plane_2D.py`、`plane_2D_Ogrid*.py`、`mesh.py` 等均需要。
- **meshio** (>=5.0.0)：VTK 读写，`plane_2D_Ogrid` 输出 VTK 必需。

### 可选依赖（强烈推荐）

- **numba** (>=0.56.0)：O 网格 FVM/源项/滤波 JIT 加速及多线程（prange）；未安装时退化为纯 numpy，计算较慢。

### 可选依赖（仅可视化与预览）

- **matplotlib** (>=3.3.0)：`signal_acoustic_1D.py`、`signal_acoustic_2D.py`、`ogrid_preview_gui.py` 需要；若只跑主模拟可不装。
- **tkinter**：`ogrid_preview_gui.py`、`main_ogrid.py` 需要，通常随 Python 安装；若缺失见下方“故障排除”。

---

## 运行说明

### 推荐：先预览再主模拟（统一入口）

```bash
python main_ogrid.py
```

流程：初始化/加载网格 → 弹窗显示 1 步后压力分布 → 点 **Continue** 后约 1.5 秒关闭窗口并启动主模拟；点 **Break** 或关窗则退出。

### 仅主模拟（跳过 GUI）

```bash
# 按 control.py 设置运行（单核/多核由 use_parallel 与网格数决定）
python plane_2D_Ogrid.py

# 指定 8 线程
python plane_2D_Ogrid.py -n 8

# 强制单核
python plane_2D_Ogrid.py --single
```

### 仅预览（不跑主模拟）

```bash
python ogrid_preview_gui.py
```

### 其他程序

```bash
# 结构化网格版本
python plane_2D.py

# 独立网格生成（输出 mesh/o_cut_annulus.vtk）
python mesh.py

# 1D/2D 可视化
python signal_acoustic_1D.py
python signal_acoustic_2D.py
```

---

## 网格缓存

- **文件**：`output_ogrid/ogrid_mesh.npz`（由 `control.py` 中 `output_dir` 决定）。
- **逻辑**：
  - 首次运行或文件不存在：生成网格，主模拟结束后**保存**。
  - 再次运行且 `R`、`r_core`、`dr`、`dtheta_base` 与文件中一致：**直接加载**，不重新生成。
  - 任一参数不一致：**重新生成**并**覆盖**该文件。
- 预览 GUI 使用同一套加载/生成逻辑，但**不会写入**网格文件。

---

## 配置参数

所有模拟参数在 `control.py` 的 `control()` 中定义，并由 `validate_params(params)` 校验。

```bash
# 查看当前参数
python control.py
```

修改参数：编辑 `control.py` 中 `control()` 的赋值，然后运行主程序或 `main_ogrid.py`。

---

## 输出与日志

- **VTK**：`output_ogrid/acoustic_ogrid_XXXX.vtk`（按 `output_time_interval` 输出）。
- **仿真日志**：`output_ogrid/simulation_log.txt`（时间步、残差、压力统计、p_mean_vol、p_rms_vol 等）。
- **源项 DC 诊断**：`output_ogrid/source_dc_diagnostics.csv`。
- **网格缓存**：`output_ogrid/ogrid_mesh.npz`。

---

## 系统要求

- Python 3.7 及以上。
- 推荐使用虚拟环境以避免依赖冲突。

---

## 故障排除

### 找不到 `meshio`

```bash
pip install meshio
```

### 找不到 `_tkinter`（运行 main_ogrid.py 或 ogrid_preview_gui.py 报错）

Python 未带 Tk 支持时会出现。可任选其一：

- **Homebrew (macOS)**：
  ```bash
  brew install python-tk@3.13
  # 或先装 Tcl/Tk 再重装 Python
  brew install tcl-tk
  brew reinstall python@3.13
  ```
- **Conda**：`conda install tk`

若暂时不用 GUI，可直接运行主模拟：

```bash
python plane_2D_Ogrid.py
```

### 计算很慢

1. 确认已安装 `numba`：`pip install numba`。
2. 网格数较大（如 >50000）时，可启用多线程：`python plane_2D_Ogrid.py -n 8` 或在 `control.py` 中设 `use_parallel=True`。
3. 适当增大 `output_time_interval`、`log_time_interval` 以减少 I/O。

### 参数校验失败

运行时会从 `control.py` 读取参数并校验。若报“参数验证失败”，请根据终端提示检查 `control.py` 中是否缺少或填错参数（如 None、非正数等）。可用 `python control.py` 查看当前参数。

### 网格检查失败

主程序会在获得网格后做一致性检查。若报“网格检查失败”，程序会打印具体错误并退出，请根据提示排查（如节点/单元数、面积、边与邻居等）。

---

## 更新依赖

```bash
pip install --upgrade -r requirements.txt
```

---

**注意**：本代码库用于学术研究，使用前请阅读 [LICENSE](LICENSE)。
