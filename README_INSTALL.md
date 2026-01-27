# RDE声学模拟项目 - 安装指南

## 快速开始

### 1. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 使用pip安装所有依赖
pip install -r requirements.txt

# 或者手动安装
pip install numpy>=1.20.0 meshio>=5.0.0 matplotlib>=3.3.0
```

### 3. 验证安装

```bash
# 测试导入
python -c "import numpy; import meshio; import matplotlib; print('所有依赖安装成功！')"
```

## 依赖说明

### 必需依赖（核心模拟）

- **numpy**: 数值计算库，用于所有数值运算
- **meshio**: 网格处理库，用于生成和读写VTK格式的网格文件

### 可选依赖（可视化）

- **matplotlib**: 用于 `signal_acoustic_1D.py` 和 `signal_acoustic_2D.py` 的可视化脚本

### Python标准库（无需安装）

以下库是Python标准库，无需额外安装：
- `os`, `sys`, `time`: 系统操作
- `argparse`: 命令行参数解析
- `multiprocessing`: 并行计算
- `functools`: 函数工具

## 使用说明

### 运行主模拟程序

```bash
# 结构化网格版本
python plane_2D.py

# 非结构化O网格版本（支持并行）
python plane_2D_Ogrid.py
python plane_2D_Ogrid.py -n 4  # 使用4个核
python plane_2D_Ogrid.py --single  # 单核运行
```

### 配置模拟参数

所有模拟参数统一在 `control.py` 文件中管理：

```bash
# 查看参数配置（会打印所有参数）
python control.py

# 修改参数：直接编辑 control.py 文件
# 然后运行主程序即可
python plane_2D_Ogrid.py
```

**参数类型**：
- 物理参数（密度、声速、黏度）
- 几何参数（内外半径、网格间距）
- 时间参数（时间步长、结束时间）
- 声源参数（爆轰波参数）
- 数值计算参数（滤波、CFL数等）
- 输出参数（输出目录、时间间隔、日志文件等）

**输出功能**：
- **进度条显示**：实时显示计算进度（百分比），包括已用时间和预计剩余时间
- **日志文件**：自动记录到 `output_ogrid/simulation_log.txt`，包含：
  - 时间步数和当前时间
  - 压力、速度残差
  - 压力统计（最小值、最大值、平均值）
- **VTK输出**：基于时间间隔输出，更加直观（由 `output_time_interval` 控制）
- **日志输出**：基于时间间隔输出（由 `log_time_interval` 控制）

### 生成网格

```bash
python mesh.py
```

### 运行可视化脚本

```bash
# 1D可视化
python signal_acoustic_1D.py

# 2D可视化
python signal_acoustic_2D.py
```

## 系统要求

- Python 3.7 或更高版本
- 推荐使用虚拟环境以避免依赖冲突

## 故障排除

### 问题：找不到meshio模块

```bash
pip install meshio
```

### 问题：matplotlib显示问题（macOS）

如果使用macOS且matplotlib无法显示窗口，可能需要安装tkinter：

```bash
# macOS (使用Homebrew)
brew install python-tk

# 或使用conda
conda install tk
```

### 问题：并行计算在Windows上不工作

确保使用 `if __name__ == "__main__":` 保护（代码中已包含）。

## 更新依赖

```bash
pip install --upgrade -r requirements.txt
```
