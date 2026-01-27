# RDE声学模拟代码库 (acoustic_RDE)

## 项目简介

本代码库提供了旋转爆轰发动机（Rotating Detonation Engine, RDE）内声学现象模拟的基础代码。代码实现了基于有限差分和有限体积方法的声学波动方程求解，用于研究RDE中旋转爆轰波产生的声场特性及其传播规律。

## 主要功能

- **声学波动方程求解**：实现线性声学方程的数值求解
- **旋转爆轰波源项**：模拟旋转爆轰波作为声源，考虑波前传播和衰减
- **多种网格类型**：支持结构化网格和非结构化O网格
- **并行计算支持**：非结构化网格版本支持多核并行计算
- **可视化输出**：生成VTK格式文件，便于ParaView等工具可视化分析

## 代码文件说明

### 核心模拟代码

#### `plane_2D.py` - 结构化网格版本
- **功能**：使用结构化矩形网格进行2D声学模拟
- **数值方法**：有限差分法（FDM）
- **网格类型**：结构化网格 + 掩码（mask）筛选环形区域
- **特点**：
  - 实现RK4时间积分方法
  - 硬壁边界条件（法向速度为零）
  - 数值滤波机制
  - CFL稳定性检查
- **适用场景**：快速原型验证、参数研究

#### `plane_2D_Ogrid.py` - 非结构化O网格版本
- **功能**：使用非结构化环形网格进行2D声学模拟
- **数值方法**：有限体积法（FVM）
- **网格类型**：非结构化O网格（环形四边形单元）
- **特点**：
  - 网格质量更高，边界处理更精确
  - 支持并行计算（multiprocessing）
  - 可通过命令行参数控制并行核数
  - 与`plane_2D.py`物理模型一致
- **适用场景**：高精度模拟、大规模计算
- **使用方法**：
  ```bash
  python plane_2D_Ogrid.py              # 使用推荐核数
  python plane_2D_Ogrid.py -n 4         # 指定4个核
  python plane_2D_Ogrid.py --single     # 单核运行
  ```

### 网格生成

#### `mesh.py` - 非结构化O网格生成
- **功能**：生成环形非结构化网格（O-cut annulus）
- **输出格式**：VTK格式，可用于ParaView可视化
- **特点**：
  - 确保网格在角度方向闭合（周期性边界）
  - 网格质量检查（面积、连接性验证）
  - 与模拟代码参数一致
- **输出文件**：`mesh/o_cut_annulus.vtk`

### 可视化脚本

#### `signal_acoustic_1D.py` - 1D信号可视化
- **功能**：展示爆轰波在圆周上的传播动画
- **模型**：陡升+指数衰减的爆轰信号
- **输出**：matplotlib动画

#### `signal_acoustic_2D.py` - 2D压力云图可视化
- **功能**：展示倾斜爆轰波在管道中的压力分布动画
- **模型**：考虑前倾角度的爆轰波传播
- **输出**：matplotlib动画

## 物理模型

### 控制方程

线性声学波动方程：

```
压力方程:  dp/dt = -ρ₀c₀²·div(u) + S
速度方程:  du/dt = -grad(p)/ρ₀ + ν·laplacian(u)
```

其中：
- `p`: 声压 [Pa]
- `u, v`: 速度分量 [m/s]
- `ρ₀`: 空气密度 [kg/m³]
- `c₀`: 声速 [m/s]
- `ν`: 运动黏度 [m²/s]（主要用于数值稳定性）
- `S`: 源项 [Pa/s]

### 源项模型

旋转爆轰波源项：
- 位置：环形区域（内半径到外半径）
- 传播：以角速度ω旋转
- 形状：高斯衰减模型
- 方向：只在波前方向（dtheta ≥ 0）

### 边界条件

- **硬壁边界条件**：壁面处法向速度为零（u·n = 0）
- **内壁和外壁**：分别处理内边界和外边界

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

详细安装说明请参考 [README_INSTALL.md](README_INSTALL.md)

### 2. 运行模拟

```bash
# 结构化网格版本
python plane_2D.py

# 非结构化O网格版本（推荐，网格质量更高）
python plane_2D_Ogrid.py

# 生成网格
python mesh.py
```

### 3. 可视化结果

使用ParaView打开 `output/` 或 `output_ogrid/` 目录中的VTK文件。

## 输出文件

- **模拟结果**：
  - `output/acoustic_XXXX.vtk` - 结构化网格版本输出
  - `output_ogrid/acoustic_ogrid_XXXX.vtk` - 非结构化网格版本输出
- **网格文件**：
  - `mesh/o_cut_annulus.vtk` - 环形网格

## 项目结构

```
acoustic_RDE/
├── README.md                 # 本文件
├── README_INSTALL.md         # 安装说明
├── LICENSE                   # MIT许可证
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略文件
│
├── plane_2D.py              # 结构化网格模拟（主程序）
├── plane_2D_Ogrid.py        # 非结构化O网格模拟（主程序，支持并行）
├── mesh.py                  # 网格生成
│
├── signal_acoustic_1D.py    # 1D可视化脚本
├── signal_acoustic_2D.py    # 2D可视化脚本
│
├── mesh/                    # 网格文件目录
│   └── o_cut_annulus.vtk
├── output/                   # 结构化网格版本输出
└── output_ogrid/            # 非结构化网格版本输出
```

## 技术特点

### 数值方法
- **时间积分**：四阶Runge-Kutta方法（RK4）
- **空间离散**：
  - 结构化网格：有限差分法（中心差分+边界单侧差分）
  - 非结构化网格：有限体积法（Green-Gauss方法）
- **边界处理**：硬壁边界条件，法向速度为零

### 数值稳定性
- CFL条件检查
- 数值滤波机制（抑制高频振荡）
- 边界处使用单侧差分

### 性能优化
- 并行计算支持（非结构化网格版本）
- 优化打印频率
- 只在有效区域计算源项

## 参数说明

### 几何参数
- `R = 0.01 m`：外半径
- `r_core = 0.002 m`：内半径（避免中心奇点）
- `dr = 0.0002 m`：网格间距

### 物理参数
- `ρ₀ = 1.225 kg/m³`：空气密度
- `c₀ = 343.0 m/s`：声速
- `ν = 1.5e-5 m²/s`：运动黏度（主要用于数值稳定性）

### 源项参数
- `wave_pressure = 10.0 Pa`：爆轰波压强幅度
- `v_rde = 1400.0 m/s`：爆轰波传播速度
- `sigma = 0.001 m`：衰减尺度

### 时间参数
- `dt = 1e-9 s`：时间步长
- `t_end = 1e-3 s`：结束时间

## 注意事项

1. **网格一致性**：确保所有文件的几何参数（R, r_core, dr）保持一致
2. **CFL条件**：代码会自动检查CFL数，如果超过限制会给出警告
3. **并行计算**：非结构化网格版本的并行计算在Windows上需要特殊处理（代码中已包含）
4. **输出文件**：输出文件可能很大，建议定期清理或使用`.gitignore`忽略

## 引用

如果使用本代码进行研究，请引用：

```
Qi Sun, RDE Acoustic Simulation Code, 2026
College of Mechanics and Engineering Science, Peking University
```

## 作者信息

**作者**：Qi Sun  
**邮箱**：ernest-llg@outlook.com  
**单位**：College of Mechanics and Engineering Science  
**机构**：Peking University  
**邮编**：100871

## 版权与许可

本项目采用 **MIT License** 开源许可证。

### 版权声明

Copyright (c) 2026 Qi Sun

### 许可证说明

MIT License 是一个宽松的开源许可证，允许：

- ✅ **商业使用**：可以用于商业项目
- ✅ **修改**：可以修改代码
- ✅ **分发**：可以分发代码
- ✅ **私人使用**：可以私人使用
- ✅ **专利使用**：可以使用专利

**唯一要求**：
- 保留版权声明和许可证声明

**不提供担保**：
- 软件按"原样"提供，不提供任何明示或暗示的担保

完整的许可证文本请参见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交Issue和Pull Request来改进代码。

## 更新日志

### 2026-01-27
- 初始版本发布
- 实现结构化网格和非结构化O网格两种版本
- 添加并行计算支持
- 完善边界条件和数值方法
- 添加完整的文档和安装说明

## 联系方式

如有问题或建议，请通过以下方式联系：

- **邮箱**：ernest-llg@outlook.com
- **单位**：北京大学 工学院 力学与工程科学系

---

**注意**：本代码库用于学术研究目的，使用前请仔细阅读LICENSE文件。
