#!/usr/bin/env python3
"""
RDE声学模拟 - 参数控制模块
用于设置和配置所有模拟参数
"""

import numpy as np


def control():
    """
    控制函数：设置所有模拟参数
    
    返回:
        params: 参数字典，包含所有模拟所需的参数
    """
    
    # ==================== 物理参数 ====================
    rho0 = 1.225        # 空气密度 [kg/m³]
    c0   = 343.0        # 声速 [m/s]
    nu   = 1.5e-5       # 运动黏度 [m²/s]
    # 注意：nu用于速度方程中的黏性项 nu*laplacian(u)
    # 对于线性声学方程，这可以视为：
    # 1) 人工数值耗散（用于稳定性），或
    # 2) 声学吸收（但通常声学吸收系数不同）
    # 当前实现主要用于数值稳定性，实际物理耗散很小
    
    # ==================== 几何参数 ====================
    R      = 0.01         # 外半径 [m]
    r_core = 0.007        # 内半径 [m]
    dr     = 0.0002       # 径向网格间距 [m]
    
    # ==================== 时间参数 ====================
    dt    = 1e-9        # 时间步长 [s]
    t_end = 1e-3        # 结束时间 [s]
    
    # ==================== 声源参数 ====================
    wave_pressure = 10.0      # 爆轰波压强幅度 [Pa]
    v_rde = 1400.0            # 爆轰波传播速度 [m/s]
    # 爆轰波在径向上的宽度 [m]，从外壁向内量
    # 物理含义：爆轰波被压缩在靠近外壁的一条窄带内
    detonation_width = 0.001   # 例如：1 mm 宽度
    # 根据爆轰波宽度自动确定源项作用的径向范围
    # 内半径不能小于 r_core，避免跨越内壁
    inner_r  = max(R - detonation_width, r_core)   # 环带内半径 [m]
    outer_r  = R                                   # 环带外半径 [m]（始终贴在外壁）
    omega    = v_rde / R      # 角速度 [rad/s]
    sigma    = 0.001          # 衰减尺度 [m]
    tau = 2 * np.pi * R / v_rde  # 特征时间 [s]
    
    # ==================== 数值计算参数 ====================
    # 数值滤波参数
    filter_strength = 0.01  # 滤波强度（0-1），0表示不滤波
    apply_filter = True     # 是否应用数值滤波
    filter_frequency = 10   # 滤波频率（每N步滤波一次）
    
    # CFL稳定性参数
    CFL_max = 0.5           # 最大CFL数（用于稳定性检查）
    
    # 边界识别容差
    boundary_tolerance = 1.5 * dr  # 边界识别容差
    
    # ==================== 输出参数 ====================
    output_dir = "output_ogrid"           # 输出目录
    output_time_interval = 1e-5           # VTK输出时间间隔 [s]（每N秒输出一次）
    # 日志输出时间间隔 [s]（每N秒输出一次日志）
    # 建议量级：1e-6 ~ 1e-5，可避免日志文件过大和频繁IO
    log_time_interval = 1e-6
    log_file = "simulation_log.txt"       # 日志文件名（保存在output_dir中）
    progress_bar_width = 50               # 进度条宽度（字符数）
    
    # ==================== 性能优化说明 ====================
    # 注意：代码已优化，不再使用multiprocessing并行计算
    # 性能优化通过以下方式实现：
    # 1. 预计算几何量（边向量、法向量、长度等）- 避免重复计算
    # 2. 预计算边-邻居映射（避免运行时查找）- 大幅提升邻居查找速度
    # 3. 向量化操作（numpy数组操作）- 替代Python循环
    # 4. 直接计算拉普拉斯（避免双重计算梯度+散度）
    # 
    # 对于小到中等规模网格（<50000单元），优化后的串行计算已经足够快
    # 预期性能提升：10-20倍（相比优化前）
    #
    # 以下参数保留以兼容旧代码，但不再使用：
    use_parallel_default = False   # （已弃用）不再使用multiprocessing
    n_cores_default = 1        # （已弃用）不再使用multiprocessing
    
    # ==================== 网格生成参数 ====================
    # 角度间距计算方式：dtheta = 0.0002 / R
    # 可以通过修改此参数来调整角度方向的网格密度
    dtheta_base = 0.0002           # 角度间距基准值 [m]
    
    # ==================== 打包所有参数 ====================
    params = {
        # 物理参数
        'rho0': rho0,
        'c0': c0,
        'nu': nu,
        
        # 几何参数
        'R': R,
        'r_core': r_core,
        'dr': dr,
        
        # 时间参数
        'dt': dt,
        't_end': t_end,
        
        # 声源参数
        'wave_pressure': wave_pressure,
        'v_rde': v_rde,
        'detonation_width': detonation_width,
        'inner_r': inner_r,
        'outer_r': outer_r,
        'omega': omega,
        'sigma': sigma,
        'tau': tau,
        
        # 数值计算参数
        'filter_strength': filter_strength,
        'apply_filter': apply_filter,
        'filter_frequency': filter_frequency,
        'CFL_max': CFL_max,
        'boundary_tolerance': boundary_tolerance,
        
        # 输出参数
        'output_dir': output_dir,
        'output_time_interval': output_time_interval,
        'log_time_interval': log_time_interval,
        'log_file': log_file,
        'progress_bar_width': progress_bar_width,
        
        # 并行计算参数
        'use_parallel_default': use_parallel_default,
        'n_cores_default': n_cores_default,
        
        # 网格生成参数
        'dtheta_base': dtheta_base,
    }
    
    return params


def validate_params(params):
    """
    验证参数字典是否包含所有必需的参数，并检查参数值的合理性
    
    参数:
        params: 参数字典
    
    返回:
        tuple: (is_valid, missing_params, invalid_params)
            is_valid: 是否所有必需参数都存在且有效
            missing_params: 缺少的参数列表（如果is_valid为False）
            invalid_params: 无效的参数列表（值为None或无效值）
    """
    # 定义所有必需参数及其描述
    required_params = {
        # 物理参数
        'rho0': '空气密度 [kg/m³]',
        'c0': '声速 [m/s]',
        'nu': '运动黏度 [m²/s]',
        # 几何参数
        'R': '外半径 [m]',
        'r_core': '内半径 [m]',
        'dr': '径向网格间距 [m]',
        # 时间参数
        'dt': '时间步长 [s]',
        't_end': '结束时间 [s]',
        # 声源参数
        'wave_pressure': '爆轰波压强幅度 [Pa]',
        'v_rde': '爆轰波传播速度 [m/s]',
        'detonation_width': '爆轰波径向宽度（从外壁向内） [m]',
        'inner_r': '环带内半径 [m]',
        'outer_r': '环带外半径 [m]',
        'omega': '角速度 [rad/s]',
        'sigma': '衰减尺度 [m]',
        'tau': '特征时间 [s]',
        # 数值计算参数
        'filter_strength': '滤波强度',
        'apply_filter': '是否应用数值滤波',
        'filter_frequency': '滤波频率',
        'CFL_max': '最大CFL数',
        'boundary_tolerance': '边界识别容差 [m]',
        # 输出参数
        'output_dir': '输出目录',
        'output_time_interval': 'VTK输出时间间隔 [s]',
        'log_time_interval': '日志输出时间间隔 [s]',
        'log_file': '日志文件名',
        'progress_bar_width': '进度条宽度',
        # 性能优化参数（已弃用，保留以兼容）
        'use_parallel_default': '（已弃用）不再使用multiprocessing并行计算',
        'n_cores_default': '（已弃用）不再使用multiprocessing并行计算',
        # 网格生成参数
        'dtheta_base': '角度间距基准值 [m]',
    }
    
    missing_params = []
    invalid_params = []
    
    # 检查参数是否存在
    for param, description in required_params.items():
        if param not in params:
            missing_params.append((param, description))
        else:
            # 检查参数值是否有效（不是None，且对于数值参数，应该是数字）
            value = params[param]
            if value is None:
                invalid_params.append((param, description, "值为None"))
            elif param in ['rho0', 'c0', 'nu', 'R', 'r_core', 'dr', 'dt', 't_end', 
                          'wave_pressure', 'v_rde', 'detonation_width',
                          'inner_r', 'outer_r', 'omega', 
                          'sigma', 'tau', 'filter_strength', 'filter_frequency', 
                          'CFL_max', 'boundary_tolerance', 'output_time_interval',
                          'log_time_interval', 'progress_bar_width', 'dtheta_base']:
                # 数值参数检查
                try:
                    float(value)
                    # 检查是否为正数（某些参数必须为正）
                    if param in ['rho0', 'c0', 'R', 'r_core', 'dr', 'dt', 't_end', 
                                'wave_pressure', 'v_rde', 'detonation_width',
                                'CFL_max', 'output_time_interval',
                                'log_time_interval', 'progress_bar_width', 'dtheta_base']:
                        if float(value) <= 0:
                            invalid_params.append((param, description, f"值必须为正数，当前值: {value}"))
                except (TypeError, ValueError):
                    invalid_params.append((param, description, f"值不是有效数字，当前值: {value}"))
            elif param == 'apply_filter' and not isinstance(value, bool):
                invalid_params.append((param, description, f"值必须是布尔类型，当前值: {value}"))
            elif param in ['output_dir', 'log_file'] and not isinstance(value, str):
                invalid_params.append((param, description, f"值必须是字符串，当前值: {value}"))
    
    is_valid = len(missing_params) == 0 and len(invalid_params) == 0
    return is_valid, missing_params, invalid_params


if __name__ == "__main__":
    # 测试：打印所有参数
    params = control()
    print("=" * 60)
    print("RDE声学模拟参数配置")
    print("=" * 60)
    print("\n物理参数:")
    print(f"  密度 (rho0): {params['rho0']} kg/m³")
    print(f"  声速 (c0): {params['c0']} m/s")
    print(f"  运动黏度 (nu): {params['nu']} m²/s")
    
    print("\n几何参数:")
    print(f"  外半径 (R): {params['R']} m")
    print(f"  内半径 (r_core): {params['r_core']} m")
    print(f"  径向网格间距 (dr): {params['dr']} m")
    
    print("\n时间参数:")
    print(f"  时间步长 (dt): {params['dt']} s")
    print(f"  结束时间 (t_end): {params['t_end']} s")
    print(f"  总步数: {int(params['t_end'] / params['dt'])}")
    
    print("\n声源参数:")
    print(f"  爆轰波压强幅度: {params['wave_pressure']} Pa")
    print(f"  爆轰波传播速度: {params['v_rde']} m/s")
    print(f"  爆轰波径向宽度: {params['detonation_width']} m (从外壁向内)")
    print(f"  角速度 (omega): {params['omega']} rad/s")
    print(f"  衰减尺度 (sigma): {params['sigma']} m")
    print(f"  特征时间 (tau): {params['tau']} s")
    
    print("\n数值计算参数:")
    print(f"  滤波强度: {params['filter_strength']}")
    print(f"  应用滤波: {params['apply_filter']}")
    print(f"  滤波频率: 每 {params['filter_frequency']} 步")
    print(f"  最大CFL数: {params['CFL_max']}")
    
    print("\n输出参数:")
    print(f"  输出目录: {params['output_dir']}")
    print(f"  VTK输出时间间隔: {params['output_time_interval']} s")
    print(f"  日志输出时间间隔: {params['log_time_interval']} s")
    print(f"  日志文件名: {params['log_file']}")
    print(f"  进度条宽度: {params['progress_bar_width']} 字符")
    
    print("\n性能优化:")
    print(f"  代码已优化，不再使用multiprocessing并行计算")
    print(f"  性能优化通过预计算和向量化实现")
    print(f"  预期性能提升: 10-20倍")
    print(f"  适用规模: <50000单元（小到中等规模网格）")
    
    print("\n" + "=" * 60)
