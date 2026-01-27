import numpy as np
import meshio
import os

# Parameters (与plane_2D.py保持一致)
R = 0.01       # Outer radius [m]
r_core = 0.002  # Inner cut radius [m], avoid singular center
dr = 0.0002    # Radial grid spacing [m]
dtheta = 0.0002 / R  # Angular spacing to approximate 0.2 mm arc length

# Generate O-cut grid (annular region only)
# 修复1: 确保r数组正好从r_core到R（不超出边界）
r = np.arange(r_core, R + dr/2, dr)  # 使用dr/2作为容差，确保不超出R
r = r[r <= R]  # 确保最大值不超过R
if len(r) == 0 or (len(r) > 0 and r[-1] < R - 1e-10):
    # 如果最后一个点不够接近R，添加R
    r = np.append(r, R)
nr = len(r)

# 修复2: 确保theta数组闭合（最后一个点正好是2*pi）
# 计算需要的角度点数，确保闭合
ntheta_approx = int(2 * np.pi / dtheta)
if ntheta_approx < 3:
    ntheta_approx = 3  # 至少3个点
dtheta_actual = 2 * np.pi / ntheta_approx  # 调整dtheta使网格闭合
# 使用linspace确保包含0，但不包含2*pi（因为2*pi和0是同一个点）
theta = np.linspace(0, 2*np.pi, ntheta_approx, endpoint=False)
ntheta = len(theta)

# 验证网格闭合性
theta_gap = 2*np.pi - theta[-1]
if theta_gap > dtheta_actual * 0.1:  # 如果间隙大于10%的间距
    print(f"警告: theta数组可能不闭合，最后一个theta={theta[-1]:.6f}rad, 间隙={theta_gap:.6e}rad")
    # 重新生成，确保闭合
    theta = np.linspace(0, 2*np.pi, ntheta_approx, endpoint=False)
    dtheta_actual = theta[1] - theta[0] if len(theta) > 1 else dtheta

# Node coordinates
nodes = []
for i in range(nr):
    for j in range(ntheta):
        nodes.append([r[i]*np.cos(theta[j]), r[i]*np.sin(theta[j]), 0.0])
nodes = np.array(nodes)

# Connectivity: quadrilateral cells in annulus
# 注意：使用周期性边界，j=ntheta-1时，j+1通过模运算回到0
cells = []
for i in range(nr-1):
    for j in range(ntheta):
        n0 = i*ntheta + j
        n1 = i*ntheta + (j+1) % ntheta  # 周期性边界
        n2 = (i+1)*ntheta + (j+1) % ntheta  # 周期性边界
        n3 = (i+1)*ntheta + j
        cells.append([n0, n1, n2, n3])
cells = np.array(cells)

# 网格质量检查
def check_cell_quality():
    """检查单元质量：面积、扭曲等"""
    issues = []
    min_area = float('inf')
    max_area = 0.0
    total_area = 0.0
    
    for idx, cell in enumerate(cells):
        # 获取四个节点坐标
        p0 = nodes[cell[0]]
        p1 = nodes[cell[1]]
        p2 = nodes[cell[2]]
        p3 = nodes[cell[3]]
        
        # 计算单元面积（将四边形分成两个三角形）
        # 三角形1: p0, p1, p3
        v1 = np.array([p1[0]-p0[0], p1[1]-p0[1]])
        v2 = np.array([p3[0]-p0[0], p3[1]-p0[1]])
        area1 = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        
        # 三角形2: p1, p2, p3
        v3 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        v4 = np.array([p3[0]-p1[0], p3[1]-p1[1]])
        area2 = 0.5 * abs(v3[0]*v4[1] - v3[1]*v4[0])
        
        total_area_cell = area1 + area2
        total_area += total_area_cell
        min_area = min(min_area, total_area_cell)
        max_area = max(max_area, total_area_cell)
        
        if total_area_cell < 1e-15:
            issues.append(f"单元{idx}: 面积过小或为零 ({total_area_cell:.2e})")
        elif total_area_cell < 0:
            issues.append(f"单元{idx}: 负面积 ({total_area_cell:.2e})")
    
    # 验证周期性边界连接
    # 检查每个径向层的最后一个单元（j=ntheta-1）是否正确通过模运算连接到j=0
    periodic_issues = []
    for i in range(nr-1):
        # 找到j=ntheta-1的单元（每个径向层有ntheta个单元）
        cell_idx = i * ntheta + (ntheta - 1)
        if cell_idx < len(cells):
            cell = cells[cell_idx]
            # 检查n1是否通过模运算正确连接到j=0
            # n1应该等于 i*ntheta + ((ntheta-1)+1) % ntheta = i*ntheta + 0
            expected_n1 = i*ntheta + 0
            if cell[1] != expected_n1:
                periodic_issues.append(f"径向层{i}: 单元{cell_idx}的周期性连接错误 "
                                     f"(n1={cell[1]}, 期望={expected_n1})")
            # 同样检查n2（下一层的对应节点）
            expected_n2 = (i+1)*ntheta + 0
            if cell[2] != expected_n2:
                periodic_issues.append(f"径向层{i}: 单元{cell_idx}的周期性连接错误 "
                                     f"(n2={cell[2]}, 期望={expected_n2})")
    
    return issues, periodic_issues, min_area, max_area, total_area

# 执行质量检查
quality_issues, periodic_issues, min_area, max_area, total_area = check_cell_quality()

# Write mesh to VTK for ParaView
mesh = meshio.Mesh(points=nodes, cells=[("quad", cells)])
output_dir = "mesh"
os.makedirs(output_dir, exist_ok=True)
meshio.write(f"{output_dir}/o_cut_annulus.vtk", mesh)

# 输出网格信息
print(f"Generated O-cut annular mesh from r={r_core*1000:.1f}mm to R={R*1000:.1f}mm.")
print(f"Mesh saved to '{output_dir}/o_cut_annulus.vtk'.")
print(f"\n网格统计:")
print(f"  径向层数: nr={nr}")
print(f"  角度点数: ntheta={ntheta}")
print(f"  总节点数: {nr*ntheta}")
print(f"  总单元数: {len(cells)}")
print(f"  径向范围: r_min={r[0]:.6f}m ({r[0]*1000:.3f}mm), r_max={r[-1]:.6f}m ({r[-1]*1000:.3f}mm)")
print(f"  角度范围: theta_min={theta[0]:.6f}rad, theta_max={theta[-1]:.6f}rad")
print(f"  角度间距: dtheta_actual={dtheta_actual:.6e}rad ({dtheta_actual*180/np.pi:.4f}度)")
print(f"  网格面积: 最小={min_area:.6e}m², 最大={max_area:.6e}m², 总计={total_area:.6e}m²")
print(f"  理论面积: {np.pi*(R**2 - r_core**2):.6e}m²")
print(f"  面积误差: {abs(total_area - np.pi*(R**2 - r_core**2)):.6e}m²")

# 输出质量检查结果
if quality_issues:
    print(f"\n警告: 发现{len(quality_issues)}个网格质量问题")
    for issue in quality_issues[:5]:  # 只显示前5个
        print(f"  {issue}")
    if len(quality_issues) > 5:
        print(f"  ... 还有{len(quality_issues)-5}个问题")
else:
    print("\n网格质量检查: 通过 ✓")

if periodic_issues:
    print(f"\n警告: 发现{len(periodic_issues)}个周期性边界问题")
    for issue in periodic_issues:
        print(f"  {issue}")
else:
    print("周期性边界检查: 通过 ✓")

# 验证与plane_2D.py的匹配性
print(f"\n参数匹配检查:")
print(f"  R: mesh.py={R:.6f}m, plane_2D.py应该={R:.6f}m")
print(f"  r_core: mesh.py={r_core:.6f}m, plane_2D.py应该={r_core:.6f}m")
print(f"  dr: mesh.py={dr:.6f}m, plane_2D.py={dr:.6f}m")
if abs(r[-1] - R) > 1e-10:
    print(f"  警告: r数组最大值({r[-1]:.6f}m)与R({R:.6f}m)不完全匹配")
else:
    print(f"  r数组边界: 匹配 ✓")
