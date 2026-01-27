import numpy as np
import meshio
import os

# Parameters
R = 0.01       # Outer radius [m]
r_core = 0.002  # Inner cut radius [m], avoid singular center
dr = 0.0002    # Radial grid spacing [m]
dtheta = 0.0002 / R  # Angular spacing to approximate 0.2 mm arc length

# Generate O-cut grid (annular region only)
r = np.arange(r_core, R + dr, dr)
theta = np.arange(0, 2*np.pi, dtheta)
nr, ntheta = len(r), len(theta)

# Node coordinates
nodes = []
for i in range(nr):
    for j in range(ntheta):
        nodes.append([r[i]*np.cos(theta[j]), r[i]*np.sin(theta[j]), 0.0])
nodes = np.array(nodes)

# Connectivity: quadrilateral cells in annulus
cells = []
for i in range(nr-1):
    for j in range(ntheta):
        n0 = i*ntheta + j
        n1 = i*ntheta + (j+1) % ntheta
        n2 = (i+1)*ntheta + (j+1) % ntheta
        n3 = (i+1)*ntheta + j
        cells.append([n0, n1, n2, n3])
cells = np.array(cells)

# Write mesh to VTK for ParaView
mesh = meshio.Mesh(points=nodes, cells=[("quad", cells)])
output_dir = "mesh"
os.makedirs(output_dir, exist_ok=True)
meshio.write(f"{output_dir}/o_cut_annulus.vtk", mesh)

print(f"Generated O-cut annular mesh from r={r_core*1000:.1f}mm to R={R*1000:.1f}mm.")
print("Mesh saved to 'mesh/o_cut_annulus.vtk'.")

