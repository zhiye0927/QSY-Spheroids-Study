
import mesh_processing
import trimesh
import igl
from trimesh.smoothing import filter_laplacian


stl_path = "E:/spheroids_analysis/core_3d/4_Box.stl"
vertices, faces = mesh_processing.clean_mesh(stl_path)

target_faces = 1000
result = igl.decimate(vertices, faces, target_faces)
decimated_vertices = result[1]
decimated_faces = result[2]
mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
filter_laplacian(mesh, iterations=3)
decimated_vertices = mesh.vertices

mesh_processing.visualize_error(vertices, decimated_vertices)

normalized_vertices = mesh_processing.normalize_mesh(decimated_vertices)

mesh_processing.visualize_with_lighting_and_wire(normalized_vertices, decimated_faces)



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from matplotlib.colors import LinearSegmentedColormap

# 定义橙色到绿色的渐变色
colors = ["green", "white", "orange"]
cmap = LinearSegmentedColormap.from_list("GreenOrange", colors, N=256)

l, m = 0, 0

phi = np.linspace(0, 2 * np.pi, 200)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

Y_lm = sph_harm(m, l, phi, theta)
r = np.real(Y_lm)
r_abs = np.abs(r)

x = r_abs * np.sin(theta) * np.cos(phi)
y = r_abs * np.sin(theta) * np.sin(phi)
z = r_abs * np.cos(theta)

norm = plt.Normalize(-r.max(), r.max())
colors_mapped = cmap(norm(r))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, facecolors=colors_mapped, rstride=1, cstride=1, linewidth=0, antialiased=False)

ax.set_axis_off()
ax.set_box_aspect([1,1,1])

plt.title(f"Spherical Harmonic $Y_{{{l}}}^{{{m}}}$ Real Part with Orange-Green")
plt.show()