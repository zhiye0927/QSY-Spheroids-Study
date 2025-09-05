import mesh_processing
import trimesh
import igl
from trimesh.smoothing import filter_laplacian
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from matplotlib.colors import LinearSegmentedColormap


def visualize_mesh(stl_path, target_faces=1000, laplacian_iterations=3):
    """
    mesh used in SPHARM_cluster
    """

    vertices, faces = mesh_processing.clean_mesh(stl_path)

    result = igl.decimate(vertices, faces, target_faces)
    decimated_vertices = result[1]
    decimated_faces = result[2]

    mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
    filter_laplacian(mesh, iterations=laplacian_iterations)
    decimated_vertices = mesh.vertices

    mesh_processing.visualize_error(vertices, decimated_vertices)

    normalized_vertices = mesh_processing.normalize_mesh(decimated_vertices)

    mesh_processing.visualize_with_lighting_and_wire(normalized_vertices, decimated_faces)

stl_file = "E:/spheroids_analysis/core_3d/4_Box.stl"
visualize_mesh(stl_file, target_faces=1000, laplacian_iterations=3)


def visualize_spherical_harmonic_real(l=2, m=2, resolution_phi=200, resolution_theta=100, colors=["#66c2a5", "white", "orange"]):
    """
    Visualize the real part of a spherical harmonic Y_l^m on a sphere
    """

    cmap = LinearSegmentedColormap.from_list("GreenOrangeCustom", colors, N=256)

    phi = np.linspace(0, 2 * np.pi, resolution_phi)
    theta = np.linspace(0, np.pi, resolution_theta)
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
    ax.set_box_aspect([1, 1, 1])
    max_range = r_abs.max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.title(f"Spherical Harmonic $Y_{{{l}}}^{{{m}}}$ Real Part with Custom Colors", fontsize=16, pad=20)
    plt.show()

visualize_spherical_harmonic_real(l=0, m=0)