import pyvista as pv
import numpy as np
from scipy.spatial import KDTree
import igl
import trimesh


def clean_mesh(filepath):
    """
    Clean a 3D mesh from an STL file.

    Parameters
    ----------
    filepath : str
        Path to the STL file.

    Returns
    -------
    tuple
        vertices : np.ndarray
            Array of vertex coordinates.
        faces : np.ndarray
            Array of triangular face indices.

    """

    # 读取网格数据-Read mesh vertices and faces
    v, f = igl.read_triangle_mesh(filepath)

    # 移除未被面片引用的顶点-Remove unreferenced vertices
    v, f, _, _ = igl.remove_unreferenced(v, f)

    # 处理异常索引-Adjust indices if minimum face index is 1
    if f.min() == 1:
        f = f - 1

    # 构建 trimesh 进行进一步清理-Convert to trimesh for further cleaning
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_infinite_values()

    # 删除退化面片-Remove degenerate faces
    mask = mesh.nondegenerate_faces().astype(bool)
    mesh.update_faces(mask)

    return mesh.vertices, mesh.faces


def hausdorff_distance(points1, points2):

    """Compute the Hausdorff distance between two point clouds. """

    tree1 = KDTree(points1)
    d1_to_2, _ = tree1.query(points2)
    tree2 = KDTree(points2)
    d2_to_1, _ = tree2.query(points1)
    return max(np.max(d1_to_2), np.max(d2_to_1))


def visualize_error(vertices, decimated_vertices):

    """Visualize the Hausdorff error between the original and decimated mesh"""

    error_mesh = pv.PolyData(vertices)
    error_mesh["Distance"] = KDTree(decimated_vertices).query(vertices)[0]

    plotter = pv.Plotter()
    plotter.add_mesh(error_mesh,
                     scalars="Distance",
                     cmap="coolwarm",
                     opacity=1.0,
                     show_edges=True,
                     scalar_bar_args={"title": "error(mm)"})
    plotter.add_mesh(pv.PolyData(decimated_vertices), color="cyan", show_edges=True, opacity=0.8)
    plotter.show()


def normalize_mesh(vertices):

    """Normalize a mesh so that it is centered at the origin and scaled to fit inside a unit sphere."""

    # centroid alignment
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # Scale to fit inside the unit sphere
    max_radius = np.max(np.linalg.norm(centered_vertices, axis=1))
    normalized_vertices = centered_vertices / max_radius

    # radii = np.linalg.norm(normalized_vertices, axis=1)
    # print(f"[Radius range after normalization]: {radii.min():.4f} ~ {radii.max():.4f}")

    return normalized_vertices


# Function to visualize normalized mesh
def visualize_normalization(normalized_vertices, decimated_faces):

    """Visualize and validate the normalized mesh"""

    # Convert triangular faces into PyVista format: [3, v1, v2, v3, ...]
    pyvista_faces = np.insert(decimated_faces.astype(np.int64), 0, 3, axis=1).ravel()

    # 验证面片数组格式-Print validation info
    print("Face array validation:")
    print(f"Original shape: {decimated_faces.shape} (expected [n_faces, 3])")
    print(f"Converted shape: {pyvista_faces.shape} (expected [n_faces*4,])")
    print(f"Index range: {decimated_faces.min()} ~ {decimated_faces.max()} "
          f"(should be < {len(normalized_vertices)})")

    # 归一化网格后可视化-visualize
    normalized_mesh = pv.PolyData(normalized_vertices, pyvista_faces)
    plotter = pv.Plotter()
    plotter.add_mesh(normalized_mesh,
                     color="lightblue",
                     show_edges=True,
                     edge_color="gray",
                     opacity=1,
                     label=f"Mesh after normalization ({len(pyvista_faces) // 4} faces)")
    plotter.add_axes(box_args={'color': 'red'})
    plotter.add_title(f"\nvertices_number: {len(normalized_vertices)}\nfaces_number: {len(pyvista_faces) // 4}")
    plotter.show()


def visualize_with_lighting_and_wire(normalized_vertices, decimated_faces):

    """Visualization used in fig_SPHARM_cluster"""

    pyvista_faces = np.insert(decimated_faces.astype(np.int64), 0, 3, axis=1).ravel()
    mesh = pv.PolyData(normalized_vertices, pyvista_faces)

    plotter = pv.Plotter(window_size=[800, 800])
    plotter.set_background('white')

    plotter.add_mesh(
        mesh,
        color='#f2f2f2',
        opacity=0.5,
        show_edges=False,
        lighting=True,
        smooth_shading=True
    )

    plotter.add_mesh(
        mesh,
        style='wireframe',
        color='#999999',
        line_width=1.2,
        lighting=False
    )

    plotter.add_axes()
    plotter.show()