import trimesh
import igl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def simplify_mesh(vertices, faces, target_faces):

    """
    simplify the mesh

    Parameters
    ----------
    vertices : numpy.ndarray, shape (N, 3)
        Original vertex array.
    faces : numpy.ndarray, shape (M, 3)
        Original triangular face array (indices into vertices).
    target_faces : int
        Target number of faces for the simplified mesh.
        Must be less than the original face count.

    Returns
    -------
    simplified_vertices : numpy.ndarray
        Vertex array of the simplified mesh.
    simplified_faces : numpy.ndarray
        Face array of the simplified mesh.
    """

    if target_faces >= len(faces):
        print("Target face count is greater than or equal to original")
        return vertices, faces

    result = igl.decimate(vertices, faces, target_faces)
    simplified_vertices = result[1]
    simplified_faces = result[2]

    return simplified_vertices, simplified_faces


def compute_curvature(mesh, k_neighbors=30):

    """
        Compute an approximate curvature for each triangle in a mesh.
        The curvature is estimated as the angle between the triangle's normal
        and the normal of a locally fitted plane through its neighboring triangle centers.

        Parameters:
        - mesh: a trimesh object with `triangles_center` and `face_normals`
        - k_neighbors: number of nearest neighbors used to fit the local plane

        Returns:
        - curvatures: numpy array of curvature angles (in degrees) for each triangle
    """

    # Get the center points of all triangles
    tri_centers = mesh.triangles_center

    # Get the normal vectors of all triangles
    tri_normals = mesh.face_normals

    # Build a nearest neighbors search for triangle centers
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(tri_centers)
    _, indices = nbrs.kneighbors(tri_centers)

    curvatures = []

    # Loop over each triangle
    for i, neighbors in enumerate(indices):
        if len(neighbors) <= 1:
            curvatures.append(np.nan)
            continue

        # Get neighbor points (excluding the triangle itself)
        neighbor_points = tri_centers[neighbors[1:]]
        # Fit a local plane using SVD
        # Center the points by subtracting the mean
        # Vt[-1] gives the direction of smallest variance => local plane normal
        U, S, Vt = np.linalg.svd(neighbor_points - neighbor_points.mean(axis=0))
        local_normal = Vt[-1]

        # Compute the angle between the actual triangle normal and local plane normal
        dot = np.dot(tri_normals[i], local_normal)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.arccos(abs(dot))

        curvature_deg = np.degrees(angle)
        curvatures.append(curvature_deg)

    return np.array(curvatures)


def batch_average_curvature(input_folder, target_faces=10000, k_neighbors=30):

    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.stl')]
    results = []

    for file in files:
        path = os.path.join(input_folder, file)
        mesh = trimesh.load(path)
        vertices, faces = mesh.vertices, mesh.faces
        simple_v, simple_f = simplify_mesh(vertices, faces, target_faces)
        simple_mesh = trimesh.Trimesh(vertices=simple_v, faces=simple_f)
        curvatures = compute_curvature(simple_mesh, k_neighbors)
        avg_curvature = np.nanmean(curvatures)
        filename_no_ext = os.path.splitext(file)[0]
        results.append({'filename': filename_no_ext, 'average_curvature_deg': avg_curvature})
        print(f"{filename_no_ext}: average curvature = {avg_curvature:.3f} degrees")

    return pd.DataFrame(results)


def visualize_curvature(mesh, curvatures, title="Curvature (degrees)"):

    """
    Visualize curvature distribution on a mesh
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=np.nanmin(curvatures), vmax=np.nanmax(curvatures))
    colors = plt.cm.viridis(norm(curvatures))

    mesh_tris = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(mesh_tris, facecolors=colors, edgecolor='none')
    ax.add_collection3d(collection)

    ax.auto_scale_xyz(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.title(title)

    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array(curvatures)
    fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Curvature (°)')

    plt.show()



if __name__ == "__main__":

    # A single stl file_process and visualize
    stl_file = r"E:\spheroids_analysis\spheroids_3d\QSY-A-2797.stl"
    mesh = trimesh.load(stl_file)
    simple_v, simple_f = simplify_mesh(mesh.vertices, mesh.faces, target_faces=10000)
    simple_mesh = trimesh.Trimesh(vertices=simple_v, faces=simple_f)
    curvatures = compute_curvature(simple_mesh)
    visualize_curvature(simple_mesh, curvatures, title="QSY-A-2797_Curvature")

    # Batch process the entire folder
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_folder = r"E:\spheroids_analysis\spheroids_3d"
    output_folder = os.path.join(base_dir, "analysis", "data", "raw_data", "SPHARM_sphericity_curvature_result")

    output_csv = os.path.join(output_folder, "curvature.csv")

    os.makedirs(output_folder, exist_ok=True)

    df = batch_average_curvature(input_folder)
    df.to_csv(output_csv, index=False)

    print(f"curvature saved to {output_csv}")