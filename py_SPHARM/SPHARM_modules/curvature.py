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
    Robustly simplify the mesh using libigl, handling different API versions.
    """
    # Quick check: if mesh is already smaller than target, do nothing
    if target_faces >= len(faces):
        return vertices, faces

    try:
        # 1. Prepare data: IGL requires Fortran order and specific dtypes
        # We use np.asarray to strip trimesh wrappers, then force dtypes and order
        v_igl = np.asfortranarray(np.asarray(vertices), dtype=np.float64)
        f_igl = np.asfortranarray(np.asarray(faces), dtype=np.int32)
        target = int(target_faces)

        # 2. Call decimate and handle varying return signatures (4 vs 5 values)
        res = igl.decimate(v_igl, f_igl, target)

        if len(res) == 5:
            # Version returns (success, V, F, face_indices, vertex_indices)
            success, simplified_vertices, simplified_faces, _, _ = res
            if not success:
                print("Warning: igl.decimate indicated failure. Returning original.")
                return vertices, faces
        else:
            # Version returns (V, F, face_indices, vertex_indices)
            simplified_vertices, simplified_faces, _, _ = res

        # 3. Final safety check: ensure the resulting mesh isn't empty
        if simplified_vertices is None or len(simplified_faces) == 0:
            return vertices, faces

        return simplified_vertices, simplified_faces

    except Exception as e:
        print(f"Decimation error: {e}. Falling back to original mesh.")
        return vertices, faces

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stl_file = os.path.join(base_dir, "..", "SPHARM_main", "3d_models_QSY_spheroids_multi_poly", "QSY-A-2797-Spheroid.stl")
    stl_file = os.path.abspath(stl_file)

    mesh = trimesh.load(stl_file)
    simple_v, simple_f = simplify_mesh(mesh.vertices, mesh.faces, target_faces=10000)
    simple_mesh = trimesh.Trimesh(vertices=simple_v, faces=simple_f)
    curvatures = compute_curvature(simple_mesh)
    visualize_curvature(simple_mesh, curvatures, title="QSY-A-2797_Curvature")

    # Batch process the entire folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "..", "SPHARM_main", "3d_models_QSY_spheroids_multi_poly")
    output_folder = os.path.join(base_dir, "..", "..", "analysis", "data", "raw_data",
                                    "SPHARM_sphericity_curvature_result")

    output_csv = os.path.join(output_folder, "curvature.csv")

    os.makedirs(output_folder, exist_ok=True)

    df = batch_average_curvature(input_folder)
    df.to_csv(output_csv, index=False)

    print(f"curvature saved to {output_csv}")
