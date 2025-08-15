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
    使用 libigl 的 decimate 函数简化网格。

    参数：
    - vertices: numpy.ndarray, shape (N, 3) 原始顶点数组
    - faces: numpy.ndarray, shape (M, 3) 原始三角面片数组
    - target_faces: int 目标面片数量，不能超过原始面片数

    返回：
    - simplified_vertices: numpy.ndarray, 简化后的顶点数组
    - simplified_faces: numpy.ndarray, 简化后的面片数组
    """
    if target_faces >= len(faces):
        print("目标面片数大于等于原始面片数，不进行简化。")
        return vertices, faces


    result = igl.decimate(vertices, faces, target_faces)
    simplified_vertices = result[1]
    simplified_faces = result[2]

    return simplified_vertices, simplified_faces


def compute_curvature(mesh, k_neighbors=30):
    tri_centers = mesh.triangles_center
    tri_normals = mesh.face_normals
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(tri_centers)
    _, indices = nbrs.kneighbors(tri_centers)
    curvatures = []
    for i, neighbors in enumerate(indices):
        if len(neighbors) <= 1:
            curvatures.append(np.nan)
            continue
        neighbor_points = tri_centers[neighbors[1:]]
        U, S, Vt = np.linalg.svd(neighbor_points - neighbor_points.mean(axis=0))
        local_normal = Vt[-1]
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
        results.append({'filename': file, 'average_curvature_deg': avg_curvature})
        print(f"{file}: average curvature = {avg_curvature:.3f} degrees")
    return pd.DataFrame(results)

def visualize_curvature(mesh, curvatures, title="Local Curvature (degrees)"):
    """
    可视化网格上的曲率分布。

    参数：
    - mesh: trimesh.Trimesh 对象，已简化网格
    - curvatures: numpy.ndarray, 每个面片的曲率值
    - title: str，可选，图像标题
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 面片颜色映射
    norm = plt.Normalize(vmin=np.nanmin(curvatures), vmax=np.nanmax(curvatures))
    colors = plt.cm.viridis(norm(curvatures))

    # 构建三角面片集合
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

# 加载单个样本，简化并计算曲率
# mesh = trimesh.load(r"E:\spheroids_analysis\spheroids_3d\QSY-A-2797.stl")
# simple_v, simple_f = simplify_mesh(mesh.vertices, mesh.faces, target_faces=10000)
# simple_mesh = trimesh.Trimesh(vertices=simple_v, faces=simple_f)
# curvatures = compute_curvature(simple_mesh)

# 调用可视化函数
# visualize_curvature(simple_mesh, curvatures, title="QSY-A-2797 Curvature")

if __name__ == "__main__":
    input_folder = r"E:\spheroids_analysis\spheroids_3d"
    output_csv = r"E:\spheroids_analysis\spheroids_3d\average_curvatures.csv"

    df = batch_average_curvature(input_folder)
    df.to_csv(output_csv, index=False)
    print(f"平均曲率已保存到 {output_csv}")