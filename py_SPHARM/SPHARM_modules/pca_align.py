import igl
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
import copy


def clean_mesh(filepath):

    v, f = igl.read_triangle_mesh(filepath)

    v, f, _, _ = igl.remove_unreferenced(v, f)

    if f.min() == 1:
        f = f - 1

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.remove_infinite_values()

    non_deg_mask = mesh.nondegenerate_faces()
    mesh.update_faces(non_deg_mask)

    return mesh.vertices, mesh.faces


def decimate_mesh(vertices, faces, target_faces=20000):

    result = igl.decimate(vertices, faces, target_faces)
    v_decim = result[1]
    f_decim = result[2]

    return v_decim, f_decim


def normalize_mesh(vertices):

    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    max_radius = np.max(np.linalg.norm(centered_vertices, axis=1))
    normalized_vertices = centered_vertices / max_radius

    radii = np.linalg.norm(normalized_vertices, axis=1)
    # print(f"r range: {radii.min():.4f} ~ {radii.max():.4f}")

    return normalized_vertices


def visualize_normalization(normalized_vertices, decimated_faces):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normalized_vertices[:, 0], normalized_vertices[:, 1], normalized_vertices[:, 2],
               s=1, c='b', alpha=0.6)
    ax.set_title("Normalized Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    pyvista_faces = np.insert(decimated_faces.astype(np.int64), 0, 3, axis=1).ravel()

    normalized_mesh = pv.PolyData(normalized_vertices, pyvista_faces)
    plotter = pv.Plotter()
    plotter.add_mesh(normalized_mesh,
                     color="lightblue",
                     show_edges=True,
                     edge_color="gray",
                     opacity=0.8)
    plotter.add_axes(box_args={'color': 'red'})
    plotter.show()


def robust_pca_alignment(points, enforce_direction=True, verbose=True):
    """
    PCA alignment for 3D point clouds

    Parameters:
    - points: (N,3) numpy array of input 3D points
    - enforce_direction: bool, whether to enforce consistent principal direction
    - verbose: bool, whether to print debug/verification info

    Returns:
    - aligned_points: numpy array of aligned points
    - rotation_matrix: 3x3 rotation matrix used for alignment
    """

    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("Input point cloud must be an Nx3 NumPy array")
    if len(points) < 3:
        raise ValueError("At least 3 points are required to compute principal axes")

    # 去中心化-Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # 使用SVD分解提高数值稳定性-Use SVD for numerical stability
    cov_matrix = np.cov(centered.T)
    U, s, Vt = np.linalg.svd(cov_matrix)

    # 右手坐标系-right-handed coordinate system
    rotation_matrix = Vt.T

    if np.linalg.det(rotation_matrix) < 0:
        #if verbose:
            #print("Detected left-handed system; correcting...")
        rotation_matrix[:, 2] *= -1

    if enforce_direction:
        projected = centered @ rotation_matrix

        if np.median(projected[:, 0]) < 0:
            #if verbose:
                #print("Detected flipped principal direction; correcting...")
            rotation_matrix[:, 0] *= -1

    aligned_points = centered @ rotation_matrix

    #if verbose:
        #print("\n===== 验证报告 =====")
        #print("旋转矩阵行列式:", np.linalg.det(rotation_matrix))
        #print("主成分方向:")
        #print(f"PC1 (X轴): {rotation_matrix[:, 0]}")
        #print(f"PC2 (Y轴): {rotation_matrix[:, 1]}")
        #print(f"PC3 (Z轴): {rotation_matrix[:, 2]}")
        #print("对齐后坐标统计:")
        #print(f"X范围: [{aligned_points[:, 0].min():.3f}, {aligned_points[:, 0].max():.3f}]")
        #print(f"Y范围: [{aligned_points[:, 1].min():.3f}, {aligned_points[:, 1].max():.3f}]")
        #print(f"Z范围: [{aligned_points[:, 2].min():.3f}, {aligned_points[:, 2].max():.3f}]")

    return aligned_points


def plot_pca_aligned_points(points):
    """
    Visualize the aligned point cloud, and draw the XYZ axes at the centroid
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=1, c='gray', alpha=0.6, label='Point Cloud')

    centroid = np.mean(points, axis=0)
    ax.scatter(centroid[0], centroid[1], centroid[2],
               s=50, c='k', marker='o', label='Centroid')

    max_range = np.max(np.ptp(points, axis=0))
    axis_length = max_range * 1.1

    ax.plot([centroid[0] - axis_length / 2, centroid[0] + axis_length / 2],
            [centroid[1], centroid[1]],
            [centroid[2], centroid[2]],
            color='r', lw=2, label='X axis')

    ax.plot([centroid[0], centroid[0]],
            [centroid[1] - axis_length / 2, centroid[1] + axis_length / 2],
            [centroid[2], centroid[2]],
            color='g', lw=2, label='Y axis')

    ax.plot([centroid[0], centroid[0]],
            [centroid[1], centroid[1]],
            [centroid[2] - axis_length / 2, centroid[2] + axis_length / 2],
            color='b', lw=2, label='Z axis')

    ax.set_box_aspect([1, 1, 1])

    ax.set_title("PCA Aligned Point Cloud with Coordinate Axes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def load_template(template_path):

    """prepare a template model"""

    v_temp, f_temp = clean_mesh(template_path)

    v_temp_decim, f_temp_decim = decimate_mesh(v_temp, f_temp, 20000)

    template_normalized = normalize_mesh(v_temp_decim)

    template_aligned, _ = robust_pca_alignment(template_normalized)

    return template_aligned


def prepare_pointcloud(points):

    """Convert a NumPy array to an Open3D PointCloud object"""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def multi_resolution_icp(source, target, init_transform=np.eye(4), verbose=True):

    """
    Multi-resolution ICP (Iterative Closest Point) registration.

    Parameters:
    - source: source point cloud to be aligned (o3d.geometry.PointCloud)
    - target: target/template point cloud (o3d.geometry.PointCloud)
    - init_transform: initial 4x4 transformation matrix
    - verbose: whether to print registration evaluation info

    Returns:
    - 4x4 optimized transformation matrix aligning source to target
    """

    # Multi-scale parameters
    voxel_sizes = [0.1, 0.05, 0.02]
    max_iterations = [200, 100, 50]

    current_transform = init_transform
    final_result = None

    for i, (voxel_size, max_iter) in enumerate(zip(voxel_sizes, max_iterations)):
        # Downsample point clouds
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # Estimate normals (required for Point-to-Plane ICP)
        source_down.estimate_normals()
        target_down.estimate_normals()

        # Perform ICP at current resolution
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=voxel_size * 2,
            init=current_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter))

        current_transform = result.transformation

        final_result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=voxel_sizes[-1],
            init=current_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        if verbose:
            evaluate_icp_result(final_result)

    return final_result.transformation


def evaluate_icp_result(result):
    """ICP evaluation function (unit checks and exception handling)"""
    try:

        print(f"Registration fitness: {result.fitness:.4f} (0-1, 1 is best)")
        print(f"Inlier RMSE: {result.inlier_rmse * 1000:.2f} mm")

        # Analyze transformation matrix
        T = result.transformation
        R = T[:3, :3]
        t = T[:3, 3]

        # Check rotation matrix
        det = np.linalg.det(R)
        if abs(det - 1) > 1e-3:
            print(f"Warning: Rotation matrix determinant abnormal {det:.6f} ")

        # translation vector
        trans_norm = np.linalg.norm(t)
        print(f"Translation vector magnitude: {trans_norm * 1000:.2f} mm")
        print(f"Translation vector: [{t[0] * 1000:.2f}, {t[1] * 1000:.2f}, {t[2] * 1000:.2f}] mm")

        # Condition number of transformation matrix
        cond_num = np.linalg.cond(T)
        print(f"Transformation matrix condition number: {cond_num:.2e}")
        if cond_num > 1e6:
            print("Warning: Matrix is near-singular, registration result may be unreliable")

        # Number of corresponding points
        if hasattr(result, "correspondence_set"):
            print(f"Number of valid correspondences: {len(result.correspondence_set)}")
        else:
            print("Correspondence information unavailable")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


def align_all_to_template(template_path, stone_paths):

    """alignment workflow for multiple models to a template"""

    # 加载模板-Load template
    template_points = load_template(template_path)
    template_pcd = prepare_pointcloud(template_points)

    # 预处理模板-Preprocess
    template_pcd.estimate_normals()

    aligned_results = []

    for path in stone_paths:

        # 1. 预处理当前标本-clean and normalize
        v_clean, f_clean = clean_mesh(path)
        v_decim, f_decim = decimate_mesh(v_clean, f_clean, 20000)
        points_norm = normalize_mesh(v_decim)

        # 2. PCA粗对齐-alignment using PCA
        points_pca, rot_matrix = robust_pca_alignment(points_norm)

        # 3. 转换为Open3D格式-change format
        stone_pcd = prepare_pointcloud(points_pca)
        stone_pcd.estimate_normals()

        # 4. ICP精对齐-alignment using multi-resolution ICP
        transform = multi_resolution_icp(stone_pcd, template_pcd)

        # 5. 传递变换矩阵到可视化函数-visualize
        visualize_alignment(stone_pcd, template_pcd, transform)
        # visualize_icp_result(stone_pcd, template_pcd, transform)

        stone_pcd.transform(transform)

        # 6. 保存结果-save
        aligned_results.append(np.asarray(stone_pcd.points))

    return aligned_results


def visualize_alignment(source_pcd, target_pcd, transform):
    """visualize alignment"""

    source_original = copy.deepcopy(source_pcd)

    source_aligned = copy.deepcopy(source_pcd)
    source_aligned.transform(transform)

    source_original.paint_uniform_color([1, 0, 0])  # red：before
    source_aligned.paint_uniform_color([0, 0, 1])  # blue: after
    target_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries(
        [source_original, source_aligned, target_pcd],
        window_name="Alignment Comparison",
        width=1200,
        height=800,
        left=200,
        top=200
    )


def visualize_icp_result(source, target, transform):
    """ visualize icp result"""

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.transform(transform)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries([source_temp, target_temp, coord_frame],
                                      window_name="ICP Result Verification",
                                      width=1200,
                                      height=800,
                                      point_show_normal=True)


