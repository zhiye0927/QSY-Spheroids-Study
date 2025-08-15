import os
import numpy as np
import pyshtools as pysh
import trimesh
import igl
from trimesh.smoothing import filter_laplacian

from SPHARM_modules import mesh_processing, pca_align, spherical_harmonics, statistics_analysis


def process_single_mesh(stl_path, output_dir, target_faces=20000, grid_size=256, lmax=20):
    """处理单个STL文件并返回特征"""
    try:
        # 1. 清理网格
        vertices, faces = mesh_processing.clean_mesh(stl_path)

        # 2. 统一面片数
        result = igl.decimate(vertices, faces, target_faces)
        decimated_vertices = result[1]
        decimated_faces = result[2]
        mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
        filter_laplacian(mesh, iterations=3)
        decimated_vertices = mesh.vertices

        # 3. 归一化处理
        normalized_vertices = mesh_processing.normalize_mesh(decimated_vertices)
        align_vertices = pca_align.robust_pca_alignment(normalized_vertices, enforce_direction=True)

        # 4. 转换到球坐标并插值
        spherical_coords = spherical_harmonics.cartesian_to_spherical(align_vertices)
        R, theta, phi = spherical_coords.T
        grid_r = spherical_harmonics.spherical_interpolate(R, theta, phi, grid_size)

        # 5. 计算球谐系数
        clm = spherical_harmonics.compute_spherical_harmonics(grid_r, normalization_method='zero-component')

        # 球谐零阶系数值
        print(f"零阶系数值: {clm[0, 0, 0]:.3f} (应为接近1)")

        clm_sh = pysh.SHCoeffs.from_array(
            clm,
            normalization='unnorm',  # 与自定义归一化一致
            csphase=1, lmax=lmax
        ).pad(lmax=lmax)

        # 计算球度（新增）
        sphericity = np.sum(np.abs(clm_sh.coeffs) ** 2)  # 直接使用系数计算

        # 直接计算零阶功率
        c00 = clm[0, 0, 0]
        manual_power_l0 = np.abs(c00) ** 2  # C_{0,0}^2
        print(f"手动计算零阶功率: {manual_power_l0:.3f}")

        # 6. 提取功率谱特征
        full, stat = spherical_harmonics.process_spherical_harmonics(clm_sh)

        # print(f"stat 的维度: {stat.shape}")  # 应为 (21,4)
        if stat.shape[1] != 4:
            raise ValueError(f"stat 列数异常！实际列数：{stat.shape[1]}")
        total_power = stat["total_power"].astype(float)

        # 保存结果
        base_name = os.path.splitext(os.path.basename(stl_path))[0]
        np.savetxt(f"{output_dir}/{base_name}_power.csv", stat, delimiter=",")
        np.savetxt(f"{output_dir}/{base_name}_coeffs.csv", spherical_harmonics.clm_to_1d_standard(clm), delimiter=",")

        print(f"\n==== 调试模型: {os.path.basename(stl_path)} ====")
        print("归一化后半径范围:", np.min(np.linalg.norm(normalized_vertices, axis=1)),
              np.max(np.linalg.norm(normalized_vertices, axis=1)))
        print("零阶功率值:", total_power[0])
        print("最大非零阶功率:", np.max(total_power[1:]))
        print("=" * 50)

        return total_power, sphericity
    except Exception as e:
        print(f"处理{stl_path}时出错: {str(e)}")
        return None


def batch_process(input_dir, output_dir):
    """批量处理目录中的所有STL文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_stats = []
    all_sphericities = []
    filenames = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.stl'):
            stl_path = os.path.join(input_dir, filename)
            power_stat, sphericity = process_single_mesh(stl_path, output_dir)
            if power_stat is not None:
                all_stats.append(power_stat)
                all_sphericities.append(sphericity)
                filenames.append(os.path.splitext(filename)[0])

    sphericity_table = np.column_stack((filenames, all_sphericities))
    np.savetxt(f"{output_dir}/sphericity_SPHARM.csv",
               sphericity_table,
               delimiter=",",
               header="ID,sphericity_SPHARM",
               fmt="%s",
               comments='')

    if len(all_stats) > 0:
        all_stats_array = np.array(all_stats)
        np.savez(f"{output_dir}/all_data.npz",
                 stats=all_stats_array,
                 sphericities=np.array(all_sphericities),
                 filenames=filenames)

        var_per_degree = np.var(all_stats_array, axis=0)
        degrees = np.arange(len(var_per_degree))
        data_to_save = np.column_stack((degrees, var_per_degree))
        np.savetxt(f"{output_dir}/variance_per_degree.csv",
                   data_to_save,
                   delimiter=",",
                   header="degree,variance",
                   comments='')

        statistics_analysis.analyze_variance(all_stats_array, filenames)
        # statistics_analysis.analyze_pca(all_stats_array, filenames)
        # statistics_analysis.calculate_power_distance(all_stats_array, filenames, output_dir, lmax=20)
        statistics_analysis.analyze_umap2(
            all_stats_array,  # 将球度作为新特征
            filenames,
            output_dir,
            lmax=20
        )



if __name__ == "__main__":
    input_directory = "E:\spheroids_analysis\core_3d"
    output_directory = "E:\spheroids_analysis\core_3d_output"
    batch_process(input_directory, output_directory)
