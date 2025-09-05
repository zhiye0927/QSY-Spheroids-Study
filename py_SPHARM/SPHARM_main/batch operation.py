import os
import numpy as np
import pyshtools as pysh
import trimesh
import igl
from trimesh.smoothing import filter_laplacian

from SPHARM_modules import mesh_processing, pca_align, spherical_harmonics, statistics_analysis


def process_single_mesh(stl_path, output_dir, target_faces=20000, grid_size=256, lmax=20):

    """ Process a single STL mesh file and compute spherical harmonics"""

    try:
        # 1. 清理网格-Clean mesh
        vertices, faces = mesh_processing.clean_mesh(stl_path)

        # 2. 统一面片数-Mesh decimation
        result = igl.decimate(vertices, faces, target_faces)
        decimated_vertices = result[1]
        decimated_faces = result[2]
        mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
        filter_laplacian(mesh, iterations=3)
        decimated_vertices = mesh.vertices

        # 3. 归一化处理-Normalize
        normalized_vertices = mesh_processing.normalize_mesh(decimated_vertices)
        align_vertices = pca_align.robust_pca_alignment(normalized_vertices, enforce_direction=True)

        # 4. 转换到球坐标并插值-Convert to spherical coordinates and interpolate
        spherical_coords = spherical_harmonics.cartesian_to_spherical(align_vertices)
        R, theta, phi = spherical_coords.T
        grid_r = spherical_harmonics.spherical_interpolate(R, theta, phi, grid_size)

        # 5. 计算球谐系数-Compute spherical harmonic coefficients
        clm = spherical_harmonics.compute_spherical_harmonics(grid_r, normalization_method='zero-component')

        clm_sh = pysh.SHCoeffs.from_array(
            clm,
            normalization='unnorm',
            csphase=1, lmax=lmax
        ).pad(lmax=lmax)

        # 6. 计算球谐函数能量-spherical harmonics energy
        SHE = np.sum(np.abs(clm_sh.coeffs) ** 2)

        # 6. 提取功率谱特征-spherical harmonics power spectrum
        full, spectrum = spherical_harmonics.process_spherical_harmonics(clm_sh)

        total_power = spectrum["total_power"].astype(float)

        # 7. 存储-save
        base_name = os.path.splitext(os.path.basename(stl_path))[0]
        np.savetxt(f"{output_dir}/{base_name}_power.csv", spectrum, delimiter=",")
        np.savetxt(f"{output_dir}/{base_name}_coeffs.csv", spherical_harmonics.clm_to_1d_standard(clm), delimiter=",")

        # Debug info
        print(f"\n==== Debug Model: {os.path.basename(stl_path)} ====")
        print("Normalized radius range:", np.min(np.linalg.norm(normalized_vertices, axis=1)),
              np.max(np.linalg.norm(normalized_vertices, axis=1)))
        print("Zero-order power:", total_power[0])
        print("Max non-zero-order power:", np.max(total_power[1:]))
        print("=" * 50)

        return total_power, SHE

    except Exception as e:
        print(f"Error processing{stl_path}: {str(e)}")
        return None


def batch_process(input_dir, output_dir):
    """批量处理目录中的所有STL文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_spectrum = []
    all_SHE = []
    filenames = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.stl'):
            stl_path = os.path.join(input_dir, filename)
            power_stat, SHE = process_single_mesh(stl_path, output_dir)
            if power_stat is not None:
                all_spectrum.append(power_stat)
                all_SHE.append(SHE)
                filenames.append(os.path.splitext(filename)[0])

    SHE_table = np.column_stack((filenames, all_SHE))
    np.savetxt(f"{output_dir}/SHE.csv",
               SHE_table,
               delimiter=",",
               header="ID,SHE",
               fmt="%s",
               comments='')

    if len(all_spectrum) > 0:
        all_stats_array = np.array(all_spectrum)
        np.savez(f"{output_dir}/all_data.npz",
                 stats=all_stats_array,
                 SHE=np.array(all_SHE),
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

        statistics_analysis.analyze_umap2(
            all_stats_array,
            filenames,
            output_dir,
            lmax=20
        )



if __name__ == "__main__":
    input_directory = "E:\spheroids_analysis\core_3d"
    output_directory = "E:\Lithic_Analysis\QSYSpheroidsStudy\py_SPHARM"
    batch_process(input_directory, output_directory)
