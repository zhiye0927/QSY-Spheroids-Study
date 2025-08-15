import trimesh
import numpy as np
import mesh_processing
import igl
from trimesh.smoothing import filter_laplacian
import os
import csv
import pandas as pd

def calculate_iso_sphericity(stl_file_path, target_faces=15000):
    """根据 ISO 标准定义计算球度和表面积，面积单位为平方厘米"""
    vertices, faces = mesh_processing.clean_mesh(stl_file_path)

    # 统一面片数
    result = igl.decimate(vertices, faces, target_faces)
    decimated_vertices = result[1]
    decimated_faces = result[2]
    mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
    filter_laplacian(mesh, iterations=3)

    # 计算体积和表面积（单位为 mm）
    volume_mm3 = mesh.volume
    surface_area_mm2 = mesh.area

    # 转换为 cm³ 和 cm²
    volume_cm3 = volume_mm3 / 1000
    surface_area_cm2 = surface_area_mm2 / 100

    # 计算球度（使用标准公式）
    numerator = (np.pi ** (1 / 3)) * ((6 * volume_cm3) ** (2 / 3))
    sphericity = numerator / surface_area_cm2

    # 中心距离
    centroid_offset = compute_centroid_offset(mesh)

    return sphericity, surface_area_cm2, centroid_offset


def compute_centroid_offset(mesh: trimesh.Trimesh):
    """
    计算质量中心与包围盒中心的欧氏距离（单位：mm）
    """
    centroid = mesh.centroid
    bbox_center = mesh.bounding_box.centroid
    distance = np.linalg.norm(centroid - bbox_center)
    return distance

    # 归一化处理：距离除以体积
    if volume_mm3 == 0:
        return np.nan
    normalized_distance = distance / volume_mm3
    return normalized_distance


def batch_calculate_sphericity(input_dir, output_csv):
    """批量计算 STL 文件的 ISO 球度和面积并保存到 CSV"""
    results = []
    sphericities = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.stl'):
            stl_path = os.path.join(input_dir, filename)
            try:
                sphericity, surface_area, centroid_offset = calculate_iso_sphericity(stl_path)
                results.append({
                    'ID': filename,
                    'sphericity': sphericity,
                    'surface_area': surface_area,
                    'centroid_offset_mm': centroid_offset,
                    'status': 'success'
                })
                print(f"成功处理：{filename} 球度={sphericity:.4f} 面积={surface_area:.2f} cm²")
                sphericities.append(sphericity)
            except Exception as e:
                results.append({
                    'ID': filename,
                    'sphericity': None,
                    'surface_area': None,
                    'status': f'error: {str(e)}'
                })
                print(f"处理失败：{filename}，错误信息：{e}")

    results.sort(key=lambda x: (x['ID'] if x['ID'] is not None else 0), reverse=True)

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'sphericity', 'surface_area', 'centroid_offset_mm', 'status'])
        writer.writeheader()
        writer.writerows(results)

    # 打印统计信息
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n处理完成！成功：{success_count}/{len(results)}，失败：{len(results)-success_count}")

    if success_count >= 2:
        mean_sp = np.mean(sphericities)
        std_sp = np.std(sphericities, ddof=1)
        cv = (std_sp / mean_sp) * 100
        print(f"\n统计结果：")
        print(f"成功样本数: {success_count}")
        print(f"球度均值: {mean_sp:.4f}")
        print(f"球度标准差: {std_sp:.4f}")
        print(f"变异系数 (CV): {cv:.2f}%")

def merge_iso_sphericity(sphericity_csv, benn_csv):
    """清理 ID 字段并合并 sphericity 到 benn_output.csv"""
    # 读取两个表
    sph_df = pd.read_csv(sphericity_csv)
    benn_df = pd.read_csv(benn_csv)

    # 清理 ID：去除 "-" 和扩展名
    sph_df['ID'] = sph_df['ID'].apply(lambda x: os.path.splitext(x.replace('-', ''))[0])

    # 合并到 benn_df 的后面
    merged_df = pd.merge(benn_df, sph_df[['ID', 'sphericity', 'surface_area']], on='ID', how='left')

    # 保存回原路径
    merged_df.to_csv(benn_csv, index=False, encoding='utf-8')
    print("成功将球度和面积信息添加到 benn_output.csv。")


def merge_spharm_sphericity(spharm_csv_path, benn_csv_path):
    """将 SPHARM 球度（来自 sphericity_summary.csv）合并到 benn_output.csv"""
    # 读取 benn 输出文件
    benn_df = pd.read_csv(benn_csv_path)
    benn_df['ID'] = benn_df['ID'].apply(lambda x: x.replace('-', '').strip())

    # 读取 SPHARM 球度表
    spharm_df = pd.read_csv(spharm_csv_path)
    spharm_df.columns = [col.lower().strip() for col in spharm_df.columns]  # 标准化列名
    spharm_df['id'] = spharm_df['id'].apply(lambda x: x.replace('-', '').strip())

    # 合并 SPHARM 球度
    merged_df = pd.merge(benn_df, spharm_df[['id', 'sphericity_spharm']], left_on='ID', right_on='id', how='left')
    merged_df.drop(columns=['id'], inplace=True)

    # 保存
    merged_df.to_csv(benn_csv_path, index=False, encoding='utf-8')
    print("SPHARM 球度已成功合并进 benn_output.csv。")


if __name__ == "__main__":
    input_directory = "E:\\spheroids_analysis\\core_3d"
    output_file = "sphericity_iso.csv"
    benn_csv_path = r"C:\Users\27086\Desktop\Sam Lin-orientation analysis\Lin_et_al_Core_scar_orientation_Data_and_R_code\Data_and_R_code\benn_output.csv"
    spharm_csv_path = r"E:\spheroids_analysis\core_3d\sphericity_SPHARM.csv"

    batch_calculate_sphericity(input_directory, output_file)
    #merge_iso_sphericity(output_file, benn_csv_path)
    #merge_spharm_sphericity(spharm_csv_path, benn_csv_path)
