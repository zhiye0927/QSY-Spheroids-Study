import trimesh
import numpy as np
import mesh_processing
import igl
from trimesh.smoothing import filter_laplacian
import os
import csv


def calculate_iso_sphericity(stl_file_path, target_faces=15000):

    """
    Calculate sphericity and surface area according to ISO standard.

    Parameters:
    - stl_file_path: str, path to the STL file
    - target_faces: int, number of faces for mesh decimation

    Returns:
    - sphericity: float, ISO-defined sphericity
    - surface_area_cm2: float, surface area in cm²
    - centroid_offset: float, distance of centroid offset
    """

    # Clean and Decimate mesh
    vertices, faces = mesh_processing.clean_mesh(stl_file_path)

    result = igl.decimate(vertices, faces, target_faces)
    decimated_vertices = result[1]
    decimated_faces = result[2]
    mesh = trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_faces)
    filter_laplacian(mesh, iterations=3)

    # Compute volume and surface area
    volume_mm3 = mesh.volume
    surface_area_mm2 = mesh.area

    volume_cm3 = volume_mm3 / 1000
    surface_area_cm2 = surface_area_mm2 / 100

    # Compute sphericity using standard ISO formula
    numerator = (np.pi ** (1 / 3)) * ((6 * volume_cm3) ** (2 / 3))
    sphericity = numerator / surface_area_cm2

    # centroid offset
    centroid_offset = compute_centroid_offset(mesh)

    return sphericity, surface_area_cm2, centroid_offset


def compute_centroid_offset(mesh: trimesh.Trimesh):
    """
    Compute the Euclidean distance between the mass centroid and the bounding box centroid
    """
    centroid = mesh.centroid
    bbox_center = mesh.bounding_box.centroid
    distance = np.linalg.norm(centroid - bbox_center)
    return distance

    # normalize
    if volume_mm3 == 0:
        return np.nan
    normalized_distance = distance / volume_mm3
    return normalized_distance


def batch_calculate_sphericity(input_dir, output_csv):
    """Batch calculate ISO sphericity and surface area for STL files and save to CSV"""

    results = []
    sphericities = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.stl'):
            stl_path = os.path.join(input_dir, filename)
            try:
                base_name = filename[:-4]
                parts = base_name.split('-')
                if len(parts) >= 2:
                    id_part = "_".join(parts[:-1])
                    typology = parts[-1]
                else:
                    id_part = base_name
                    typology = ''

                # Compute sphericity and surface area
                sphericity, surface_area, centroid_offset = calculate_iso_sphericity(stl_path)

                results.append({
                    'ID': id_part,
                    'Typology': typology,
                    'sphericity': sphericity,
                    'surface_area': surface_area,
                    'centroid_offset_mm': centroid_offset,
                    'status': 'success'
                })
                print(f"Processed: {filename} | Sphericity={sphericity:.4f} | Area={surface_area:.2f} cm²")
                sphericities.append(sphericity)

            except Exception as e:
                results.append({
                    'ID': filename,
                    'Typology': '',
                    'sphericity': None,
                    'surface_area': None,
                    'centroid_offset_mm': None,
                    'status': f'error: {str(e)}'
                })
                print(f"Failed: {filename}, Error: {e}")

    results.sort(key=lambda x: (x['ID'] if x['ID'] is not None else ''), reverse=True)

    # Save to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'Typology', 'sphericity', 'surface_area', 'centroid_offset_mm', 'status'])
        writer.writeheader()
        writer.writerows(results)

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nFinished! Success: {success_count}/{len(results)}, Failed: {len(results)-success_count}")


if __name__ == "__main__":

    input_directory = "E:\\spheroids_analysis\\core_3d"
    output_file = "sphericity_iso.csv"

    batch_calculate_sphericity(input_directory, output_file)
