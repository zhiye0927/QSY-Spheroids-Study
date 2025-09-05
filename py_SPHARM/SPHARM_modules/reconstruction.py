import pyvista as pv
import numpy as np


def visualize_spherical_harmonics_reconstruction(grid_sh):

    """Visualize a 3D shape reconstructed from spherical harmonics"""

    grid_data = np.real(grid_sh.data)
    grid_size = grid_data.shape[0]

    theta = np.linspace(0, np.pi, grid_size, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, grid_size, endpoint=True)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    r = grid_data
    x = (r * np.sin(theta_grid) * np.cos(phi_grid)).T
    y = (r * np.sin(theta_grid) * np.sin(phi_grid)).T
    z = (r * np.cos(theta_grid)).T

    grid = pv.StructuredGrid(x, y, z)

    plotter = pv.Plotter()
    plotter.add_mesh(grid,
                     scalars=r.flatten(),
                     cmap="coolwarm",
                     opacity=1.0,
                     show_edges=False,
                     specular=0.8)
    plotter.add_axes(box_args={'color': 'red'})
    plotter.add_title(f"\nResolution: {grid_size}x{grid_size}")
    plotter.show()