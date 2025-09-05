import os
import numpy as np
from scipy.interpolate import griddata
import pyvista as pv
import pyshtools.expand as shtools
import pyshtools as pysh
import pandas as pd
import matplotlib.pyplot as plt


def cartesian_to_spherical(normalized_vertices):
    """convert Cartesian coordinates to spherical coordinates"""
    spherical_coordinates = np.zeros((len(normalized_vertices), 3))
    for i, vertex in enumerate(normalized_vertices):
        x, y, z = vertex
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.where(r == 0, 0.00001, r)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        phi = phi % (2 * np.pi)
        spherical_coordinates[i] = [r, theta, phi]

    return spherical_coordinates


def spherical_interpolate(R, theta, phi, grid_size):

    """
    Interpolate data onto a regular grid

     Parameters
    ----------
    R : array-like
        Radii values corresponding to the spherical coordinates.
    theta : array-like
        Colatitude angles (in radians, range [0, π]).
    phi : array-like
        Longitude angles (in radians, range [0, 2π)).
    grid_size : int
        Size of the output square grid (grid_size x grid_size).

    Returns
    -------
    grid : ndarray
        Interpolated grid of shape (grid_size, grid_size).
    """

    if len(R) < 4:
        return None

    # Create regular grid in spherical coordinates
    I = np.linspace(0, np.pi, grid_size, endpoint=False)
    J = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    J, I = np.meshgrid(J, I)

    # Original data
    values = R
    points = np.array([theta, phi]).T

    # Add poles (theta = 0 and theta = π)
    points = np.concatenate((points,
                             np.array([[0, 0], [0, 2 * np.pi], [np.pi, 0], [np.pi, 2 * np.pi]])), axis=0)
    rmin = np.mean(R[theta == theta.min()])
    rmax = np.mean(R[theta == theta.max()])
    values = np.concatenate((values, [rmin, rmin, rmax, rmax]))

    # Handle periodicity in longitude (phi)
    points = np.concatenate((points, points - [0, 2 * np.pi], points + [0, 2 * np.pi]), axis=0)
    values = np.concatenate((values, values, values))

    # Generate interpolation points
    xi = np.array([[I[i, j], J[i, j]] for i in range(grid_size) for j in range(grid_size)])

    # interpolate
    grid = griddata(points, values, xi, method='linear')
    grid = grid.reshape((grid_size, grid_size))
    grid[:, -1] = grid[:, 0]

    return grid


def visualize_interpolated(grid_r):

    """Visualize the interpolated regular spherical grid"""

    # Get grid size (assuming square grid)
    grid_size = grid_r.shape[0]

    # Generate spherical coordinates
    theta = np.linspace(0, np.pi, grid_size, endpoint=True)  # colatitude [0, π)
    phi = np.linspace(0, 2 * np.pi, grid_size, endpoint=True)  # longitude [0, 2π)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    print("Sampling check:")
    print(f"theta range: {theta[0]:.3f} ~ {theta[-1]:.3f} (expected: 0 ~ π)")
    print(f"phi range: {phi[0]:.3f} ~ {phi[-1]:.3f} (expected: 0 ~ 2π)")

    # Convert spherical to Cartesian coordinates
    x = grid_r * np.sin(theta_grid) * np.cos(phi_grid)
    y = grid_r * np.sin(theta_grid) * np.sin(phi_grid)
    z = grid_r * np.cos(theta_grid)

    # Closure validation at phi=0 vs phi=2π
    idx_phi0 = 0
    idx_phi2pi = -1
    tolerance = 1e-3
    x_diff = np.max(np.abs(x[:, idx_phi0] - x[:, idx_phi2pi]))
    y_diff = np.max(np.abs(y[:, idx_phi0] - y[:, idx_phi2pi]))
    z_diff = np.max(np.abs(z[:, idx_phi0] - z[:, idx_phi2pi]))
    print("Closure check (φ=0 vs φ=2π):")
    print(f"Max coordinate difference: x={x_diff:.3e}, y={y_diff:.3e}, z={z_diff:.3e}")
    if x_diff > tolerance or y_diff > tolerance or z_diff > tolerance:
        print("Warning: Coordinates at phi=0 and phi=2π do not match!")

    grid = pv.StructuredGrid(x, y, z)
    grid["Radius"] = grid_r.T.ravel()
    plotter = pv.Plotter()
    plotter.add_mesh(grid,
                     color='cyan',
                     opacity=0.8,
                     show_edges=True,
                     scalars=grid_r.flatten(),
                     cmap='viridis')
    plotter.add_title(f"({grid_size}x{grid_size})")
    plotter.show()


def compute_spherical_harmonics(surface, normalize=True, normalization_method='zero-component'):
    """
    Compute spherical harmonic coefficients from a 2D surface grid.

    Parameters
    ----------
    surface : numpy.ndarray, shape (n, n) or (n, 2n)
        2D grid sampled according to the Driscoll–Healy sampling theorem.
        Both dimensions must be even.
    normalize : bool, optional, default=True
        Whether to normalize the coefficients.
    normalization_method : {'zero-component', 'mean-radius'}, optional, default='zero-component'
        Method of normalization:
        - 'zero-component': normalize by the l=0, m=0 coefficient
        - 'mean-radius': normalize by the mean radius of the surface

    Returns
    -------
    harmonics : numpy.ndarray

    """

    if surface.shape[1] % 2 or surface.shape[0] % 2:
        raise ValueError("Latitude and longitude samples (n) must be even")

    # Determine grid type
    if surface.shape[1] == surface.shape[0]:
        sampling = 1  # 等采样网格-equally spaced in colatitude and longitude
    elif surface.shape[1] == 2 * surface.shape[0]:
        sampling = 2  # 等间距网格-equally spaced in latitude and longitude
    else:
        raise ValueError("Grid must be (N, N) or (N, 2N)")

    # Preprocessing
    processed_surface = surface.copy()
    if normalize and normalization_method == 'mean-radius':
        processed_surface /= np.mean(np.abs(processed_surface))

    # spherical harmonic expansion
    harmonics = shtools.SHExpandDHC(processed_surface, sampling=sampling)

    if normalize and normalization_method == 'zero-component':
        harmonics = harmonics / harmonics[0][0, 0]
        print(harmonics)

    return harmonics


def clm_to_1d_standard(clm, target_l_max=30):

    """
    Convert spherical harmonic coefficients `clm`
    (with shape (2, l_max_input+1, l_max_input+1)) into a 1D real-valued array.

    Parameters
    ----------
    clm : numpy.ndarray, shape (2, l_max_input+1, l_max_input+1)
        - clm[0, l, m] stores coefficients with m >= 0
        - clm[1, l, m] stores coefficients with m < 0
          (where clm[1, l, 1] corresponds to m=-1, ..., clm[1, l, l] to m=-l)
    target_l_max : int, default=30
        The maximum degree l to export (coefficients from l=0 up to target_l_max are included).

    Returns
    -------
    numpy.ndarray, shape ((target_l_max+1)**2,)
        1D real-valued array of coefficients, ordered as:
        for each l: [c(l,-l), c(l,-l+1), ..., c(l,-1), c(l,0), c(l,1), ..., c(l,l)]

    """

    if clm.shape[1] - 1 < target_l_max:
        raise ValueError("Input coefficients l_max is smaller than target_l_max")

    coeffs = []
    for l in range(target_l_max + 1):
        # m = -l, -l+1, …, -1
        for m in range(l, 0, -1):
            coeffs.append(clm[1, l, m])
        # m = 0
        coeffs.append(clm[0, l, 0])
        # m = 1, 2, …, l
        for m in range(1, l + 1):
            coeffs.append(clm[0, l, m])

    # Convert to numpy array and keep only the real part
    return np.real(np.array(coeffs))


def process_spherical_harmonics(clm, output_path=None):
    """
    Analyze spherical harmonic coefficients and compute the power spectrum

    Parameters
    ----------
    clm : pysh.SHCoeffs or numpy.ndarray
        Input spherical harmonic coefficients. Supported formats:
        - SHCoeffs object from pyshtools
        - numpy array of shape (2, lmax+1, lmax+1)
    output_path : str
        Path to save the results. Supported formats:
        - .xlsx : multi-sheet Excel file containing full coefficients and power spectrum
        - .csv  : two CSV files (one for full coefficients, one for power spectrum)
        - None  : do not save any files (default)

    Returns
    -------
    tuple of (full_df,spectrum_df)
    full_df : DataFrame
        Contains the complete spherical harmonic coefficients, including
        degree, order, complex value, amplitude, power, real, imaginary part, and a harmonic label.
    spectrum_df : DataFrame
        Contains the energy spectrum statistics aggregated by degree, including
        total power, maximum amplitude, and total amplitude per degree.

    """

    # Input validation and preprocessing
    if isinstance(clm, pysh.SHCoeffs):
        coeffs = clm.to_array()
        lmax = clm.lmax
    elif isinstance(clm, np.ndarray):
        if clm.ndim != 3 or clm.shape[0] != 2:
            raise ValueError("Numpy array input must have shape (2, lmax+1, lmax+1)")
        coeffs = clm
        lmax = clm.shape[1] - 1
    else:
        raise TypeError("Unsupported input type, please provide SHCoeffs object or numpy array")

    if not np.iscomplexobj(coeffs):
        raise ValueError("Input coefficients must contain complex numbers")

    # data table
    n_records = (lmax + 1) ** 2

    data = {
        'degree': np.empty(n_records, dtype=np.int32),
        'order': np.empty(n_records, dtype=np.int32),
        'value': np.empty(n_records, dtype=np.complex128)
    }

    idx = 0
    for l in range(lmax + 1):
        if l == 0:
            data['degree'][idx] = 0
            data['order'][idx] = 0
            data['value'][idx] = coeffs[0, 0, 0]
            idx += 1
        else:
            # m＜0
            for m in range(l, 0, -1):
                data['degree'][idx] = l
                data['order'][idx] = -m
                data['value'][idx] = coeffs[1, l, m]
                idx += 1

            # m=0
            data['degree'][idx] = l
            data['order'][idx] = 0
            data['value'][idx] = coeffs[0, l, 0]
            idx += 1

            # m＞0
            for m in range(1, l + 1):
                data['degree'][idx] = l
                data['order'][idx] = m
                data['value'][idx] = coeffs[0, l, m]
                idx += 1

    full_df = pd.DataFrame(data)

    full_df['amplitude'] = np.abs(full_df['value'])
    full_df['power'] = full_df['amplitude'] ** 2
    full_df['real'] = np.real(full_df['value'])
    full_df['imag'] = np.imag(full_df['value'])
    full_df['harmonic'] = "l=" + full_df['degree'].astype(str) + " m=" + full_df['order'].astype(str)

    degrees = np.arange(lmax + 1)
    total_power = np.zeros_like(degrees, dtype=np.float64)
    max_amplitude = np.zeros_like(degrees, dtype=np.float64)

    for l in degrees:
        mask = full_df['degree'] == l
        amplitudes = full_df.loc[mask, 'amplitude'].values

        total_power[l] = np.sum(amplitudes ** 2)
        max_amplitude[l] = np.max(amplitudes) if amplitudes.size > 0 else 0.0

    spectrum_df = pd.DataFrame({
        'degree': degrees,
        'total_power': total_power,
        'max_amplitude': max_amplitude,
        'total_amplitude': np.sqrt(total_power)
    })

    if output_path:
        base_name, ext = os.path.splitext(output_path)

        if ext.lower() == '.xlsx':
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                full_df.to_excel(writer, sheet_name='Full_Coefficients', index=False)
                spectrum_df.to_excel(writer, sheet_name='Power_Spectrum', index=False)
                print(f"saved to: {output_path}")

        elif ext.lower() == '.csv':
            full_path = f"{base_name}_full.csv"
            spectrum_path = f"{base_name}_spectrum.csv"

            full_df.to_csv(full_path, index=False)
            spectrum_df.to_csv(spectrum_path, index=False)
            print(f"Full dataset saved at: {full_path}")
            print(f"Spherical harmonics power spectrum saved at: {spectrum_path}")

        else:
            raise ValueError(".xlsx or.csv is accepted")

    return full_df, spectrum_df


def visualize_power_spectrum(spectrum_df, max_degree=None, log_scale=True, filename=None):
    """
    Visualize the spherical harmonic power spectrum for a 3D model
    """

    df = spectrum_df.copy()
    if max_degree is not None:
        df = df[df['degree'] <= max_degree]

    plt.figure(figsize=(12, 6))

    plt.plot(df['degree'], df['total_power'],
             marker='o',
             linestyle='-',
             color='#2c7bb6',
             markersize=6,
             linewidth=2,
             label='Total Power')

    plt.plot(df['degree'], df['max_amplitude'],
             marker='s',
             linestyle='--',
             color='#d7191c',
             markersize=5,
             linewidth=1.5,
             alpha=0.7,
             label='Max Amplitude')

    plt.xlabel('Spherical Harmonic Degree (l)', fontsize=12, labelpad=10)
    plt.ylabel('Power / Amplitude' + (' (log scale)' if log_scale else ''), fontsize=12, labelpad=10)
    plt.title('Spherical Harmonic Energy Spectrum', fontsize=14, pad=20)

    plt.xticks(np.arange(0, df['degree'].max() + 1, 5 if df['degree'].max() > 20 else 2))
    plt.xlim(-0.5, df['degree'].max() + 0.5)

    if log_scale:
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
    else:
        plt.grid(True, axis='y', ls="--", alpha=0.3)

    plt.legend(fontsize=10, frameon=True, loc='upper right')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"spectrum was saved to：{filename}")
    else:
        plt.show()

