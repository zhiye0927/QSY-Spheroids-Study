import numpy as np
from sklearn.decomposition import PCA
import os
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pandas as pd
import umap
import re


def analyze_variance(all_stats, filenames):
    """
    Perform variance analysis and PCA-related summary on spherical harmonic coefficients.
    """

    all_stats = np.array(all_stats)
    print("\n==== all_stats Data Structure ====")
    print("Data type:", type(all_stats))
    print("Array shape:", all_stats.shape)
    print("Sample data (first 3 samples):\n", all_stats[:3])

    # Variance analysis
    variances = np.var(all_stats, axis=0)
    variances_plus = variances * 10
    plt.plot(variances_plus)
    plt.title('Variance per Degree')
    plt.xlabel('Spherical Harmonic Degree')
    plt.show()

    print("\n==== Variance Analysis Summary ====")
    print(f"Number of samples: {all_stats.shape[0]}")
    print(f"Maximum variance: {np.max(variances):.4f} (degree {np.argmax(variances)})")
    print(f"Minimum variance: {np.min(variances):.4f} (degree {np.argmin(variances)})")
    print(f"Mean variance: {np.mean(variances):.4f}")
    print(f"Variance standard deviation: {np.std(variances):.4f}")

    # Variance ranking
    sorted_indices = np.argsort(variances)[::-1]
    print("\nTop 5 degrees by variance")
    for i, idx in enumerate(sorted_indices[:5]):
        print(f"Rank {i + 1}: degree {idx} -> {variances[idx]:.4f}")


def analyze_pca(all_stats, filenames):

    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(all_stats)

    if filenames is not None:
        pca_table = np.column_stack((
            filenames,
            pca_result[:, 0],  # PC1
            pca_result[:, 1],  # PC2
            pca_result[:, 2],  # PC3
            pca_result[:, 3]  # PC4
        ))
        np.savetxt("pca_all_components.csv", pca_table,
                   fmt="%s", delimiter=",",
                   header="Filename,PC1,PC2,PC3,PC4")

    print("\n==== PCA Loading Matrix ====")
    print("Shape:", pca.components_.shape)

    # Print feature contributions for each principal component (sorted by importance)
    for i in range(4):
        sorted_indices = np.argsort(np.abs(pca.components_[i]))[::-1]
        print(f"\nPC{i + 1} loadings (sorted by absolute value):")
        for deg in sorted_indices:
            print(f"degree {deg}: {pca.components_[i, deg]:.3f}")

    plot_pca_components(pca_result, pca, filenames, x_component=0, y_component=1)  # PC1-PC2
    plot_pca_components(pca_result, pca, filenames, x_component=2, y_component=3)  # PC3-PC4


def plot_pca_components(pca_result, pca, filenames=None, x_component=0, y_component=1):

    plt.figure(figsize=(10, 8))
    plt.scatter(
        pca_result[:, x_component],
        pca_result[:, y_component],
        alpha=0.7,
        edgecolors='w',
        s=40
    )

    if filenames is not None:
        for i, (x, y) in enumerate(zip(pca_result[:, x_component], pca_result[:, y_component])):
            plt.text(
                x, y,
                os.path.splitext(filenames[i])[0],
                fontsize=7,
                alpha=0.8,
                rotation=20,
                ha='left', va='center'
            )

    explained = pca.explained_variance_ratio_
    plt.xlabel(f'PC{x_component + 1} ({explained[x_component]:.1%})')
    plt.ylabel(f'PC{y_component + 1} ({explained[y_component]:.1%})')
    plt.title(f'PCA Components {x_component + 1}-{y_component + 1}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def calculate_power_distance(all_stats, filenames, output_dir, lmax=20):
    """
    Compute pairwise power spectrum distances between models and generate a heatmap
    """

    power_spectra = all_stats[:, :lmax + 1]

    min_val = np.min(power_spectra, axis=0)
    max_val = np.max(power_spectra, axis=0)
    power_spectra_norm = (power_spectra - min_val) / (max_val - min_val + 1e-10)

    distance_matrix = squareform(pdist(power_spectra_norm, metric='euclidean'))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=filenames,
        yticklabels=filenames
    )
    plt.title(f"Shape Distance Heatmap (Lmax={lmax})")
    plt.xlabel("Models")
    plt.ylabel("Models")

    heatmap_path = os.path.join(output_dir, f"power_distance_heatmap_lmax{lmax}.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"saved to: {heatmap_path}")


def analyze_umap(all_stats, filenames, output_dir, lmax=20):

    """
    Perform UMAP dimensionality reduction on power spectra and plot labeled 2D embedding.
    """

    power_spectra = all_stats[:, :lmax + 1]

    n_samples = power_spectra.shape[0]
    n_neighbors = min(6, n_samples - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.06,
        metric='cosine',
        random_state=42,
    )
    umap_result = reducer.fit_transform(power_spectra)

    standard_models = {"1_Sphere", "2_Ellipsoid", "3_Rounded_cube", "4_Box", "5_Discoid"}

    plt.figure(figsize=(15, 12))
    ax = plt.gca()

    for i, (x, y) in enumerate(umap_result):
        label = filenames[i]
        color = 'red' if label in standard_models else 'black'
        ax.scatter(x, y, color=color, s=100, alpha=0.8)

        plt.text(
            x + 0.02,
            y + 0.02,
            label,
            fontsize=9,
            ha='left',
            va='bottom'
        )

    plt.title(f"UMAP Projection with Labels (Lmax={lmax})", fontsize=14)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)

    output_path = os.path.join(output_dir, f"umap_updated_lmax{lmax}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"UMAP_figure was saved to: {output_path}")


def analyze_umap2(all_stats, filenames, output_dir, lmax=20):
    """
    Perform UMAP dimensionality reduction and export the 2D embedding as a CSV.
    """

    power_spectra = all_stats[:, :lmax + 1]
    n_samples = power_spectra.shape[0]
    n_neighbors = min(6, n_samples - 1)

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=2, # reduce to 2 dimensions for visualization
        n_neighbors=n_neighbors, # number of nearest neighbors considered for manifold structure
        min_dist=0.06, # minimum distance between embedded points (controls clustering tightness)
        metric='cosine', # distance metric used. Using cosine focuses more on the intrinsic shape patterns
        random_state=42,
    )
    umap_result = reducer.fit_transform(power_spectra)

    def get_category(label):
        if "Multifacial" in label:
            return "Multifacial"
        elif "Subspheroid" in label:
            return "Subspheroid"
        elif "Spheroid" in label:
            return "Spheroid"
        elif "Polyhedron" in label:
            return "Polyhedron"
        else:
            return "idealmodel"

    umap_df = pd.DataFrame({
        'x': umap_result[:, 0],
        'y': umap_result[:, 1],
        'filename': [
        re.sub(r'-[^-]+$', '', re.sub(r'^\d+_', '', f))
        for f in filenames
    ],
        'category': [get_category(f) for f in filenames]
    })

    csv_path = os.path.join(output_dir, f"umap_lmax{lmax}.csv")
    umap_df.to_csv(csv_path, index=False)
    print(f"UMAP_data was saved to: {csv_path}")
