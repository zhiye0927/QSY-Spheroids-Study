import numpy as np
from sklearn.decomposition import PCA
import os
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import umap
import matplotlib.pyplot as plt
import pandas as pd

def analyze_variance(all_stats, filenames):
    all_stats = np.array(all_stats)
    print("\n==== all_stats 数据结构 ====")
    print("数据类型:", type(all_stats))
    print("数组维度:", all_stats.shape)
    print("示例数据 (前3个样本):\n", all_stats[:3])
    """执行方差分析和PCA"""

    # 方差分析
    variances = np.var(all_stats, axis=0)
    variances_plus = variances * 10
    plt.plot(variances_plus)
    plt.title('Variance per Degree')
    plt.xlabel('Spherical Harmonic Degree')
    plt.show()

    # 2. 控制台打印统计摘要
    print("\n==== 方差分析结果 ====")
    print(f"分析样本数: {all_stats.shape[0]}")
    print(f"最高方差: {np.max(variances):.4f} (degree {np.argmax(variances)})")
    print(f"最低方差: {np.min(variances):.4f} (degree {np.argmin(variances)})")
    print(f"平均方差: {np.mean(variances):.4f}")
    print(f"方差标准差: {np.std(variances):.4f}")

    # 3. 打印方差排名（前5和后5）
    sorted_indices = np.argsort(variances)[::-1]
    print("\n方差排名 Top 5:")
    for i, idx in enumerate(sorted_indices[:5]):
        print(f"Rank {i + 1}: degree {idx} -> {variances[idx]:.4f}")

    # 4. 输出完整度数方差表
    print("\n完整方差表：")
    print("Degree | Variance   | Percentage")
    print("-------------------------------")
    total_variance = np.sum(variances)
    for deg, var in enumerate(variances):
        print(f"{deg:5d} | {var:.6f} | {var / total_variance * 100:6.2f}%")


def analyze_pca(all_stats, filenames):

    # PCA分析（获取前4个主成分）
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(all_stats)

    # ==== 新增保存全部PCA坐标 ====
    if filenames is not None:
        pca_table = np.column_stack((
            filenames,
            pca_result[:, 0],  # PC1
            pca_result[:, 1],  # PC2
            pca_result[:, 2],  # PC3
            pca_result[:, 3]  # PC4
        ))
        #np.savetxt("pca_all_components.csv", pca_table,
                   #fmt="%s", delimiter=",",
                   #header="Filename,PC1,PC2,PC3,PC4")

    print("\n==== PCA载荷矩阵 ====")
    print("Shape:", pca.components_.shape)  # 应为(2,21)
    print("PC1载荷（按重要性排序）:")
    sorted_pc1 = np.argsort(np.abs(pca.components_[0]))[::-1]
    for deg in sorted_pc1:
        print(f"degree {deg}: {pca.components_[0, deg]:.3f}")

    print("\nPC2载荷（按重要性排序）:")
    sorted_pc2 = np.argsort(np.abs(pca.components_[1]))[::-1]
    for deg in sorted_pc2:
        print(f"degree {deg}: {pca.components_[1, deg]:.3f}")

    print("\nPC3载荷（按重要性排序）:")
    sorted_pc3 = np.argsort(np.abs(pca.components_[2]))[::-1]
    for deg in sorted_pc3:
        print(f"degree {deg}: {pca.components_[2, deg]:.3f}")

    print("\nPC4载荷（按重要性排序）:")
    sorted_pc4 = np.argsort(np.abs(pca.components_[3]))[::-1]
    for deg in sorted_pc4:
        print(f"degree {deg}: {pca.components_[3, deg]:.3f}")

    # 在analyze_features中调用
    plot_pca_components(pca_result, pca, filenames, x_component=0, y_component=1)  # PC1-PC2
    plot_pca_components(pca_result, pca, filenames, x_component=2, y_component=3)  # PC3-PC4


def plot_pca_components(pca_result, pca, filenames=None, x_component=0, y_component=1):
    """通用PCA绘图函数"""
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        pca_result[:, x_component],
        pca_result[:, y_component],
        alpha=0.7,
        edgecolors='w',  # 白色边缘增强对比
        s=40            # 控制点大小
    )

    # 标注部分文件名
    if filenames is not None:
        for i, (x, y) in enumerate(zip(pca_result[:, x_component],
                                       pca_result[:, y_component])):
            plt.text(
                x, y,
                os.path.splitext(filenames[i])[0],
                fontsize=7,  # 更小字体
                alpha=0.8,  # 更高透明度
                rotation=20,  # 倾斜角度
                ha='left', va='center'
            )

    # 添加方差解释率
    explained = pca.explained_variance_ratio_
    plt.xlabel(f'PC{x_component + 1} ({explained[x_component]:.1%})')
    plt.ylabel(f'PC{y_component + 1} ({explained[y_component]:.1%})')

    plt.title(f'PCA Components {x_component + 1}-{y_component + 1}')
    plt.show()


def calculate_power_distance(all_stats, filenames, output_dir, lmax=20):
    """
    计算模型间功率谱差异并生成热图

    参数：
    all_stats : numpy.ndarray, 形状 (N, lmax+1)
        所有模型的功率谱数组
    filenames : list of str
        模型文件名列表（用于热图标签）
    output_dir : str
        输出目录路径
    lmax : int
        最大球谐阶数
    """
    # 提取功率谱（仅保留有效阶数）
    power_spectra = all_stats[:, :lmax + 1]

    # 标准化（Min-Max归一化，避免高阶项主导）
    min_val = np.min(power_spectra, axis=0)
    max_val = np.max(power_spectra, axis=0)
    power_spectra_norm = (power_spectra - min_val) / (max_val - min_val + 1e-10)  # 防止除零

    # 计算欧氏距离矩阵
    distance_matrix = squareform(pdist(power_spectra_norm, metric='euclidean'))

    # 生成热图
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

    # 保存热图
    heatmap_path = os.path.join(output_dir, f"power_distance_heatmap_lmax{lmax}.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"热图已保存至: {heatmap_path}")


def analyze_umap(all_stats, filenames, output_dir, lmax=20):
    power_spectra = all_stats[:, :lmax + 1]

    # 动态调整参数
    n_samples = power_spectra.shape[0]
    n_neighbors = min(6, n_samples - 1)

    # 初始化并拟合UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.06,
        metric='cosine',
        random_state=42,
    )
    embedding = reducer.fit_transform(power_spectra)

    # 定义标准模型名称
    standard_models = {"1_Sphere", "2_Ellipsoid", "3_Rounded_cube", "4_Box", "5_Discoid"}

    # 创建图像
    plt.figure(figsize=(15, 12))
    ax = plt.gca()

    # 绘制每个点（单独控制颜色）
    for i, (x, y) in enumerate(embedding):
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

    # 标题和标签
    plt.title(f"UMAP Projection with Labels (Lmax={lmax})", fontsize=14)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)

    # 保存图像
    output_path = os.path.join(output_dir, f"umap_updated_lmax{lmax}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"带标注的UMAP图表已保存至: {output_path}")


def analyze_umap2(all_stats, filenames, output_dir, lmax=20):
    power_spectra = all_stats[:, :lmax + 1]
    n_samples = power_spectra.shape[0]
    n_neighbors = min(6, n_samples - 1)

    # 执行UMAP降维
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.06,
        metric='cosine',
        random_state=42,
    )
    embedding = reducer.fit_transform(power_spectra)

    # 保存为 CSV 以供 R 等使用
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
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'filename': filenames,
        'category': [get_category(f) for f in filenames]
    })

    csv_path = os.path.join(output_dir, f"umap_embedding_lmax{lmax}.csv")
    umap_df.to_csv(csv_path, index=False)
    print(f"UMAP数据导出到: {csv_path}")

    # （可选）绘图部分，如不需要可以忽略以下内容
    """
    plt.figure(figsize=(15, 12))
    ax = plt.gca()

    for i, (x, y) in enumerate(embedding):
        full_label = filenames[i]
        label_parts = full_label.split('-')
        base_label = '-'.join(label_parts[:3]) if len(label_parts) >= 3 else full_label

        color = "#666666"
        if "Multifacial" in full_label:
            color = "lightgray"
        elif "Subspheroid" in full_label:
            color = "orange"
        elif "Spheroid" in full_label:
            color = "red"

        ax.scatter(x, y, color=color, s=100, alpha=0.85, edgecolor='k', linewidth=0.3)
        plt.text(x + 0.02, y + 0.02, base_label, fontsize=9, ha='left', va='bottom')

    plt.title(f"UMAP Projection with Labels (Lmax={lmax})", fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=14)
    plt.ylabel("UMAP Component 2", fontsize=14)

    output_path = os.path.join(output_dir, f"umap_updated_lmax{lmax}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"带标注的UMAP图表已保存至: {output_path}")
    """