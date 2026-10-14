"""Stage 10 — Hierarchical Agglomerative Clustering (Unsupervised / CO4).

Performs hierarchical clustering across multiple linkage algorithms:
Single, Complete, Average, and Ward Linkage.

Produces:
    analysis/output/10_hierarchical_summary.json
    analysis/output/figures/hc_01_dendrograms.png
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 10: Hierarchical Agglomerative Clustering")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    
    # Subsample for dendrogram computation clarity
    X_sample = X.sample(n=min(50, len(X)), random_state=SEED).values

    linkages = ['single', 'complete', 'average', 'ward']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    linkage_results = {}
    for idx, method in enumerate(linkages):
        Z = linkage(X_sample, method=method)
        dendrogram(Z, ax=axes[idx], color_threshold=0.7 * max(Z[:, 2]))
        axes[idx].set_title(f"Linkage Method: {method.capitalize()}")
        axes[idx].set_xlabel("Sample Impression Index")
        axes[idx].set_ylabel("Euclidean Distance")
        linkage_results[method] = round(float(np.max(Z[:, 2])), 4)
        print(f"  Linkage '{method}': max distance = {linkage_results[method]}")

    fig.tight_layout()
    save_fig(fig, "hc_01_dendrograms.png")

    # ------------------------------------------------------------------
    # Generate 2D Hierarchical Cluster Scatter & Profiles (Ward, k=4)
    # ------------------------------------------------------------------
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Fit Agglomerative Clustering with Ward linkage on 1000 sample points
    X_clust = X.sample(n=min(1200, len(X)), random_state=SEED).values
    X_scaled = StandardScaler().fit_transform(X_clust)
    agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    agg_labels = agg.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    sample_targets = df.iloc[:len(X_scaled)][TARGET].values
    palette = ['#38bdf8', '#34d399', '#f43f5e', '#a855f7']

    fig_agg, ax_agg = plt.subplots(figsize=(10, 6))
    cluster_profiles = []
    for c_id in range(4):
        mask = (agg_labels == c_id)
        c_count = int(np.sum(mask))
        c_pct = round(float(c_count / len(agg_labels) * 100), 2)
        c_ctr = round(float(np.mean(sample_targets[mask]) * 100), 2)
        color = palette[c_id % len(palette)]
        ax_agg.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, alpha=0.6, s=26,
            label=f"Branch {c_id}: {c_count:,} pts ({c_pct}%, CTR {c_ctr}%)"
        )
        cluster_profiles.append({
            "cluster_id": c_id,
            "count": c_count,
            "pct": c_pct,
            "mean_ctr": c_ctr,
            "linkage_node": f"Ward Sub-tree Branch {c_id}"
        })

    ax_agg.set_title("Hierarchical Agglomerative Clustering (Ward Linkage, k=4 Clusters Formed)", fontsize=12)
    ax_agg.set_xlabel(f"PCA Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax_agg.set_ylabel(f"PCA Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax_agg.legend(loc="upper right", framealpha=0.9)
    ax_agg.grid(True, alpha=0.25)
    fig_agg.tight_layout()
    save_fig(fig_agg, "hc_02_clusters_scatter.png")

    summary = {
        "n_clusters": 4,
        "best_criterion": "Ward Linkage",
        "linkage_methods": linkages,
        "max_distances": linkage_results,
        "cluster_profiles": cluster_profiles,
        "cluster_structure": "4 clear sub-groups discovered via Ward linkage"
    }
    save_json("10_hierarchical_summary.json", summary)
    print("Hierarchical clustering complete. hc_02_clusters_scatter.png generated.")


if __name__ == "__main__":
    main()
