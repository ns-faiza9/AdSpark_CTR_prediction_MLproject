"""Stage 11 — DBSCAN Density-Based Clustering (Unsupervised / CO4).

Uses Nearest Neighbors to compute the k-distance graph for epsilon choice,
then fits DBSCAN to discover arbitrary-shaped clusters and noise points.

Produces:
    analysis/output/11_dbscan_summary.json
    analysis/output/figures/db_01_kdistance_eps.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 11: DBSCAN Density-Based Clustering")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    
    from sklearn.preprocessing import StandardScaler
    X_raw = X.sample(n=min(1500, len(X)), random_state=SEED).values
    X_sample = StandardScaler().fit_transform(X_raw)

    # k-distance graph calculation (MinPts = 5)
    min_pts = 5
    nn = NearestNeighbors(n_neighbors=min_pts).fit(X_sample)
    distances, _ = nn.kneighbors(X_sample)
    k_distances = np.sort(distances[:, -1])[::-1]

    # Dynamically select elbow or robust epsilon (typically 3.0 to 4.0 in standardized 23-d space)
    eps_cutoff = float(np.percentile(k_distances, 75))
    eps_cutoff = round(eps_cutoff, 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_distances, 'g-', lw=2)
    ax.axhline(y=eps_cutoff, color='r', linestyle='--', label=f'Epsilon threshold = {eps_cutoff}')
    ax.set_title(f"DBSCAN k-Distance Neighborhood Graph (MinPts={min_pts})")
    ax.set_xlabel("Points sorted by 5th nearest neighbor distance")
    ax.set_ylabel("5-NN Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "db_01_kdistance_eps.png")

    db = DBSCAN(eps=eps_cutoff, min_samples=min_pts).fit(X_sample)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(list(labels).count(-1))
    noise_pct = round(n_noise / len(X_sample) * 100, 2)

    # ------------------------------------------------------------------
    # 2D DBSCAN Cluster & Outlier Visualization
    # ------------------------------------------------------------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_sample)

    sample_df = df.iloc[:len(X_sample)]
    sample_targets = sample_df[TARGET].values

    fig_db, ax_db = plt.subplots(figsize=(10, 6))
    
    # Plot noise points first in subtle gray
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        ax_db.scatter(
            X_pca[noise_mask, 0], X_pca[noise_mask, 1],
            c='#64748b', marker='x', alpha=0.45, s=20,
            label=f"Noise Outliers: {n_noise:,} pts ({noise_pct}%)"
        )

    # Plot ALL discovered clusters with a 20-color discrete colormap
    import matplotlib as mpl
    cluster_ids = [c for c in sorted(set(labels)) if c != -1]
    tab20_colors = mpl.colormaps['tab20'].colors
    
    cluster_profiles = []
    for idx, c_id in enumerate(cluster_ids):  # All clusters formed (0 to n_clusters-1)
        mask = (labels == c_id)
        c_count = int(np.sum(mask))
        c_pct = round(float(c_count / len(labels) * 100), 2)
        c_ctr = round(float(np.mean(sample_targets[mask]) * 100), 2)
        color = tab20_colors[idx % len(tab20_colors)]
        
        ax_db.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=color, alpha=0.75, s=26,
            label=f"Cluster {c_id}: {c_count} pts ({c_pct}%, CTR {c_ctr}%)"
        )
        cluster_profiles.append({
            "cluster_id": c_id,
            "count": c_count,
            "pct": c_pct,
            "mean_ctr": c_ctr,
            "density_type": "Dense Core Segment" if c_count >= 50 else ("Micro-Cluster" if c_count >= 10 else "Sparse Density Group")
        })

    ax_db.set_title(f"DBSCAN Density-Based Clusters (eps={eps_cutoff}, MinPts={min_pts}) — All {n_clusters} Clusters Formed", fontsize=12)
    ax_db.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax_db.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax_db.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., framealpha=0.9, fontsize=8, ncol=1)
    ax_db.grid(True, alpha=0.25)
    fig_db.tight_layout()
    save_fig(fig_db, "db_02_clusters_scatter.png")

    print(f"  DBSCAN Epsilon={eps_cutoff}: Clusters={n_clusters}, Noise points={n_noise} ({noise_pct}%)")

    summary = {
        "epsilon": eps_cutoff,
        "min_samples": min_pts,
        "n_clusters": n_clusters,
        "estimated_clusters": n_clusters,
        "noise_points": n_noise,
        "noise_percentage": noise_pct,
        "cluster_profiles": cluster_profiles
    }
    save_json("11_dbscan_summary.json", summary)
    print("DBSCAN clustering complete. db_02_clusters_scatter.png generated.")


if __name__ == "__main__":
    main()
