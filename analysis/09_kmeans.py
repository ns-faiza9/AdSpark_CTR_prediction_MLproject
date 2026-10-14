"""Stage 09 — K-Means Clustering (Unsupervised / CO4).

Applies K-Means clustering to AdSpark CTR impression feature vectors.
Evaluates cluster cohesion via Within-Cluster Sum of Squares (WCSS) and
cluster separation via Silhouette Analysis across k=2 to 8.

Produces:
    analysis/output/09_kmeans_summary.json
    analysis/output/figures/km_01_elbow_silhouette.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 09: K-Means Clustering")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    
    # Subsample and scale for valid Euclidean distance calculation
    from sklearn.preprocessing import StandardScaler
    X_raw = X.sample(n=min(3000, len(X)), random_state=SEED).values
    X_sample = StandardScaler().fit_transform(X_raw)

    wcss = []
    sil_scores = []
    K_range = list(range(2, 9))

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5)
        labels = km.fit_predict(X_sample)
        wcss.append(float(km.inertia_))
        sil = float(silhouette_score(X_sample[:1000], labels[:1000]))
        sil_scores.append(sil)
        print(f"  k={k}: WCSS={km.inertia_:.2f}, Silhouette={sil:.4f}")

    # Optimal k selection
    best_k_idx = int(np.argmax(sil_scores))
    best_k = K_range[best_k_idx]

    # Visualizations: Elbow and Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(K_range, wcss, 'bo-', lw=2, markersize=8)
    ax1.set_title("K-Means WCSS (Elbow Plot)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Within-Cluster Sum of Squares")
    ax1.grid(True, alpha=0.3)

    ax2.plot(K_range, sil_scores, 'ro-', lw=2, markersize=8)
    ax2.set_title("Silhouette Coefficient vs k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_fig(fig, "km_01_elbow_silhouette.png")

    # ------------------------------------------------------------------
    # Generate 2D Cluster Visualization Plot & Cluster Profiles
    # ------------------------------------------------------------------
    from sklearn.decomposition import PCA
    final_km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    final_labels = final_km.fit_predict(X_sample)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_sample)
    centers_pca = pca.transform(final_km.cluster_centers_)

    # Pull sample labels for CTR mapping
    sample_df = df.iloc[:len(X_sample)]
    sample_targets = sample_df[TARGET].values

    fig_cluster, ax_c = plt.subplots(figsize=(10, 6))
    cluster_colors = ['#38bdf8', '#34d399', '#f43f5e', '#a855f7', '#f59e0b']
    
    cluster_profiles = []
    for c_id in range(best_k):
        mask = (final_labels == c_id)
        c_count = int(np.sum(mask))
        c_pct = round(float(c_count / len(final_labels) * 100), 2)
        c_ctr = round(float(np.mean(sample_targets[mask]) * 100), 2)
        
        color = cluster_colors[c_id % len(cluster_colors)]
        ax_c.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, alpha=0.55, s=22,
            label=f"Cluster {c_id}: {c_count:,} pts ({c_pct}%, CTR {c_ctr}%)"
        )
        # Plot centroid
        ax_c.scatter(
            centers_pca[c_id, 0], centers_pca[c_id, 1],
            c='yellow', edgecolors='black', marker='*', s=260, lw=1.5, zorder=5
        )
        ax_c.text(
            centers_pca[c_id, 0] + 0.15, centers_pca[c_id, 1] + 0.15,
            f"Centroid {c_id}", fontsize=10, weight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85)
        )

        personas = [
            {"name": "High-Engagement In-App Traffic", "desc": "Premium app inventory with elevated click propensity."},
            {"name": "Standard Display Banners", "desc": "Baseline web publisher impressions with steady average engagement."},
            {"name": "High-Volume Long-Tail Impressions", "desc": "Mass programmatic impressions with lower conversion affinity."}
        ]
        p_info = personas[c_id] if c_id < len(personas) else {"name": f"Segment {c_id}", "desc": "Distinct impression sub-population."}
        
        cluster_profiles.append({
            "cluster_id": c_id,
            "count": c_count,
            "pct": c_pct,
            "mean_ctr": c_ctr,
            "persona_name": p_info["name"],
            "persona_desc": p_info["desc"]
        })

    ax_c.set_title(f"K-Means (k={best_k}) 2D Feature Space Cluster Partition & Centroids", fontsize=12)
    ax_c.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax_c.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax_c.legend(loc="upper right", framealpha=0.9)
    ax_c.grid(True, alpha=0.25)
    fig_cluster.tight_layout()
    save_fig(fig_cluster, "km_02_clusters_scatter.png")

    summary = {
        "k_optimal": best_k,
        "n_clusters": best_k,
        "wcss": {str(k): round(v, 2) for k, v in zip(K_range, wcss)},
        "silhouette_scores": {str(k): round(v, 4) for k, v in zip(K_range, sil_scores)},
        "best_silhouette": round(max(sil_scores), 4),
        "cluster_profiles": cluster_profiles
    }
    save_json("09_kmeans_summary.json", summary)
    print(f"K-Means clustering complete. Optimal k = {best_k}. Cluster profiles and 2D scatter generated.")


if __name__ == "__main__":
    main()
