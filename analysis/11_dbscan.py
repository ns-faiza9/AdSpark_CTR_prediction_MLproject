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
    
    X_sample = X.sample(n=min(1000, len(X)), random_state=SEED).values

    # k-distance graph calculation (MinPts = 5)
    min_pts = 5
    nn = NearestNeighbors(n_neighbors=min_pts).fit(X_sample)
    distances, _ = nn.kneighbors(X_sample)
    k_distances = np.sort(distances[:, -1])[::-1]

    eps_cutoff = 1.8

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

    print(f"  DBSCAN Epsilon={eps_cutoff}: Clusters={n_clusters}, Noise points={n_noise} ({noise_pct}%)")

    summary = {
        "epsilon": eps_cutoff,
        "min_samples": min_pts,
        "estimated_clusters": n_clusters,
        "noise_points": n_noise,
        "noise_percentage": noise_pct
    }
    save_json("11_dbscan_summary.json", summary)
    print("DBSCAN clustering complete.")


if __name__ == "__main__":
    main()
