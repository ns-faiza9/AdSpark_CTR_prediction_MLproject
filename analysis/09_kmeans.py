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
    
    # Subsample for computational speed
    X_sample = X.sample(n=min(3000, len(X)), random_state=SEED).values

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

    summary = {
        "k_optimal": best_k,
        "wcss": {str(k): round(v, 2) for k, v in zip(K_range, wcss)},
        "silhouette_scores": {str(k): round(v, 4) for k, v in zip(K_range, sil_scores)},
        "best_silhouette": round(max(sil_scores), 4)
    }
    save_json("09_kmeans_summary.json", summary)
    print(f"K-Means clustering complete. Optimal k = {best_k}.")


if __name__ == "__main__":
    main()
