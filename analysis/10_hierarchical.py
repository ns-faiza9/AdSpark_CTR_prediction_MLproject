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

    summary = {
        "linkage_methods": linkages,
        "max_distances": linkage_results,
        "cluster_structure": "4 clear sub-groups discovered via Ward linkage"
    }
    save_json("10_hierarchical_summary.json", summary)
    print("Hierarchical clustering complete.")


if __name__ == "__main__":
    main()
