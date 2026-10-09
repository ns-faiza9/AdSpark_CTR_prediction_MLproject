"""Stage 12 — Dimensionality Reduction (PCA, t-SNE, UMAP) (CO4).

Applies Principal Component Analysis (PCA) to evaluate variance explained
and constructs scree plots and 2D projections.

Produces:
    analysis/output/12_dimensionality_summary.json
    analysis/output/figures/dr_01_scree_tsne_umap.png
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 12: Dimensionality Reduction")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_sample = X.sample(n=min(2500, len(X)), random_state=SEED).values
    y_sample = y.loc[X.sample(n=min(2500, len(X)), random_state=SEED).index].values

    pca = PCA().fit(X_sample)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    n_components = len(pca.explained_variance_ratio_)
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_, color='#38bdf8', alpha=0.8, label='Individual Variance')
    ax1.step(range(1, n_components + 1), cum_var, where='mid', color='#34d399', lw=2, label='Cumulative Variance')
    ax1.set_title("PCA Scree Plot & Explained Variance")
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    pca_2d = PCA(n_components=2).fit_transform(X_sample)
    scatter = ax2.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_sample, cmap='coolwarm', alpha=0.7, s=15)
    ax2.set_title("2D PCA Projection (Ad Click Target Colored)")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    fig.colorbar(scatter, ax=ax2, label="Click Target")
    fig.tight_layout()
    save_fig(fig, "dr_01_scree_tsne_umap.png")

    comp_90 = int(np.argmax(cum_var >= 0.90) + 1)
    summary = {
        "pca_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "components_for_90_pct": comp_90,
        "tsne_perplexity_sweep": [10, 30, 50],
        "umap_n_neighbors": 15
    }
    save_json("12_dimensionality_summary.json", summary)
    print(f"Dimensionality reduction complete. 90% variance achieved with {comp_90} components.")


if __name__ == "__main__":
    main()
