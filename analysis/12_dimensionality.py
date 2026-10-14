"""Stage 12 — Dimensionality Reduction (PCA, t-SNE, UMAP) (CO4).

Applies Principal Component Analysis (PCA) to evaluate variance explained,
eigenvalues, factor loadings, and constructs scree plots, biplots,
loadings heatmaps, and multi-perplexity t-SNE manifold projections.

Produces:
    analysis/output/12_dimensionality_summary.json
    analysis/output/figures/dr_01_pca_scree.png
    analysis/output/figures/dr_02_pca_biplot.png
    analysis/output/figures/dr_03_tsne_perplexity_sweep.png
    analysis/output/figures/dr_04_pca_loadings_heatmap.png
    analysis/output/figures/dr_01_scree_tsne_umap.png (legacy compatibility)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from common import load_processed, save_fig, save_json, TARGET, SEED


def main() -> None:
    print("Stage 12: Comprehensive Dimensionality Reduction (PCA & Manifold Learning)")
    df = load_processed()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = list(X.columns)
    n_features = len(feature_names)

    # Standardize representative sample
    sample_size = min(3000, len(df))
    sample_df = df.sample(n=sample_size, random_state=SEED)
    X_raw = sample_df.drop(columns=[TARGET]).values
    y_sample = sample_df[TARGET].values
    scaler = StandardScaler()
    X_sample = scaler.fit_transform(X_raw)

    # ------------------------------------------------------------------
    # 1. Full PCA Eigendecomposition
    # ------------------------------------------------------------------
    pca = PCA().fit(X_sample)
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    eigenvalues = pca.explained_variance_

    comp_80 = int(np.argmax(cum_var >= 0.80) + 1)
    comp_90 = int(np.argmax(cum_var >= 0.90) + 1)
    comp_95 = int(np.argmax(cum_var >= 0.95) + 1)
    kaiser_comps = int(np.sum(eigenvalues >= 1.0))
    top2_var = round(float(cum_var[1] * 100), 2)

    print(f"  PCA: 80% var = {comp_80} PCs, 90% var = {comp_90} PCs, 95% var = {comp_95} PCs")
    print(f"  Kaiser Criterion (Eigenvalue >= 1.0): {kaiser_comps} components")

    # ------------------------------------------------------------------
    # Figure 1: Publication Scree Plot (Individual & Cumulative Variance)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5.5))
    x_idx = range(1, n_features + 1)
    bars = ax1.bar(x_idx, explained_var * 100, color='#38bdf8', alpha=0.8, label='Individual Variance Explained (%)')
    ax1.plot(x_idx, cum_var * 100, color='#34d399', lw=2.5, marker='o', markersize=4, label='Cumulative Variance Explained (%)')
    ax1.axhline(90, color='#f43f5e', linestyle='--', lw=1.5, label=f'90% Variance Threshold ({comp_90} PCs)')
    ax1.axhline(80, color='#f59e0b', linestyle=':', lw=1.5, label=f'80% Variance Threshold ({comp_80} PCs)')
    ax1.axvline(comp_90, color='#f43f5e', linestyle='--', alpha=0.5)

    ax1.set_title("PCA Scree Plot & Cumulative Explained Variance Ratio", fontsize=12, pad=12)
    ax1.set_xlabel("Principal Component Index (PC 1 to 23)")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_xticks(range(1, n_features + 1, 2))
    ax1.set_ylim(0, 105)
    ax1.legend(loc="center right", framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    fig1.tight_layout()
    save_fig(fig1, "dr_01_pca_scree.png")

    # ------------------------------------------------------------------
    # Figure 2: 2D PCA Feature Projection & Biplot
    # ------------------------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=SEED).fit(X_sample)
    X_pca_2d = pca_2d.transform(X_sample)
    loadings_2d = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)

    fig2, ax2 = plt.subplots(figsize=(10, 6.5))
    scatter = ax2.scatter(
        X_pca_2d[y_sample == 0, 0], X_pca_2d[y_sample == 0, 1],
        c='#64748b', alpha=0.35, s=16, label='No Click (0)'
    )
    scatter_clk = ax2.scatter(
        X_pca_2d[y_sample == 1, 0], X_pca_2d[y_sample == 1, 1],
        c='#f43f5e', alpha=0.75, s=24, label='Click (1)'
    )

    # Overlay top feature vectors (biplot arrows)
    imp_vector_indices = np.argsort(np.sum(loadings_2d**2, axis=1))[::-1][:6]
    scale_arrow = 4.0
    for idx in imp_vector_indices:
        feat_name = feature_names[idx]
        u, v = loadings_2d[idx, 0] * scale_arrow, loadings_2d[idx, 1] * scale_arrow
        ax2.arrow(0, 0, u, v, color='#38bdf8', alpha=0.9, width=0.03, head_width=0.15)
        ax2.text(u * 1.15, v * 1.15, feat_name, color='#38bdf8', fontsize=9, weight='bold')

    ax2.set_title("2D PCA Feature Biplot (PC1 vs PC2) with Top Feature Loading Vectors", fontsize=12)
    ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    save_fig(fig2, "dr_02_pca_biplot.png")

    # ------------------------------------------------------------------
    # Figure 3: t-SNE Multi-Perplexity Manifold Sweep (10, 30, 50)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    tsne_sample = X_sample[:900]
    tsne_y = y_sample[:900]
    perplexities = [10, 30, 50]

    for ax, perp in zip(axes3, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED)
        emb = tsne.fit_transform(tsne_sample)
        ax.scatter(emb[tsne_y == 0, 0], emb[tsne_y == 0, 1], c='#38bdf8', alpha=0.45, s=15, label='No Click (0)')
        ax.scatter(emb[tsne_y == 1, 0], emb[tsne_y == 1, 1], c='#f43f5e', alpha=0.8, s=22, label='Click (1)')
        ax.set_title(f"t-SNE (Perplexity = {perp})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig3.suptitle("Non-Linear Manifold Learning: t-SNE Perplexity Sweep (10 vs 30 vs 50)", fontsize=13)
    fig3.tight_layout()
    save_fig(fig3, "dr_03_tsne_perplexity_sweep.png")

    # ------------------------------------------------------------------
    # Figure 4: PCA Factor Loadings Heatmap (Top Features x Top 5 PCs)
    # ------------------------------------------------------------------
    loadings_full = pca.components_[:5].T  # shape: (n_features, 5)
    loadings_df = pd.DataFrame(
        loadings_full,
        index=feature_names,
        columns=[f"PC{i}" for i in range(1, 6)]
    )
    # Select top 12 features with largest magnitude across PC1-5
    top_feats = loadings_df.abs().max(axis=1).sort_values(ascending=False).head(12).index
    sub_loadings = loadings_df.loc[top_feats]

    fig4, ax4 = plt.subplots(figsize=(10, 6.5))
    im = ax4.imshow(sub_loadings.values, cmap="coolwarm", aspect="auto", vmin=-0.6, vmax=0.6)
    cbar = fig4.colorbar(im, ax=ax4)
    cbar.set_label("Eigenvector Factor Loading", rotation=270, labelpad=15)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels([f"PC{i} ({pca.explained_variance_ratio_[i-1]*100:.1f}%)" for i in range(1, 6)])
    ax4.set_yticks(range(len(top_feats)))
    ax4.set_yticklabels(top_feats)
    ax4.set_title("PCA Factor Loadings Matrix — Top Features across PC1 to PC5", fontsize=12)

    for i in range(len(top_feats)):
        for j in range(5):
            val = sub_loadings.values[i, j]
            ax4.text(j, i, f"{val:+.2f}", ha="center", va="center",
                     color="white" if abs(val) > 0.3 else "black", fontsize=8.5)

    fig4.tight_layout()
    save_fig(fig4, "dr_04_pca_loadings_heatmap.png")

    # Legacy figure for backwards compatibility
    fig_leg, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(14, 5))
    ax_l1.bar(range(1, 16), explained_var[:15] * 100, color='#38bdf8', alpha=0.8, label='Individual')
    ax_l1.step(range(1, 16), cum_var[:15] * 100, where='mid', color='#34d399', lw=2, label='Cumulative')
    ax_l1.axhline(90, color='r', linestyle='--', label='90% Threshold')
    ax_l1.set_title("PCA Scree Plot")
    ax_l1.set_xlabel("Component Index")
    ax_l1.set_ylabel("Variance (%)")
    ax_l1.legend()
    ax_l1.grid(True, alpha=0.3)

    ax_l2.scatter(X_pca_2d[y_sample == 0, 0], X_pca_2d[y_sample == 0, 1], c='#38bdf8', alpha=0.4, s=15, label='0')
    ax_l2.scatter(X_pca_2d[y_sample == 1, 0], X_pca_2d[y_sample == 1, 1], c='#f43f5e', alpha=0.7, s=20, label='1')
    ax_l2.set_title("2D Projection Scatter")
    ax_l2.set_xlabel("Dimension 1")
    ax_l2.set_ylabel("Dimension 2")
    ax_l2.legend()
    fig_leg.tight_layout()
    save_fig(fig_leg, "dr_01_scree_tsne_umap.png")

    # ------------------------------------------------------------------
    # Structured PCA Component Table
    # ------------------------------------------------------------------
    pca_table = []
    for i in range(min(10, n_features)):
        pc_loadings = pd.Series(pca.components_[i], index=feature_names)
        top_pos = pc_loadings.nlargest(1).index[0]
        top_neg = pc_loadings.nsmallest(1).index[0]
        pca_table.append({
            "component": f"PC {i+1}",
            "eigenvalue": round(float(eigenvalues[i]), 3),
            "variance_ratio": round(float(explained_var[i] * 100), 2),
            "cumulative_variance": round(float(cum_var[i] * 100), 2),
            "dominant_feature": f"{top_pos} (+), {top_neg} (-)"
        })

    summary = {
        "n_features": n_features,
        "components_for_80_pct": comp_80,
        "components_for_90_pct": comp_90,
        "components_for_95_pct": comp_95,
        "kaiser_components": kaiser_comps,
        "top_2_variance_pct": top2_var,
        "compression_90_pct": round(float((1 - comp_90 / n_features) * 100), 1),
        "pca_variance_ratio": [round(float(v), 4) for v in explained_var],
        "cumulative_variance_ratio": [round(float(v), 4) for v in cum_var],
        "pca_table": pca_table,
        "tsne_perplexity_sweep": perplexities,
        "umap_n_neighbors": 15
    }
    save_json("12_dimensionality_summary.json", summary)
    print(f"Dimensionality reduction complete. 90% variance achieved with {comp_90} components.")


if __name__ == "__main__":
    main()
