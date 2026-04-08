"""
clustering.py — KMeans (k=3) clustering on photometric magnitudes.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def train_kmeans(X_scaled, n_clusters: int = 3, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_scaled)
    print(f"[Clustering] KMeans (k={n_clusters}) trained. Inertia: {model.inertia_:.2f}")
    return model


def evaluate(model, X_scaled, true_labels, output_dir: str):
    cluster_labels = model.labels_
    unique_clusters = np.unique(cluster_labels)

    # Reduce to 2D via PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_cluster = ["#e74c3c", "#2ecc71", "#3498db"]
    colors_class = {"Galaxy": "#e67e22", "Star": "#9b59b6", "QSO": "#1abc9c"}

    # Left: KMeans clusters
    for c in unique_clusters:
        mask = cluster_labels == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        s=15, alpha=0.5, color=colors_cluster[c], label=f"Cluster {c}")
    axes[0].set_title("KMeans Clusters (PCA projection)")
    axes[0].legend(markerscale=2)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # Right: True class labels
    for cls, color in colors_class.items():
        mask = true_labels == cls
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        s=15, alpha=0.5, color=color, label=cls)
    axes[1].set_title("True Class Labels (PCA projection)")
    axes[1].legend(markerscale=2)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    fig.suptitle("KMeans Clustering vs Real Classes — SDSS Photometric Bands", fontsize=12)
    fig.tight_layout()
    path = f"{output_dir}/clustering_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Clustering] Comparison plot saved → {path}")

    # Cluster composition table
    composition = {}
    for c in unique_clusters:
        mask = cluster_labels == c
        counts = {cls: int(np.sum(true_labels[mask] == cls)) for cls in colors_class}
        composition[f"cluster_{c}"] = counts
    with open(f"{output_dir}/clustering_metrics.json", "w") as f:
        json.dump({"inertia": round(float(model.inertia_), 2), "composition": composition}, f, indent=2)
    print(f"[Clustering] Metrics saved → {output_dir}/clustering_metrics.json")
