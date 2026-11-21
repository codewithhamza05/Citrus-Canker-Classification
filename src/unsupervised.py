# unsupervised.py
# ------------------------------------------------------------------
# PCA and K-Means. These are kept separate from the 5 supervised models.
#
#   python src/unsupervised.py
#
# PCA     = dimensionality reduction. It squeezes the 42 features into a few
#           new ones (components). It never sees the labels. We only use the
#           labels afterwards to COLOUR the scatter plot.
# K-Means = unsupervised clustering. It groups similar images WITHOUT ever
#           seeing the labels. It is NOT a classifier. We only compare its
#           clusters with the true labels afterwards, for discussion.
#
# Makes (all in results/):
#   pca_scree.png  pca_scatter.png  pca_loadings.png  pca_loadings.csv
#   kmeans_elbow_silhouette.png  kmeans_vs_actual.png  kmeans_contingency.csv
#   unsupervised.json
# ------------------------------------------------------------------
import json

import matplotlib
matplotlib.use("Agg")                    # save plots to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES

COLORS = {"Healthy": "#2a78d6", "Canker": "#eb6834"}
CLUSTER_COLORS = ["#1baf7a", "#6250d6"]      # different colours: clusters are not classes

data = pd.read_csv("data/features.csv")
labels = data["label"]

# both methods need scaled features (otherwise big-number features dominate)
X = StandardScaler().fit_transform(data[FEATURE_NAMES])

# ==================================================================
# PART 1: PCA
# ==================================================================
pca = PCA()
components = pca.fit_transform(X)                 # labels are NOT given to PCA
explained = pca.explained_variance_ratio_ * 100   # % of variance per component
cumulative = np.cumsum(explained)                 # running total
n_for_90 = int(np.argmax(cumulative >= 90) + 1)   # components needed to reach 90%

# scree plot: bars = each component, line = running total
numbers = np.arange(1, 21)
plt.figure(figsize=(9, 5))
plt.bar(numbers, explained[:20], color="#2a78d6", label="Each component")
plt.plot(numbers, cumulative[:20], color="#eb6834", marker="o", label="Cumulative")
plt.axhline(90, color="gray", linestyle="--", linewidth=1)
plt.text(18.7, 92, "90%")
plt.xticks(numbers)
plt.xlabel("Principal component")
plt.ylabel("Variance explained (%)")
plt.title("PCA scree plot (" + str(n_for_90) + " components reach 90%)")
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig("results/pca_scree.png", dpi=150)
plt.close()

# 2-D scatter of PC1 vs PC2. Labels are used only for the colours.
pc1 = components[:, 0]
pc2 = components[:, 1]
plt.figure(figsize=(8, 6))
for class_name in ["Healthy", "Canker"]:
    rows = labels == class_name
    plt.scatter(pc1[rows], pc2[rows], color=COLORS[class_name], label=class_name,
                alpha=0.8, edgecolors="white")
plt.xlabel("PC1 (" + str(round(explained[0], 1)) + "% of variance)")
plt.ylabel("PC2 (" + str(round(explained[1], 1)) + "% of variance)")
plt.title("PCA: every image placed by its first two components")
plt.legend(title="True label (colour only)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/pca_scatter.png", dpi=150)
plt.close()

# loadings: how much each original feature contributes to PC1 and PC2
loadings = pd.DataFrame({"PC1": pca.components_[0], "PC2": pca.components_[1]}, index=FEATURE_NAMES)
loadings.round(4).to_csv("results/pca_loadings.csv", index_label="feature")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pc in [(axes[0], "PC1"), (axes[1], "PC2")]:
    strongest = loadings[pc].abs().sort_values(ascending=False).index[:12]     # 12 biggest loadings
    values = loadings.loc[strongest, pc][::-1]
    ax.barh(values.index, values.values, color="#2a78d6")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Loading")
    ax.set_title(pc + ": 12 strongest features")
plt.tight_layout()
plt.savefig("results/pca_loadings.png", dpi=150)
plt.close()

# ==================================================================
# PART 2: K-MEANS
# ==================================================================
# try k = 2 ... 6 and record inertia (for the elbow plot) and silhouette score
k_values = [2, 3, 4, 5, 6]
inertias = []
silhouettes = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_of_image = model.fit_predict(X)       # labels are NOT given to K-Means
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, cluster_of_image))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(k_values, inertias, marker="o", color="#2a78d6")
axes[0].set_xticks(k_values)
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow method")
axes[0].grid(alpha=0.3)
axes[1].plot(k_values, silhouettes, marker="o", color="#2a78d6")
axes[1].set_xticks(k_values)
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score (higher = better separated clusters)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/kmeans_elbow_silhouette.png", dpi=150)
plt.close()

# final clustering with k = 2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# compare the clusters with the true labels (only now are the labels used)
ari = adjusted_rand_score(labels, clusters)       # 1 = perfect match, about 0 = random
cluster_names = pd.Series(clusters).map({0: "Cluster 0", 1: "Cluster 1"})
contingency = pd.crosstab(labels, cluster_names.values, rownames=["True label"], colnames=["K-Means cluster"])
contingency.to_csv("results/kmeans_contingency.csv")

# clusters (left) next to true labels (right), both drawn on PC1 / PC2
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for c in [0, 1]:
    rows = clusters == c
    axes[0].scatter(pc1[rows], pc2[rows], color=CLUSTER_COLORS[c], label="Cluster " + str(c),
                    alpha=0.8, edgecolors="white")
axes[0].set_title("K-Means clusters (k = 2, no labels used)   ARI = " + str(round(ari, 3)))
for class_name in ["Healthy", "Canker"]:
    rows = labels == class_name
    axes[1].scatter(pc1[rows], pc2[rows], color=COLORS[class_name], label=class_name,
                    alpha=0.8, edgecolors="white")
axes[1].set_title("True labels (for comparison only)")
for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/kmeans_vs_actual.png", dpi=150)
plt.close()

# ==================================================================
# Save the key numbers (the dashboard reads this file)
# ==================================================================
summary = {
    "pca_pc1_percent": round(float(explained[0]), 2),
    "pca_pc2_percent": round(float(explained[1]), 2),
    "pca_pc1_plus_pc2_percent": round(float(cumulative[1]), 2),
    "pca_components_for_90_percent": n_for_90,
    "kmeans_k": 2,
    "kmeans_adjusted_rand_index": round(float(ari), 4),
    "kmeans_silhouette_k2": round(float(silhouettes[0]), 4),
}
with open("results/unsupervised.json", "w") as f:
    json.dump(summary, f, indent=2)

print("PCA: PC1 =", round(explained[0], 1), "%  PC2 =", round(explained[1], 1),
      "%  together =", round(cumulative[1], 1), "% ;", n_for_90, "components reach 90%")
print("K-Means (k=2): ARI =", round(ari, 3), " silhouette =", round(silhouettes[0], 3))
print(contingency)
print("Saved PCA and K-Means plots to results/")
