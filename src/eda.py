# eda.py
# ------------------------------------------------------------------
# Exploratory Data Analysis: look at the data before modelling.
#
#   python src/eda.py
#
# Makes (all in results/eda/):
#   class_balance.png               how many images / fruits per class
#   sample_images.png               example photos from both classes
#   segmentation_examples.png       photo next to the same photo without background
#   feature_distributions.png       histograms of 8 key features, by class
#   feature_boxplots.png            the same 8 features as boxplots
#   correlation_heatmap.png         how the features relate to each other
#   feature_target_correlation.png  how each feature relates to the label
#   summary_statistics.csv          mean, std, min, max ... of every feature
#   summary_by_class.csv            mean of every feature for each class
# ------------------------------------------------------------------
import os

import cv2
import matplotlib
matplotlib.use("Agg")                    # save plots to files, do not open windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features import FEATURE_NAMES, load_image, remove_background

os.makedirs("results/eda", exist_ok=True)

# the same colour for the same class in every plot
COLORS = {"Healthy": "#2a78d6", "Canker": "#eb6834"}

data = pd.read_csv("data/features.csv")
metadata = pd.read_csv("data/metadata.csv")

# ------------------------------------------------------------------
# 1. Summary statistics tables
# ------------------------------------------------------------------
data[FEATURE_NAMES].describe().T.round(4).to_csv("results/eda/summary_statistics.csv", index_label="feature")
data.groupby("label")[FEATURE_NAMES].mean().T.round(4).to_csv("results/eda/summary_by_class.csv", index_label="feature")

# ------------------------------------------------------------------
# 2. Class balance: images per class and fruits per class
# ------------------------------------------------------------------
images_per_class = data["label"].value_counts()
fruits_per_class = data.groupby("label")["sample_id"].nunique()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, counts, title in [(axes[0], images_per_class, "Images per class"),
                          (axes[1], fruits_per_class, "Physical samples (fruits) per class")]:
    counts = counts[["Healthy", "Canker"]]
    bars = ax.bar(counts.index, counts.values, color=[COLORS["Healthy"], COLORS["Canker"]], width=0.5)
    ax.bar_label(bars)                   # write the number on top of each bar
    ax.set_title(title)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("results/eda/class_balance.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. Grid of sample images (6 different fruits from each class)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 6, figsize=(13, 5))
row = 0
for class_name in ["Canker", "Healthy"]:
    one_per_fruit = metadata[metadata["label"] == class_name].drop_duplicates("sample_id")
    chosen = one_per_fruit.sample(6, random_state=42)
    for col in range(6):
        filename = chosen.iloc[col]["filename"]
        image = load_image("data/raw/" + filename)
        axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(class_name + "\n" + filename, fontsize=8)
        axes[row, col].axis("off")
    row += 1
plt.tight_layout()
plt.savefig("results/eda/sample_images.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 4. Background removal examples: original on top, fruit only below
# ------------------------------------------------------------------
examples = pd.concat([
    metadata[metadata["label"] == "Canker"].sample(2, random_state=1),
    metadata[metadata["label"] == "Healthy"].sample(2, random_state=1),
])
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for col in range(4):
    image = load_image("data/raw/" + examples.iloc[col]["filename"])
    mask = remove_background(image)
    fruit_only = image.copy()
    fruit_only[mask == 0] = 255          # paint the background white

    axes[0, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(examples.iloc[col]["label"] + ": original", fontsize=9)
    axes[0, col].axis("off")
    axes[1, col].imshow(cv2.cvtColor(fruit_only, cv2.COLOR_BGR2RGB))
    axes[1, col].set_title("background removed", fontsize=9)
    axes[1, col].axis("off")
plt.tight_layout()
plt.savefig("results/eda/segmentation_examples.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 5. Which features are most related to the label?
# Correlation of every feature with the label (Canker = 1, Healthy = 0)
# ------------------------------------------------------------------
target = data["label"].map({"Healthy": 0, "Canker": 1})
correlation_with_label = data[FEATURE_NAMES].corrwith(target).sort_values()

bar_colors = []
for value in correlation_with_label:
    if value > 0:
        bar_colors.append(COLORS["Canker"])      # higher in Canker
    else:
        bar_colors.append(COLORS["Healthy"])     # higher in Healthy

plt.figure(figsize=(8, 10))
plt.barh(correlation_with_label.index, correlation_with_label.values, color=bar_colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Correlation with the label (Canker = 1, Healthy = 0)")
plt.title("Orange = higher in Canker, blue = higher in Healthy")
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("results/eda/feature_target_correlation.png", dpi=150)
plt.close()

# the 8 features with the strongest correlation (positive or negative)
top8 = correlation_with_label.abs().sort_values(ascending=False).index[:8].tolist()

# ------------------------------------------------------------------
# 6. Histograms and boxplots of those 8 features, by class
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
axes = axes.flatten()
for i in range(8):
    sns.histplot(data=data, x=top8[i], hue="label", hue_order=["Healthy", "Canker"],
                 palette=COLORS, bins=18, ax=axes[i])
    axes[i].set_title(top8[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Images")
plt.suptitle("Distribution of the 8 features most related to the label")
plt.tight_layout()
plt.savefig("results/eda/feature_distributions.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
axes = axes.flatten()
for i in range(8):
    sns.boxplot(data=data, x="label", y=top8[i], hue="label", order=["Healthy", "Canker"],
                palette=COLORS, legend=False, ax=axes[i])
    axes[i].set_title(top8[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
plt.suptitle("Boxplots of the same 8 features (dots are outliers)")
plt.tight_layout()
plt.savefig("results/eda/feature_boxplots.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 7. Correlation heatmap between all features
# ------------------------------------------------------------------
plt.figure(figsize=(14, 12))
sns.heatmap(data[FEATURE_NAMES].corr(), cmap="coolwarm", vmin=-1, vmax=1, center=0,
            xticklabels=True, yticklabels=True, cbar_kws={"label": "Correlation"})
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title("Feature correlation heatmap (red = move together, blue = move in opposite directions)")
plt.tight_layout()
plt.savefig("results/eda/correlation_heatmap.png", dpi=150)
plt.close()

print("Images:", len(data), " fruits:", data["sample_id"].nunique(), " features:", len(FEATURE_NAMES))
print(images_per_class)
print("8 features most related to the label:")
print(correlation_with_label[top8].round(3))
print("Saved EDA plots and tables to results/eda/")
