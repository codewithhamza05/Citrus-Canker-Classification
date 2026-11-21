# preprocess.py
# ------------------------------------------------------------------
# Data cleaning checks, outlier check and the train/test split.
#
#   python src/preprocess.py
#
# Makes:
#   results/outliers.csv          images with unusual feature values
#   results/cleaning_report.txt   every cleaning decision, written down
#   data/split.csv                which image is in train and which is in test
#
# Note: scaling (StandardScaler) is NOT done here. It is done inside the
# model pipeline in train.py, so it is fitted on the training data only.
# ------------------------------------------------------------------
import os

import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import FEATURE_NAMES

os.makedirs("results", exist_ok=True)

metadata = pd.read_csv("data/metadata.csv")
data = pd.read_csv("data/features.csv")
report = []          # lines of text for the cleaning report

# ------------------------------------------------------------------
# STEP 1: check every image (readable? blurry? has a sample_id and a label?)
# ------------------------------------------------------------------
unreadable = []
blurry = []
for i in range(len(metadata)):
    filename = metadata.loc[i, "filename"]
    gray = cv2.imread("data/raw/" + filename, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        unreadable.append(filename)
        continue
    # sharpness = variance of the Laplacian. A blurry photo has few edges, so a low value.
    sharpness = cv2.Laplacian(cv2.resize(gray, (512, 512)), cv2.CV_64F).var()
    if sharpness < 20:
        blurry.append(filename)

missing_sample_id = metadata["sample_id"].isna().sum()
wrong_labels = (~metadata["label"].isin(["Canker", "Healthy"])).sum()
missing_values = data[FEATURE_NAMES].isna().sum().sum()

report.append("DATA CLEANING REPORT")
report.append("")
report.append("Images in the dataset: " + str(len(metadata)))
report.append("Canker images: " + str((metadata["label"] == "Canker").sum()))
report.append("Healthy images: " + str((metadata["label"] == "Healthy").sum()))
report.append("Physical samples (different fruits): " + str(metadata["sample_id"].nunique()))
report.append("")
report.append("Unreadable images: " + str(len(unreadable)))
report.append("Images without a sample_id: " + str(missing_sample_id))
report.append("Images with a wrong label: " + str(wrong_labels))
report.append("Missing feature values: " + str(missing_values))
report.append("Blurry images (sharpness below 20): " + str(blurry))
report.append("  Decision: looked at it. It is slightly soft but the fruit surface is still visible, so it is KEPT.")

# files that were removed earlier, when the dataset was organised
removed = pd.read_csv("data/excluded/excluded_log.csv")
n_duplicates = removed["reason"].str.startswith("exact duplicate").sum()
n_conflicts = removed["reason"].str.startswith("label conflict").sum()
report.append("")
report.append("Removed before feature extraction (see data/excluded/excluded_log.csv): " + str(len(removed)))
report.append("  exact duplicate files: " + str(n_duplicates))
report.append("  same fruit found in BOTH class folders (label cannot be trusted): " + str(n_conflicts))

# ------------------------------------------------------------------
# STEP 2: outliers with z-scores
# z-score = how many standard deviations a value is away from the mean.
# |z| > 3 is an extreme value. An image is flagged when 5 or more of its
# features are extreme.
# ------------------------------------------------------------------
values = data[FEATURE_NAMES]
z_scores = (values - values.mean()) / values.std()
extreme = z_scores.abs() > 3
data["n_extreme_features"] = extreme.sum(axis=1)

outliers = data[data["n_extreme_features"] >= 5]
outliers = outliers[["image_id", "sample_id", "label", "n_extreme_features"]]
outliers = outliers.sort_values("n_extreme_features", ascending=False)
outliers.to_csv("results/outliers.csv", index=False)
data = data.drop(columns="n_extreme_features")

report.append("")
report.append("Outliers (5 or more features with |z-score| > 3): " + str(len(outliers)) + " images")
report.append("  image ids: " + str(outliers["image_id"].tolist()))
report.append("  Decision: looked at them. They are mostly the studio-style healthy photos (small fruit on a")
report.append("  white background) and one yellow-green canker fruit. They are real photos, not errors, so they are KEPT.")

# ------------------------------------------------------------------
# STEP 3: train/test split BY PHYSICAL SAMPLE, made only once
# All photos of the same fruit (same sample_id) go to the same side.
# Otherwise the model could "recognise" a fruit it has already seen (leakage).
# ------------------------------------------------------------------
if os.path.exists("data/split.csv"):
    print("data/split.csv already exists, keeping it (the split is made only ONCE).")
    print("Delete the file first if the dataset changed and you really need a new split.")
    split = pd.read_csv("data/split.csv")
else:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rows, test_rows = next(splitter.split(data, data["label"], groups=data["sample_id"]))

    # proof that no fruit is in both sets
    train_samples = set(data.iloc[train_rows]["sample_id"])
    test_samples = set(data.iloc[test_rows]["sample_id"])
    assert train_samples.isdisjoint(test_samples)

    split = data[["image_id", "sample_id", "label"]].copy()
    split["split"] = "train"
    split.iloc[test_rows, split.columns.get_loc("split")] = "test"
    split.to_csv("data/split.csv", index=False)
    print("Saved data/split.csv")

train = split[split["split"] == "train"]
test = split[split["split"] == "test"]

# both classes must be in the test set
assert test["label"].nunique() == 2

report.append("")
report.append("Train/test split: by sample_id, GroupShuffleSplit(test_size=0.2, random_state=42)")
report.append("  train: " + str(len(train)) + " images from " + str(train["sample_id"].nunique()) + " fruits")
report.append("  test:  " + str(len(test)) + " images from " + str(test["sample_id"].nunique()) + " fruits")
report.append("  test set has " + str((test["label"] == "Canker").sum()) + " Canker and "
              + str((test["label"] == "Healthy").sum()) + " Healthy images")
report.append("  fruits in both sets: " + str(len(set(train["sample_id"]) & set(test["sample_id"]))))
report.append("")
report.append("Label encoding: Canker = 1, Healthy = 0")
report.append("Scaling: StandardScaler inside the model pipeline (fitted on training data only)")

report_text = "\n".join(report)
with open("results/cleaning_report.txt", "w") as f:
    f.write(report_text + "\n")
print(report_text)
