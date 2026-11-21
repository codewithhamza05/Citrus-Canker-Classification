# evaluate.py
# ------------------------------------------------------------------
# Tests the 5 trained models on the held-out TEST set and compares them.
#
#   python src/evaluate.py
#
# Makes (all in results/):
#   metrics.json                     all numbers for every model
#   cm_<model>.png                   confusion matrix of every model
#   classification_reports.txt
#   model_comparison.csv / .png      table + grouped bar chart of all 5 models
#   cv_comparison.png                cross-validation F1 (mean +/- std)
#   feature_importance_<model>.png   for Random Forest and XGBoost
# ------------------------------------------------------------------
import json

import joblib
import matplotlib
matplotlib.use("Agg")                    # save plots to files, do not open windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

from features import FEATURE_NAMES

MODEL_NAMES = ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "XGBoost"]

# ------------------------------------------------------------------
# STEP 1: load the test data
# ------------------------------------------------------------------
data = pd.read_csv("data/features.csv")
split = pd.read_csv("data/split.csv")
data = data.merge(split[["image_id", "split"]], on="image_id")

test = data[data["split"] == "test"]
X_test = test[FEATURE_NAMES]
y_test = test["label"].map({"Healthy": 0, "Canker": 1})

with open("results/cv_results.json") as f:
    cv_results = json.load(f)

print("Test images:", len(test), " fruits:", test["sample_id"].nunique())

# ------------------------------------------------------------------
# STEP 2: evaluate every model
# ------------------------------------------------------------------
metrics = {}
table_rows = []
reports_text = ""

for name in MODEL_NAMES:
    file_key = name.lower().replace(" ", "_")
    model = joblib.load("models/" + file_key + ".joblib")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Canker"], zero_division=0)

    metrics[name] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),           # rows = actual, columns = predicted
        "classification_report": report,
        "best_params": cv_results[name]["best_params"],
        "cv_f1_mean": cv_results[name]["cv_f1_mean"],
        "cv_f1_std": cv_results[name]["cv_f1_std"],
    }
    table_rows.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "CV F1 mean": cv_results[name]["cv_f1_mean"],
        "CV F1 std": cv_results[name]["cv_f1_std"],
    })
    reports_text += "===== " + name + " =====\n" + report + "\n"

    # confusion matrix picture
    display = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Canker"])
    display.plot(cmap="Blues", colorbar=False)
    plt.title(name + "\naccuracy " + str(round(accuracy * 100, 1)) + "%   F1 " + str(round(f1 * 100, 1)) + "%")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/cm_" + file_key + ".png", dpi=150)
    plt.close()

    # feature importance (only the tree ensembles have it)
    if name in ["Random Forest", "XGBoost"]:
        classifier = model.named_steps["clf"]
        importance = pd.DataFrame({"feature": FEATURE_NAMES, "importance": classifier.feature_importances_})
        importance = importance.sort_values("importance", ascending=False)
        importance.to_csv("results/feature_importance_" + file_key + ".csv", index=False)

        top15 = importance.head(15).sort_values("importance")
        plt.figure(figsize=(7, 5))
        plt.barh(top15["feature"], top15["importance"], color="#2a78d6")
        plt.xlabel("Importance")
        plt.title(name + ": 15 most important features")
        plt.tight_layout()
        plt.savefig("results/feature_importance_" + file_key + ".png", dpi=150)
        plt.close()

# ------------------------------------------------------------------
# STEP 3: save the numbers
# ------------------------------------------------------------------
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("results/classification_reports.txt", "w") as f:
    f.write(reports_text)

table = pd.DataFrame(table_rows)
table.to_csv("results/model_comparison.csv", index=False)

# ------------------------------------------------------------------
# STEP 4: comparison charts
# ------------------------------------------------------------------
# grouped bar chart: 4 test metrics for each model
chart = table.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
chart.plot(kind="bar", figsize=(10, 5), color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], width=0.8)
plt.ylim(0, 1.1)
plt.ylabel("Score on the test set")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Model comparison on the test set")
plt.grid(axis="y", alpha=0.3)
plt.legend(ncol=4, loc="upper right")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()

# cross-validation F1: one dot per model, the line shows +/- 1 standard deviation
plt.figure(figsize=(8, 4.5))
plt.errorbar(table["Model"], table["CV F1 mean"], yerr=table["CV F1 std"],
             fmt="o", color="#2a78d6", ecolor="gray", capsize=5, markersize=8)
plt.ylim(0.5, 1.1)
plt.ylabel("F1 in 5-fold cross-validation")
plt.title("Cross-validation F1 (mean +/- std over 5 folds)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/cv_comparison.png", dpi=150)
plt.close()

print(reports_text)
print(table.to_string(index=False))
print("Saved results/metrics.json, confusion matrices and comparison charts")
