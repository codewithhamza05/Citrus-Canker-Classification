# train.py
# ------------------------------------------------------------------
# Trains and tunes all 5 models in the same way.
#
#   python src/train.py
#
# For every model:
#   1. build a pipeline: StandardScaler -> model
#   2. try several settings with GridSearchCV (5-fold cross-validation)
#   3. keep the settings with the best F1 score
#   4. save the whole pipeline (scaler + model) to models/
#
# Only the TRAINING set is used here. The test set is used once, in evaluate.py.
# ------------------------------------------------------------------
import json
import os
import warnings

# scikit-learn prints the same harmless warning for every single fit. Hide them.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import FEATURE_NAMES

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------
# STEP 1: load the training data
# ------------------------------------------------------------------
data = pd.read_csv("data/features.csv")
split = pd.read_csv("data/split.csv")
data = data.merge(split[["image_id", "split"]], on="image_id")

train = data[data["split"] == "train"]
X_train = train[FEATURE_NAMES]
y_train = train["label"].map({"Healthy": 0, "Canker": 1})     # Canker = 1, Healthy = 0
groups = train["sample_id"]                                   # which fruit each image shows

print("Training images:", len(train), " fruits:", groups.nunique(), " features:", len(FEATURE_NAMES))

# ------------------------------------------------------------------
# STEP 2: the 5 models and the settings to try for each one
# ("clf__" just means: this setting belongs to the classifier step of the pipeline)
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=5000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, n_jobs=1),
}

settings_to_try = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],          # small C = stronger regularisation
        "clf__l1_ratio": [0, 1],                    # the penalty: 0 = L2 (ridge), 1 = L1 (lasso)
    },
    "SVM": {
        "clf__kernel": ["linear", "rbf"],           # straight or curved boundary
        "clf__C": [0.1, 1, 10],                     # how hard it tries to fit every training point
        "clf__gamma": ["scale", 0.01, 0.1],         # how far one point's influence reaches (rbf)
    },
    "Decision Tree": {
        "clf__max_depth": [2, 3, 5, 7, None],       # how deep the tree may grow
        "clf__min_samples_leaf": [1, 3, 5, 10],     # smallest group allowed at a leaf
        "clf__criterion": ["gini", "entropy"],      # how a split is scored
    },
    "Random Forest": {
        "clf__n_estimators": [100, 200, 300],       # number of trees
        "clf__max_depth": [None, 5, 10],            # depth of each tree
        "clf__max_features": ["sqrt", "log2", 0.5], # features each split may look at
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200],            # number of trees
        "clf__learning_rate": [0.05, 0.1, 0.2],     # how much each new tree corrects the old ones
        "clf__max_depth": [2, 3, 5],                # depth of each tree
        "clf__subsample": [0.8, 1.0],               # share of the images each tree sees
    },
}

# GroupKFold: 5 folds, and all photos of one fruit always stay in the same fold.
# This stops leakage inside the tuning.
folds = GroupKFold(n_splits=5)

# ------------------------------------------------------------------
# STEP 3: tune, train and save each model
# ------------------------------------------------------------------
cv_results = {}

for name in models:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),      # scaling is fitted on the training folds only
        ("clf", models[name]),
    ])

    search = GridSearchCV(pipeline, settings_to_try[name], cv=folds, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train, groups=groups)

    # cross-validation F1 of the best settings: mean and standard deviation over the 5 folds
    best = search.best_index_
    cv_mean = search.cv_results_["mean_test_score"][best]
    cv_std = search.cv_results_["std_test_score"][best]

    # tidy the names of the best settings, e.g. "clf__C" -> "C"
    best_settings = {}
    for key in search.best_params_:
        best_settings[key.replace("clf__", "")] = search.best_params_[key]

    cv_results[name] = {
        "best_params": best_settings,
        "cv_f1_mean": round(float(cv_mean), 4),
        "cv_f1_std": round(float(cv_std), 4),
    }

    # save the whole pipeline (scaler + best model), e.g. models/random_forest.joblib
    filename = "models/" + name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(search.best_estimator_, filename)

    print(name, "| CV F1 =", round(cv_mean, 3), "+/-", round(cv_std, 3), "| best settings:", best_settings)

with open("results/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

print("Saved 5 pipelines to models/ and the CV scores to results/cv_results.json")
