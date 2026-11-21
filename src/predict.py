# predict.py
# ------------------------------------------------------------------
# Predicts Canker / Healthy for a NEW photo.
#
#   python src/predict.py photo.jpg                  (uses SVM)
#   python src/predict.py photo.jpg "Random Forest"  (choose the model)
#   python src/predict.py photo.jpg all              (ask all 5 models)
#
# What happens:
#   new photo -> extract_features() (same function as for the dataset)
#             -> saved pipeline (scaler + model)
#             -> class + probability
# ------------------------------------------------------------------
import sys

import joblib
import pandas as pd

from features import FEATURE_NAMES, extract_features

MODEL_NAMES = ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "XGBoost"]


def predict_image(image_path, model_name="SVM"):
    """Returns (predicted class, probability of that class, probability of Canker)."""
    # 1. photo -> 42 features, as a one-row table
    features = extract_features(image_path)
    X = pd.DataFrame([features])[FEATURE_NAMES]

    # 2. load the saved pipeline (scaler + model)
    model = joblib.load("models/" + model_name.lower().replace(" ", "_") + ".joblib")

    # 3. predict. predict_proba gives [P(Healthy), P(Canker)]
    probability_canker = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])          # 1 = Canker, 0 = Healthy

    if prediction == 1:
        return "Canker", probability_canker, probability_canker
    else:
        return "Healthy", 1 - probability_canker, probability_canker


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py photo.jpg ["Model Name" or all]')
        sys.exit()

    image_path = sys.argv[1]

    if len(sys.argv) < 3:
        chosen_models = ["SVM"]
    elif sys.argv[2] == "all":
        chosen_models = MODEL_NAMES
    else:
        chosen_models = [sys.argv[2]]

    print("Image:", image_path)
    for name in chosen_models:
        if name not in MODEL_NAMES:
            print("Unknown model:", name, "| choose one of:", MODEL_NAMES)
            continue
        label, probability, probability_canker = predict_image(image_path, name)
        print(name, "->", label, "| probability:", str(round(probability * 100, 1)) + "%",
              "| P(Canker):", str(round(probability_canker * 100, 1)) + "%")
