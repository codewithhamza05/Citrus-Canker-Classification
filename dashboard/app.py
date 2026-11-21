# app.py  (the dashboard)
# ------------------------------------------------------------------
#   streamlit run dashboard/app.py
#
# The dashboard only SHOWS files that the scripts in src/ already made.
# The one thing it calculates live is the prediction for an uploaded photo.
#
# The 8 required functions and where they are:
#   1. select / load dataset        -> tab "Dataset"
#   2. select trained model         -> dropdown in the sidebar
#   3. run processing / prediction  -> button "Run prediction"
#   4. view evaluation results      -> tab "Evaluation"
#   5. view visualizations          -> tab "Visualizations"
#   6. view model comparison        -> tab "Model comparison"
#   7. upload a new photo           -> tab "Predict a new photo"
#   8. show predicted class         -> Canker / Healthy + probability
# ------------------------------------------------------------------
import json
import os
import sys

import cv2
import pandas as pd
import streamlit as st

sys.path.append("src")                   # so we can import our own files from src/
from features import load_image, remove_background
from predict import MODEL_NAMES, predict_image

st.set_page_config(page_title="Citrus Canker Classifier", layout="wide")
st.title("Citrus Canker Classification")
st.caption("Machine Learning, Project B: Canker vs Healthy from image features")


def show_plot(path, explanation):
    """Show a saved plot with a one-line explanation under it."""
    st.image(path, width="stretch")
    st.caption(explanation)


# ------------------------------------------------------------------
# Load the files made by the scripts
# ------------------------------------------------------------------
features_table = pd.read_csv("data/features.csv")
metadata_table = pd.read_csv("data/metadata.csv")
split_table = pd.read_csv("data/split.csv")
comparison_table = pd.read_csv("results/model_comparison.csv")

with open("results/metrics.json") as f:
    metrics = json.load(f)
with open("results/unsupervised.json") as f:
    unsupervised = json.load(f)

# ------------------------------------------------------------------
# Sidebar: select the trained model (function 2)
# ------------------------------------------------------------------
st.sidebar.header("Trained model")
model_name = st.sidebar.selectbox("Select model", MODEL_NAMES, index=MODEL_NAMES.index("SVM"))
model_key = model_name.lower().replace(" ", "_")        # e.g. "Random Forest" -> "random_forest"
model_metrics = metrics[model_name]

st.sidebar.write("Best settings found by GridSearchCV:")
st.sidebar.json(model_metrics["best_params"])
st.sidebar.caption("This model is used in the Evaluation tab and for predicting a new photo.")

tab_data, tab_eval, tab_compare, tab_plots, tab_predict = st.tabs(
    ["Dataset", "Evaluation", "Model comparison", "Visualizations", "Predict a new photo"])

# ------------------------------------------------------------------
# TAB 1: Dataset (function 1)
# ------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset")
    choice = st.selectbox("Select / load dataset", ["Features", "Metadata", "Train/test split"])
    if choice == "Features":
        table = features_table
    elif choice == "Metadata":
        table = metadata_table
    else:
        table = split_table

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images", len(features_table))
    col2.metric("Canker", int((features_table["label"] == "Canker").sum()))
    col3.metric("Healthy", int((features_table["label"] == "Healthy").sum()))
    col4.metric("Physical samples (fruits)", features_table["sample_id"].nunique())

    st.dataframe(table, width="stretch", height=350)
    st.caption("The selected table. image_id links the three files; images of the same fruit share a sample_id.")

    st.write("**Class counts in train and test**")
    st.dataframe(pd.crosstab(split_table["label"], split_table["split"]))
    st.caption("Number of images of each class in the training set and in the test set.")

    with st.expander("Data cleaning report"):
        with open("results/cleaning_report.txt") as f:
            st.text(f.read())

# ------------------------------------------------------------------
# TAB 2: Evaluation of the selected model (function 4)
# ------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation: " + model_name)
    st.caption("Results on the held-out test set. The positive class is Canker.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", str(round(model_metrics["accuracy"] * 100, 1)) + "%")
    col2.metric("Precision", str(round(model_metrics["precision"] * 100, 1)) + "%")
    col3.metric("Recall", str(round(model_metrics["recall"] * 100, 1)) + "%")
    col4.metric("F1", str(round(model_metrics["f1"] * 100, 1)) + "%")
    col5.metric("CV F1 (mean ± std)",
                str(model_metrics["cv_f1_mean"]) + " ± " + str(model_metrics["cv_f1_std"]))

    left, right = st.columns(2)
    with left:
        show_plot("results/cm_" + model_key + ".png",
                  "Confusion matrix: rows are the true class, columns are the predicted class.")
    with right:
        st.write("**Classification report**")
        st.text(model_metrics["classification_report"])
        st.caption("Precision, recall and F1 for each class; support = number of test images.")

        importance_plot = "results/feature_importance_" + model_key + ".png"
        if os.path.exists(importance_plot):
            show_plot(importance_plot, "The features this model relies on most.")

# ------------------------------------------------------------------
# TAB 3: Model comparison (function 6)
# ------------------------------------------------------------------
with tab_compare:
    st.subheader("Model comparison: all 5 models")
    st.dataframe(comparison_table, width="stretch", hide_index=True)
    st.caption("Test-set scores of every model, next to its cross-validation F1 (mean and std over 5 folds).")

    left, right = st.columns(2)
    with left:
        show_plot("results/model_comparison.png",
                  "Accuracy, precision, recall and F1 of every model on the same test set.")
    with right:
        show_plot("results/cv_comparison.png",
                  "Cross-validation F1: the dot is the mean, the line is ± 1 standard deviation. "
                  "Overlapping lines mean the models are not clearly different.")
    st.info("The test set is small, so one test score is noisy. "
            "The cross-validation mean ± std is the more reliable comparison.")

# ------------------------------------------------------------------
# TAB 4: Visualizations: EDA, PCA, K-Means (function 5)
# ------------------------------------------------------------------
with tab_plots:
    st.subheader("Visualizations")
    section = st.radio("Section", ["EDA", "PCA", "K-Means"], horizontal=True)

    if section == "EDA":
        show_plot("results/eda/class_balance.png",
                  "Class balance: number of images and of different fruits in each class.")
        show_plot("results/eda/sample_images.png",
                  "Example photos from both classes.")
        show_plot("results/eda/segmentation_examples.png",
                  "Background removal: the features are measured only on the fruit shown in the bottom row.")
        show_plot("results/eda/feature_distributions.png",
                  "Histograms of the 8 features most related to the label. The less the two colours overlap, the more useful the feature.")
        show_plot("results/eda/feature_boxplots.png",
                  "The same 8 features as boxplots: the box is the middle 50% of images, the line is the median.")
        show_plot("results/eda/correlation_heatmap.png",
                  "Correlation between every pair of features. Strong red or blue blocks are features that carry similar information.")
        show_plot("results/eda/feature_target_correlation.png",
                  "Correlation of each feature with the label: orange is higher in Canker, blue is higher in Healthy.")
        st.write("**Summary statistics**")
        st.dataframe(pd.read_csv("results/eda/summary_statistics.csv"), width="stretch", hide_index=True)
        st.caption("Count, mean, standard deviation, minimum, quartiles and maximum of every feature.")

    elif section == "PCA":
        st.info("PCA is dimensionality reduction. It summarises the features and never sees the labels. "
                "The labels are only used to colour the scatter plot.")
        col1, col2, col3 = st.columns(3)
        col1.metric("PC1", str(unsupervised["pca_pc1_percent"]) + "%")
        col2.metric("PC1 + PC2", str(unsupervised["pca_pc1_plus_pc2_percent"]) + "%")
        col3.metric("Components for 90%", unsupervised["pca_components_for_90_percent"])
        show_plot("results/pca_scree.png",
                  "Scree plot: how much of the variance each component explains, and the running total.")
        show_plot("results/pca_scatter.png",
                  "Every image placed by its first two components and coloured by its true label.")
        show_plot("results/pca_loadings.png",
                  "Loadings: the original features that contribute most to PC1 and PC2.")

    else:
        st.info("K-Means is unsupervised clustering. It groups similar images without ever seeing the labels. "
                "It is NOT a classifier. The comparison with the true labels is done afterwards, for discussion only.")
        col1, col2 = st.columns(2)
        col1.metric("Adjusted Rand Index (k = 2)", unsupervised["kmeans_adjusted_rand_index"])
        col2.metric("Silhouette score (k = 2)", unsupervised["kmeans_silhouette_k2"])
        show_plot("results/kmeans_vs_actual.png",
                  "Left: the two clusters K-Means found. Right: the true labels. If they looked the same, the clusters would match the classes.")
        show_plot("results/kmeans_elbow_silhouette.png",
                  "Elbow (inertia) and silhouette score for k = 2 to 6, used to judge how many clusters fit the data.")
        st.write("**Clusters compared with the true labels**")
        st.dataframe(pd.read_csv("results/kmeans_contingency.csv", index_col=0))
        st.caption("Contingency table: how many images of each true class landed in each cluster. "
                   "ARI = 1 means a perfect match, ARI near 0 means no better than chance.")

# ------------------------------------------------------------------
# TAB 5: Upload a photo, run the prediction, show the class (functions 7, 3, 8)
# ------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict a new citrus photo")
    st.write("Model in use: **" + model_name + "** (change it in the sidebar)")

    uploaded_file = st.file_uploader("Upload a citrus fruit photo", type=["jpg", "jpeg", "png"])
    run_clicked = st.button("Run prediction")

    if run_clicked and uploaded_file is None:
        st.warning("Please upload a photo first.")

    if run_clicked and uploaded_file is not None:
        # save the upload to a temporary file so OpenCV can read it
        temp_path = "uploaded_photo_temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        label, probability, probability_canker = predict_image(temp_path, model_name)

        image = load_image(temp_path)
        mask = remove_background(image)
        fruit_only = image.copy()
        fruit_only[mask == 0] = 255          # paint the background white

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption("The uploaded photo, resized to 512 x 512.")
        with col2:
            st.image(cv2.cvtColor(fruit_only, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption("Background removed. The 42 features are measured on this part only.")
        with col3:
            percent = str(round(probability * 100, 1)) + "%"
            if label == "Canker":
                st.error("Predicted class: Canker  \nProbability: " + percent)
            else:
                st.success("Predicted class: Healthy  \nProbability: " + percent)
            st.caption("P(Canker) = " + str(round(probability_canker * 100, 1)) + "% according to " + model_name + ".")

            # what the other models say about the same photo
            rows = []
            for name in MODEL_NAMES:
                other_label, _, other_p_canker = predict_image(temp_path, name)
                rows.append({"Model": name, "Prediction": other_label,
                             "P(Canker)": str(round(other_p_canker * 100, 1)) + "%"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("The prediction of all 5 trained models for this photo.")

        os.remove(temp_path)
