# Citrus Canker Classification

An end-to-end image classification project that uses machine learning and computer vision to distinguish between **citrus fruits affected by Canker** and **Healthy citrus fruits**.

The project combines image preprocessing, background removal, handcrafted feature extraction, supervised machine learning, and unsupervised learning. It also includes model evaluation, image-based predictions, and an interactive Streamlit dashboard.

---

## Project Overview

Citrus canker is a bacterial disease that affects citrus plants and can cause visible damage to fruit and leaves. Identifying affected fruit through image analysis can help demonstrate how computer vision and machine learning can support agricultural applications.

This project develops a classification pipeline that analyzes citrus fruit images and predicts whether an image belongs to the Canker or Healthy category.

The complete workflow covers dataset organization, image processing, feature engineering, model training, evaluation, and interactive prediction.

## Key Features

- **Image Processing:** Image loading, preprocessing, and background removal.
- **Feature Engineering:** Extraction of 42 handcrafted image features.
- **Supervised Learning:** Training and comparison of five machine learning classifiers.
- **Model Evaluation:** Performance analysis using classification metrics and confusion matrices.
- **Unsupervised Learning:** PCA and K-Means clustering for exploratory analysis.
- **Image Prediction:** Classification of new citrus fruit images.
- **Interactive Dashboard:** Streamlit-based interface for exploring results and making predictions.
- **Reproducible Workflow:** Organized scripts, saved models, datasets, and generated results.

## Dataset

The project uses a cleaned dataset of **229 citrus fruit images**.

| Category | Number of Images |
|---|---:|
| Canker | 120 |
| Healthy | 109 |
| **Total** | **229** |

Additional dataset information:

- Raw images are stored in `data/raw/`.
- Image metadata is stored in `data/metadata.csv`.
- Extracted features are stored in `data/features.csv`.
- Dataset partitions are recorded in `data/split.csv`.
- Excluded images and their removal log are stored in `data/excluded/`.

The dataset contains images collected from online sources and organized for this classification task.

## Machine Learning Models

Five supervised learning algorithms are implemented and evaluated:

| Model | Approach |
|---|---|
| Logistic Regression | Linear classification |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble learning |
| Support Vector Machine (SVM) | Margin-based classification |
| XGBoost | Gradient boosting |

The models are trained using the extracted image features. Their performance is compared using evaluation metrics and visualizations.

## Methodology

The project follows an end-to-end machine learning workflow:

1. **Dataset Organization:** Organize and validate the citrus fruit images.
2. **Image Preprocessing:** Prepare images and remove backgrounds.
3. **Feature Extraction:** Extract 42 handcrafted features from each image.
4. **Exploratory Data Analysis:** Examine feature distributions, class balance, and relationships.
5. **Data Preparation:** Perform cleaning, outlier analysis, and train/test splitting.
6. **Model Training:** Train and tune five supervised learning models.
7. **Model Evaluation:** Compare model performance on the held-out test set.
8. **Unsupervised Analysis:** Apply PCA and K-Means clustering.
9. **Prediction:** Use trained models to classify new images.
10. **Dashboard:** Present project results through an interactive Streamlit application.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| OpenCV | Image processing |
| Scikit-image | Image analysis |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient boosting |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |

## Project Structure

```text
Citrus-Canker-Classification/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── excluded/
│   │   └── excluded_log.csv
│   ├── metadata.csv
│   ├── features.csv
│   └── split.csv
│
├── models/
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   └── xgboost.joblib
│
├── results/
│   ├── eda/
│   ├── classification_reports.txt
│   ├── cleaning_report.txt
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── cv_results.json
│   └── unsupervised.json
│
├── src/
│   ├── organize_dataset.py
│   ├── features.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── unsupervised.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/codewithhamza05/Citrus-Canker-Classification.git
cd Citrus-Canker-Classification
```

Replace `codewithhamza05` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run the following commands from the project root directory.

### Step 1: Feature Extraction

```bash
python src/features.py
```

Extracts image features and generates `data/features.csv`.

### Step 2: Data Preprocessing

```bash
python src/preprocess.py
```

Performs data cleaning, outlier analysis, and train/test splitting.

### Step 3: Exploratory Data Analysis

```bash
python src/eda.py
```

Generates exploratory data analysis reports and visualizations.

### Step 4: Model Training

```bash
python src/train.py
```

Trains and tunes the five supervised machine learning models.

### Step 5: Model Evaluation

```bash
python src/evaluate.py
```

Evaluates the trained models and generates comparison reports and confusion matrices.

### Step 6: Unsupervised Learning

```bash
python src/unsupervised.py
```

Performs PCA and K-Means clustering.

### Step 7: Image Prediction

```bash
python src/predict.py photo.jpg
```

Predicts whether the provided image belongs to the Canker or Healthy category using the SVM model.

To run predictions using all five models:

```bash
python src/predict.py photo.jpg all
```

### Step 8: Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

**Note:** The dashboard uses previously generated analysis and evaluation results. Running the prediction feature on a newly uploaded image performs a live prediction.

## Results and Evaluation

Model evaluation outputs are saved in the `results/` directory.

The project includes:

- Accuracy, precision, recall, and F1-score.
- Classification reports.
- Confusion matrices for all five models.
- Cross-validation comparison.
- Model performance comparison.
- Feature importance analysis.
- PCA visualizations and loadings.
- K-Means clustering analysis.
- Outlier analysis and dataset summaries.

Refer to `results/metrics.json`, `results/model_comparison.csv`, and the generated visualizations for the actual experimental results.

## Reproducibility

The project maintains a structured workflow with separate scripts for data preparation, feature extraction, model training, evaluation, and prediction.

Trained models are saved in the `models/` directory, while generated metrics and visualizations are stored in `results/`.

For reproducible experiments, use the same dataset, dependencies, and configuration used during the original training process.

## Future Improvements

Potential directions for extending this project include:

- Implementing deep learning architectures such as CNNs.
- Exploring transfer learning with pretrained image classification models.
- Expanding the dataset with more diverse citrus images.
- Improving generalization across different lighting and background conditions.
- Deploying the prediction system as a web application or API.
- Exploring real-time citrus disease detection.

## Project Scope and Limitations

This project is an experimental machine learning classification system developed using a relatively small image dataset.

Its predictions should not be treated as a substitute for professional agricultural diagnosis. Performance on new images may vary depending on image quality, lighting, background, and dataset diversity.

## Author

**Machine Learning & AI Projects**

Interested in machine learning, computer vision, and building practical AI-powered applications.

---

**If you find this project interesting, feel free to explore the code, review the results, and experiment with the classification pipeline.**
