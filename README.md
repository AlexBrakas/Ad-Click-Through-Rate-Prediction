# Ad Click-Through Rate (CTR) Prediction

## 1. Project Overview

This project aims to develop a machine learning model to predict the Click-Through Rate (CTR) of digital advertisements. The goal is to build a classifier that can accurately predict whether a user will click on an ad based on a variety of features related to the ad, the user, and the context. This is a fundamental task in computational advertising and is critical for ad ranking, revenue optimization, and user experience.

---

## 2. Dataset

* **Source:** Click-Through Rate Prediction
* **Platform:** Kaggle
* **Link:** [Dataset](https://www.kaggle.com/datasets/swekerr/click-through-rate-prediction)
* **License:** Apache License, Version 2.0
* **Description:** The dataset contains 10,000 of ad impressions, with features including user information, ad characteristics, and contextual data. The target variable is `Clicked on Ad` (1 if the ad was clicked, 0 otherwise).

---

## 3. Methodology
1.  **Exploratory Data Analysis (EDA):**
    * Analyzed feature distributions and checked for missing values.
    * Visualized relationships between key features and the `is_clicked` target variable.
    * Investigated feature cardinality and correlations.

2.  **Feature Engineering & Preprocessing:**
    * Handled missing data using appropriate imputation techniques.
    * Encoded categorical features using One-Hot Encoding.
    * Scaled numerical features using StandardScaler to prevent model bias.

3.  **Model Selection & Training:**
    * **Baseline Model:** A Logistic Regression model was used to establish a baseline performance.
    * **Primary Model:** An XGBoost Classifier (`XGBClassifier`) was chosen for its high performance on tabular data and its ability to handle large datasets.
    * The data was split into training (80%) and testing (20%) sets.

4.  **Evaluation:**
    * The model's performance was evaluated on the unseen test set using the following metrics:
        * **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** The primary metric, as it is well-suited for imbalanced classification problems like CTR prediction.
        * **Accuracy:** To provide a general sense of correctness.
        * **Classification Report:** Including precision, recall, and F1-score for a detailed view of performance.

---

## 4. Repository Structure
```bash
├── data/
│   └── ad_10000records.csv         
├── EDA.ipynb        
├── train_model.py     
├── requirements.txt
└── README.md
```
---

## 5. Setup & Usage
Python 12.8

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AlexBrakas/Ad-Click-Through-Rate-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the training script:**
    * Place your Kaggle CSV file inside a `data/` directory.
    * Update the file path in `train_model.py`.
    * Run the script from the root directory:
    ```bash
    python train_model.py
    ```

---

## 6. Results

The final XGBoost model achieved the following performance on the test set:
* **AUC-ROC Score:** 0.9407
* **Accuracy:** 0.8645

---

## 7. Future Work

* Perform extensive hyperparameter tuning using GridSearch or Optuna.
* Experiment with other gradient boosting models like LightGBM or CatBoost.
* Develop more sophisticated features from user interaction history or timestamps.