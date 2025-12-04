# Optimizing Ad Spend: Click-Through Rate (CTR) Prediction

## 📌 Context & Business Problem

In digital advertising, allocating budget efficiently is the difference between profit and loss. **Click-Through Rate (CTR)** is the critical metric that determines ad ranking and cost-per-click.

This project implements a production-ready **XGBoost Classifier** to predict the probability of a user clicking an ad. By accurately identifying high-intent impressions, this model enables:

  * **Budget Optimization:** Bidding higher only on high-probability clicks.
  * **Audience Segmentation:** Understanding which demographics (Age, Income, Internet Usage) drive conversion.

## 🚀 Key Results

The model was evaluated on a hold-out test set (20% split) and achieved high discrimination capability:

| Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.9407** | Excellent ability to rank high-intent users vs. non-clickers. |
| **Accuracy** | **86.45%** | Correctly predicts user action in \>8 out of 10 cases. |

-----

## 🛠️ Technical Implementation

### 1\. The Pipeline (`train_model.py`)

I implemented a robust `sklearn.pipeline.Pipeline` to prevent data leakage and ensure reproducibility in production:

```python
# Modular preprocessing for heterogeneous data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Gradient Boosting with XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'))
])
```

### 2\. Exploratory Analysis (`EDA.ipynb`)

Key insights derived from the data:

  * **Income vs. Clicks:** Lower-income segments showed a surprisingly higher click propensity.
  * **Time Sensitivity:** Ad engagement peaks during late-night hours, suggesting potential for day-parting strategies.
  * **Usage Correlation:** Users with *lower* daily internet usage were more likely to click (less "ad blindness").

## 📂 Repository Structure

  * **`train_model.py`**: The main driver for training, evaluation, and serialization.
  * **`EDA.ipynb`**: Detailed data investigation and feature validation.
  * **`data/`**: Contains the dataset (`ad_10000records.csv`).
  * **`ctr_model.joblib`**: The serialized production model (generated after training).

## 💻 Setup & Usage

To reproduce the environment and results:

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    This script handles loading, preprocessing, training, and evaluation automatically.

    ```bash
    python train_model.py
    ```

3.  **Output:**
    The script will output the classification report and save the trained artifact:

    ```text
    --- Model Evaluation Results ---
    Accuracy: 0.8645
    AUC-ROC Score: 0.9407
    Saving model to ctr_model.joblib...
    ```

## 🧠 Future Improvements

  * **Hyperparameter Tuning:** Implementing `GridSearchCV` or `Optuna` to fine-tune XGBoost parameters (learning rate, tree depth).
  * **Feature Engineering:** Extracting keywords from `Ad Topic Line` using TF-IDF or Word2Vec to capture semantic meaning.
  * **Serving:** Wrapping the model in a FastAPI endpoint for real-time inference.
