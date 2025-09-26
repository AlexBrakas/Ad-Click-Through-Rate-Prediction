import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def main():
    file_path = 'data/ad_10000records.csv' 
    df = load_data(file_path)

    target_column = 'Clicked on Ad' 
    categorical_features = ['Ad Topic Line', 'City', 'Country', 'Gender']
    numerical_features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']
    
    df = df.drop(columns=['Timestamp'])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print("Setting up preprocessing pipeline...")
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
                               
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    model_filename = 'ctr_model.joblib'
    print(f"Saving the trained model to {model_filename}...")
    joblib.dump(pipeline, model_filename)
    
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()