import os
import requests
import pandas as pd
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, r2_score
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

URL_STRING = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'

def download_data_if_needed(file_path='data.csv'):
    if not os.path.exists(file_path):
        print("Downloading dataset...")
        url_content = requests.get(URL_STRING).content
        with open(file_path, 'wb') as data_file:
            data_file.write(url_content)

def load_and_preprocess_data(file_path='data.csv'):
    df = pd.read_csv(file_path)
    
    # Drop name column that provides no useful predictive information
    if 'name' in df.columns:
        df.drop(['name'], axis=1, inplace=True)
    
    df['status'] = df['status'].astype('uint8')
    
    X = df.drop(['status'], axis=1)
    y = df['status']
    
    # Train-test split FIRST to avoid data leakage (Critical ML improvement)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)
    
    print(f"Original Train Shape: {X_train.shape}, Target True Count: {sum(y_train)}")
    
    # SMOTE only on training data
    sm = SMOTE(random_state=300)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Resampled Train Shape: {X_train_res.shape}, Target True Count: {sum(y_train_res)}")
    
    # Scaling using training data parameters
    scaler = MinMaxScaler((-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_res)
    # Apply transformation to the test dataset (DO NOT fit test dataset)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_res, y_test

def get_models():
    # Defining models and their parameters for GridSearchCV
    models = {
        'Decision Tree': (DecisionTreeClassifier(random_state=120), {
            'max_features': ['sqrt', 'log2'],
            'max_depth': range(1, 10),
            'criterion': ['gini', 'entropy']
        }),
        'Random Forest': (RandomForestClassifier(random_state=200), {
            'n_estimators': [100, 150, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': range(4, 9),
            'criterion': ['gini', 'entropy']
        }),
        'Logistic Regression': (LogisticRegression(max_iter=1000), {}),
        'SVM': (svm.SVC(probability=True), {
            'kernel': ['linear', 'rbf'],
            'C': [0.5, 1, 10]
        }),
        'Naive Bayes': (GaussianNB(), {}),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': range(2, 10)
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'max_depth': range(4, 8),
            'eta': [0.1, 0.3],
            'random_state': [300]
        })
    }
    return models

def train_and_evaluate(X_train, y_train, X_test, y_test):
    os.makedirs('ML_Models', exist_ok=True)
    metrics_data = []
    
    models = get_models()
    for name, (model, params) in models.items():
        print(f"\nTraining {name}...")
        if params:
            # Parallel processing (n_jobs=-1) dramatically boosts grid search training
            grid = GridSearchCV(model, param_grid=params, scoring='f1', cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"  Best Params: {grid.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Save model
        filename = f"ML_Models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(best_model, filename)
        
        # Predict on Test Set
        y_pred = best_model.predict(X_test)
        
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        metrics_data.append({
            'Model': name,
            'Accuracy': acc,
            'F1-Score': f1,
            'Recall': rec,
            'Precision': prec,
            'R2-Score': r2
        })
    
    chart_df = pd.DataFrame(metrics_data)
    # Sorting by F1-Score (The primary measure for imbalanced targets)
    chart_df = chart_df.sort_values('F1-Score', ascending=False)
    chart_df.to_csv("model_metrics_comparison.csv", index=False)
    print("\nMetrics exported successfully to 'model_metrics_comparison.csv'")
    return chart_df

if __name__ == '__main__':
    download_data_if_needed()
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    metrics_df = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print("\n[Summary of Top 3 Models]")
    print(metrics_df.head(3).to_string(index=False))
