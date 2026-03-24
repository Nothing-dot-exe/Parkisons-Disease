# Parkinson's Disease Detection: How It Works

This document provides a detailed, step-by-step technical explanation of the pipeline used to detect Parkinson's disease from vocal features, as implemented in `main.py`.

---

## 1. The Dataset
The dataset utilized is the **UCI Machine Learning Repository Parkinson's Dataset**.
It consists of 195 biomedical voice measurements taken from 31 individuals, 23 of whom have been diagnosed with Parkinson's Disease (PD).

The columns in our `data.csv` consist of various voice frequency properties:
*   **MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz):** Average, maximum, and minimum vocal fundamental frequencies.
*   **MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP:** Various measures of variance in the fundamental frequency.
*   **MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA:** Measures of variance in vocal amplitude.
*   **NHR, HNR:** Measures of the ratio of noise to tonal components in the voice.
*   **status:** The target label (`1` for Parkinson's, `0` for healthy).

---

## 2. Data Preprocessing Pipeline

The preprocessing is handled in the `load_and_preprocess_data` function inside `main.py`.

### Step 1: Cleaning
The `name` column (the subject's identifier string) is dropped because it provides no statistical predictive value and cannot be fed into numerical machine learning models.

### Step 2: Prevention of Data Leakage (Train-Test Split)
Before any further mathematical transformations occur, the script immediately splits the data into two sets:
*   **Training Set (80%):** Used strictly for the machine to learn patterns.
*   **Testing Set (20%):** Locked away and completely unseen by the model until the very end.
*   *Why?* If we transform the data before splitting it, information from the testing set "leaks" into the training phase, artificially inflating the model's accuracy.

### Step 3: SMOTE (Class Balancing)
The dataset is imbalanced (there are far more PD patients than healthy ones). If trained raw, models would be biased to always guess "PD".
*   We use **SMOTE (Synthetic Minority Over-sampling Technique)**.
*   SMOTE creates synthetic, biologically plausible data points for the minority class (healthy patients) by drawing vectors between existing data points.
*   **Crucial Detail:** SMOTE is *only* applied to the training set.

### Step 4: Min-Max Scaling
Machine learning algorithms are sensitive to the absolute scale of numbers. For instance, frequency `MDVP:Fo` might be around 200 Hz, while `MDVP:Jitter` is 0.005. The algorithm might incorrectly prioritize the frequency just because it's a larger number.
*   We use `MinMaxScaler` to tightly compress all features proportionally between `-1` and `1`.
*   **Crucial Detail:** The scaler calculates its limits (min and max) *only* off the training data, applying that math to both sets.

---

## 3. Modeling & Optimization

The machine learning core runs in the `train_and_evaluate_models` function.

We deploy several advanced classifiers:
*   **Decision Trees & Random Forests**: Rule-based logic algorithms.
*   **Support Vector Machines (SVM)**: Finds the mathematical hyperplane that perfectly divides healthy vs. sick points in complex dimensional space.
*   **XGBoost:** A powerful gradient-boosting neural tree algorithm known to excel in competitions.
*   **Naive Bayes, KNN, Logistic Regression**: Used as strong baseline models.

### Hyperparameter Tuning via GridSearchCV
Machine learning models have "settings" (hyperparameters). E.g., How deep should a Random Forest grow? What is the penalty threshold for SVMs?
*   Instead of randomly guessing, the script uses `GridSearchCV`.
*   It automatically tries thousands of combinations of settings.
*   **Parallel Execution (`n_jobs=-1`):** It runs these tests simultaneously across all processing cores on your CPU, drastically cutting down training time.
*   **Cross-Validation (`cv=3`):** It splits the training data into 3 chunks internally, tests on itself repeatedly, and promotes only the mathematical settings that achieve the highest overall `F1-score` across all chunks.

---

## 4. Evaluation and Output

Once the "perfect" settings are found for a model, it takes a final exam using the unseen **Testing Set** (from Step 2).

It evaluates its results based on rigorous clinical metrics:
*   **Accuracy:** Total predictions perfectly correct.
*   **Recall (Sensitivity):** Out of all people who genuinely have PD, how many did the model catch? (Highly critical in medicine).
*   **Precision:** Out of all the people the model claimed had PD, how many actually did?
*   **F1-Score:** The harmonized average of Precision and Recall. It is the gold standard for judging medical models over simple Accuracy.

### Final Outputs:
1.  **Metric Report (`model_metrics_comparison.csv`):** The script saves all metric performances into this file so you can open it in Excel and easily compare model rankings.
2.  **Model Storage (`ML_Models` folder):** Every fully trained, optimized model is serialized and permanently saved as a `.pkl` (pickle) file using the `joblib` library. In a production environment, you would load these files instantly to predict a new patient's voice clip without needing to retrain it!
