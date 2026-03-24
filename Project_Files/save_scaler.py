import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

df = pd.read_csv('c:/Users/kadam/OneDrive/Parkinson-Disease-Detection/Project_Files/data.csv')
if 'name' in df.columns: df.drop(['name'], axis=1, inplace=True)
X = df.drop(['status'], axis=1)
y = df['status'].astype('uint8')
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=20)
sm = SMOTE(random_state=300)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
scaler = MinMaxScaler((-1, 1))
scaler.fit(X_train_res)
os.makedirs('c:/Users/kadam/OneDrive/Parkinson-Disease-Detection/Project_Files/ML_Models', exist_ok=True)
joblib.dump(scaler, 'c:/Users/kadam/OneDrive/Parkinson-Disease-Detection/Project_Files/ML_Models/scaler.pkl')
print("Scaler saved successfully!")
