import os
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ✅ Ensure 'model' directory exists
os.makedirs("model", exist_ok=True)

df = pd.read_csv("students_old one.csv")

df = pd.get_dummies(df, columns=["gender"], drop_first=True)

print(df.head())

x = df.drop("score", axis=1)
y = df["score"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# ✅ Save to the correct path after directory is confirmed
joblib.dump(model, "model/score_model.pkl")

print("✅ Model saved successfully to 'model/score_model.pkl'")
