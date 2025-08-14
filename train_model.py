import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('processed_netflix_titles.csv')
features = ['release_year', 'duration_numeric', 'genre_count']
df_model = df[df['duration_numeric'].notna()][features + ['type']].copy()
le = LabelEncoder()
df_model['type'] = le.fit_transform(df_model['type'])
X = df_model[features]
y = df_model['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
print("Training Accuracy: {:.4f}".format(accuracy_score(y_train, y_train_pred)))
print("Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_test_pred)))
print("\nTraining Classification Report:\n", classification_report(y_train, y_train_pred, target_names=le.classes_))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
