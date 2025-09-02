import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import numpy as np

# Load dataset
df = pd.read_csv("/Users/zeeshankhan/Desktop/botnet-detector/ml/dataset/semantic_features.csv")

# Features aur target
X = df.drop(columns=["ID", "ROBOT"])
y = df["ROBOT"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Optionally, save the model
import joblib
joblib.dump(model, "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/xgboost_model.pkl")
