import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
# Load dataset
df = pd.read_csv("/Users/zeeshankhan/Desktop/botnet-detector/ml/dataset/semantic_features.csv")

# Features aur target
X = df.drop(columns=["ID", "ROBOT"])
y = df["ROBOT"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
classes = np.array([0, 1])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# Model with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Model ko Save krraha hu

joblib.dump(model, "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/bot_detector.pkl")
print("Random Forest model saved successfully!")