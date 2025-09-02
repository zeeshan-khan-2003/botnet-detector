import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import random

# -------------------------
# Absolute Paths
# -------------------------
DATASET_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/dataset/semantic_features.csv"
RETRAIN_CSV = "/Users/zeeshankhan/Desktop/botnet-detector/ml/retrain_data/live_logs.csv"
RF_MODEL_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/bot_detector.pkl"
XGB_MODEL_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/xgboost_model.pkl"

# Ensure retrain_data folder exists
Path("/Users/zeeshankhan/Desktop/botnet-detector/ml/retrain_data").mkdir(parents=True, exist_ok=True)

# -------------------------
# Load original dataset & split
# -------------------------
df_original = pd.read_csv(DATASET_PATH)
X = df_original.drop(columns=["ID","ROBOT"])
y = df_original["ROBOT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

FEATURES = X_train.columns

# -------------------------
# Parameters
# -------------------------
BATCH_SIZE = 512

# -------------------------
# Helper functions
# -------------------------
def load_models():
    with open(RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    return rf_model, xgb_model

def save_models(rf_model, xgb_model):
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)

def retrain_models(df_batch):
    # Prepare features & labels
    X_batch = df_batch.drop(columns=["ROBOT", "ID"], errors='ignore')
    y_batch = df_batch["ROBOT"]

    # Class weights
    classes = np.array([0,1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_batch)
    class_weights = {0: weights[0], 1: weights[1]}

    # Retrain models
    rf_model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
    rf_model.fit(X_batch, y_batch)

    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_batch, y_batch)

    # Save models
    save_models(rf_model, xgb_model)

    # -------------------------
    # Evaluate on original test set
    # -------------------------
    print("\n=== Retrain Metrics on ORIGINAL TEST SET ===")
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    print("Random Forest:")
    print(classification_report(y_test, y_pred_rf))
    print("XGBoost:")
    print(classification_report(y_test, y_pred_xgb))

# -------------------------
# Continuous collection & retraining
# -------------------------
def run_continuous_retrain():
    buffer = []
    while True:
        new_row = {}
        for col in FEATURES:
            if df_original[col].dtype in [np.float64, np.int64]:
                new_row[col] = np.random.normal(df_original[col].mean(), df_original[col].std())
            else:
                probs = df_original[col].value_counts(normalize=True)
                new_row[col] = np.random.choice(probs.index, p=probs.values)

        new_row["ROBOT"] = np.random.choice([0,1], p=[0.8,0.2])
        new_row["ID"] = len(buffer) + 1

        buffer.append(new_row)

        # Save batch to CSV
        df_buffer = pd.DataFrame(buffer)
        if Path(RETRAIN_CSV).exists():
            df_buffer.to_csv(RETRAIN_CSV, mode='a', header=False, index=False)
        else:
            df_buffer.to_csv(RETRAIN_CSV, index=False)

        # Retrain if batch full
        if len(buffer) >= BATCH_SIZE:
            print(f"\nBatch reached ({BATCH_SIZE} rows). Retraining models...")
            retrain_models(pd.DataFrame(buffer))
            buffer = []

        time.sleep(random.uniform(0.5,1.5)) # random time pe generate karega

if __name__ == "__main__":
    run_continuous_retrain()
