import pandas as pd
import numpy as np
import requests
import time
import random
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# Paths
# -------------------------
DATASET_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/dataset/semantic_features.csv"
MASTER_RF_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/bot_detector.pkl"
MASTER_XGB_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/xgboost_model.pkl"
LIVE_RF_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/bot_detector_live.pkl"
LIVE_XGB_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/model/xgboost_model_live.pkl"

# -------------------------
# Load dataset & feature stats
# -------------------------
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=["ID", "ROBOT"])
feature_stats = {}

for col in X.columns:
    if X[col].dtype in [np.float64, np.int64]:
        feature_stats[col] = (X[col].mean(), X[col].std())
    else:
        feature_stats[col] = X[col].value_counts(normalize=True).to_dict()

# -------------------------
# Model loader with auto-init
# -------------------------
def load_model(rf_path, xgb_path):
    Path(rf_path).parent.mkdir(parents=True, exist_ok=True)

    # Random Forest
    if Path(rf_path).exists():
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    else:
        rf_model = RandomForestClassifier(
            n_estimators=100, warm_start=True, class_weight='balanced', random_state=42
        )
        with open(rf_path, "wb") as f: pickle.dump(rf_model, f)
        print(f"Live RF model initialized and saved at {rf_path}")

    # XGBoost
    if Path(xgb_path).exists():
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
    else:
        xgb_model = XGBClassifier(
            n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42
        )
        with open(xgb_path, "wb") as f: pickle.dump(xgb_model, f)
        print(f"Live XGB model initialized and saved at {xgb_path}")

    return rf_model, xgb_model

# -------------------------
# Traffic simulation & API
# -------------------------
def run_simulation(api_url="http://127.0.0.1:8000/predict", model_type="live"):
    if model_type == "live":
        rf_path, xgb_path = LIVE_RF_PATH, LIVE_XGB_PATH
    else:
        rf_path, xgb_path = MASTER_RF_PATH, MASTER_XGB_PATH

    rf_model, xgb_model = load_model(rf_path, xgb_path)
    print(f"Using {model_type} model for prediction.")

    while True:
        row = {}
        for col, stats in feature_stats.items():
            if isinstance(stats, tuple):  # numeric
                row[col] = np.random.normal(stats[0], stats[1])
            else:  # categorical
                row[col] = np.random.choice(list(stats.keys()), p=list(stats.values()))

        df_row = pd.DataFrame([row])

        # Send to API
        try:
            response = requests.post(api_url, json=df_row.to_dict(orient="records"))
            print(f"Prediction ({model_type} model):", response.json())
        except Exception as e:
            print("Error sending request:", e)

        time.sleep(random.uniform(0.01, 0.3))

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Choose "live" or "master"
    run_simulation(model_type="live")
