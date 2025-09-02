import pandas as pd
import numpy as np
import requests
import time
import random
# -------------------------
# 1️⃣ Load dataset to mimic patterns
# -------------------------
df = pd.read_csv("ml/dataset/semantic_features.csv")
X = df.drop(columns=["ID", "ROBOT"])

# -------------------------
# 2️⃣ Analyze feature distributions
# -------------------------
feature_stats = {}
for col in X.columns:
    if X[col].dtype in [np.float64, np.int64]:
        # For numeric features, use mean & std
        feature_stats[col] = (X[col].mean(), X[col].std())
    else:
        # For categorical features, use value probabilities
        feature_stats[col] = X[col].value_counts(normalize=True).to_dict()

# -------------------------
# 3️⃣ Simulator loop (generate traffic)
# -------------------------
while True:
    row = {}
    for col, stats in feature_stats.items():
        if isinstance(stats, tuple):  # numeric
            row[col] = np.random.normal(stats[0], stats[1])
        else:  # categorical
            row[col] = np.random.choice(list(stats.keys()), p=list(stats.values()))
    
    df_row = pd.DataFrame([row])
    
    # -------------------------
    # 4️⃣ Send to FastAPI
    # -------------------------
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=df_row.to_dict(orient="records"))
        print("Prediction:", response.json())
    except Exception as e:
        print("Error sending request:", e)
    
    # -------------------------
    # 5️⃣ Wait before next row
    # -------------------------
    time.sleep(random.uniform(0.01,0.3))  # random seconds generate karrahe hai from this range
