import pandas as pd
import os
import re

# Paths
RAW_DATA_PATH = os.path.join("ml", "dataset", "weblog.csv")
OUTPUT_PATH = os.path.join("ml", "dataset", "parsed_labeled_logs.csv")

def label_traffic(user_agent: str) -> str:
    """Label request as bot or human based on User-Agent string."""
    if pd.isna(user_agent):
        return "human"
    ua = user_agent.lower()
    # Common bot indicators
    if any(bot_word in ua for bot_word in ["bot", "crawler", "spider", "crawl", "slurp", "bingpreview"]):
        return "bot"
    return "human"

def main():
    print("[INFO] Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Ensure expected columns
    expected_cols = ["ip", "timestamp", "method", "url", "status", "user_agent"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Label bot/human
    print("[INFO] Labeling traffic...")
    df["label"] = df["user_agent"].apply(label_traffic)

    # Save parsed & labeled logs
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Parsed and labeled logs saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
