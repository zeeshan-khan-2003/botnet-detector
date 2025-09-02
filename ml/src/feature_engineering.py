import pandas as pd
import os

DATA_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/dataset/semantic_features.csv"
OUTPUT_PATH = "/Users/zeeshankhan/Desktop/botnet-detector/ml/dataset/processed_features.csv"

def load_and_process_features(path=DATA_PATH, save_path=OUTPUT_PATH):
    # Load dataset
    df = pd.read_csv(path)

    # Drop ID column if it exists
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Ensure label column exists
    if 'ROBOT' not in df.columns:
        raise ValueError("Label column 'ROBOT' not found in dataset!")

    # Split features and labels
    X = df.drop(columns=['ROBOT'])
    y = df['ROBOT']

    # Save processed features
    processed = X.copy()
    processed['ROBOT'] = y  # keep target column
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed.to_csv(save_path, index=False)

    return X, y


if __name__ == "__main__":
    X, y = load_and_process_features()
    print("✅ Features and labels processed & saved!")
    print("Features shape:", X.shape)
    print("Labels distribution:\n", y.value_counts())
    print(f"Processed file saved at: {OUTPUT_PATH}")
