flowchart TD
    A[Original Dataset CSV<br>(semantic_features.csv)] --> B[Train/Test Split<br>80% train, 20% test]
    B --> C[Initial Model Training<br>Random Forest & XGBoost]
    C --> D[Save Models to Disk<br>(Pickle files)]

    E[Live Traffic Simulator<br>- Stratified batch<br>- Random features<br>- 20% bots, 80% human] --> F[Batch Buffer<br>Collect BATCH_SIZE rows]
    F --> G[Retraining Trigger<br>- Class weights applied<br>- Stratified batch]
    G --> H[Retrain Models<br>RF + XGB<br>Evaluate on Original Test Set]
    H --> D

    D --> I[Continuous Loop<br>New simulated traffic appended to batch]
    I --> E
