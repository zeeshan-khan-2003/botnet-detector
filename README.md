# Botnet Detector

A Python-based system that detects bot traffic on web applications using machine learning. This project includes a traffic simulator to generate live traffic, a trained ML model for bot detection, and a FastAPI-based prediction API for real-time monitoring.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Traffic Simulator](#traffic-simulator)
- [Machine Learning Model](#machine-learning-model)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project is designed to detect bot traffic on websites by analyzing web logs. It uses a supervised ML approach (Random Forest classifier) trained on labeled web log datasets. The system can simulate realistic traffic patterns to test and improve detection performance.

---

## Features
- **Bot Detection:** Classifies web traffic into human or bot.
- **Traffic Simulator:** Generates live traffic based on real log patterns.
- **FastAPI Integration:** Real-time prediction API.
- **Retraining Pipeline:** Automatically feeds simulated traffic to improve the ML model.
- **Visualization Ready:** Output can be used in dashboards.

---

## Project Structure

- **backend/**
  - **src/** – FastAPI backend code
    - `main.py` – API endpoints
- **frontend/** – Optional dashboard/frontend
  - **src/**
- **ml/**
  - **dataset/** – Original & parsed datasets
  - **model/** – Trained ML model (.pkl)
  - **retrain_data/** – Live logs for retraining
  - **src/** – ML scripts
    - `feature_engineering.py`
    - `live_predictor.py`
    - `parse_logs.py`
    - `train_model.py`
- **traffic-simulator/**
  - `simulator.py` – Generates live traffic
---

## Setup & Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/botnet-detector.git
cd botnet-detector
Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

pip install -r requirements.txt


Ensure datasets and trained models are in place:

ml/dataset/parsed_labeled_logs.csv
ml/model/bot_detector.pkl

Usage
1. Run the FastAPI server
uvicorn backend.src.main:app --reload


API endpoint for prediction: POST /predict

Example request payload:

{
  "ip": "192.168.1.1",
  "user_agent": "Mozilla/5.0",
  "timestamp": "2025-09-02T22:00:00",
  "url": "/login"
}


Response:

{
  "prediction": "bot"
}

2. Run the Traffic Simulator
python traffic-simulator/simulator.py


Generates synthetic traffic for both humans and bots.

Sends requests to FastAPI /predict endpoint.

Optionally saves simulated logs for retraining.

3. Retrain ML Model
python ml/src/train_model.py


Uses ml/retrain_data/live_logs.csv combined with original dataset.

Supports stratified batch generation for balanced training.

Saves updated model to ml/model/bot_detector.pkl.

Contributing

Fork the repo

Create a branch (feature/new-feature)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature/new-feature)

Create a Pull Request
---
License

MIT License © 2025 Zeeshan Khan
---