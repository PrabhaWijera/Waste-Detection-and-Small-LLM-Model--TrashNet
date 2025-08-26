# app/services/training_manager.py
import os
import subprocess
import threading
from pathlib import Path

# Directory where new data is stored
NEW_DATA_DIR = "data/new_waste"
MIN_SAMPLES = 1  # retrain after every 5 submissions

def count_new_samples():
    """Count total new images submitted by public users."""
    total = 0
    for root, _, files in os.walk(NEW_DATA_DIR):
        total += len([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    return total

def run_retrain():
    """Runs retrain.py script as subprocess."""
    try:
        cmd = [
            "python", "retrain.py",
            "--data", "data/archive/processed_dataset",   # ✅ adjust if needed
            "--base", "yolov8n-cls.pt",
            "--out-model", "models/new_model.pt"
        ]
        subprocess.run(cmd, check=True)
        print("Retraining complete ✅")
    except Exception as e:
        print(f"Retraining failed ❌: {e}")

def start_retrain_async():
    """Start retraining in background thread so API is non-blocking."""
    t = threading.Thread(target=run_retrain, daemon=True)
    t.start()
    return "Retraining started in background."
