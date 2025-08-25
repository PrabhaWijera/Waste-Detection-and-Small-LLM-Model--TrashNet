import os
import shutil
import torch
from app.services.train_model import train_model

NEW_DATA_DIR = "data/new_waste"
MODEL_PATH = "models/latest_model.pth"
TRAIN_DATA_DIR = "data/archive/processed_dataset/train"
MIN_SAMPLES = 1

def count_new_samples():
    total = 0
    for root, _, files in os.walk(NEW_DATA_DIR):
        total += len([f for f in files if f.endswith((".jpg", ".png"))])
    return total

def archive_new_samples():
    """
    Move all new samples to the main training dataset
    and preserve their class folder structure if present.
    """
    if not os.path.exists(TRAIN_DATA_DIR):
        os.makedirs(TRAIN_DATA_DIR)

    for root, dirs, files in os.walk(NEW_DATA_DIR):
        for file in files:
            if file.endswith((".jpg", ".png")):
                class_folder = os.path.basename(root)
                dest_dir = os.path.join(TRAIN_DATA_DIR, class_folder)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(os.path.join(root, file), os.path.join(dest_dir, file))

def check_and_trigger_retrain():
    """
    Called after each new submission.
    If enough samples exist, retrains model.
    """
    sample_count = count_new_samples()
    if sample_count >= MIN_SAMPLES:
        print(f"🔄 Retraining model with {sample_count} new samples + TrashNet...")
        train_model(NEW_DATA_DIR, MODEL_PATH)
        archive_new_samples()
        print("✅ Retraining complete and new samples archived")
        return "Retraining triggered successfully."

    return f"Not enough new samples yet ({sample_count}/{MIN_SAMPLES})"
