import os
import threading

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "new")  # adjust if needed
MIN_SAMPLES = 10  # Minimum new samples required to trigger retraining


# === Utility Functions ===
def count_new_samples():
    """
    Count how many new samples exist in NEW_DATA_DIR.
    Assumes NEW_DATA_DIR has subfolders (like dataset classes).
    """
    if not os.path.exists(NEW_DATA_DIR):
        return 0

    count = 0
    for root, _, files in os.walk(NEW_DATA_DIR):
        count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return count


# === Retraining Logic ===
def run_retrain():
    """
    Run the retraining pipeline.
    Replace this placeholder with actual YOLO / ML training logic.
    """
    print("🔄 Starting retraining...")
    samples = count_new_samples()
    print(f"📊 Found {samples} new samples in {NEW_DATA_DIR}")

    if samples < MIN_SAMPLES:
        print(f"❌ Not enough samples. Need at least {MIN_SAMPLES}.")
        return

    # TODO: Add your YOLOv8 / ML retraining code here
    print("✅ Retraining completed (placeholder).")


# === Async Wrapper ===
def start_retrain_async():
    """
    Start retraining in background so API is non-blocking.
    """
    t = threading.Thread(target=run_retrain, daemon=True)
    t.start()
    return "Retraining started in background."
