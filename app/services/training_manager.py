# training_manager.py
import os
import shutil
import random
import threading
from pathlib import Path
import yaml
from ultralytics import YOLO

from app.models.waste_classifier import WasteClassifier, reload_classifier

# === Paths ===
BASE_DIR = Path(__file__).parent
NEW_DATA_DIR = BASE_DIR / ".." / "data" / "new_waste"
ARCHIVE_DIR = BASE_DIR / ".." / "data" / "archive" / "dataset-resized"
SPLIT_DIR = BASE_DIR / ".." / "data" / "archive" / "dataset_split"
OUTPUT_MODEL = BASE_DIR / ".." / "models" / "new_model.pt"
MIN_SAMPLES = 1  # retrain threshold

# Initialize classifier
classifier = WasteClassifier()

# === Utility Functions ===
def count_new_samples():
    if not NEW_DATA_DIR.exists():
        return 0
    return sum(1 for f in NEW_DATA_DIR.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"])

def merge_new_data():
    """Move new images into the main dataset by class, creating folders if needed."""
    if not NEW_DATA_DIR.exists():
        return

    for cls_dir in NEW_DATA_DIR.iterdir():
        if not cls_dir.is_dir():
            continue

        # Ensure the class folder exists in ARCHIVE_DIR
        dest_dir = ARCHIVE_DIR / cls_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move images into the class folder
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                shutil.move(str(img_file), dest_dir / img_file.name)

    # Remove NEW_DATA_DIR after merging
    shutil.rmtree(NEW_DATA_DIR)

def split_dataset(train_ratio=0.8):
    """Split dataset into train/val folders and create YAML with absolute paths"""
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    train_dir = SPLIT_DIR / "train"
    val_dir = SPLIT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted([d.name for d in ARCHIVE_DIR.iterdir() if d.is_dir()])
    for cls in class_names:
        cls_dir = ARCHIVE_DIR / cls
        images = [img for img in cls_dir.glob("*.*") if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

        if len(images) == 0:
            print(f"⚠️ Warning: No images in class {cls}")
            continue

        random.shuffle(images)

        # Compute split index
        split_idx = max(1, int(len(images) * train_ratio))
        if split_idx >= len(images):
            split_idx = len(images) - 1  # leave at least 1 for val

        cls_train_dir = train_dir / cls
        cls_val_dir = val_dir / cls
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # If a split is empty, duplicate images to ensure every split has at least one
        if len(train_imgs) == 0:
            train_imgs = val_imgs[:1]
        if len(val_imgs) == 0:
            val_imgs = train_imgs[:1]

        for img in train_imgs:
            shutil.copy(img, cls_train_dir / img.name)
        for img in val_imgs:
            shutil.copy(img, cls_val_dir / img.name)

    # Create YAML file for YOLO
    yaml_file = SPLIT_DIR / "waste_data.yaml"
    data_yaml = {
        "train": train_dir.resolve().as_posix(),
        "val": val_dir.resolve().as_posix(),
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }
    with open(yaml_file, "w") as f:
        yaml.dump(data_yaml, f)

    print("✅ Dataset split complete")
    print("Train path:", data_yaml["train"])
    print("Val path:", data_yaml["val"])
    return yaml_file


# === Retraining Logic ===
def run_retrain():
    samples = count_new_samples()
    print(f"📊 Found {samples} new samples.")
    if samples < MIN_SAMPLES:
        print(f"❌ Not enough samples to retrain. Need at least {MIN_SAMPLES}.")
        return

    print("🔄 Merging new data into dataset...")
    merge_new_data()

    print("🔄 Splitting dataset and creating YAML...")
    yaml_file = split_dataset()

    print("🔄 Starting YOLOv8 training...")
    model = YOLO("yolov8n-cls.pt")  # base model

    # Train directly using the SPLIT_DIR (train/val subfolders)
    model.train(
        data=str(SPLIT_DIR),
        task="classify",
        epochs=20,
        batch=16,
        imgsz=224,
        name="waste_classification",
        device=0,
        save_period=1
    )

    # Save best weights
    final_weights = model.path / "weights" / "best.pt"
    if final_weights.exists():
        shutil.copy(final_weights, OUTPUT_MODEL)
        print(f"✅ Retraining complete! Model saved at {OUTPUT_MODEL}")
        try:
            reload_classifier(str(OUTPUT_MODEL))
            print("🔄 Classifier reloaded with new model.")
        except Exception as e:
            print(f"⚠️ Failed to reload classifier: {e}")
    else:
        print("❌ Training completed but final weights not found!")

# === Async Wrapper ===
def start_retrain_async():
    t = threading.Thread(target=run_retrain, daemon=True)
    t.start()
    return "Retraining started in background."












# import os
# import threading

# # === Configuration ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# NEW_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "new")  # adjust if needed
# MIN_SAMPLES = 10  # Minimum new samples required to trigger retraining


# # === Utility Functions ===
# def count_new_samples():
#     """
#     Count how many new samples exist in NEW_DATA_DIR.
#     Assumes NEW_DATA_DIR has subfolders (like dataset classes).
#     """
#     if not os.path.exists(NEW_DATA_DIR):
#         return 0

#     count = 0
#     for root, _, files in os.walk(NEW_DATA_DIR):
#         count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
#     return count


# # === Retraining Logic ===
# def run_retrain():
#     """
#     Run the retraining pipeline.
#     Replace this placeholder with actual YOLO / ML training logic.
#     """
#     print("🔄 Starting retraining...")
#     samples = count_new_samples()
#     print(f"📊 Found {samples} new samples in {NEW_DATA_DIR}")

#     if samples < MIN_SAMPLES:
#         print(f"❌ Not enough samples. Need at least {MIN_SAMPLES}.")
#         return

#     # TODO: Add your YOLOv8 / ML retraining code here
#     print("✅ Retraining completed (placeholder).")


# # === Async Wrapper ===
# def start_retrain_async():
#     """
#     Start retraining in background so API is non-blocking.
#     """
#     t = threading.Thread(target=run_retrain, daemon=True)
#     t.start()
#     return "Retraining started in background."
