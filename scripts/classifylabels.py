import os
import shutil
import random

# ---------------- CONFIG ----------------
dataset_dir = r"E:/ICBT/CIS-6002-final presatation ML/FinalProject Backend/TrashNet-OK/waste_management_system/data/archive/dataset-resized"
output_dir = r"E:/ICBT/CIS-6002-final presatation ML/FinalProject Backend/TrashNet-OK/waste_management_system/data/archive/processed_dataset"

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # replace with your actual classes
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
# ----------------------------------------

# Create output folder structure
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Loop through each class folder
for cls in classes:
    class_folder = os.path.join(dataset_dir, cls)
    if not os.path.exists(class_folder):
        print(f"Warning: class folder {class_folder} does not exist")
        continue

    images = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # Move images
    for img in train_imgs:
        shutil.copy(os.path.join(class_folder, img), os.path.join(output_dir, "train", cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_folder, img), os.path.join(output_dir, "val", cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_folder, img), os.path.join(output_dir, "test", cls, img))

print("✅ Dataset reorganized for YOLO classification successfully!")
