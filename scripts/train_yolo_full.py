from pathlib import Path
import shutil, random, yaml
from ultralytics import YOLO

# Base dataset path
dataset_dir = Path(r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\data\archive\dataset-resized")
output_dir = Path(r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\data\archive\dataset_split")
yaml_file = output_dir / "waste_data.yaml"

train_dir = output_dir / "train"
val_dir = output_dir / "val"

# Check dataset exists
if not dataset_dir.exists():
    raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

# Remove old splits
if output_dir.exists():
    shutil.rmtree(output_dir)

# Create directories
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Get class names
class_names = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
class_names.sort()

# Split images into train/val
for cls in class_names:
    cls_train_dir = train_dir / cls
    cls_val_dir = val_dir / cls
    cls_train_dir.mkdir(parents=True, exist_ok=True)
    cls_val_dir.mkdir(parents=True, exist_ok=True)

    images = list((dataset_dir / cls).glob("*.*"))
    if len(images) == 0:
        raise RuntimeError(f"No images found in class folder: {cls}")

    random.shuffle(images)
    split_idx = int(len(images) * 0.8)

    for img in images[:split_idx]:
        shutil.copy(img, cls_train_dir / img.name)
    for img in images[split_idx:]:
        shutil.copy(img, cls_val_dir / img.name)

    print(f"{cls}: {len(images[:split_idx])} train, {len(images[split_idx:])} val images")

# Write YAML for YOLOv8
data_yaml = {
    "train": train_dir.as_posix(),
    "val": val_dir.as_posix(),
    "nc": len(class_names),
    "names": {i: name for i, name in enumerate(class_names)}
}

with open(yaml_file, "w") as f:
    yaml.dump(data_yaml, f)

print(f"YAML created at {yaml_file}")
print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}")

# # Train YOLOv8 classification model
# model = YOLO("yolov8n.pt")
# model.train(
#     data=yaml_file.as_posix(),
#     task="classify",
#     epochs=20,
#     batch=16,
#     imgsz=224,
#     name="waste_classification_final",
#     device=0
# )
