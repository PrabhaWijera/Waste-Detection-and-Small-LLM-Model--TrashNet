from ultralytics import YOLO
from pathlib import Path

# Path to your dataset YAML
yaml_path = Path(r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_data.yaml").as_posix()

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Start training
model.train(
    data=yaml_path,          # dataset YAML
    task="classify",         # classification task
    epochs=20,               # number of epochs
    batch=16,                # batch size (reduce if GPU memory is limited)
    imgsz=224,               # image size
    name="waste_classification_final",  # run name
    device=0,                # GPU device index (0 = first GPU)
    save=True,               # save checkpoints
    save_period=1,           # save every epoch
    plots=True,              # show training plots
    verbose=True,            # detailed logs
    patience=5               # early stopping patience
)

print("Training started! Check results in runs/classify/waste_classification_final/")
