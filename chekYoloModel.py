# from ultralytics import YOLO

# # Use the actual full path to your trained best.pt
# model = YOLO("E:/ICBT/CIS-6002-final presatation ML/FinalProject Backend/TrashNet-OK/waste_management_system/runs/classify/waste_classification/weights/best.pt")


# # Test prediction on an image
# img_path = r"F:/cc.webp"
# results = model(img_path)

# print("Prediction:", results[0].probs.top1)
# print("Confidence:", results[0].probs.top1conf.item())

# from ultralytics import YOLO

# model = YOLO(r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\runs\classify\waste_classification\weights\best.pt")

# print("Class names mapping:", model.names)



# chekYoloModel.py
from app.models.yolo_wrapper import YOLOWrapper
from PIL import Image
m = YOLOWrapper(model_path="models/current.pt")
img = Image.open(r"F:\cc.webp").convert("RGB")
label, conf, probs = m.predict(img, imgsz=224)
print("Prediction:", label, "Confidence:", conf)
print("All probs:", probs)
print("Class map:", m._model.names)
