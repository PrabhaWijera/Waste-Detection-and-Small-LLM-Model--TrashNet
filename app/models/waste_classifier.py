# import os
# import json
# import torch
# import torchvision.transforms as T
# from torchvision import models
# from PIL import Image
# from typing import Tuple

# ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "models/artifacts")
# MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pt")
# LABELS_PATH = os.path.join(ARTIFACT_DIR, "label_map.json")

# class WasteClassifier:
#     def __init__(self, device: str = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self._model = None
#         self._labels = None

#         self.transform = T.Compose([
#             T.Resize((224, 224)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     @property
#     def model_loaded(self) -> bool:
#         return self._model is not None

#     def _load(self):
#         if self._model is None:
#             # Load labels
#             with open(LABELS_PATH, "r") as f:
#                 self._labels = json.load(f)
#             num_classes = len(self._labels)
#             model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#             model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#             # state = torch.load(MODEL_PATH, map_location=self.device)
#             state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
#             model.load_state_dict(state)
#             model.eval().to(self.device)
#             self._model = model

#     def predict(self, img: Image.Image) -> Tuple[str, float]:
#         self._load()
#         with torch.no_grad():
#             x = self.transform(img).unsqueeze(0).to(self.device)
#             logits = self._model(x)
#             probs = torch.softmax(logits, dim=1)[0]
#             conf, idx = torch.max(probs, dim=0)
#             label = self._labels[str(int(idx))]
#             return label, float(conf.item())

from ultralytics import YOLO
import numpy as np
from PIL import Image

class WasteClassifier:
    def __init__(self):
        # Load your trained YOLO classification model
        self.model = YOLO(
            r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\runs\classify\waste_classification\weights\best.pt"
        )

    def predict(self, image_path: str):
        """
        Predict the class of an image.

        Args:
            image_path (str): Path to the image or PIL.Image

        Returns:
            tuple: (predicted_class: str, confidence: float)
        """
        # If input is a PIL image, convert to numpy
        if isinstance(image_path, Image.Image):
            image_path = np.array(image_path)

        results = self.model.predict(image_path, verbose=False)
        if not results or results[0].probs is None:
            return None, 0.0

        # Convert probabilities to numpy
        probs = results[0].probs.cpu().numpy()
        class_idx = probs.argmax()
        confidence = float(probs[class_idx])
        predicted_class = self.model.names[class_idx]

        return predicted_class, confidence
