# app/models/yolo_wrapper.py
import os
import threading
import numpy as np
from ultralytics import YOLO
from PIL import Image

class YOLOWrapper:
    """
    Loads a classification YOLO model from a path (default models/current.pt).
    Auto-reloads if file mtime changes.
    """
    def __init__(self, model_path: str = "models/current.pt", device: str = None):
        self.model_path = model_path
        self.device = device  # ultralytics will choose cuda if available by default
        self._model = None
        self._mt = None
        self._lock = threading.RLock()
        self._load_model()

    def _file_mtime(self):
        try:
            return os.path.getmtime(self.model_path)
        except Exception:
            return None

    def _load_model(self):
        with self._lock:
            mtime = self._file_mtime()
            if mtime is None:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            print(f"[YOLOWrapper] Loading model: {self.model_path}")
            self._model = YOLO(self.model_path)
            self._mt = mtime
            print("[YOLOWrapper] class names:", self._model.names)

    def maybe_reload(self):
        mtime = self._file_mtime()
        if mtime is None:
            return
        if self._mt is None or mtime != self._mt:
            try:
                self._load_model()
            except Exception as e:
                print(f"[YOLOWrapper] Failed to reload model: {e}")

    def predict(self, img, imgsz=224):
        """
        img: PIL.Image OR numpy array OR path
        returns (label:str, confidence:float, probs:list)
        """
        self.maybe_reload()

        # convert PIL to numpy
        if isinstance(img, Image.Image):
            img = np.array(img)

        with self._lock:
            results = self._model.predict(img, imgsz=imgsz)

        if not results or results[0].probs is None:
            return None, None, None

        # Use the new Probs API, move tensor to CPU first
        probs_tensor = results[0].probs.data  # torch tensor on GPU or CPU
        probs_array = probs_tensor.cpu().numpy()  # move to CPU and convert to numpy
        top_idx = int(results[0].probs.top1)       # top1 index
        conf = float(results[0].probs.top1conf)   # top1 confidence
        label = self._model.names.get(top_idx, str(top_idx))

        return label, conf, probs_array.tolist()
