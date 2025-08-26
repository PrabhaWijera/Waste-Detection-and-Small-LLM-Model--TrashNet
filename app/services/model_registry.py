import os
from app.models.yolo_wrapper import YOLOWrapper

# Path to the trained YOLO classification weights
DEFAULT_WEIGHTS = r"E:\ICBT\CIS-6002-final presatation ML\FinalProject Backend\TrashNet-OK\waste_management_system\runs\classify\waste_classification\weights\best.pt"

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        # Use model_path (not weights_path) as required by YOLOWrapper
        _classifier = YOLOWrapper(model_path=DEFAULT_WEIGHTS)
    return _classifier

def reload_classifier(new_model_path: str):
    """Reloads the classifier with a new model path."""
    global _classifier
    if _classifier is None:
        _classifier = YOLOWrapper(model_path=new_model_path)
    else:
        # Directly replace the internal model path and reload
        _classifier.model_path = new_model_path
        _classifier.maybe_reload()
    return True
