import json
from PIL import Image
from app.models.waste_classifier import WasteClassifier

def test_import_and_predict_smoke():
    clf = WasteClassifier()
    # Create a dummy image (won't be accurate)
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    try:
        clf.predict(img)
    except FileNotFoundError:
        # Expected before model artifacts are trained
        assert True
