# tests/test_classify.py
# tests/test_llm_recommend.py
# tests/test_monthly_report.py

import sys
import os
import pytest

# Add project root to sys.path so we can import run.py and app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from fastapi.testclient import TestClient
from run import app  # import your FastAPI app

client = TestClient(app)

# Replace this with your valid JWT token
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NSwicm9sZSI6InVyYmFuIiwiZXhwIjoxNzU2OTQ4MzU5fQ.CzfjjXRT3OvS5jTL60Ys0Qz_i-T0DhDXaN9JJLXnP7k"

def test_classify_image():
    headers = {"Authorization": f"Bearer {jwt_token}"}

    # Use a sample image file from your project (place test.jpg in tests/ folder)
    file_path = os.path.join(os.path.dirname(__file__), "test.jpg")
    assert os.path.exists(file_path), "Test image file not found!"

    with open(file_path, "rb") as img_file:
        files = {"image": ("test.jpg", img_file, "image/jpeg")}
        response = client.post("/classify", headers=headers, files=files)

    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    json_data = response.json()
    assert "class" in json_data  # or whatever your classify API returns
    assert isinstance(json_data["class"], str)
