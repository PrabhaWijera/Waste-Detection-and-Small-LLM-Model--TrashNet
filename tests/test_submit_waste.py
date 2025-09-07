# tests/test_submit_waste.py
# tests/test_monthly_report.py

import sys
import os
import pytest

# Add project root to sys.path so we can import run.py and app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from run import app  # Make sure this points to your FastAPI app
from io import BytesIO

client = TestClient(app)

def test_submit_waste():
    # Prepare form-data including a fake image file
    files = {
        "image": ("F:/plastic.jpg", BytesIO(b"fake image content"), "image/jpeg")
    }
    data = {
        "waste_type": "plastic",
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "0771234567",
        "location": "Colombo",
        "urban_center": "WesternProvince"
    }

    response = client.post("/api/public/submit_waste", files=files, data=data)

    # Debug prints
    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    json_data = response.json()
    assert "saved_file" in json_data
    assert "message" in json_data
    assert json_data["message"] == "Submission received. User registered (if new)."
