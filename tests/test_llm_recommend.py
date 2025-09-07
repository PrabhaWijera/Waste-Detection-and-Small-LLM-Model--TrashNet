# tests/test_llm_recommend.py
# tests/test_monthly_report.py

import sys
import os
import pytest

# Add project root to sys.path so we can import run.py and app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from run import app  # import your FastAPI app from run.py

client = TestClient(app)

# Replace this with your valid JWT token
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NSwicm9sZSI6InVyYmFuIiwiZXhwIjoxNzU2OTQ4MzU5fQ.CzfjjXRT3OvS5jTL60Ys0Qz_i-T0DhDXaN9JJLXnP7k"

def test_llm_recommend():
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"description": "plastic bottle"}

    response = client.post("/llm_recommend", headers=headers, data=data)

    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    json_data = response.json()
    assert "recommendation" in json_data
    assert len(json_data["recommendation"]) > 0
