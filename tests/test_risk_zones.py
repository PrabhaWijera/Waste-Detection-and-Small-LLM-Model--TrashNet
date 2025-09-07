# tests/test_risk_zones.py
# tests/test_my_submissions.py

import sys
import os
import pytest

# Ensure Python can find run.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from run import app  # Make sure run.py is in your project root

client = TestClient(app)

# Replace this with a valid JWT token
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NSwicm9sZSI6InVyYmFuIiwiZXhwIjoxNzU2OTQ4MzU5fQ.CzfjjXRT3OvS5jTL60Ys0Qz_i-T0DhDXaN9JJLXnP7k"

def test_get_risk_zones():
    headers = {"Authorization": f"Bearer {jwt_token}"}
    response = client.get("/risk_zones", headers=headers)

    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)  # assuming risk zones return a list
    assert len(data) > 0           # optional: check that the list is not empty
