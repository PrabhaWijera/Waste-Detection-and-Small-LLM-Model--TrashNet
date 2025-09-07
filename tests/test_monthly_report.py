# tests/test_monthly_report.py

import sys
import os
import pytest

# Add project root to sys.path so we can import run.py and app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app  # Now this should work
from fastapi.testclient import TestClient

# JWT token for authentication
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NSwicm9sZSI6InVyYmFuIiwiZXhwIjoxNzU2OTQ4MzU5fQ.CzfjjXRT3OvS5jTL60Ys0Qz_i-T0DhDXaN9JJLXnP7k"

# Create TestClient instance
client = TestClient(app)

def test_monthly_report_with_jwt():
    headers = {"Authorization": f"Bearer {jwt_token}"}

    response = client.get("/api/admin/waste_report/monthly", headers=headers)

    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    assert response.status_code == 200
    data = response.json()
    assert "monthly_report" in data
    assert isinstance(data["monthly_report"], list)
    if data["monthly_report"]:
        assert "total" in data["monthly_report"][0]
