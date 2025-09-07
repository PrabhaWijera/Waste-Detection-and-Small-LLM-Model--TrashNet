# tests/test_register_user.py
import sys
import os

# Add project root so Python can find run.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app

# tests/test_normal_submissions.py

import pytest
from fastapi.testclient import TestClient
from run import app  # Make sure this points to your FastAPI app

client = TestClient(app)

# Replace with your actual JWT token
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Niwicm9sZSI6InVyYmFuIiwiZXhwIjoxNzU2OTYwMzA0fQ.HTedz-ewXGXE14iixI3iqK47lE5PIJ5p2TnbpRJy8DY"

def test_normal_submissions_with_jwt():
    headers = {"Authorization": f"Bearer {jwt_token}"}
    response = client.get("/api/admin/normal_submissions", headers=headers)
    
    # Debug prints
    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "submissions" in data
    assert len(data["submissions"]) > 0
