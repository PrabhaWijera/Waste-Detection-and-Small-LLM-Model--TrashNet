# tests/test_login_user.py
# tests/test_register_user.py
import sys
import os

# Add project root so Python can find run.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app
import pytest
from fastapi.testclient import TestClient
from run import app  # make sure this points to your FastAPI app

client = TestClient(app)

def test_login_user():
    payload = {
        "username": "testuser1",
        "password": "password1232"
    }

    response = client.post("/api/auth/login", json=payload)
    
    # Debug prints
    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"
