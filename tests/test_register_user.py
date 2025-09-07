# tests/test_register_user.py
import sys
import os

# Add project root so Python can find run.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app

from fastapi.testclient import TestClient
from run import app  # make sure this points to your FastAPI app

client = TestClient(app)

def test_register_user():
    url = "/api/auth/register"
    payload = {
        "username": "testuser1",
        "password": "password1232",
        "email": "test1@example.com"
    }

    response = client.post(url, json=payload)

    # Print for debugging
    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Urban user registered successfully"
