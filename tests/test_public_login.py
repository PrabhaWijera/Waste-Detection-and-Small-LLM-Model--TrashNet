# tests/test_login.py

import sys
import os
import pytest

# Ensure Python can find run.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.parametrize("identifier", [
    "john@example.com",  # test with email
    "0771234567"         # test with phone
])
def test_login(identifier):
    """
    Test the /api/public/login endpoint.
    """

    payload = {"identifier": identifier}

    response = client.post(
        "/api/public/login",
        json=payload
    )

    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "email" in data
    assert data["email"] == "john@example.com"  # adjust if testing phone login returns email
