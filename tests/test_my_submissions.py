# tests/test_my_submissions.py

import sys
import os
import pytest

# Ensure Python can find run.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Replace with a valid JWT token for your test user
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huQGV4YW1wbGUuY29tIiwicGhvbmUiOiIwNzcxMjM0NTY3IiwiZXhwIjoxNzU2OTYxMTgzfQ.oeAQ2EMaUV2iyf06Np5ebPyZpCXhZSRmh8RXZktQxLA"

def test_my_submissions():
    """
    Test the /api/public/my_submissions endpoint using JWT auth.
    """

    headers = {"Authorization": f"Bearer {JWT_TOKEN}"}

    response = client.get("/api/public/my_submissions", headers=headers)

    print("\nStatus code:", response.status_code)
    print("Response JSON:", response.json())

    assert response.status_code == 200
    data = response.json()
    assert "submissions" in data
    assert isinstance(data["submissions"], list)
