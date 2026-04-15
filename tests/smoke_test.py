"""
Smoke tests for the Climate RAG backend.
Tests only endpoints that don't require a live DB query.

Usage:
    pytest tests/smoke_test.py -v

Requires backend running before running tests.
Set BACKEND_URL env var to override default:
    BACKEND_URL=https://your-aws-url.com pytest tests/smoke_test.py -v
"""

import os
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:3001").rstrip("/")


def test_health():
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_root():
    res = requests.get(f"{BASE_URL}/")
    assert res.status_code == 200
    assert "message" in res.json()


def test_history_returns_list():
    res = requests.get(f"{BASE_URL}/history")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_history_entry_structure():
    res = requests.get(f"{BASE_URL}/history")
    data = res.json()
    if len(data) == 0:
        pytest.skip("No history yet.")
    entry = data[0]
    for key in ["chat_id", "title", "updated_at", "messages"]:
        assert key in entry, f"Missing key '{key}' in history entry"