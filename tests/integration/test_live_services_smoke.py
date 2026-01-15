#!/usr/bin/env python3
"""
Smoke tests for live services (API server and Web UI).
Skips when endpoints are unreachable.
"""

import os
import pytest
import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8711")
WEB_UI_URL = os.getenv("WEB_UI_URL", "http://localhost:8501")


def _reachable(url: str) -> bool:
    try:
        response = requests.get(url, timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False


def test_api_health():
    if not _reachable(f"{API_BASE_URL}/health"):
        pytest.skip("API server not reachable")
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    assert response.status_code == 200


def test_web_ui_home():
    if not _reachable(WEB_UI_URL):
        pytest.skip("Web UI not reachable")
    response = requests.get(WEB_UI_URL, timeout=5)
    assert response.status_code == 200
