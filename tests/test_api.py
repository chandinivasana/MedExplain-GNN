import pytest
import sys
import os
import asyncio
import httpx
import importlib.util

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # Ensure sys.path contains module directory for its relative imports
    mod_dir = os.path.dirname(file_path)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec.loader.exec_module(module)
    return module

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
backend_main = load_module('backend_app_module', os.path.join(base_dir, 'backend', 'main.py'))
ai_main = load_module('ai_app_module', os.path.join(base_dir, 'ai_engine', 'main.py'))

def async_request(app, method, url, **kwargs):
    async def _invoke():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(_invoke())

def test_backend_health_check():
    response = async_request(backend_main.app, "GET", "/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MedExplain-GNN Gateway"

def test_backend_empty_prediction_rejected():
    response = async_request(backend_main.app, "POST", "/predict-disease", json={"text": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_backend_excessive_length_rejected():
    response = async_request(backend_main.app, "POST", "/predict-disease", json={"text": "fever " * 2000})
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()

def test_backend_history_endpoint():
    response = async_request(backend_main.app, "GET", "/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_ai_service_health():
    response = async_request(ai_main.app, "GET", "/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "queue_size" in data
