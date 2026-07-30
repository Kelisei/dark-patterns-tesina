import json
import warnings

import pytest
from fastapi.testclient import TestClient

from app.main import app

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_shaming_v02(client):
    with open("data/ejemplos.json", encoding="utf-8") as f:
        data = json.load(f)

    texts = [
        {"text": t["Text"], "id": t["ID"], "path": data["Path"]}
        for t in data.get("Texts", [])
    ]

    data_schema = {"Version": "0.2", "texts": texts}

    response = client.post("/shaming", json=data_schema)
    assert response.status_code == 200

    resp_json = response.json()
    assert "version" in resp_json
    assert resp_json["version"] == "0.2"
    assert "instances" in resp_json

    for instance in resp_json["instances"]:
        assert "text" in instance
        assert "id" in instance
        assert "has_shaming" in instance
        assert isinstance(instance["has_shaming"], bool)


def test_urgency(client):
    with open("data/ejemplos_urgency.json", encoding="utf-8") as f:
        data = json.load(f)

    response = client.post("/urgency", json=data)
    assert response.status_code == 200

    resp = response.json()
    assert "urgency_instances" in resp
    assert resp["version"] == data["version"]
    assert isinstance(resp["urgency_instances"], list)

    for instance in resp["urgency_instances"]:
        assert "text" in instance
        assert "has_urgency" in instance
        assert isinstance(instance["has_urgency"], bool)


def test_scarcity(client):
    with open("data/ejemplos_scarcity.json", encoding="utf-8") as f:
        data = json.load(f)

    response = client.post("/scarcity", json=data)
    assert response.status_code == 200

    resp_json = response.json()
    assert "instances" in resp_json
    assert resp_json["version"] == "1.0"

    instances = resp_json["instances"]
    for inst in instances:
        assert "text" in inst
        assert "has_scarcity" in inst
        assert isinstance(inst["has_scarcity"], bool)


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_detect_endpoint(client):
    data = {
        "texts": [
            {"text": "Oferta relámpago, últimas unidades", "id": "1"},
            {"text": "Texto inofensivo de ejemplo", "id": "2"}
        ]
    }
    response = client.post("/detect", json=data)
    assert response.status_code == 200
    
    resp_json = response.json()
    assert "version" in resp_json
    assert len(resp_json["instances"]) == 2
    
    for inst in resp_json["instances"]:
        assert "detected" in inst
        assert "labels" in inst
        assert isinstance(inst["labels"], list)

def test_invalid_payload_returns_422(client):
    # Enviar un JSON que no cumple con el esquema Pydantic (falta 'version' y 'texts' tiene mal formato)
    data = {
        "textos": ["Esto está mal"]
    }
    response = client.post("/urgency", json=data)
    # FastAPI/Pydantic devuelve 422 Unprocessable Entity automáticamente
    assert response.status_code == 422

def test_empty_texts_payload(client):
    data = {
        "version": "1.0",
        "texts": []
    }
    response = client.post("/urgency", json=data)
    assert response.status_code == 200
    resp_json = response.json()
    assert len(resp_json["urgency_instances"]) == 0
