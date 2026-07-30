import time
from fastapi.testclient import TestClient
from app.main import app
import contextlib

client = TestClient(app)

# Warmup model loading via lifespan
with client:
    payload = {
        "texts": [
            {"text": "¡Solo quedan 2 lugares! Apúrate o te lo pierdes. Compra ya 12:30:10", "id": "1", "path": "/div"}
        ]
    }
    
    # Cold request
    start = time.perf_counter()
    resp_cold = client.post("/detect", json=payload)
    end = time.perf_counter()
    cold_latency_ms = (end - start) * 1000
    
    # Cached request
    start2 = time.perf_counter()
    resp_cached = client.post("/detect", json=payload)
    end2 = time.perf_counter()
    cached_latency_ms = (end2 - start2) * 1000
    
    print(f"COLD_MS:{cold_latency_ms:.3f}")
    print(f"CACHED_MS:{cached_latency_ms:.3f}")
