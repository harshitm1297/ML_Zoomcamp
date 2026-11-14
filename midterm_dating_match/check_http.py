"""Start the API locally, send one valid and one invalid HTTP request, then stop it."""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx


root = Path(__file__).resolve().parent
creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "predict:app", "--host", "127.0.0.1", "--port", "9696"],
    cwd=root,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=creationflags,
)
try:
    with httpx.Client(timeout=3) as client:
        for _ in range(40):
            try:
                response = client.get("http://127.0.0.1:9696/health")
                if response.status_code == 200:
                    break
            except httpx.RequestError:
                time.sleep(0.25)
        else:
            raise RuntimeError("Server did not become ready")
        payload = json.loads((root / "example_request.json").read_text())
        valid = client.post("http://127.0.0.1:9696/predict", json=payload)
        invalid = client.post(
            "http://127.0.0.1:9696/predict",
            json={"male": {"age": 10}, "female": {"age": 30}},
        )
        valid.raise_for_status()
        assert invalid.status_code == 422
        result = valid.json()
        assert 0 <= result["mutual_interest_probability"] <= 1
        evidence = {"health": response.json(), "prediction": result, "invalid_status": invalid.status_code}
        output = root / "results" / "api_smoke.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2))
finally:
    server.terminate()
    server.wait(timeout=10)
