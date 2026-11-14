"""Send a documented example request to a running prediction service."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9696")
    parser.add_argument("--evidence", type=Path, help="Write HTTP results to this JSON file")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    payload = json.loads(Path(__file__).with_name("example_request.json").read_text())
    with httpx.Client(timeout=10) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        assert health.json() == {"status": "ok"}
        response = client.post(f"{base_url}/predict", json=payload)
        response.raise_for_status()
        prediction = response.json()
        invalid = json.loads(json.dumps(payload))
        invalid["male"]["age"] = 10
        rejected = client.post(f"{base_url}/predict", json=invalid)
    assert 0 <= prediction["mutual_interest_probability"] <= 1
    assert rejected.status_code == 422
    evidence = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": base_url,
        "health_status": health.status_code,
        "health_response": health.json(),
        "prediction_status": response.status_code,
        "prediction_response": prediction,
        "invalid_input_status": rejected.status_code,
    }
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
