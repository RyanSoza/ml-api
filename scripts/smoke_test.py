"""Check that a deployed service runs the expected version and can predict.

Usage: python3 scripts/smoke_test.py <base-url> <expected-version>
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

ATTEMPTS = int(os.environ.get("SMOKE_ATTEMPTS", "40"))
DELAY_SECONDS = float(os.environ.get("SMOKE_DELAY", "15"))

VALID = {
    "median_income": 3.5,
    "house_age": 20,
    "avg_rooms": 5.0,
    "avg_bedrooms": 1.0,
    "population": 1200,
    "avg_occupancy": 3.0,
    "latitude": 34.0,
    "longitude": -118.0,
}


def call(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    """Get URl or POST the payload as JSON, then return the JSON response"""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_version(base_url: str, expected: str) -> bool:
    """Poll /health until it reports the expected version."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            version = call(f"{base_url}/health")["version"]
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"attempt {attempt}: not reachable ({error})", flush=True)
        else:
            print(f"attempt {attempt}: version {version}", flush=True)
            if version == expected:
                return True
        time.sleep(DELAY_SECONDS)
    return False


def main() -> None:
    base_url, expected = sys.argv[1].rstrip("/"), sys.argv[2]
    if not wait_for_version(base_url, expected):
        sys.exit(f"FAILED: {base_url} never reported version {expected}")
    prediction = call(f"{base_url}/predict", payload=VALID)["prediction"]
    if not isinstance(prediction, float):
        sys.exit(f"FAILED: {base_url} returned a non-float value")
    print(f"OK: {base_url} runs {expected} and predicted {prediction:.3f}")


if __name__ == "__main__":
    main()
