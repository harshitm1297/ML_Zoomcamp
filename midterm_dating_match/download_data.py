"""Download the two source CSVs from a pinned public Mixmosa revision."""

import hashlib
from pathlib import Path
from urllib.request import urlopen


REVISION = "48676f40689fe5f41faa9d875e18320880725633"
FILES = {
    "interactions.csv": "319588de5d1b30f86f9a0f6d1954a838d0d6c1f78fbb69e021a2ded0510dc1ae",
    "user_features.csv": "3445a2dea858b0551d65cde4f7e6aad49a37ef077a2e4532f734578522a22a7d",
}
DATA_DIR = Path(__file__).resolve().parent / "data"


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in FILES.items():
        target = DATA_DIR / filename
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
            print(f"Verified {target}")
            continue
        url = (
            "https://raw.githubusercontent.com/dankras/mixmosa-anonymized-data/"
            f"{REVISION}/csv/{filename}"
        )
        with urlopen(url, timeout=30) as response:
            content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Unexpected SHA256 for {filename}: {actual_hash}")
        target.write_bytes(content)
        print(f"Downloaded {target}")


if __name__ == "__main__":
    download()
