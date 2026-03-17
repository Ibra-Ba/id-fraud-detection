"""
Download and extract IDNet dataset from Zenodo.
Only downloads the European subset to save space.
"""

import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_URL = os.getenv("IDNET_ZENODO_URL", "https://zenodo.org/records/10570393")
RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))

# European country files available in IDNet
EUROPEAN_FILES = [
    "France.zip",
    "Germany.zip",
    "Italy.zip",
    "Spain.zip",
]


def _get_download_links() -> dict[str, str]:
    """Fetch file download links from Zenodo record API."""
    record_id = ZENODO_URL.rstrip("/").split("/")[-1]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    return {f["key"]: f["links"]["self"] for f in files}


def download_file(url: str, dest: Path) -> None:
    """Stream-download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip and remove archive after extraction."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()


def download_idnet(countries: list[str] = EUROPEAN_FILES) -> Path:
    """
    Download IDNet European subset.
    Returns path to raw data directory.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching Zenodo file index...")
    links = _get_download_links()

    for filename in countries:
        dest_zip = RAW_DIR / filename
        country_dir = RAW_DIR / filename.replace(".zip", "")

        if country_dir.exists():
            print(f"[SKIP] {filename} already extracted")
            continue

        if filename not in links:
            print(f"[WARN] {filename} not found in Zenodo record")
            continue

        print(f"[DOWN] Downloading {filename}...")
        download_file(links[filename], dest_zip)
        print(f"[EXTR] Extracting {filename}...")
        extract_zip(dest_zip, RAW_DIR)

    print(f"[OK] IDNet data available at {RAW_DIR}")
    return RAW_DIR


if __name__ == "__main__":
    download_idnet()
