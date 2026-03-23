"""
Download and extract IDNet dataset from Zenodo.
Uses ESP (Spain) subset from part3 record — European ID cards.
"""

import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# part3 record contient ESP.zip (Spain) — 7GB
ZENODO_RECORD_ID = os.getenv("IDNET_ZENODO_RECORD", "13852734")
RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))

# Fichiers à télécharger depuis le record
TARGET_FILES = ["ESP.zip"]


def _get_download_links() -> dict[str, str]:
    """Fetch file download links from Zenodo record API."""
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    links = {f["key"]: f["links"]["self"] for f in files}
    print(f"[INFO] Fichiers disponibles : {list(links.keys())}")
    return links


def download_file(url: str, dest: Path) -> None:
    """Stream-download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=65536):  # 64KB chunks
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip and remove archive after extraction."""
    print(f"[EXTR] Extraction de {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()
    print(f"[OK] Extraction terminée → {dest}")


def download_idnet(files: list[str] = TARGET_FILES) -> Path:
    """
    Download IDNet ESP subset from Zenodo part3.
    Returns path to raw data directory.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Fetching Zenodo record {ZENODO_RECORD_ID}...")
    links = _get_download_links()

    for filename in files:
        dest_zip = RAW_DIR / filename
        country_dir = RAW_DIR / filename.replace(".zip", "")

        if country_dir.exists():
            print(f"[SKIP] {filename} déjà extrait → {country_dir}")
            continue

        if filename not in links:
            print(f"[WARN] {filename} introuvable dans le record {ZENODO_RECORD_ID}")
            print(f"       Fichiers disponibles : {list(links.keys())}")
            continue

        print(f"[DOWN] Téléchargement {filename} (~7GB, patience...)...")
        download_file(links[filename], dest_zip)
        extract_zip(dest_zip, RAW_DIR)

    print(f"\n[OK] IDNet ESP disponible dans {RAW_DIR}")
    return RAW_DIR


if __name__ == "__main__":
    download_idnet()
