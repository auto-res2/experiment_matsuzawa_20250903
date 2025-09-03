"""
preprocess.py – dataset downloading / extraction utilities
"""
from __future__ import annotations
import tarfile, zipfile, requests
from pathlib import Path

# data root directory shared across modules
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
# Download helper with progress information & error handling
# -------------------------------------------------------------------------

def _download(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    print(f"[DL] {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(target, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Downloading {url} failed") from e

# -------------------------------------------------------------------------
# Archive extraction helper
# -------------------------------------------------------------------------

def _extract(archive: Path, dest: Path):
    if dest.exists():
        return
    print(f"[EXTRACT] {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if archive.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(archive, 'r:gz') as t:
                t.extractall(dest)
        elif archive.suffix == '.zip':
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            raise RuntimeError('Unknown archive type')
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        raise RuntimeError(f'Failed to extract {archive}') from e
