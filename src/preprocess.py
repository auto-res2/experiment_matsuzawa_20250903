"""src/preprocess.py
Data acquisition / extraction helper functions that are shared across all
experiments.
"""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Union

import requests

# Root directory where raw datasets will be stored
DATA = Path("data")
DATA.mkdir(parents=True, exist_ok=True)

__all__ = ["DATA", "download", "extract"]


def download(url: str, target: Union[str, Path]) -> Path:
    """Download *url* to *target* if the file does not yet exist."""
    target = Path(target)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DL] {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
    except Exception as e:
        # Clean up partially downloaded files so that future retries work.
        if target.exists():
            target.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    return target


def extract(archive: Union[str, Path], dest: Union[str, Path]) -> Path:
    """Extract *archive* into *dest* unless already extracted."""
    archive, dest = Path(archive), Path(dest)
    if dest.exists():
        return dest

    print(f"[EXTRACT] {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        if archive.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(dest)
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
        else:
            raise RuntimeError("Unsupported archive format: " + archive.suffix)
    except Exception as e:
        # In case of failure wipe the destination to avoid corrupt partial
        # extractions.
        if dest.exists():
            for p in dest.rglob("*"):
                p.unlink(missing_ok=True)
            dest.rmdir()
        raise

    return dest
