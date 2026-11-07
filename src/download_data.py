"""
Download and extract the DLR Urban Traffic (DLR-UT) dataset from Zenodo.

The public fragment (24.09.2023, 96 files of 15 minutes) is downloaded
as a single ZIP archive and extracted into data/.

Usage:
    python src/download_data.py [--force]

Options:
    --force   Re-download and re-extract even if the dataset already exists.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    DATA_DIR,
    DLRUT_ARCHIVE_NAME,
    DLRUT_EXTRACTED_DIR_NAME,
    DLRUT_ROOT,
    DLRUT_TRAJECTORIES_DIR,
    DLRUT_ZENODO_URL,
    N_FILES,
)


def download_archive(url: str, dest: Path, force: bool = False) -> None:
    """Download a file from `url` to `dest` with a progress bar."""
    if dest.exists() and not force:
        print(f"[skip] Archive already exists: {dest}")
        return

    print(f"[download] {url}")
    print(f"[download] -> {dest}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive: Path, dest_dir: Path, force: bool = False) -> None:
    """Extract a ZIP archive to `dest_dir`."""
    target = dest_dir / DLRUT_EXTRACTED_DIR_NAME
    if target.exists() and not force:
        print(f"[skip] Dataset already extracted: {target}")
        return

    if target.exists() and force:
        print(f"[remove] Removing existing directory: {target}")
        shutil.rmtree(target)

    print(f"[extract] {archive} -> {dest_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.namelist()
        with tqdm(total=len(members), unit="file", desc="Extracting") as pbar:
            for member in members:
                zf.extract(member, dest_dir)
                pbar.update(1)


def verify_dataset() -> None:
    """Check that the trajectory directory contains the expected number of files."""
    if not DLRUT_TRAJECTORIES_DIR.exists():
        raise FileNotFoundError(
            f"Trajectory directory not found: {DLRUT_TRAJECTORIES_DIR}"
        )

    files = sorted(DLRUT_TRAJECTORIES_DIR.glob("trajectories_*.csv"))
    print(f"[verify] Found {len(files)} trajectory files in {DLRUT_TRAJECTORIES_DIR}")

    if len(files) != N_FILES:
        print(
            f"[warn] Expected {N_FILES} files, found {len(files)}. "
            "The analysis may still work, but results could differ from the paper."
        )
    else:
        print(f"[verify] OK: {N_FILES} files found.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract the DLR UT dataset from Zenodo."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the dataset already exists.",
    )
    args = parser.parse_args()

    archive_path = DATA_DIR / DLRUT_ARCHIVE_NAME

    try:
        download_archive(DLRUT_ZENODO_URL, archive_path, force=args.force)
        extract_archive(archive_path, DATA_DIR, force=args.force)
        verify_dataset()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"\n[done] Dataset is available at: {DLRUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
