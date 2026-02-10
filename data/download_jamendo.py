#!/usr/bin/env python3
"""
Download MTG-Jamendo EDM tracks.

Uses the official MTG-Jamendo download infrastructure.
Tracks metadata already extracted at: ../djtransgan/data/edm_tracks.tsv
"""

import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent
TRACKS_DIR = DATA_DIR / "tracks"
OLD_DATA_DIR = Path(__file__).parent.parent.parent / "djtransgan" / "data"
TSV_PATH = OLD_DATA_DIR / "edm_tracks.tsv"

# MTG-Jamendo download base URLs
# Low quality MP3 (96kbps) - smaller, faster
JAMENDO_MP3_BASE = "https://mp3l.jamendo.com/?trackid={track_num}&format=mp31"
# Full quality MP3 (320kbps) - better for training
JAMENDO_HQ_BASE = "https://prod-1.storage.jamendo.com/?trackid={track_num}&format=mp32"

# Rate limiting
DOWNLOAD_DELAY = 0.5  # Seconds between downloads
MAX_CONCURRENT = 4


def log(message: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(DATA_DIR / "PIPELINE_LOG.md", "a") as f:
        f.write(log_line + "\n")


def load_track_metadata(tsv_path: Path) -> list[dict]:
    """Load EDM tracks metadata from TSV."""
    tracks = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Parse track ID to get numeric part
            track_id = row["TRACK_ID"]
            track_num = int(track_id.replace("track_", ""))
            
            tracks.append({
                "track_id": track_id,
                "track_num": track_num,
                "artist_id": row["ARTIST_ID"],
                "album_id": row["ALBUM_ID"],
                "path": row["PATH"],  # e.g., "73/773.mp3"
                "duration": float(row["DURATION"]),
                "tags": row.get("TAGS", "").split("\t") if row.get("TAGS") else [],
            })
    return tracks


def download_track(track: dict, output_dir: Path, use_hq: bool = True) -> Optional[str]:
    """Download a single track from Jamendo."""
    track_id = track["track_id"]
    track_num = track["track_num"]
    output_path = output_dir / f"{track_id}.mp3"
    
    # Skip if already exists
    if output_path.exists():
        return str(output_path)
    
    # Build URL
    base_url = JAMENDO_HQ_BASE if use_hq else JAMENDO_MP3_BASE
    url = base_url.format(track_num=track_num)
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "audio" not in content_type and "octet-stream" not in content_type:
            # Might be an error page
            return None
        
        # Stream to file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file size (should be > 100KB for a real audio file)
        if output_path.stat().st_size < 100000:
            output_path.unlink()
            return None
        
        return str(output_path)
    
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        return None


def download_batch(tracks: list[dict], output_dir: Path, use_hq: bool = True) -> tuple[int, int]:
    """Download a batch of tracks with progress bar."""
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {}
        for track in tracks:
            future = executor.submit(download_track, track, output_dir, use_hq)
            futures[future] = track
            time.sleep(DOWNLOAD_DELAY / MAX_CONCURRENT)  # Stagger submissions
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            track = futures[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
    
    return successful, failed


def download_jamendo_edm(max_tracks: int = None, use_hq: bool = True):
    """Main download function."""
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting MTG-Jamendo EDM download")
    
    # Load metadata
    if not TSV_PATH.exists():
        log(f"ERROR: Track metadata not found at {TSV_PATH}")
        log("Please ensure edm_tracks.tsv exists from the previous djtransgan project")
        return
    
    tracks = load_track_metadata(TSV_PATH)
    log(f"Loaded {len(tracks)} EDM track metadata entries")
    
    # Check already downloaded
    existing = set(f.stem for f in TRACKS_DIR.glob("*.mp3"))
    to_download = [t for t in tracks if t["track_id"] not in existing]
    
    log(f"Already downloaded: {len(existing)}")
    log(f"Remaining: {len(to_download)}")
    
    if max_tracks:
        to_download = to_download[:max_tracks]
        log(f"Limited to {max_tracks} tracks")
    
    if not to_download:
        log("Nothing to download!")
        return
    
    # Download
    log(f"Starting download of {len(to_download)} tracks...")
    log(f"Quality: {'High (320kbps)' if use_hq else 'Low (96kbps)'}")
    
    successful, failed = download_batch(to_download, TRACKS_DIR, use_hq)
    
    # Summary
    log("=" * 60)
    log(f"Download complete:")
    log(f"  Successful: {successful}")
    log(f"  Failed: {failed}")
    log(f"  Total in library: {len(list(TRACKS_DIR.glob('*.mp3')))}")


def verify_downloads():
    """Verify downloaded files and report stats."""
    tracks = list(TRACKS_DIR.glob("*.mp3"))
    log(f"Verifying {len(tracks)} downloaded tracks...")
    
    total_size = 0
    total_duration = 0
    corrupted = []
    
    for track_path in tqdm(tracks, desc="Verifying"):
        size = track_path.stat().st_size
        total_size += size
        
        # Quick corruption check - file should be > 100KB
        if size < 100000:
            corrupted.append(track_path.name)
    
    log(f"Total size: {total_size / (1024**3):.2f} GB")
    log(f"Corrupted/incomplete: {len(corrupted)}")
    
    if corrupted:
        log("Corrupted files:")
        for f in corrupted[:10]:
            log(f"  - {f}")
        if len(corrupted) > 10:
            log(f"  ... and {len(corrupted) - 10} more")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download MTG-Jamendo EDM tracks")
    parser.add_argument("--max", type=int, default=None, help="Max tracks to download")
    parser.add_argument("--low-quality", action="store_true", help="Use low quality (96kbps)")
    parser.add_argument("--verify", action="store_true", help="Verify existing downloads")
    args = parser.parse_args()
    
    if args.verify:
        verify_downloads()
    else:
        download_jamendo_edm(max_tracks=args.max, use_hq=not args.low_quality)
