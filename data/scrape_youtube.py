#!/usr/bin/env python3
"""
Scrape YouTube DJ sets with tracklists in description.

Boiler Room, Cercle, and other channels often include full tracklists
with timestamps in video descriptions.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent
MIXES_DIR = DATA_DIR / "raw" / "mixes"
METADATA_DIR = DATA_DIR / "metadata" / "youtube"
RATE_LIMIT = 2  # Seconds between requests


def log(message: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(DATA_DIR / "PIPELINE_LOG.md", "a") as f:
        f.write(log_line + "\n")


def get_video_info(url: str) -> Optional[dict]:
    """Get video metadata using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            log(f"yt-dlp error: {result.stderr[:200]}")
    except Exception as e:
        log(f"Error getting video info: {e}")
    
    return None


def parse_tracklist_from_description(description: str, duration: int) -> list[dict]:
    """Parse tracklist from video description."""
    tracklist = []
    
    if not description:
        return tracklist
    
    lines = description.split("\n")
    
    # Common timestamp patterns
    # 00:00 Artist - Track
    # 0:00:00 Artist - Track
    # [00:00] Artist - Track
    # 00:00 - Artist - Track
    timestamp_patterns = [
        r"^\s*\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–—]?\s*(.+)",
        r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)",
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        for pattern in timestamp_patterns:
            match = re.match(pattern, line)
            if match:
                time_str, track_text = match.groups()
                
                # Parse timestamp
                seconds = parse_timestamp(time_str)
                if seconds is None or seconds > duration:
                    continue
                
                # Parse artist - title
                artist, title = parse_track_text(track_text)
                
                if artist or title:
                    tracklist.append({
                        "position": len(tracklist) + 1,
                        "artist": artist,
                        "title": title,
                        "timestamp": time_str,
                        "timestamp_seconds": seconds,
                    })
                break
    
    return tracklist


def parse_timestamp(time_str: str) -> Optional[int]:
    """Convert timestamp string to seconds."""
    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return None


def parse_track_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse 'Artist - Title' text."""
    text = text.strip()
    
    # Remove common suffixes
    text = re.sub(r'\s*\(.*?\)\s*$', '', text)  # Remove (Remix) etc
    text = re.sub(r'\s*\[.*?\]\s*$', '', text)  # Remove [Label] etc
    
    # Split on separator
    separators = [' - ', ' – ', ' — ', ' // ']
    for sep in separators:
        if sep in text:
            parts = text.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    
    # No separator found - might just be title
    return None, text


def generate_mix_id(data: dict) -> str:
    """Generate unique ID for a mix."""
    parts = []
    if data.get("dj_name"):
        parts.append(re.sub(r"[^\w]", "", data["dj_name"].lower())[:20])
    if data.get("date"):
        parts.append(re.sub(r"[^\d]", "", str(data["date"]))[:8])
    if data.get("mix_title"):
        parts.append(re.sub(r"[^\w]", "", data["mix_title"].lower())[:30])
    
    import hashlib
    url_hash = hashlib.md5(data.get("source_url", "").encode()).hexdigest()[:8]
    parts.append(url_hash)
    
    return "_".join(filter(None, parts)) or url_hash


def download_audio(url: str, output_dir: Path, mix_id: str) -> Optional[str]:
    """Download audio using yt-dlp."""
    output_path = output_dir / f"{mix_id}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", str(output_path),
        "--no-playlist",
        url,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            for f in output_dir.glob(f"{mix_id}.*"):
                if f.suffix in [".mp3", ".m4a", ".opus", ".webm", ".wav"]:
                    return str(f)
        else:
            log(f"yt-dlp error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log(f"Download timeout for {url}")
    except Exception as e:
        log(f"Download error: {e}")
    
    return None


def search_youtube_mixes(query: str, max_results: int = 20) -> list[str]:
    """Search YouTube for DJ mixes."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        f"ytsearch{max_results}:{query}",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            urls = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    data = json.loads(line)
                    url = data.get("url") or data.get("webpage_url")
                    if url:
                        if not url.startswith("http"):
                            url = f"https://www.youtube.com/watch?v={data.get('id', url)}"
                        urls.append(url)
            return urls
    except Exception as e:
        log(f"Search error: {e}")
    
    return []


def get_search_queries() -> list[str]:
    """Search queries for finding DJ mixes with tracklists IN the description."""
    # These searches find fan uploads and compilations that actually include
    # tracklists with timestamps in the video description
    return [
        "techno mix timestamps 2024",
        "techno mix timestamps 2025",
        "house mix tracklist 2024",
        "tech house mix tracklist",
        "dark techno mix tracklist",
        "melodic techno mix tracklist",
        "progressive house mix tracklist",
        "trance mix timestamps",
        "drum and bass mix tracklist",
        "minimal techno mix tracklist",
        "deep house mix tracklist",
        "hardgroove mix tracklist",
        "industrial techno mix tracklist",
        "acid techno mix tracklist",
        "peak time techno mix tracklist",
    ]


def scrape_youtube(max_mixes: int = 100, download_audio_flag: bool = True):
    """Main scraping function."""
    MIXES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting YouTube DJ mix scraper")
    log(f"Target: {max_mixes} mixes")
    
    collected_mixes = []
    seen_urls = set()
    
    # Load existing
    for f in METADATA_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                seen_urls.add(data.get("source_url"))
        except:
            pass
    
    log(f"Already have {len(seen_urls)} mixes")
    
    # Search for videos
    search_queries = get_search_queries()
    all_urls = []
    
    for query in tqdm(search_queries, desc="Searching"):
        urls = search_youtube_mixes(query, max_results=15)
        for url in urls:
            if url not in seen_urls and url not in all_urls:
                all_urls.append(url)
        time.sleep(RATE_LIMIT)
        
        if len(all_urls) >= max_mixes * 2:
            break
    
    log(f"Found {len(all_urls)} candidate videos")
    
    # Process each
    successful = 0
    for url in tqdm(all_urls, desc="Processing"):
        if successful >= max_mixes:
            break
        
        if url in seen_urls:
            continue
        
        # Get video info
        info = get_video_info(url)
        if not info:
            continue
        
        duration = info.get("duration", 0)
        
        # Skip short videos (probably not full sets)
        if duration < 1800:  # 30 minutes minimum
            continue
        
        # Parse tracklist from description
        description = info.get("description", "")
        tracklist = parse_tracklist_from_description(description, duration)
        
        # Skip if no tracklist found
        if len(tracklist) < 5:
            log(f"Skipping {info.get('title', url)[:40]}... - only {len(tracklist)} tracks in description")
            continue
        
        # Build metadata
        data = {
            "source": "youtube",
            "source_url": url,
            "video_id": info.get("id"),
            "scraped_at": datetime.now().isoformat(),
            "dj_name": info.get("uploader"),
            "mix_title": info.get("title"),
            "duration_seconds": duration,
            "date": info.get("upload_date"),
            "description": description[:2000],  # Truncate
            "genres": info.get("tags", [])[:10],
            "tracklist": tracklist,
            "audio_links": [url],
            "channel": info.get("channel"),
            "view_count": info.get("view_count"),
        }
        
        mix_id = generate_mix_id(data)
        data["mix_id"] = mix_id
        
        # Download audio
        if download_audio_flag:
            audio_path = download_audio(url, MIXES_DIR, mix_id)
            if audio_path:
                data["audio_file"] = audio_path
                log(f"Downloaded: {audio_path}")
            else:
                log(f"Audio download failed for {url}")
        
        # Save metadata
        metadata_path = METADATA_DIR / f"{mix_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
        
        collected_mixes.append(data)
        seen_urls.add(url)
        successful += 1
        
        log(f"Scraped: {data['dj_name']} - {data['mix_title'][:30]}... ({len(tracklist)} tracks)")
        time.sleep(RATE_LIMIT)
    
    # Summary
    with_audio = sum(1 for m in collected_mixes if m.get("audio_file"))
    total_tracks = sum(len(m.get("tracklist", [])) for m in collected_mixes)
    
    log("=" * 60)
    log(f"YouTube scraping complete: {len(collected_mixes)} mixes")
    log(f"  With audio: {with_audio}")
    log(f"  Metadata only: {len(collected_mixes) - with_audio}")
    log(f"  Total tracks identified: {total_tracks}")
    
    return collected_mixes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape YouTube DJ mixes")
    parser.add_argument("--max", type=int, default=50, help="Max mixes")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio download")
    args = parser.parse_args()
    
    scrape_youtube(max_mixes=args.max, download_audio_flag=not args.no_audio)
