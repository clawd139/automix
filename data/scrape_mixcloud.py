#!/usr/bin/env python3
"""
Mixcloud scraper for fetching DJ mixes with tracklists.

Uses REST API for metadata + web scraping for tracklists.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent
MIXES_DIR = DATA_DIR / "mixes"
METADATA_DIR = DATA_DIR / "metadata" / "mixcloud"
RATE_LIMIT = 2  # Seconds between requests

API_BASE = "https://api.mixcloud.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def log(message: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(DATA_DIR / "PIPELINE_LOG.md", "a") as f:
        f.write(log_line + "\n")


def api_request(endpoint: str) -> Optional[dict]:
    """Make a request to Mixcloud REST API."""
    try:
        url = f"{API_BASE}{endpoint}" if endpoint.startswith("/") else endpoint
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        time.sleep(RATE_LIMIT)
        return response.json()
    except Exception as e:
        log(f"API error: {e}")
        return None


def search_mixes(query: str, limit: int = 20) -> list[dict]:
    """Search for mixes by keyword."""
    endpoint = f"/search/?q={query.replace(' ', '+')}&type=cloudcast&limit={limit}"
    result = api_request(endpoint)
    if not result:
        return []
    return result.get("data", [])


def scrape_tracklist_from_page(url: str) -> list[dict]:
    """Scrape tracklist from Mixcloud page (web scraping fallback)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        
        tracklist = []
        
        # Look for tracklist in JSON-LD
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "MusicRecording":
                    # Found track info
                    pass
            except:
                pass
        
        # Look for track elements in page
        track_elements = soup.select('[class*="track"], [data-track], .tracklist-item')
        for i, elem in enumerate(track_elements):
            track = {
                "position": i + 1,
                "artist": None,
                "title": None,
                "timestamp_seconds": None,
            }
            
            # Try various selectors
            artist = elem.select_one('[class*="artist"], .artist')
            title = elem.select_one('[class*="title"], .title, .track-name')
            time_elem = elem.select_one('[class*="time"], .timestamp')
            
            if artist:
                track["artist"] = artist.get_text(strip=True)
            if title:
                track["title"] = title.get_text(strip=True)
            if time_elem:
                time_text = time_elem.get_text(strip=True)
                track["timestamp"] = time_text
                track["timestamp_seconds"] = parse_timestamp_str(time_text)
            
            if track["artist"] or track["title"]:
                tracklist.append(track)
        
        return tracklist
    except Exception as e:
        log(f"Page scrape error: {e}")
        return []


def parse_timestamp_str(time_str: str) -> Optional[int]:
    """Parse timestamp string to seconds."""
    if not time_str:
        return None
    parts = re.findall(r'\d+', time_str)
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 1:
        return int(parts[0])
    return None


def get_cloudcast_details(key: str) -> Optional[dict]:
    """Get detailed info for a cloudcast."""
    result = api_request(key)
    if not result:
        return None
    
    # Try to get tracklist from page
    url = result.get("url", f"https://www.mixcloud.com{key}")
    tracklist = scrape_tracklist_from_page(url)
    result["scraped_tracklist"] = tracklist
    
    return result


def parse_mixcloud_data(cloudcast: dict) -> dict:
    """Parse Mixcloud cloudcast data into our standard format."""
    url = cloudcast.get("url", "")
    if not url.startswith("http"):
        url = f"https://www.mixcloud.com{url}"
    
    data = {
        "source": "mixcloud",
        "source_url": url,
        "scraped_at": datetime.now().isoformat(),
        "dj_name": None,
        "mix_title": cloudcast.get("name"),
        "duration_seconds": cloudcast.get("audio_length"),
        "date": cloudcast.get("created_time"),
        "description": cloudcast.get("description"),
        "genres": [],
        "tracklist": [],
        "audio_links": [url],
    }
    
    # DJ name from user field (REST API format)
    user = cloudcast.get("user", {})
    data["dj_name"] = user.get("name") or user.get("username")
    
    # Tags/genres (REST API format)
    tags = cloudcast.get("tags", [])
    if isinstance(tags, list):
        data["genres"] = [t.get("name") for t in tags if t.get("name")]
    
    # Tracklist from scraped data
    scraped_tracklist = cloudcast.get("scraped_tracklist", [])
    if scraped_tracklist:
        data["tracklist"] = scraped_tracklist
    
    return data


def format_timestamp(seconds: Optional[int]) -> Optional[str]:
    """Convert seconds to MM:SS or HH:MM:SS."""
    if seconds is None:
        return None
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def generate_mix_id(data: dict) -> str:
    """Generate unique ID for a mix."""
    parts = []
    if data.get("dj_name"):
        parts.append(re.sub(r"[^\w]", "", data["dj_name"].lower())[:20])
    if data.get("date"):
        parts.append(re.sub(r"[^\d]", "", str(data["date"]))[:8])
    if data.get("mix_title"):
        parts.append(re.sub(r"[^\w]", "", data["mix_title"].lower())[:20])
    
    import hashlib
    url_hash = hashlib.md5(data["source_url"].encode()).hexdigest()[:8]
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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


def get_search_queries() -> list[str]:
    """Search queries for finding good DJ mixes."""
    return [
        "techno mix 2024",
        "house mix 2024", 
        "tech house mix",
        "deep house mix",
        "progressive house mix",
        "drum and bass mix",
        "trance mix",
        "minimal techno mix",
        "boiler room",
        "essential mix",
        "resident advisor podcast",
        "fabric london",
        "berghain",
        "amelie lens",
        "charlotte de witte",
        "adam beyer",
        "carl cox",
        "solomun",
        "tale of us",
        "boris brejcha",
    ]


def scrape_mixcloud(max_mixes: int = 100, download_audio_flag: bool = True):
    """Main scraping function."""
    MIXES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting Mixcloud scraper")
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
    
    # Search for mixes
    search_queries = get_search_queries()
    all_cloudcasts = []
    
    for query in tqdm(search_queries, desc="Searching"):
        results = search_mixes(query, limit=30)
        for cc in results:
            url = f"https://www.mixcloud.com{cc.get('url', '')}"
            if url not in seen_urls:
                all_cloudcasts.append(cc)
                seen_urls.add(url)  # Dedupe
        
        if len(all_cloudcasts) >= max_mixes * 2:
            break
    
    log(f"Found {len(all_cloudcasts)} candidate mixes")
    
    # Process each
    successful = 0
    for cc in tqdm(all_cloudcasts, desc="Processing"):
        if successful >= max_mixes:
            break
        
        key = cc.get("key", "")
        url = cc.get("url", f"https://www.mixcloud.com{key}")
        
        # Get detailed info with tracklist
        detailed = get_cloudcast_details(key)
        if not detailed:
            continue
        
        data = parse_mixcloud_data(detailed)
        
        # Note: Mixcloud often doesn't expose tracklist in API
        # We still collect metadata for potential manual tracklist addition
        # or tracklist scraping from other sources
        
        mix_id = generate_mix_id(data)
        data["mix_id"] = mix_id
        
        # Download audio (Mixcloud audio is valuable even without tracklist)
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
        successful += 1
        
        track_count = len(data.get("tracklist", []))
        log(f"Scraped: {data['dj_name']} - {data['mix_title']} ({track_count} tracks, {data.get('duration_seconds', 0)/60:.0f} min)")
    
    # Summary
    with_audio = sum(1 for m in collected_mixes if m.get("audio_file"))
    log("=" * 60)
    log(f"Mixcloud scraping complete: {len(collected_mixes)} mixes")
    log(f"  With audio: {with_audio}")
    log(f"  Metadata only: {len(collected_mixes) - with_audio}")
    
    return collected_mixes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape Mixcloud")
    parser.add_argument("--max", type=int, default=50, help="Max mixes")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio download")
    args = parser.parse_args()
    
    scrape_mixcloud(max_mixes=args.max, download_audio_flag=not args.no_audio)
