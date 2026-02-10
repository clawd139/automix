#!/usr/bin/env python3
"""
Scraper for 1001tracklists.com DJ mixes with tracklists.

Respects robots.txt crawl-delay of 8 seconds.
Saves metadata even if audio download fails.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
BASE_URL = "https://www.1001tracklists.com"
DATA_DIR = Path(__file__).parent
MIXES_DIR = DATA_DIR / "mixes"
METADATA_DIR = DATA_DIR / "metadata" / "1001tracklists"
CRAWL_DELAY = 10  # Be extra respectful (robots.txt says 8)

# Browser-like headers (they block CPython user-agent)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def log(message: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(DATA_DIR / "PIPELINE_LOG.md", "a") as f:
        f.write(log_line + "\n")


def fetch_page(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """Fetch a page with rate limiting."""
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        time.sleep(CRAWL_DELAY)
        return BeautifulSoup(response.text, "lxml")
    except Exception as e:
        log(f"Error fetching {url}: {e}")
        return None


def parse_tracklist_page(soup: BeautifulSoup, url: str) -> dict:
    """Extract tracklist data from a mix page."""
    data = {
        "source_url": url,
        "scraped_at": datetime.now().isoformat(),
        "dj_name": None,
        "mix_title": None,
        "date": None,
        "genres": [],
        "tracklist": [],
        "audio_links": [],
    }

    # Extract title
    title_elem = soup.select_one("h1#pageTitle, h1.tlTitle")
    if title_elem:
        data["mix_title"] = title_elem.get_text(strip=True)

    # Extract DJ name from meta or title
    meta_dj = soup.select_one('meta[property="og:title"]')
    if meta_dj:
        content = meta_dj.get("content", "")
        if " @ " in content:
            data["dj_name"] = content.split(" @ ")[0].strip()
        elif " - " in content:
            data["dj_name"] = content.split(" - ")[0].strip()

    # Extract date
    date_elem = soup.select_one(".tl_date, time.publishedDate")
    if date_elem:
        data["date"] = date_elem.get_text(strip=True)

    # Extract genres/tags
    genre_elems = soup.select(".genre a, .styles a, a[href*='/genre/']")
    data["genres"] = list(set(g.get_text(strip=True) for g in genre_elems))

    # Extract tracklist with timestamps
    track_rows = soup.select(".tlpItem, .tlpTog, tr.tlpItem")
    
    for i, row in enumerate(track_rows):
        track = {
            "position": i + 1,
            "artist": None,
            "title": None,
            "label": None,
            "timestamp": None,
            "timestamp_seconds": None,
        }

        # Try different selectors for artist/title
        artist_elem = row.select_one(".trackValue a[href*='/artist/'], .artistval a")
        title_elem = row.select_one(".trackValue a[href*='/track/'], .trackval a")
        
        if artist_elem:
            track["artist"] = artist_elem.get_text(strip=True)
        if title_elem:
            track["title"] = title_elem.get_text(strip=True)
        
        # Fallback: try combined track text
        if not track["artist"] and not track["title"]:
            track_text = row.select_one(".trackValue, .tlTrack")
            if track_text:
                text = track_text.get_text(strip=True)
                if " - " in text:
                    parts = text.split(" - ", 1)
                    track["artist"] = parts[0].strip()
                    track["title"] = parts[1].strip()

        # Extract timestamp
        time_elem = row.select_one(".cueValue, .cue, .tlpCue, [class*='cue']")
        if time_elem:
            time_text = time_elem.get_text(strip=True)
            track["timestamp"] = time_text
            track["timestamp_seconds"] = parse_timestamp(time_text)

        # Extract label
        label_elem = row.select_one("a[href*='/label/']")
        if label_elem:
            track["label"] = label_elem.get_text(strip=True)

        if track["artist"] or track["title"]:
            data["tracklist"].append(track)

    # Extract audio links (SoundCloud, Mixcloud, YouTube)
    audio_patterns = [
        r"soundcloud\.com",
        r"mixcloud\.com", 
        r"youtube\.com",
        r"youtu\.be",
    ]
    
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        for pattern in audio_patterns:
            if re.search(pattern, href, re.I):
                data["audio_links"].append(href)
                break

    # Also check iframes for embeds
    for iframe in soup.select("iframe[src]"):
        src = iframe.get("src", "")
        for pattern in audio_patterns:
            if re.search(pattern, src, re.I):
                # Extract actual URL from embed
                if "soundcloud" in src:
                    match = re.search(r"url=([^&]+)", src)
                    if match:
                        from urllib.parse import unquote
                        data["audio_links"].append(unquote(match.group(1)))
                else:
                    data["audio_links"].append(src)
                break

    data["audio_links"] = list(set(data["audio_links"]))
    return data


def parse_timestamp(time_str: str) -> Optional[int]:
    """Convert timestamp string to seconds."""
    if not time_str:
        return None
    
    time_str = time_str.strip()
    parts = time_str.split(":")
    
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
    except ValueError:
        pass
    return None


def download_audio(url: str, output_dir: Path, mix_id: str) -> Optional[str]:
    """Download audio using yt-dlp."""
    output_path = output_dir / f"{mix_id}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", str(output_path),
        "--no-playlist",
        url,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Find the downloaded file
            for ext in [".mp3", ".m4a", ".opus", ".webm"]:
                potential = output_dir / f"{mix_id}{ext}"
                if potential.exists():
                    return str(potential)
            # Check with pattern
            for f in output_dir.glob(f"{mix_id}.*"):
                if f.suffix in [".mp3", ".m4a", ".opus", ".webm", ".wav"]:
                    return str(f)
        else:
            log(f"yt-dlp error for {url}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log(f"Download timeout for {url}")
    except Exception as e:
        log(f"Download error for {url}: {e}")
    
    return None


def get_genre_list_urls() -> list[str]:
    """Get URLs for genre listing pages to find mixes."""
    return [
        f"{BASE_URL}/genre/1/techno.html",
        f"{BASE_URL}/genre/4/house.html",
        f"{BASE_URL}/genre/2/tech-house.html",
        f"{BASE_URL}/genre/3/deep-house.html",
        f"{BASE_URL}/genre/11/trance.html",
        f"{BASE_URL}/genre/12/progressive-house.html",
        f"{BASE_URL}/genre/7/drum-and-bass.html",
        f"{BASE_URL}/genre/17/dubstep.html",
        f"{BASE_URL}/genre/5/electro-house.html",
        f"{BASE_URL}/genre/15/minimal.html",
    ]


def scrape_mix_urls_from_page(soup: BeautifulSoup) -> list[str]:
    """Extract mix URLs from a listing page."""
    urls = []
    for link in soup.select("a[href*='/tracklist/']"):
        href = link.get("href", "")
        if "/tracklist/" in href and href not in urls:
            full_url = urljoin(BASE_URL, href)
            urls.append(full_url)
    return urls


def generate_mix_id(data: dict) -> str:
    """Generate a unique ID for a mix."""
    parts = []
    if data.get("dj_name"):
        parts.append(re.sub(r"[^\w]", "", data["dj_name"].lower())[:20])
    if data.get("date"):
        parts.append(re.sub(r"[^\d]", "", data["date"])[:8])
    if data.get("mix_title"):
        parts.append(re.sub(r"[^\w]", "", data["mix_title"].lower())[:20])
    
    # Add hash of URL for uniqueness
    import hashlib
    url_hash = hashlib.md5(data["source_url"].encode()).hexdigest()[:8]
    parts.append(url_hash)
    
    return "_".join(filter(None, parts)) or url_hash


def scrape_1001tracklists(max_mixes: int = 100, download_audio_flag: bool = True):
    """Main scraping function."""
    MIXES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting 1001tracklists scraper")
    log(f"Target: {max_mixes} mixes")
    
    session = requests.Session()
    collected_mixes = []
    seen_urls = set()
    
    # Load existing metadata to avoid re-scraping
    for f in METADATA_DIR.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                seen_urls.add(data.get("source_url"))
        except:
            pass
    
    log(f"Already have {len(seen_urls)} mixes scraped")
    
    # Collect mix URLs from genre pages
    mix_urls = []
    genre_urls = get_genre_list_urls()
    
    for genre_url in tqdm(genre_urls, desc="Scanning genres"):
        soup = fetch_page(genre_url, session)
        if soup:
            new_urls = scrape_mix_urls_from_page(soup)
            mix_urls.extend([u for u in new_urls if u not in seen_urls])
            log(f"Found {len(new_urls)} mixes from {genre_url}")
        
        if len(mix_urls) >= max_mixes * 2:  # Get extra in case some fail
            break
    
    mix_urls = list(set(mix_urls))[:max_mixes * 2]
    log(f"Total unique mix URLs to process: {len(mix_urls)}")
    
    # Process each mix
    successful = 0
    for url in tqdm(mix_urls, desc="Scraping mixes"):
        if successful >= max_mixes:
            break
        
        if url in seen_urls:
            continue
        
        soup = fetch_page(url, session)
        if not soup:
            continue
        
        data = parse_tracklist_page(soup, url)
        
        # Skip if no tracklist found
        if len(data["tracklist"]) < 5:
            log(f"Skipping {url} - only {len(data['tracklist'])} tracks")
            continue
        
        # Skip if no timestamps
        tracks_with_timestamps = sum(1 for t in data["tracklist"] if t["timestamp_seconds"])
        if tracks_with_timestamps < len(data["tracklist"]) * 0.5:
            log(f"Skipping {url} - only {tracks_with_timestamps}/{len(data['tracklist'])} tracks have timestamps")
            continue
        
        mix_id = generate_mix_id(data)
        data["mix_id"] = mix_id
        
        # Try to download audio
        if download_audio_flag and data["audio_links"]:
            for audio_url in data["audio_links"]:
                log(f"Attempting download: {audio_url}")
                audio_path = download_audio(audio_url, MIXES_DIR, mix_id)
                if audio_path:
                    data["audio_file"] = audio_path
                    log(f"Downloaded: {audio_path}")
                    break
        
        # Save metadata (even without audio)
        metadata_path = METADATA_DIR / f"{mix_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
        
        collected_mixes.append(data)
        seen_urls.add(url)
        successful += 1
        
        log(f"Scraped: {data['dj_name']} - {data['mix_title']} ({len(data['tracklist'])} tracks)")
    
    # Summary
    with_audio = sum(1 for m in collected_mixes if m.get("audio_file"))
    log("=" * 60)
    log(f"Scraping complete: {len(collected_mixes)} mixes")
    log(f"  With audio: {with_audio}")
    log(f"  Metadata only: {len(collected_mixes) - with_audio}")
    
    return collected_mixes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape 1001tracklists.com")
    parser.add_argument("--max", type=int, default=50, help="Max mixes to scrape")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio download")
    args = parser.parse_args()
    
    scrape_1001tracklists(max_mixes=args.max, download_audio_flag=not args.no_audio)
