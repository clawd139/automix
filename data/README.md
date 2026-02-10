# DJ Mix Data Pipeline

Complete pipeline for collecting DJ transitions for training DJTransGAN.

## Overview

```
data/
├── mixes/           # Downloaded DJ mix audio files
├── tracks/          # Individual tracks (MTG-Jamendo EDM)
├── transitions/     # Extracted transition segments
├── metadata/        # Scraped tracklist metadata
│   ├── 1001tracklists/
│   └── mixcloud/
├── analysis/        # Beat/BPM/key analysis results
├── stems/           # Demucs stem separations (optional)
│
├── scrape_1001tracklists.py
├── scrape_mixcloud.py
├── scrape_youtube.py
├── download_jamendo.py
├── extract_transitions.py
├── analyze_tracks.py
├── requirements.txt
├── PIPELINE_LOG.md  # Progress log
└── README.md
```

## Setup

```bash
cd /Users/clawd/.openclaw/workspace/djtransgan-v2/data

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install demucs for stem separation
pip install demucs

# Optional: Install essentia for better key detection
pip install essentia-tensorflow
```

**Note:** yt-dlp must be installed system-wide for audio downloading.

## Pipeline Steps

### 1. Scrape DJ Mix Metadata + Audio

#### 1001tracklists.com
```bash
# Scrape 50 mixes (metadata + audio)
python scrape_1001tracklists.py --max 50

# Metadata only (faster, retry audio later)
python scrape_1001tracklists.py --max 100 --no-audio
```

#### Mixcloud
```bash
# Scrape 50 mixes
python scrape_mixcloud.py --max 50

# Metadata only
python scrape_mixcloud.py --max 100 --no-audio
```

#### YouTube (Boiler Room, etc.)
```bash
# Scrape YouTube DJ sets with tracklists in description
python scrape_youtube.py --max 50

# Metadata only
python scrape_youtube.py --max 100 --no-audio
```

**Best source!** Many YouTube DJ sets include full tracklists with timestamps 
in video descriptions. The scraper parses these automatically.

### 2. Download MTG-Jamendo EDM Tracks

Uses metadata from `../djtransgan/data/edm_tracks.tsv` (5,650 tracks).

```bash
# Download all EDM tracks (high quality)
python download_jamendo.py

# Download limited batch
python download_jamendo.py --max 500

# Low quality (faster, smaller)
python download_jamendo.py --low-quality

# Verify existing downloads
python download_jamendo.py --verify
```

### 3. Extract Transitions

Extracts transition segments from mixes with tracklists.

```bash
# Extract from all scraped mixes
python extract_transitions.py

# With demucs stem separation (slower, better quality)
python extract_transitions.py --demucs

# Process specific mix
python extract_transitions.py --single metadata/mixcloud/some_mix.json
```

**Output per transition:**
- `track_a_clean.mp3` — Clean portion of outgoing track
- `transition.mp3` — The actual DJ transition (~2 min)
- `track_b_clean.mp3` — Clean portion of incoming track
- `metadata.json` — Track info, timestamps, durations

### 4. Analyze Audio Features

```bash
# Analyze all extracted transitions
python analyze_tracks.py --transitions

# Analyze Jamendo tracks
python analyze_tracks.py --tracks --max 500

# Analyze single file
python analyze_tracks.py --single path/to/audio.mp3
```

**Features extracted:**
- BPM (madmom if available, else librosa)
- Beat positions
- Key (essentia if available, else librosa)
- Energy-based structure (intro/drop/breakdown)

## Data Sources

### 1001tracklists.com
- Largest DJ tracklist database
- Has timestamps for track changes
- Links to SoundCloud/Mixcloud/YouTube for audio
- Crawl delay: 10 seconds (respecting robots.txt)

### Mixcloud
- Has public GraphQL API
- Many sets have tracklists with timestamps
- Audio downloadable via yt-dlp

### MTG-Jamendo
- 5,650 EDM-tagged Creative Commons tracks
- Full metadata available
- Good for individual track analysis

## Rate Limiting & Ethics

- 1001tracklists: 10 second delay between requests
- Mixcloud: 2 second delay
- Jamendo: 0.5 second delay, max 4 concurrent
- All scraped metadata saved even if audio download fails
- Browser-like User-Agent used (CPython blocked by some sites)

## Output Format

### Transition Metadata
```json
{
  "transition_id": "djname_20240101_mixname_abc123_trans_001",
  "source_mix": "djname_20240101_mixname_abc123",
  "track_a": {
    "artist": "Artist A",
    "title": "Track A",
    "position": 5
  },
  "track_b": {
    "artist": "Artist B", 
    "title": "Track B",
    "position": 6
  },
  "transition_point": 1234.5,
  "windows": {
    "a_clean": [900, 1174.5],
    "transition": [1174.5, 1294.5],
    "b_clean": [1294.5, 1500]
  },
  "durations": {
    "a_clean": 274.5,
    "transition": 120,
    "b_clean": 205.5
  }
}
```

### Analysis Output
```json
{
  "file": "path/to/audio.mp3",
  "duration": 120.5,
  "bpm": 128.0,
  "bpm_method": "madmom",
  "beat_times": [0.5, 0.97, 1.44, ...],
  "key": {
    "key": "A",
    "scale": "minor",
    "confidence": 0.85,
    "method": "essentia"
  },
  "structure": [
    {"type": "intro", "start": 0, "end": 15.5},
    {"type": "drop", "start": 15.5, "end": 45.2},
    ...
  ]
}
```

## Targets

| Dataset | Target | Notes |
|---------|--------|-------|
| DJ Mixes with tracklists | 200+ | Priority: get timestamps |
| Individual EDM tracks | 5,000+ | From MTG-Jamendo |
| Extracted transitions | 500+ | Requires good tracklists |
| Analyzed segments | All | BPM/key/beats for everything |

## Troubleshooting

### yt-dlp errors
Some platforms may block certain regions or require authentication.
Try updating: `pip install -U yt-dlp`

### Demucs memory issues
Demucs needs significant RAM for long mixes.
Try: `demucs --segment 30` for smaller chunks.

### Essentia import errors
Essentia can be tricky to install. Falls back to librosa automatically.

### Rate limiting
If getting blocked, increase delays in the scripts.

## Progress

Check `PIPELINE_LOG.md` for detailed progress and any errors.
