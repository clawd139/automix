# DJ Mix Data Pipeline Log

## 2026-02-09

### Pipeline Created
- Created `scrape_1001tracklists.py` — 1001tracklists.com scraper
- Created `scrape_mixcloud.py` — Mixcloud API client (REST + scraping)
- Created `scrape_youtube.py` — YouTube DJ mix scraper with tracklist parsing
- Created `download_jamendo.py` — MTG-Jamendo EDM downloader
- Created `extract_transitions.py` — Transition extraction from mixes
- Created `analyze_tracks.py` — Feature extraction (BPM/key/beats/structure)
- Created `requirements.txt` and `README.md`

### Directory Structure
```
data/
├── mixes/           # DJ mix audio
├── tracks/          # Individual tracks
├── transitions/     # Extracted transitions
├── metadata/        # Tracklist metadata
│   ├── 1001tracklists/
│   ├── mixcloud/
│   └── youtube/
├── analysis/        # Analysis results
└── stems/           # Demucs stems
```

### Components Tested ✓
1. **YouTube Scraper** — Works! Found mixes with tracklists in descriptions
   - Best source for timestamp-labeled tracklists
   - Scraped 9 mixes with 200+ track entries

2. **Jamendo Downloader** — Works! Downloaded test track successfully
   - 5,650 EDM tracks available
   - Downloaded: `tracks/track_0000773.mp3` (3.2MB)

3. **Audio Analysis** — Works! 
   - BPM detection (librosa): 99.4 BPM
   - Key detection: D minor (78.5% confidence)
   - Beat tracking: 430+ beat positions
   - Structure analysis: Detected intro/drop/breakdown segments

### Dependencies Installed
- requests, beautifulsoup4, lxml, tqdm
- librosa, soundfile, pydub, numpy, scipy
- Virtual environment: `../venv/`

### Notes
- Boiler Room/Cercle don't include tracklists in video descriptions
- Fan-uploaded mixes with "timestamps" in search work best
- Mixcloud API doesn't expose tracklist data publicly

---
[2026-02-09 18:44:17] ============================================================
[2026-02-09 18:44:17] Starting YouTube DJ mix scraper
[2026-02-09 18:44:17] Target: 5 mixes
[2026-02-09 18:44:17] Already have 0 mixes
[2026-02-09 18:44:21] Found 15 candidate videos
[2026-02-09 18:44:23] Skipping Fred again.. | Boiler Room: London... - only 0 tracks in description
[2026-02-09 18:44:26] Skipping Nightmares On Wax | Boiler Room: London... - only 0 tracks in description
[2026-02-09 18:44:28] Skipping Zack Fox | Boiler Room: New York... - only 0 tracks in description
[2026-02-09 18:44:31] Skipping Sammy Virji | Boiler Room: Denver... - only 0 tracks in description
[2026-02-09 18:44:33] Skipping Kaytranada | Boiler Room: Montreal... - only 0 tracks in description
[2026-02-09 18:44:35] Skipping Charli xcx | Boiler Room & Charli xcx Pr... - only 0 tracks in description
[2026-02-09 18:44:38] Skipping Charli xcx | Boiler Room & Charli xcx pr... - only 0 tracks in description
[2026-02-09 18:44:40] Skipping Underworld | Boiler Room: London... - only 0 tracks in description
[2026-02-09 18:44:42] Skipping ¥ØU$UK€ ¥UK1MAT$U | Boiler Room: Tokyo... - only 0 tracks in description
[2026-02-09 18:44:45] Skipping nasthug | Boiler Room Tokyo: Tohji Prese... - only 0 tracks in description
[2026-02-09 18:44:47] Skipping riria | Boiler Room: Tokyo... - only 0 tracks in description
[2026-02-09 18:44:49] Skipping Folamour | Boiler Room x Sugar Mountain ... - only 0 tracks in description
[2026-02-09 18:44:52] Skipping Solomun | Boiler Room: Tulum... - only 0 tracks in description
[2026-02-09 18:44:54] Skipping I Hate Models | Boiler Room x Teletech F... - only 0 tracks in description
[2026-02-09 18:44:56] Skipping 3ballMTY | Boiler Room SYSTEM: Mexico Ci... - only 0 tracks in description
[2026-02-09 18:44:56] ============================================================
[2026-02-09 18:44:56] YouTube scraping complete: 0 mixes
[2026-02-09 18:44:56]   With audio: 0
[2026-02-09 18:44:56]   Metadata only: 0
[2026-02-09 18:44:56]   Total tracks identified: 0
[2026-02-09 18:45:10] ============================================================
[2026-02-09 18:45:10] Starting YouTube DJ mix scraper
[2026-02-09 18:45:10] Target: 5 mixes
[2026-02-09 18:45:10] Already have 0 mixes
[2026-02-09 18:45:14] Found 15 candidate videos
[2026-02-09 18:45:17] Scraped: Techno Mix - TECHNO MIX 2025 🔥 Remixes Of P... (30 tracks)
[2026-02-09 18:45:23] Scraped: Techno Mix - TECHNO MIX 2025 🤩 Remixes Of P... (30 tracks)
[2026-02-09 18:45:29] Scraped: STiF - AMAZING TRANCE 6 🔥 Best New Tr... (12 tracks)
[2026-02-09 18:45:34] Scraped: donit - Ultimate Rave Mix 2025... (34 tracks)
[2026-02-09 18:46:07] ============================================================
[2026-02-09 18:46:07] Starting YouTube DJ mix scraper
[2026-02-09 18:46:07] Target: 5 mixes
[2026-02-09 18:46:07] Already have 4 mixes
[2026-02-09 18:46:11] Found 12 candidate videos
[2026-02-09 18:46:14] Skipping Techno Rave Mix 2024 - Best Techno Set &... - only 0 tracks in description
[2026-02-09 18:46:16] Skipping 24 HOUR TECHNO/TRANCE MIX 2020 1/2... - only 0 tracks in description
[2026-02-09 18:46:18] Skipping Best Dark Techno Rave Mix 2024 - Mind-Bl... - only 0 tracks in description
[2026-02-09 18:46:20] Scraped: Techno Mix - TECHNO MIX 2025 😎⚡Remixes Of P... (25 tracks)
[2026-02-09 18:46:26] Scraped: Tony - Melodic Techno / Progressive H... (29 tracks)
[2026-02-09 18:46:31] Scraped: STiF - AMAZING TRANCE 17 🔥 Best New T... (13 tracks)
[2026-02-09 18:46:37] Skipping PEZSI [DJ-SET 2024] x OBSBOT - Melodic T... - only 1 tracks in description
[2026-02-09 18:46:39] Scraped: ECHOMIRA - TECHNO MIX 2024 | NATURE REBOR... (12 tracks)
[2026-02-09 18:46:44] Skipping Subtronics Live @ Lost Lands 2024 - Full... - only 0 tracks in description
[2026-02-09 18:46:46] Scraped: Techno Mix - TECHNO MIX 2025 🔥⚡Remixes Of P... (33 tracks)
[2026-02-09 18:46:50] ============================================================
[2026-02-09 18:46:50] YouTube scraping complete: 5 mixes
[2026-02-09 18:46:50]   With audio: 0
[2026-02-09 18:46:50]   Metadata only: 5
[2026-02-09 18:46:50]   Total tracks identified: 112
