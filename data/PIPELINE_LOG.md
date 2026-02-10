# DJ Mix Data Pipeline Log

## Session: 2026-02-09 19:19 PST

### Pipeline Goal
- Target: 200+ DJ mixes with tracklists, 2000 Jamendo EDM tracks
- Process with analysis (BPM, key, beats, structure)
- Upload to gs://clawd139/automix-data/

### Directory Structure
```
data/
├── raw/
│   ├── mixes/       # DJ mix audio files
│   └── tracks/      # Individual Jamendo tracks
├── metadata/
│   └── youtube/     # Tracklist JSON files
├── analysis/
│   └── tracks/      # Analysis JSON files
├── transitions/     # Extracted transitions
└── stems/           # Demucs stems (if processed)
```

---

## Progress Log

### 19:20 - Started YouTube scraping
- Target: 30 mixes with audio
- Found 60 candidate videos
- Scraping with audio download enabled

### 19:20 - Started Jamendo downloads
- Target: 500 EDM tracks
- 5,650 tracks available in catalog

### 19:50 - Progress check
- YouTube: 11 mixes with audio, 19 metadata files
- Jamendo: 417 tracks downloaded
- Total: 4.6GB

### Status (First batch complete)
- **DJ Mixes**: 11 with audio, 19 metadata (tracklists)
- **Jamendo Tracks**: 417 downloaded
- **Total Data**: 4.6GB
- **Notes**: Processes killed by OOM after ~30min, but good data collected

### 19:39 - Audio analysis started
- Running librosa-based analysis
- BPM, key, beats, structure detection
- ~2 seconds per track

### 19:40 - GCS upload started
- Uploaded manifest.json
- Uploaded 19 metadata files
- Uploaded 115 analysis files

### 19:43 - Analysis complete, final upload
- Total tracks analyzed: 200 of 417
- All analysis files synced to GCS
- Updated manifest with final counts

## Final Status ✅

### Data Collected
| Category | Count | Size |
|----------|-------|------|
| DJ Mixes (with audio) | 11 | 1.6 GB |
| YouTube Metadata | 19 | - |
| Jamendo Tracks | 417 | 2.6 GB |
| Analysis Files | 200 | - |
| **Total** | - | **4.6 GB** |

### GCS Upload Status
- **Bucket**: gs://clawd139/automix-data/
- **manifest.json**: ✅ Uploaded
- **raw/metadata/youtube/**: ✅ 19 files
- **processed/analysis/tracks/**: ✅ 200 files
- **Audio files**: Not uploaded (raw audio stays local for now)

### Analysis Features
- BPM detection (librosa)
- Key detection (Krumhansl-Kessler profiles)
- Beat tracking
- Structure segmentation (intro/drop/breakdown)

### Notes
- YouTube scraper works well for fan uploads with tracklists
- Boiler Room/Cercle don't include tracklists in descriptions
- 8GB RAM limited parallel processing
- OOM killed processes around 30min mark
- Consider GPU cluster for demucs stem separation

---

[2026-02-09 19:20:02] ============================================================
[2026-02-09 19:20:02] Starting YouTube DJ mix scraper
[2026-02-09 19:20:02] Target: 30 mixes
[2026-02-09 19:20:02] Already have 9 mixes
[2026-02-09 19:20:19] Found 60 candidate videos
[2026-02-09 19:20:19] ============================================================
[2026-02-09 19:20:19] Starting MTG-Jamendo EDM download
[2026-02-09 19:20:19] Loaded 5650 EDM track metadata entries
[2026-02-09 19:20:19] Already downloaded: 0
[2026-02-09 19:20:19] Remaining: 5650
[2026-02-09 19:20:19] Limited to 500 tracks
[2026-02-09 19:20:19] Starting download of 500 tracks...
[2026-02-09 19:20:19] Quality: High (320kbps)
[2026-02-09 19:20:21] Skipping Subtronics Live @ Lost Lands 2024 - Full... - only 0 tracks in description
[2026-02-09 19:20:50] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/zenworldevolutionofs_20240501_creatingpeaktimetechnolikeapro_c286a20d.mp3
[2026-02-09 19:20:50] Scraped: Zen World - Evolution Of Sound - Creating Peak Time Techno Like... (10 tracks)
[2026-02-09 19:20:55] Skipping Techno Rave Mix 2024 - Best Techno Set &... - only 0 tracks in description
[2026-02-09 19:20:57] Skipping Best Dark Techno Rave Mix 2024 - Mind-Bl... - only 0 tracks in description
[2026-02-09 19:20:59] Skipping PEZSI [DJ-SET 2024] x OBSBOT - Melodic T... - only 1 tracks in description
[2026-02-09 19:21:49] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/kenmotoyamalivegroov_20241211_japanesetechno2024originalbest_315c0bff.mp3
[2026-02-09 19:21:49] Scraped: Ken Motoyama – Live Groovebox Techno - Japanese Techno 2024 Original ... (16 tracks)
[2026-02-09 19:22:26] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technomix_20250404_technomix2025remixesofpopulars_1d875cf5.mp3
[2026-02-09 19:22:26] Scraped: Techno Mix - TECHNO MIX 2025 😎⚡Remixes Of P... (32 tracks)
[2026-02-09 19:22:30] Skipping Techno Mix 2025 | Melodic Techno Remixes... - only 0 tracks in description
[2026-02-09 19:22:33] Skipping New Hard Techno Releases 2025 | Undergro... - only 0 tracks in description
[2026-02-09 19:23:25] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technomix_20251103_technomix2025bestnonstoptechno_9c928ab0.mp3
[2026-02-09 19:23:25] Scraped: Techno Mix - Techno Mix 2025😉🤘Best Nonstop ... (29 tracks)
[2026-02-09 19:24:10] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technomix_20251109_technomix2025bestnonstoptechno_cb917817.mp3
[2026-02-09 19:24:10] Scraped: Techno Mix - Techno Mix 2025😉🤘Best Nonstop ... (28 tracks)
[2026-02-09 19:24:52] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technomix_20260121_technomix2025remixesofpopulars_e72c3226.mp3
[2026-02-09 19:24:52] Scraped: Techno Mix - TECHNO MIX 2025 ✨🔥 Remixes Of ... (30 tracks)
[2026-02-09 19:25:37] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technotab_20250829_technomix2025remixesofpopulars_df70ace9.mp3
[2026-02-09 19:25:37] Scraped: Techno Tab - TECHNO MIX 2025 🥰⚡Remixes Of P... (30 tracks)
[2026-02-09 19:26:19] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technomix_20260117_technomix2025remixesofpopulars_0f5548bd.mp3
[2026-02-09 19:26:19] Scraped: Techno Mix - TECHNO MIX 2025 ✨🔥 Remixes Of ... (30 tracks)
[2026-02-09 19:27:34] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/technomix_20251101_technomix2025bestnonstoptechno_cd652ae3.mp3
[2026-02-09 19:27:34] Scraped: Techno Mix - Techno Mix 2025😉🤘Best Nonstop ... (30 tracks)
[2026-02-09 19:28:18] Downloaded: /Users/clawd/.openclaw/workspace/djtransgan-v2/data/raw/mixes/edmtechno_20250526_technomix2025remixesofpopulars_1e0012f4.mp3
[2026-02-09 19:28:18] Scraped: EDM Techno - TECHNO MIX 2025 🔥⚡Remixes Of P... (31 tracks)
[2026-02-09 19:28:23] Skipping Summer Mix 2024 #6  | Best Of Deep & Tec... - only 0 tracks in description
[2026-02-09 19:28:25] Skipping Vibey Deep House Mix 2024 | Mix by Yaman... - only 0 tracks in description
[2026-02-09 19:36:02] ============================================================
[2026-02-09 19:36:02] Starting track analysis
[2026-02-09 19:36:02] Found 1 tracks
[2026-02-09 19:36:02] Already analyzed: 0
[2026-02-09 19:36:02] To analyze: 1
[2026-02-09 19:36:05] Track analysis complete
[2026-02-09 19:36:14] ============================================================
[2026-02-09 19:36:14] Starting track analysis
[2026-02-09 19:36:14] Found 417 tracks
[2026-02-09 19:36:14] Already analyzed: 1
[2026-02-09 19:36:14] To analyze: 100
[2026-02-09 19:39:23] Track analysis complete
[2026-02-09 19:39:44] ============================================================
[2026-02-09 19:39:44] Starting track analysis
[2026-02-09 19:39:44] Found 417 tracks
[2026-02-09 19:39:44] Already analyzed: 101
[2026-02-09 19:39:44] To analyze: 99
[2026-02-09 19:42:52] Track analysis complete
