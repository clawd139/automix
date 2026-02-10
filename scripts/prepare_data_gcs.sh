#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# prepare_data_gcs.sh — Run on a GCP VM to collect, process, and upload
# training data to GCS for the automix training cluster.
#
# Usage:
#   GCS_BUCKET=gs://your-bucket ./scripts/prepare_data_gcs.sh
#
# Prerequisites:
#   - Google Cloud VM with gcloud authenticated
#   - gsutil available
#   - At least 200GB disk (SSD recommended)
#   - 8+ CPU cores recommended (demucs + analysis are CPU-heavy)
#   - GPU optional but speeds up demucs significantly
###############################################################################

GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET=gs://your-bucket}"
WORKDIR="${WORKDIR:-/tmp/automix-data}"
MAX_MIXES="${MAX_MIXES:-500}"
MAX_JAMENDO="${MAX_JAMENDO:-5000}"
MAX_PAIRS="${MAX_PAIRS:-10000}"
N_WORKERS="${N_WORKERS:-$(nproc)}"

echo "=== AutoMix Data Pipeline ==="
echo "Bucket:     $GCS_BUCKET"
echo "Workdir:    $WORKDIR"
echo "Max mixes:  $MAX_MIXES"
echo "Max tracks: $MAX_JAMENDO"
echo "Workers:    $N_WORKERS"
echo ""

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ─── 1. Install automix ─────────────────────────────────────────────────────
echo ">>> Step 1: Installing automix..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d automix ]; then
    git clone https://github.com/clawd139/automix.git
fi
cd automix

# Verify install
uv run automix info
echo ""

# ─── 2. Install extra data pipeline deps ────────────────────────────────────
echo ">>> Step 2: Installing data pipeline dependencies..."
uv pip install yt-dlp beautifulsoup4 requests 2>/dev/null || \
    uv add yt-dlp beautifulsoup4 requests

# ─── 3. Scrape DJ mixes from YouTube ────────────────────────────────────────
echo ">>> Step 3: Scraping DJ mixes with tracklists..."
MIXES_DIR="$WORKDIR/raw/mixes"
mkdir -p "$MIXES_DIR/audio" "$MIXES_DIR/metadata"

# Search queries that reliably return DJ sets with timestamps
QUERIES=(
    "techno DJ mix tracklist timestamps 2024"
    "house DJ mix tracklist timestamps 2024"
    "drum and bass mix tracklist timestamps"
    "EDM DJ set tracklist timestamps"
    "deep house mix timestamps 2024"
    "minimal techno mix timestamps"
    "progressive house mix tracklist"
    "trance DJ set timestamps 2024"
    "UK garage mix tracklist timestamps"
    "jungle drum and bass mix timestamps"
)

for query in "${QUERIES[@]}"; do
    echo "  Searching: $query"
    uv run python data/scrape_youtube.py \
        --query "$query" \
        --max "$((MAX_MIXES / ${#QUERIES[@]}))" \
        --output "$MIXES_DIR" \
        2>&1 | tail -1 || true
    sleep 2
done

MIX_COUNT=$(find "$MIXES_DIR/audio" -name "*.mp3" -o -name "*.m4a" -o -name "*.opus" | wc -l | tr -d ' ')
echo "  Downloaded $MIX_COUNT mixes"

# ─── 4. Download MTG-Jamendo EDM tracks ─────────────────────────────────────
echo ">>> Step 4: Downloading MTG-Jamendo EDM tracks..."
TRACKS_DIR="$WORKDIR/raw/tracks"
mkdir -p "$TRACKS_DIR"

uv run python data/download_jamendo.py \
    --max "$MAX_JAMENDO" \
    --output "$TRACKS_DIR" \
    --workers "$N_WORKERS" \
    2>&1 | tail -5

TRACK_COUNT=$(find "$TRACKS_DIR" -name "*.mp3" | wc -l | tr -d ' ')
echo "  Downloaded $TRACK_COUNT tracks"

# ─── 5. Process everything: stems + analysis ────────────────────────────────
echo ">>> Step 5: Processing data (demucs stems + analysis)..."
PROCESSED_DIR="$WORKDIR/processed"
mkdir -p "$PROCESSED_DIR"

# Process individual tracks
echo "  Processing individual tracks..."
uv run automix prepare \
    --tracks "$TRACKS_DIR" \
    --output "$PROCESSED_DIR/tracks" \
    --max "$MAX_JAMENDO" \
    --workers "$N_WORKERS" \
    2>&1 | tail -5

# Extract and process transitions from mixes
echo "  Extracting transitions from mixes..."
uv run python data/extract_transitions.py \
    --mixes "$MIXES_DIR" \
    --output "$PROCESSED_DIR/transitions" \
    --workers "$N_WORKERS" \
    2>&1 | tail -5

# ─── 6. Generate training pairs ─────────────────────────────────────────────
echo ">>> Step 6: Generating training pairs..."
uv run python data/create_pairs.py \
    --tracks "$PROCESSED_DIR/tracks" \
    --transitions "$PROCESSED_DIR/transitions" \
    --output "$PROCESSED_DIR/pairs" \
    --max "$MAX_PAIRS" \
    2>&1 | tail -5 || echo "  (create_pairs.py not found, using prepare output directly)"

# ─── 7. Upload to GCS ───────────────────────────────────────────────────────
echo ">>> Step 7: Uploading to GCS..."
PAIR_COUNT=$(find "$PROCESSED_DIR" -name "analysis.json" 2>/dev/null | wc -l | tr -d ' ')
echo "  Processed pairs/tracks: $PAIR_COUNT"

# Create a manifest
cat > "$PROCESSED_DIR/manifest.json" <<EOF
{
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "mixes_scraped": $MIX_COUNT,
    "tracks_downloaded": $TRACK_COUNT,
    "processed_items": $PAIR_COUNT,
    "source": "youtube+jamendo",
    "version": "2.0.0"
}
EOF

echo "  Uploading processed data..."
gsutil -m rsync -r "$PROCESSED_DIR" "$GCS_BUCKET/automix-data/processed/"

echo "  Uploading raw metadata (no audio)..."
gsutil -m rsync -r "$MIXES_DIR/metadata" "$GCS_BUCKET/automix-data/raw/metadata/"

echo ""
echo "=== Done ==="
echo "Data uploaded to: $GCS_BUCKET/automix-data/"
echo ""
echo "To train:"
echo "  gsutil -m rsync -r $GCS_BUCKET/automix-data/processed/ /data/processed/"
echo "  cd automix && uv run automix train --data /data/processed --steps 100000 --batch-size 32"
