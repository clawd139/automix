#!/bin/bash
# Check if prepare is done, upload to GCS if so
cd /tmp/automix

# Check if process is still running
if pgrep -f "automix prepare" > /dev/null 2>&1; then
    # Still running - count progress
    PAIRS=$(ls -d data/processed/*/analysis.json 2>/dev/null | wc -l | tr -d ' ')
    echo "RUNNING: $PAIRS/514 pairs done"
    exit 0
fi

# Process finished - check if we have data
PAIRS=$(ls -d data/processed/*/analysis.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$PAIRS" -gt "0" ]; then
    echo "DONE: $PAIRS pairs. Uploading to GCS..."
    gsutil -m rsync -r data/processed/ gs://clawd139/automix-data/processed/
    echo "Upload complete"
    exit 0
else
    echo "FAILED: No pairs generated. Check prepare.log"
    exit 1
fi
