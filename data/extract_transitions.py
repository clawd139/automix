#!/usr/bin/env python3
"""
Extract DJ transitions from mixes with tracklists.

For each transition point:
1. Run demucs to separate stems (optional)
2. Extract transition segment (configurable window around switch point)
3. Extract clean portions of tracks before/after transition
4. Save structured dataset
"""

import json
import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None
    print("Warning: soundfile not installed, using pydub fallback")

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Configuration
DATA_DIR = Path(__file__).parent
MIXES_DIR = DATA_DIR / "mixes"
METADATA_DIR = DATA_DIR / "metadata"
TRANSITIONS_DIR = DATA_DIR / "transitions"
STEMS_DIR = DATA_DIR / "stems"

# Transition extraction parameters
TRANSITION_WINDOW = 120  # Seconds around transition point
CLEAN_SEGMENT_MIN = 30  # Minimum seconds for "clean" track segment
SAMPLE_RATE = 44100


def log(message: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(DATA_DIR / "PIPELINE_LOG.md", "a") as f:
        f.write(log_line + "\n")


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, return (samples, sample_rate)."""
    if sf:
        data, sr = sf.read(str(path))
        return data, sr
    elif AudioSegment:
        audio = AudioSegment.from_file(str(path))
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        return samples / 32768.0, sr  # Normalize to [-1, 1]
    else:
        raise ImportError("Need either soundfile or pydub installed")


def save_audio(path: Path, data: np.ndarray, sample_rate: int):
    """Save audio to file."""
    if sf:
        sf.write(str(path), data, sample_rate)
    elif AudioSegment:
        # Convert back to int16
        data_int = (data * 32767).astype(np.int16)
        if len(data.shape) == 2:
            channels = 2
        else:
            channels = 1
            data_int = data_int.reshape(-1)
        
        audio = AudioSegment(
            data_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=channels,
        )
        audio.export(str(path), format="mp3")
    else:
        raise ImportError("Need either soundfile or pydub installed")


def run_demucs(audio_path: Path, output_dir: Path) -> Optional[Path]:
    """Run demucs stem separation."""
    cmd = [
        "demucs",
        "-n", "htdemucs",  # Best quality model
        "--two-stems", "vocals",  # Just vocals + other (faster)
        "-o", str(output_dir),
        str(audio_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            # Find output directory
            stem_name = audio_path.stem
            potential_dirs = [
                output_dir / "htdemucs" / stem_name,
                output_dir / "htdemucs_ft" / stem_name,
            ]
            for d in potential_dirs:
                if d.exists():
                    return d
        else:
            log(f"Demucs error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log("Demucs timeout (30 min limit)")
    except FileNotFoundError:
        log("Demucs not installed. Install with: pip install demucs")
    except Exception as e:
        log(f"Demucs error: {e}")
    
    return None


def extract_segment(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Extract a segment from audio array."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    return audio[start_sample:end_sample]


def calculate_transition_windows(tracklist: list[dict], total_duration: float) -> list[dict]:
    """Calculate extraction windows for each transition."""
    transitions = []
    
    for i in range(len(tracklist) - 1):
        track_a = tracklist[i]
        track_b = tracklist[i + 1]
        
        # Get timestamps
        a_start = track_a.get("timestamp_seconds", 0) or 0
        b_start = track_b.get("timestamp_seconds")
        
        if b_start is None:
            continue
        
        # Transition point is where track B starts
        transition_point = b_start
        
        # Track A's clean segment: from its start to (transition - TRANSITION_WINDOW/2)
        a_clean_start = a_start
        a_clean_end = max(a_start, transition_point - TRANSITION_WINDOW / 2)
        
        # Track B's clean segment: from (transition + TRANSITION_WINDOW/2) to next transition (or end)
        b_clean_start = transition_point + TRANSITION_WINDOW / 2
        if i + 2 < len(tracklist) and tracklist[i + 2].get("timestamp_seconds"):
            b_clean_end = tracklist[i + 2]["timestamp_seconds"] - TRANSITION_WINDOW / 2
        else:
            b_clean_end = total_duration
        
        # Transition segment
        trans_start = max(0, transition_point - TRANSITION_WINDOW / 2)
        trans_end = min(total_duration, transition_point + TRANSITION_WINDOW / 2)
        
        transition = {
            "index": i,
            "track_a": {
                "artist": track_a.get("artist"),
                "title": track_a.get("title"),
                "position": track_a.get("position"),
            },
            "track_b": {
                "artist": track_b.get("artist"),
                "title": track_b.get("title"),
                "position": track_b.get("position"),
            },
            "transition_point": transition_point,
            "windows": {
                "a_clean": (a_clean_start, a_clean_end),
                "transition": (trans_start, trans_end),
                "b_clean": (b_clean_start, b_clean_end),
            },
        }
        
        # Only include if clean segments are long enough
        if (a_clean_end - a_clean_start >= CLEAN_SEGMENT_MIN and 
            b_clean_end - b_clean_start >= CLEAN_SEGMENT_MIN):
            transitions.append(transition)
    
    return transitions


def process_mix(metadata_path: Path, use_demucs: bool = False) -> list[dict]:
    """Process a single mix and extract transitions."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    mix_id = metadata.get("mix_id", metadata_path.stem)
    audio_path = metadata.get("audio_file")
    
    if not audio_path or not Path(audio_path).exists():
        log(f"No audio file for {mix_id}")
        return []
    
    audio_path = Path(audio_path)
    tracklist = metadata.get("tracklist", [])
    
    if len(tracklist) < 3:
        log(f"Tracklist too short for {mix_id}")
        return []
    
    # Load audio
    log(f"Loading {audio_path.name}...")
    try:
        audio, sr = load_audio(audio_path)
    except Exception as e:
        log(f"Error loading audio: {e}")
        return []
    
    total_duration = len(audio) / sr
    log(f"Audio loaded: {total_duration/60:.1f} minutes, {sr}Hz")
    
    # Calculate transition windows
    transitions = calculate_transition_windows(tracklist, total_duration)
    log(f"Found {len(transitions)} extractable transitions")
    
    if not transitions:
        return []
    
    # Create output directory for this mix
    mix_output_dir = TRANSITIONS_DIR / mix_id
    mix_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optionally run demucs
    stems_audio = None
    if use_demucs:
        stems_dir = run_demucs(audio_path, STEMS_DIR)
        if stems_dir:
            # Load the "no_vocals" stem (instrumental)
            no_vocals_path = stems_dir / "no_vocals.wav"
            if no_vocals_path.exists():
                stems_audio, _ = load_audio(no_vocals_path)
                log("Using demucs instrumental stem")
    
    # Use stems if available, otherwise original
    working_audio = stems_audio if stems_audio is not None else audio
    
    # Extract each transition
    extracted = []
    for trans in transitions:
        trans_id = f"{mix_id}_trans_{trans['index']:03d}"
        trans_dir = mix_output_dir / trans_id
        trans_dir.mkdir(exist_ok=True)
        
        windows = trans["windows"]
        
        try:
            # Extract segments
            a_clean = extract_segment(working_audio, sr, *windows["a_clean"])
            transition_seg = extract_segment(working_audio, sr, *windows["transition"])
            b_clean = extract_segment(working_audio, sr, *windows["b_clean"])
            
            # Save audio files
            save_audio(trans_dir / "track_a_clean.mp3", a_clean, sr)
            save_audio(trans_dir / "transition.mp3", transition_seg, sr)
            save_audio(trans_dir / "track_b_clean.mp3", b_clean, sr)
            
            # Save metadata
            trans_metadata = {
                "transition_id": trans_id,
                "source_mix": mix_id,
                "source_metadata": str(metadata_path),
                **trans,
                "files": {
                    "track_a_clean": str(trans_dir / "track_a_clean.mp3"),
                    "transition": str(trans_dir / "transition.mp3"),
                    "track_b_clean": str(trans_dir / "track_b_clean.mp3"),
                },
                "durations": {
                    "a_clean": windows["a_clean"][1] - windows["a_clean"][0],
                    "transition": windows["transition"][1] - windows["transition"][0],
                    "b_clean": windows["b_clean"][1] - windows["b_clean"][0],
                },
            }
            
            with open(trans_dir / "metadata.json", "w") as f:
                json.dump(trans_metadata, f, indent=2)
            
            extracted.append(trans_metadata)
            log(f"Extracted: {trans_id}")
            
        except Exception as e:
            log(f"Error extracting {trans_id}: {e}")
    
    return extracted


def extract_all_transitions(use_demucs: bool = False, max_mixes: int = None):
    """Process all mixes and extract transitions."""
    TRANSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting transition extraction")
    
    # Find all metadata files
    metadata_files = []
    for subdir in METADATA_DIR.iterdir():
        if subdir.is_dir():
            metadata_files.extend(subdir.glob("*.json"))
    
    log(f"Found {len(metadata_files)} mix metadata files")
    
    if max_mixes:
        metadata_files = metadata_files[:max_mixes]
    
    all_transitions = []
    
    for metadata_path in tqdm(metadata_files, desc="Processing mixes"):
        transitions = process_mix(metadata_path, use_demucs=use_demucs)
        all_transitions.extend(transitions)
    
    # Save master index
    index_path = TRANSITIONS_DIR / "transitions_index.json"
    with open(index_path, "w") as f:
        json.dump({
            "total_transitions": len(all_transitions),
            "extracted_at": datetime.now().isoformat(),
            "transitions": all_transitions,
        }, f, indent=2)
    
    log("=" * 60)
    log(f"Extraction complete: {len(all_transitions)} transitions")
    log(f"Index saved to: {index_path}")
    
    return all_transitions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DJ transitions")
    parser.add_argument("--demucs", action="store_true", help="Use demucs for stem separation")
    parser.add_argument("--max", type=int, default=None, help="Max mixes to process")
    parser.add_argument("--single", type=str, help="Process single metadata file")
    args = parser.parse_args()
    
    if args.single:
        transitions = process_mix(Path(args.single), use_demucs=args.demucs)
        print(f"Extracted {len(transitions)} transitions")
    else:
        extract_all_transitions(use_demucs=args.demucs, max_mixes=args.max)
