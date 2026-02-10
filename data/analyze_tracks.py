#!/usr/bin/env python3
"""
Audio analysis pipeline for extracted tracks and transitions.

Extracts:
- BPM estimation
- Beat/downbeat positions
- Key detection
- Energy-based structure analysis
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import librosa
except ImportError:
    librosa = None
    print("ERROR: librosa required. Install with: pip install librosa")

try:
    import madmom
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
    HAVE_MADMOM = True
except ImportError:
    HAVE_MADMOM = False
    print("Warning: madmom not available, using librosa for beat tracking")

# Try essentia for key detection
try:
    import essentia.standard as es
    HAVE_ESSENTIA = True
except ImportError:
    HAVE_ESSENTIA = False
    print("Warning: essentia not available, using librosa for key detection")

# Configuration
DATA_DIR = Path(__file__).parent
TRANSITIONS_DIR = DATA_DIR / "transitions"
TRACKS_DIR = DATA_DIR / "tracks"
ANALYSIS_DIR = DATA_DIR / "analysis"


def log(message: str):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(DATA_DIR / "PIPELINE_LOG.md", "a") as f:
        f.write(log_line + "\n")


def estimate_bpm_librosa(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Estimate BPM using librosa."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo), beat_times


def estimate_bpm_madmom(audio_path: str) -> tuple[float, np.ndarray]:
    """Estimate BPM using madmom (more accurate for EDM)."""
    # Beat tracking
    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(audio_path)
    beat_times = proc(act)
    
    # Calculate BPM from beat intervals
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval
    else:
        bpm = 120.0
    
    return bpm, beat_times


def detect_key_essentia(audio_path: str) -> dict:
    """Detect musical key using essentia."""
    audio = es.MonoLoader(filename=audio_path)()
    
    # Key detection
    key_extractor = es.KeyExtractor()
    key, scale, strength = key_extractor(audio)
    
    return {
        "key": key,
        "scale": scale,
        "confidence": float(strength),
        "method": "essentia",
    }


def detect_key_librosa(y: np.ndarray, sr: int) -> dict:
    """Detect musical key using librosa chroma features."""
    # Compute chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Average chroma across time
    chroma_avg = np.mean(chroma, axis=1)
    
    # Key profiles (Krumhansl-Kessler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_corr = -1
    best_key = 'C'
    best_scale = 'major'
    
    for i in range(12):
        rotated_chroma = np.roll(chroma_avg, -i)
        
        major_corr = np.corrcoef(rotated_chroma, major_profile)[0, 1]
        minor_corr = np.corrcoef(rotated_chroma, minor_profile)[0, 1]
        
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = key_names[i]
            best_scale = 'major'
        
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = key_names[i]
            best_scale = 'minor'
    
    return {
        "key": best_key,
        "scale": best_scale,
        "confidence": float(best_corr),
        "method": "librosa",
    }


def analyze_structure_energy(y: np.ndarray, sr: int) -> list[dict]:
    """Simple energy-based structure analysis."""
    # Compute RMS energy
    frame_length = int(sr * 0.5)  # 500ms frames
    hop_length = int(sr * 0.25)   # 250ms hop
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Normalize
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    
    # Detect significant changes
    threshold_low = 0.3
    threshold_high = 0.7
    
    segments = []
    current_type = "intro" if rms_norm[0] < threshold_low else "body"
    current_start = 0
    
    for i, (t, e) in enumerate(zip(times, rms_norm)):
        new_type = None
        
        if current_type == "intro" and e > threshold_high:
            new_type = "drop"
        elif current_type == "drop" and e < threshold_low:
            new_type = "breakdown"
        elif current_type == "breakdown" and e > threshold_high:
            new_type = "drop"
        elif current_type == "body" and e < threshold_low:
            new_type = "breakdown"
        elif current_type == "body" and e > threshold_high:
            new_type = "drop"
        
        if new_type and i > 0:
            segments.append({
                "type": current_type,
                "start": float(current_start),
                "end": float(t),
                "duration": float(t - current_start),
            })
            current_type = new_type
            current_start = t
    
    # Add final segment
    segments.append({
        "type": current_type,
        "start": float(current_start),
        "end": float(times[-1]),
        "duration": float(times[-1] - current_start),
    })
    
    return segments


def analyze_audio_file(audio_path: Path) -> dict:
    """Run full analysis on an audio file."""
    analysis = {
        "file": str(audio_path),
        "analyzed_at": datetime.now().isoformat(),
        "bpm": None,
        "beat_times": [],
        "key": None,
        "structure": [],
        "duration": None,
    }
    
    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)
        analysis["duration"] = float(len(y) / sr)
        analysis["sample_rate"] = sr
        
        # BPM estimation
        if HAVE_MADMOM:
            try:
                bpm, beat_times = estimate_bpm_madmom(str(audio_path))
                analysis["bpm"] = float(bpm)
                analysis["beat_times"] = beat_times.tolist()
                analysis["bpm_method"] = "madmom"
            except Exception as e:
                log(f"Madmom failed, falling back to librosa: {e}")
                bpm, beat_times = estimate_bpm_librosa(y, sr)
                analysis["bpm"] = float(bpm)
                analysis["beat_times"] = beat_times.tolist()
                analysis["bpm_method"] = "librosa"
        else:
            bpm, beat_times = estimate_bpm_librosa(y, sr)
            analysis["bpm"] = float(bpm)
            analysis["beat_times"] = beat_times.tolist()
            analysis["bpm_method"] = "librosa"
        
        # Key detection
        if HAVE_ESSENTIA:
            try:
                analysis["key"] = detect_key_essentia(str(audio_path))
            except Exception as e:
                log(f"Essentia key detection failed: {e}")
                analysis["key"] = detect_key_librosa(y, sr)
        else:
            analysis["key"] = detect_key_librosa(y, sr)
        
        # Structure analysis
        analysis["structure"] = analyze_structure_energy(y, sr)
        
    except Exception as e:
        analysis["error"] = str(e)
        log(f"Analysis error for {audio_path}: {e}")
    
    return analysis


def analyze_transition(trans_dir: Path) -> dict:
    """Analyze all files in a transition directory."""
    results = {
        "transition_id": trans_dir.name,
        "analyzed_at": datetime.now().isoformat(),
        "track_a_clean": None,
        "transition": None,
        "track_b_clean": None,
    }
    
    for name, key in [("track_a_clean.mp3", "track_a_clean"),
                       ("transition.mp3", "transition"),
                       ("track_b_clean.mp3", "track_b_clean")]:
        audio_path = trans_dir / name
        if audio_path.exists():
            results[key] = analyze_audio_file(audio_path)
    
    return results


def analyze_all_transitions(max_transitions: int = None):
    """Analyze all extracted transitions."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting transition analysis")
    
    # Find all transition directories
    trans_dirs = [d for d in TRANSITIONS_DIR.iterdir() 
                  if d.is_dir() and d.name != "." and not d.name.startswith(".")]
    
    # Flatten to individual transitions
    all_trans = []
    for mix_dir in trans_dirs:
        for trans_dir in mix_dir.iterdir():
            if trans_dir.is_dir() and (trans_dir / "metadata.json").exists():
                all_trans.append(trans_dir)
    
    log(f"Found {len(all_trans)} transitions to analyze")
    
    if max_transitions:
        all_trans = all_trans[:max_transitions]
    
    analyzed = []
    for trans_dir in tqdm(all_trans, desc="Analyzing"):
        analysis = analyze_transition(trans_dir)
        
        # Save alongside transition
        analysis_path = trans_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        analyzed.append(analysis)
    
    # Summary
    log("=" * 60)
    log(f"Analysis complete: {len(analyzed)} transitions")
    
    # Stats
    bpms = [a["transition"]["bpm"] for a in analyzed 
            if a.get("transition") and a["transition"].get("bpm")]
    if bpms:
        log(f"BPM range: {min(bpms):.1f} - {max(bpms):.1f}")
        log(f"BPM median: {np.median(bpms):.1f}")


def analyze_all_tracks(max_tracks: int = None):
    """Analyze all individual tracks from Jamendo."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Starting track analysis")
    
    track_files = list(TRACKS_DIR.glob("*.mp3"))
    log(f"Found {len(track_files)} tracks")
    
    if max_tracks:
        track_files = track_files[:max_tracks]
    
    # Check which already analyzed
    analysis_subdir = ANALYSIS_DIR / "tracks"
    analysis_subdir.mkdir(exist_ok=True)
    
    existing = set(f.stem for f in analysis_subdir.glob("*.json"))
    to_analyze = [f for f in track_files if f.stem not in existing]
    
    log(f"Already analyzed: {len(existing)}")
    log(f"To analyze: {len(to_analyze)}")
    
    for track_path in tqdm(to_analyze, desc="Analyzing tracks"):
        analysis = analyze_audio_file(track_path)
        
        analysis_path = analysis_subdir / f"{track_path.stem}.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
    
    log(f"Track analysis complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze audio features")
    parser.add_argument("--transitions", action="store_true", help="Analyze transitions")
    parser.add_argument("--tracks", action="store_true", help="Analyze Jamendo tracks")
    parser.add_argument("--max", type=int, default=None, help="Max items to analyze")
    parser.add_argument("--single", type=str, help="Analyze single file")
    args = parser.parse_args()
    
    if args.single:
        result = analyze_audio_file(Path(args.single))
        print(json.dumps(result, indent=2))
    elif args.transitions:
        analyze_all_transitions(max_transitions=args.max)
    elif args.tracks:
        analyze_all_tracks(max_tracks=args.max)
    else:
        print("Specify --transitions or --tracks")
