#!/usr/bin/env python3
"""
DJTransGAN v2 Data Pipeline

Full data preparation pipeline:
1. Downloads mixes (from metadata) or uses existing audio
2. Extracts transitions from mixes with tracklists
3. Separates stems using demucs
4. Analyzes audio features (BPM, key, beats, structure)
5. Saves in structured format for the Dataset class

Output structure:
processed/
  {pair_id}/
    track_a_stems/
      drums.wav, bass.wav, vocals.wav, other.wav
    track_b_stems/
      drums.wav, bass.wav, vocals.wav, other.wav
    transition.wav
    analysis.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import torch
    import torchaudio
except ImportError:
    print("ERROR: torch and torchaudio required")
    sys.exit(1)

try:
    import librosa
except ImportError:
    librosa = None
    print("WARNING: librosa not available, some analysis features will be limited")

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_audio_file(path: Path, target_sr: int = 44100) -> tuple:
    """
    Load audio file with fallback to librosa for mp3 files.
    
    Returns:
        (waveform, sample_rate) - waveform is [channels, samples] tensor
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    # Try torchaudio first for wav files (most reliable)
    if suffix == '.wav':
        try:
            waveform, sr = torchaudio.load(str(path))
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            return waveform, target_sr
        except Exception:
            pass
    
    # Use librosa for mp3 and as fallback
    if librosa is not None:
        try:
            y, sr = librosa.load(str(path), sr=target_sr, mono=False)
            if y.ndim == 1:
                y = y[np.newaxis, :]  # Add channel dimension
            waveform = torch.from_numpy(y).float()
            return waveform, target_sr
        except Exception:
            pass
    
    # Last resort - try torchaudio anyway
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def save_audio_file(path: Path, waveform: torch.Tensor, sample_rate: int):
    """
    Save audio file using soundfile (more reliable than torchaudio for newer versions).
    
    Args:
        path: Output path (should be .wav)
        waveform: [channels, samples] tensor
        sample_rate: Sample rate
    """
    path = Path(path)
    
    # Ensure wav extension
    if path.suffix.lower() != '.wav':
        path = path.with_suffix('.wav')
    
    # Convert to numpy for soundfile
    audio_np = waveform.numpy()
    
    # soundfile expects [samples, channels]
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    
    if sf is not None:
        sf.write(str(path), audio_np, sample_rate)
    else:
        # Fallback to torchaudio
        torchaudio.save(str(path), waveform, sample_rate)


# Configuration
DEFAULT_OUTPUT_DIR = Path("./processed")
DEFAULT_SAMPLE_RATE = 44100
TRANSITION_DURATION = 60.0  # seconds
CLEAN_SEGMENT_MIN = 30.0  # minimum seconds for clean segment


def log(message: str, log_file: Optional[Path] = None):
    """Log message to console and optionally to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_line + "\n")


def run_demucs(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
    device: str = "auto",
) -> Optional[Path]:
    """
    Run demucs stem separation.
    
    Args:
        audio_path: Path to input audio
        output_dir: Output directory for stems
        model: Demucs model name (htdemucs, htdemucs_ft, mdx_extra)
        device: Device (cpu, cuda, mps, auto)
    
    Returns:
        Path to stems directory, or None if failed
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", str(output_dir),
        "-d", device,
        str(audio_path),
    ]
    
    try:
        # Run demucs - don't capture stderr as it contains progress bars
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            timeout=1800,  # 30 min timeout
        )
        
        # Find output directory - check if it was created successfully
        stem_name = audio_path.stem
        stems_dir = output_dir / model / stem_name
        
        if stems_dir.exists() and any(stems_dir.glob("*.wav")):
            return stems_dir
        
        # If directory doesn't exist but return code is 0, something went wrong
        if result.returncode != 0:
            log(f"Demucs error (code {result.returncode})")
            return None
        
        log(f"Stems directory not found: {stems_dir}")
        return None
        
    except subprocess.TimeoutExpired:
        log("Demucs timeout (30 min)")
        return None
    except FileNotFoundError:
        log("Demucs not installed. Install with: pip install demucs")
        return None
    except Exception as e:
        log(f"Demucs error: {e}")
        return None


def analyze_audio(
    audio_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dict:
    """
    Analyze audio file for BPM, beats, key, structure.
    
    Returns dict with analysis results.
    """
    analysis = {
        "file": str(audio_path),
        "analyzed_at": datetime.now().isoformat(),
        "sample_rate": sample_rate,
        "bpm": None,
        "beat_times": [],
        "key": None,
        "structure": [],
        "duration": None,
    }
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        # Convert to mono numpy
        audio_mono = waveform.mean(dim=0).numpy()
        duration = len(audio_mono) / sample_rate
        analysis["duration"] = duration
        
        if librosa:
            # BPM and beats
            tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
            analysis["bpm"] = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            analysis["beat_times"] = librosa.frames_to_time(beats, sr=sample_rate).tolist()
            
            # Key detection via chroma
            chroma = librosa.feature.chroma_cqt(y=audio_mono, sr=sample_rate)
            chroma_avg = np.mean(chroma, axis=1)
            
            # Simple key detection
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            best_corr, best_key, best_scale = -1, 'C', 'major'
            for i in range(12):
                rot = np.roll(chroma_avg, -i)
                maj_corr = np.corrcoef(rot, major_profile)[0, 1]
                min_corr = np.corrcoef(rot, minor_profile)[0, 1]
                if maj_corr > best_corr:
                    best_corr, best_key, best_scale = maj_corr, key_names[i], 'major'
                if min_corr > best_corr:
                    best_corr, best_key, best_scale = min_corr, key_names[i], 'minor'
            
            analysis["key"] = {"key": best_key, "scale": best_scale, "confidence": float(best_corr)}
            
            # Energy-based structure analysis
            rms = librosa.feature.rms(y=audio_mono, frame_length=22050, hop_length=11025)[0]
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=11025)
            
            segments = []
            current_type = "intro" if rms_norm[0] < 0.3 else "body"
            current_start = 0
            
            for t, e in zip(times, rms_norm):
                new_type = None
                if current_type in ["intro", "breakdown"] and e > 0.7:
                    new_type = "drop"
                elif current_type == "drop" and e < 0.3:
                    new_type = "breakdown"
                
                if new_type:
                    segments.append({
                        "type": current_type,
                        "start": float(current_start),
                        "end": float(t),
                    })
                    current_type, current_start = new_type, t
            
            segments.append({
                "type": current_type,
                "start": float(current_start),
                "end": float(times[-1] if len(times) > 0 else duration),
            })
            analysis["structure"] = segments
        else:
            # Fallback without librosa
            analysis["bpm"] = 128.0
            analysis["beat_times"] = list(np.arange(0, duration, 60/128))
            analysis["key"] = {"key": "C", "scale": "major", "confidence": 0.5}
            analysis["structure"] = [{"type": "body", "start": 0, "end": duration}]
    
    except Exception as e:
        analysis["error"] = str(e)
        log(f"Analysis error: {e}")
    
    return analysis


def extract_audio_segment(
    audio_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """Extract a segment from audio file."""
    waveform, _ = load_audio_file(audio_path, sample_rate)
    
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    
    segment = waveform[:, start_sample:end_sample]
    
    save_audio_file(output_path, segment, sample_rate)


def process_transition_pair(
    pair_id: str,
    track_a_audio: Path,
    track_b_audio: Path,
    transition_audio: Path,
    output_dir: Path,
    demucs_model: str = "htdemucs",
    device: str = "auto",
    log_file: Optional[Path] = None,
) -> bool:
    """
    Process a single transition pair.
    
    1. Run demucs on both tracks
    2. Analyze both tracks and transition
    3. Save in structured format
    """
    pair_dir = output_dir / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Processing pair: {pair_id}", log_file)
    
    # Create temp dir for demucs output
    temp_stems_dir = pair_dir / "_temp_stems"
    temp_stems_dir.mkdir(exist_ok=True)
    
    try:
        # Run demucs on track A
        log(f"  Running demucs on track A...", log_file)
        stems_a = run_demucs(track_a_audio, temp_stems_dir, demucs_model, device)
        if stems_a is None:
            log(f"  Failed to separate track A", log_file)
            return False
        
        # Run demucs on track B
        log(f"  Running demucs on track B...", log_file)
        stems_b = run_demucs(track_b_audio, temp_stems_dir, demucs_model, device)
        if stems_b is None:
            log(f"  Failed to separate track B", log_file)
            return False
        
        # Copy stems to output directory
        track_a_stems_dir = pair_dir / "track_a_stems"
        track_b_stems_dir = pair_dir / "track_b_stems"
        track_a_stems_dir.mkdir(exist_ok=True)
        track_b_stems_dir.mkdir(exist_ok=True)
        
        for stem_name in ["drums", "bass", "vocals", "other"]:
            # Track A
            src = stems_a / f"{stem_name}.wav"
            if src.exists():
                shutil.copy(src, track_a_stems_dir / f"{stem_name}.wav")
            
            # Track B
            src = stems_b / f"{stem_name}.wav"
            if src.exists():
                shutil.copy(src, track_b_stems_dir / f"{stem_name}.wav")
        
        # Copy transition audio
        trans_output = pair_dir / "transition.wav"
        if transition_audio.suffix.lower() == ".wav":
            shutil.copy(transition_audio, trans_output)
        else:
            # Convert to wav
            waveform, sr = load_audio_file(transition_audio, DEFAULT_SAMPLE_RATE)
            save_audio_file(trans_output, waveform, sr)
        
        # Analyze all audio
        log(f"  Analyzing audio...", log_file)
        analysis_a = analyze_audio(track_a_audio)
        analysis_b = analyze_audio(track_b_audio)
        analysis_trans = analyze_audio(trans_output)
        
        # Save combined analysis
        combined_analysis = {
            "pair_id": pair_id,
            "processed_at": datetime.now().isoformat(),
            "track_a": analysis_a,
            "track_b": analysis_b,
            "transition": analysis_trans,
            "transition_duration": analysis_trans.get("duration", 0),
        }
        
        with open(pair_dir / "analysis.json", "w") as f:
            json.dump(combined_analysis, f, indent=2)
        
        # Clean up temp directory
        shutil.rmtree(temp_stems_dir, ignore_errors=True)
        
        log(f"  Complete: {pair_id}", log_file)
        return True
        
    except Exception as e:
        log(f"  Error processing {pair_id}: {e}", log_file)
        shutil.rmtree(temp_stems_dir, ignore_errors=True)
        return False


def process_extracted_transitions(
    transitions_dir: Path,
    output_dir: Path,
    demucs_model: str = "htdemucs",
    device: str = "auto",
    max_items: Optional[int] = None,
    log_file: Optional[Path] = None,
):
    """
    Process transitions extracted by extract_transitions.py.
    
    Expects structure:
    transitions_dir/
      {mix_id}/
        {transition_id}/
          track_a_clean.mp3
          track_b_clean.mp3
          transition.mp3
          metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60, log_file)
    log("Processing extracted transitions", log_file)
    
    # Find all transition directories
    transitions = []
    for mix_dir in transitions_dir.iterdir():
        if mix_dir.is_dir():
            for trans_dir in mix_dir.iterdir():
                if trans_dir.is_dir() and (trans_dir / "metadata.json").exists():
                    transitions.append(trans_dir)
    
    log(f"Found {len(transitions)} transitions", log_file)
    
    if max_items:
        transitions = transitions[:max_items]
    
    success_count = 0
    
    for trans_dir in tqdm(transitions, desc="Processing"):
        pair_id = trans_dir.name
        
        # Check if already processed
        if (output_dir / pair_id / "analysis.json").exists():
            log(f"Skipping {pair_id} (already processed)", log_file)
            success_count += 1
            continue
        
        # Find audio files
        track_a = trans_dir / "track_a_clean.mp3"
        track_b = trans_dir / "track_b_clean.mp3"
        transition = trans_dir / "transition.mp3"
        
        if not all(p.exists() for p in [track_a, track_b, transition]):
            log(f"Missing files for {pair_id}", log_file)
            continue
        
        success = process_transition_pair(
            pair_id=pair_id,
            track_a_audio=track_a,
            track_b_audio=track_b,
            transition_audio=transition,
            output_dir=output_dir,
            demucs_model=demucs_model,
            device=device,
            log_file=log_file,
        )
        
        if success:
            success_count += 1
    
    log("=" * 60, log_file)
    log(f"Processing complete: {success_count}/{len(transitions)} successful", log_file)


def _worker_process_pair(args):
    """Worker function for parallel pair processing."""
    track_a, track_b, pair_id, output_dir, demucs_model, device, temp_dir = args
    
    try:
        waveform_a, _ = load_audio_file(track_a, DEFAULT_SAMPLE_RATE)
        waveform_b, _ = load_audio_file(track_b, DEFAULT_SAMPLE_RATE)
        
        trans_samples = int(TRANSITION_DURATION * DEFAULT_SAMPLE_RATE)
        
        len_a = waveform_a.shape[1]
        len_b = waveform_b.shape[1]
        
        if len_a < trans_samples or len_b < trans_samples:
            return False
        
        seg_a = waveform_a[:, -trans_samples:]
        seg_b = waveform_b[:, :trans_samples]
        
        t = torch.linspace(0, 1, trans_samples)
        fade_out = torch.cos(t * np.pi / 2).unsqueeze(0)
        fade_in = torch.sin(t * np.pi / 2).unsqueeze(0)
        transition = seg_a * fade_out + seg_b * fade_in
        
        temp_a = Path(temp_dir) / f"{pair_id}_a.wav"
        temp_b = Path(temp_dir) / f"{pair_id}_b.wav"
        temp_trans = Path(temp_dir) / f"{pair_id}_trans.wav"
        
        save_audio_file(temp_a, seg_a, DEFAULT_SAMPLE_RATE)
        save_audio_file(temp_b, seg_b, DEFAULT_SAMPLE_RATE)
        save_audio_file(temp_trans, transition, DEFAULT_SAMPLE_RATE)
        
        success = process_transition_pair(
            pair_id=pair_id,
            track_a_audio=temp_a,
            track_b_audio=temp_b,
            transition_audio=temp_trans,
            output_dir=Path(output_dir),
            demucs_model=demucs_model,
            device=device,
        )
        
        temp_a.unlink(missing_ok=True)
        temp_b.unlink(missing_ok=True)
        temp_trans.unlink(missing_ok=True)
        
        return success
    except Exception as e:
        print(f"Error with {pair_id}: {e}")
        return False


def process_track_library(
    tracks_dir: Path,
    output_dir: Path,
    demucs_model: str = "htdemucs",
    device: str = "auto",
    max_pairs: Optional[int] = None,
    log_file: Optional[Path] = None,
    n_gpus: int = 1,
    n_workers: int = 1,
):
    """
    Create synthetic transition pairs from a track library.
    
    For each pair of tracks, creates a synthetic crossfade transition
    and processes with stem separation.
    
    Multi-GPU: pairs are sharded across GPUs, each GPU runs demucs in parallel.
    n_workers controls CPU parallelism for audio loading/analysis per GPU.
    """
    import random
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60, log_file)
    log("Processing track library for synthetic transitions", log_file)
    
    # Find all audio files
    audio_files = list(tracks_dir.rglob("*.mp3")) + list(tracks_dir.rglob("*.wav")) + list(tracks_dir.rglob("*.flac"))
    log(f"Found {len(audio_files)} tracks", log_file)
    
    if len(audio_files) < 2:
        log("Need at least 2 tracks", log_file)
        return
    
    # Auto-detect GPUs if not specified
    if device == "auto" or device == "cuda":
        if torch.cuda.is_available():
            detected_gpus = torch.cuda.device_count()
            if n_gpus <= 1:
                n_gpus = detected_gpus
            else:
                n_gpus = min(n_gpus, detected_gpus)
        else:
            n_gpus = 1
    
    # Generate random pairs
    n_pairs = max_pairs or len(audio_files)
    pairs = []
    
    for _ in range(n_pairs):
        a, b = random.sample(audio_files, 2)
        pair_id = f"{a.stem}___{b.stem}"
        if not (output_dir / pair_id / "analysis.json").exists():
            pairs.append((a, b, pair_id))
    
    log(f"Processing {len(pairs)} new pairs with {n_gpus} GPU(s) and {n_workers} CPU worker(s) per GPU", log_file)
    
    if n_gpus <= 1 and n_workers <= 1:
        # Single GPU, single worker — original sequential path
        temp_dir = output_dir / "_temp"
        temp_dir.mkdir(exist_ok=True)
        
        success_count = 0
        gpu_device = "cuda:0" if torch.cuda.is_available() else device
        
        for track_a, track_b, pair_id in tqdm(pairs, desc="Processing pairs"):
            result = _worker_process_pair(
                (track_a, track_b, pair_id, str(output_dir), demucs_model, gpu_device, str(temp_dir))
            )
            if result:
                success_count += 1
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Multi-GPU parallel processing
        # Shard pairs across GPUs, use ProcessPoolExecutor for parallelism
        total_workers = n_gpus * n_workers
        
        # Build work items with GPU assignment
        work_items = []
        for i, (track_a, track_b, pair_id) in enumerate(pairs):
            gpu_id = i % n_gpus
            gpu_device = f"cuda:{gpu_id}" if torch.cuda.is_available() else device
            temp_dir = output_dir / f"_temp_gpu{gpu_id}"
            temp_dir.mkdir(exist_ok=True)
            work_items.append(
                (track_a, track_b, pair_id, str(output_dir), demucs_model, gpu_device, str(temp_dir))
            )
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            futures = {executor.submit(_worker_process_pair, item): item for item in work_items}
            
            with tqdm(total=len(futures), desc=f"Processing pairs ({n_gpus} GPUs)") as pbar:
                for future in as_completed(futures):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        log(f"Worker error: {e}", log_file)
                    pbar.update(1)
        
        # Clean up temp dirs
        for gpu_id in range(n_gpus):
            shutil.rmtree(output_dir / f"_temp_gpu{gpu_id}", ignore_errors=True)
    
    log("=" * 60, log_file)
    log(f"Processing complete: {success_count}/{len(pairs)} successful", log_file)


def main():
    parser = argparse.ArgumentParser(
        description="DJTransGAN v2 Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process extracted transitions (from extract_transitions.py)
  python pipeline.py --transitions data/transitions --output processed

  # Process a track library (creates synthetic transitions)
  python pipeline.py --tracks ~/Music/DJ_Library --output processed --max 100

  # Process a single pair
  python pipeline.py --single track_a.mp3 track_b.mp3 transition.mp3 --output processed
        """,
    )
    
    parser.add_argument(
        "--transitions",
        type=str,
        help="Path to extracted transitions directory",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        help="Path to track library for synthetic transitions",
    )
    parser.add_argument(
        "--single",
        nargs=3,
        metavar=("TRACK_A", "TRACK_B", "TRANSITION"),
        help="Process a single pair (track_a, track_b, transition)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of items to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"],
        help="Demucs model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for demucs",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    log_file = Path(args.log) if args.log else None
    
    if args.single:
        # Process single pair
        track_a, track_b, transition = [Path(p) for p in args.single]
        pair_id = f"{track_a.stem}___{track_b.stem}"
        
        success = process_transition_pair(
            pair_id=pair_id,
            track_a_audio=track_a,
            track_b_audio=track_b,
            transition_audio=transition,
            output_dir=output_dir,
            demucs_model=args.model,
            device=args.device,
            log_file=log_file,
        )
        sys.exit(0 if success else 1)
    
    elif args.transitions:
        # Process extracted transitions
        process_extracted_transitions(
            transitions_dir=Path(args.transitions),
            output_dir=output_dir,
            demucs_model=args.model,
            device=args.device,
            max_items=args.max,
            log_file=log_file,
        )
    
    elif args.tracks:
        # Process track library
        process_track_library(
            tracks_dir=Path(args.tracks),
            output_dir=output_dir,
            demucs_model=args.model,
            device=args.device,
            max_pairs=args.max,
            log_file=log_file,
        )
    
    else:
        parser.print_help()
        print("\nError: Specify --transitions, --tracks, or --single")
        sys.exit(1)


if __name__ == "__main__":
    main()
