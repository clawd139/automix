#!/usr/bin/env python3
"""
DJTransGAN v2 Inference Script

Generate DJ transitions from two audio files:
1. Run demucs stem separation
2. Analyze tracks (BPM, key, structure)
3. Run model to predict effect parameter curves
4. Apply curves via mixer
5. Output: mixed audio + visualization

Works on M2 Mac (MPS device).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:
    print("ERROR: torchaudio required. Install with: pip install torchaudio")
    sys.exit(1)

try:
    import librosa
except ImportError:
    librosa = None
    print("WARNING: librosa not available, using basic analysis")

from .model import (
    Config,
    DJTransitionModel,
    TrackFeatures,
    load_model,
    get_inference_config,
)
from .effects import (
    DifferentiableDJMixer,
    MixerConfig,
    STEM_NAMES,
    N_TOTAL_PARAMS,
)


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_demucs(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
    device: str = "auto",
) -> Optional[Path]:
    """Run demucs stem separation."""
    if device == "auto":
        device = get_device()
    
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", str(output_dir),
        "-d", device if device != "mps" else "cpu",  # demucs doesn't support MPS directly
        str(audio_path),
    ]
    
    print(f"Running demucs on {audio_path.name}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        
        if result.returncode != 0:
            print(f"Demucs error: {result.stderr[:500]}")
            return None
        
        # Find output
        stems_dir = output_dir / model / audio_path.stem
        if stems_dir.exists():
            return stems_dir
        
        print(f"Stems not found at {stems_dir}")
        return None
        
    except subprocess.TimeoutExpired:
        print("Demucs timeout")
        return None
    except FileNotFoundError:
        print("Demucs not installed. Run: pip install demucs")
        return None
    except Exception as e:
        print(f"Demucs error: {e}")
        return None


def load_stems(stems_dir: Path, sample_rate: int = 44100) -> Dict[str, torch.Tensor]:
    """Load all stems from directory."""
    stems = {}
    
    for stem_name in STEM_NAMES:
        stem_path = stems_dir / f"{stem_name}.wav"
        if not stem_path.exists():
            stem_path = stems_dir / f"{stem_name}.mp3"
        
        if stem_path.exists():
            waveform, sr = torchaudio.load(str(stem_path))
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            # Convert to mono
            stems[stem_name] = waveform.mean(dim=0)
        else:
            print(f"Warning: {stem_name} stem not found")
            stems[stem_name] = torch.zeros(sample_rate * 30)  # 30s silence
    
    return stems


def analyze_track(
    audio_path: Path,
    sample_rate: int = 44100,
) -> Dict:
    """Analyze track for BPM, key, beats, structure."""
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    audio = waveform.mean(dim=0).numpy()
    duration = len(audio) / sample_rate
    
    analysis = {
        "duration": duration,
        "sample_rate": sample_rate,
    }
    
    if librosa:
        # BPM and beats
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        analysis["bpm"] = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        analysis["beat_times"] = librosa.frames_to_time(beats, sr=sample_rate).tolist()
        
        # Key detection
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
        chroma_avg = np.mean(chroma, axis=1)
        
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
        
        analysis["key"] = {"key": best_key, "scale": best_scale}
        
        # Simple structure
        rms = librosa.feature.rms(y=audio, frame_length=22050, hop_length=11025)[0]
        analysis["energy"] = rms.tolist()
        analysis["structure"] = [{"type": "body", "start": 0, "end": duration}]
    else:
        # Fallback
        analysis["bpm"] = 128.0
        analysis["beat_times"] = list(np.arange(0, duration, 60/128))
        analysis["key"] = {"key": "C", "scale": "major"}
        analysis["structure"] = [{"type": "body", "start": 0, "end": duration}]
    
    return analysis


def build_track_features(
    stems: Dict[str, torch.Tensor],
    analysis: Dict,
    n_frames: int,
    config: Config,
    device: str,
) -> TrackFeatures:
    """Build TrackFeatures from stems and analysis."""
    sample_rate = config.audio.sample_rate
    hop_length = config.audio.hop_length
    n_mels = config.audio.n_mels
    n_fft = config.audio.n_fft
    
    # Compute mel spectrograms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
    stem_mels = []
    for stem_name in STEM_NAMES:
        audio = stems[stem_name]
        mel = mel_transform(audio.unsqueeze(0))
        mel = amp_to_db(mel)
        mel = (mel + 40) / 40  # Normalize
        mel = mel.clamp(-1, 1)
        stem_mels.append(mel.squeeze(0))
    
    # Stack: [n_stems, n_mels, n_frames_raw]
    stem_mels = torch.stack(stem_mels, dim=0)
    
    # Interpolate to target frames
    stem_mels = F.interpolate(
        stem_mels.unsqueeze(0),
        size=n_frames,
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    
    # Structure
    structure_labels = config.audio.structure_labels
    n_labels = len(structure_labels)
    structure = torch.zeros(n_frames, n_labels)
    
    # Default to body
    body_idx = structure_labels.index('body') if 'body' in structure_labels else 0
    structure[:, body_idx] = 1.0
    
    # Beats
    beat_times = analysis.get("beat_times", [])
    duration = analysis.get("duration", 30)
    
    beats = torch.zeros(n_frames)
    for bt in beat_times:
        frame_idx = int(bt / duration * n_frames)
        if 0 <= frame_idx < n_frames:
            beats[frame_idx] = 1.0
    
    # Downbeats (every 4th beat)
    downbeat_times = beat_times[::4] if beat_times else []
    downbeats = torch.zeros(n_frames)
    for dt in downbeat_times:
        frame_idx = int(dt / duration * n_frames)
        if 0 <= frame_idx < n_frames:
            downbeats[frame_idx] = 1.0
    
    # BPM
    bpm = torch.tensor([analysis.get("bpm", 128.0)])
    
    # Key
    key_info = analysis.get("key", {})
    key_name = key_info.get("key", "C")
    scale = key_info.get("scale", "major")
    
    key_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
               'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
               'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    key_idx = key_map.get(key_name, 0)
    if scale == "minor":
        key_idx += 12
    
    key = torch.zeros(config.audio.n_keys)
    key[key_idx] = 1.0
    
    # Energy
    if "energy" in analysis:
        energy_raw = torch.tensor(analysis["energy"])
        energy = F.interpolate(
            energy_raw.unsqueeze(0).unsqueeze(0),
            size=n_frames,
            mode='linear',
            align_corners=False,
        ).squeeze()
    else:
        total = sum(stems.values())
        rms = torch.sqrt(torch.mean(total ** 2))
        energy = torch.full((n_frames,), rms.item())
    
    return TrackFeatures(
        stem_mels=stem_mels.to(device),
        structure=structure.to(device),
        beats=beats.to(device),
        downbeats=downbeats.to(device),
        bpm=bpm.to(device),
        key=key.to(device),
        energy=energy.to(device),
    )


def generate_transition(
    model: DJTransitionModel,
    mixer: DifferentiableDJMixer,
    track_a_stems: Dict[str, torch.Tensor],
    track_b_stems: Dict[str, torch.Tensor],
    track_a_features: TrackFeatures,
    track_b_features: TrackFeatures,
    n_frames: int,
    device: str,
    guidance_scale: float = 2.0,
    n_inference_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate transition using model and mixer.
    
    Returns:
        mixed_audio: [samples] tensor
        params: [n_frames, n_params] parameter curves
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension to features
        def add_batch(f: TrackFeatures) -> TrackFeatures:
            return TrackFeatures(
                stem_mels=f.stem_mels.unsqueeze(0),
                structure=f.structure.unsqueeze(0),
                beats=f.beats.unsqueeze(0),
                downbeats=f.downbeats.unsqueeze(0),
                bpm=f.bpm.unsqueeze(0),
                key=f.key.unsqueeze(0),
                energy=f.energy.unsqueeze(0),
            )
        
        track_a_feat = add_batch(track_a_features)
        track_b_feat = add_batch(track_b_features)
        
        # Generate parameters
        params = model.generate(
            track_a_feat,
            track_b_feat,
            n_frames=n_frames,
            guidance_scale=guidance_scale,
            n_steps=n_inference_steps,
        )  # [1, n_frames, n_params]
        
        # Prepare stems for mixer
        stems_a = {name: s.unsqueeze(0).to(device) for name, s in track_a_stems.items()}
        stems_b = {name: s.unsqueeze(0).to(device) for name, s in track_b_stems.items()}
        
        # Apply mixer
        mixed_audio, intermediates = mixer(
            stems_a, stems_b, params,
            return_intermediates=True,
        )
        
        # Remove batch dimension
        mixed_audio = mixed_audio.squeeze(0)
        params = params.squeeze(0)
        
        # Convert stereo to mono for output
        if mixed_audio.dim() == 2:
            mixed_audio = mixed_audio.mean(dim=0)
    
    return mixed_audio.cpu(), params.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Generate DJ transition between two tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference.py track_a.mp3 track_b.mp3 -o output/transition.wav

  # With custom model
  python inference.py track_a.mp3 track_b.mp3 --model checkpoints/best_model.pt

  # Adjust transition duration
  python inference.py track_a.mp3 track_b.mp3 --duration 45
        """,
    )
    
    parser.add_argument(
        "track_a",
        type=str,
        help="Path to first (outgoing) track",
    )
    parser.add_argument(
        "track_b",
        type=str,
        help="Path to second (incoming) track",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/transition.wav",
        help="Output path for mixed audio",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Transition duration in seconds",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate parameter visualization",
    )
    parser.add_argument(
        "--no-demucs",
        action="store_true",
        help="Skip demucs (assume stems already exist)",
    )
    parser.add_argument(
        "--stems-dir",
        type=str,
        default=None,
        help="Directory containing pre-separated stems",
    )
    parser.add_argument(
        "--save-params",
        type=str,
        default=None,
        help="Save parameter curves to JSON",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if args.device != "auto" else get_device()
    print(f"Using device: {device}")
    
    track_a_path = Path(args.track_a)
    track_b_path = Path(args.track_b)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = get_inference_config()
    
    # Ensure parameter count matches
    assert config.effects.n_params == N_TOTAL_PARAMS, \
        f"Config has {config.effects.n_params} params but mixer expects {N_TOTAL_PARAMS}"
    
    # Load or create model
    if args.model and Path(args.model).exists():
        print(f"Loading model from {args.model}")
        model = load_model(args.model, device=device)
    else:
        print("No model checkpoint provided, using random initialization")
        print("(Results will be random - train a model first!)")
        from .model import create_model
        model = create_model(config)
        model.to(device)
    
    # Create mixer
    mixer_config = MixerConfig(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
    )
    mixer = DifferentiableDJMixer(mixer_config)
    mixer.to(device)
    
    # Stem separation
    if args.stems_dir:
        stems_dir = Path(args.stems_dir)
        stems_a_dir = stems_dir / track_a_path.stem
        stems_b_dir = stems_dir / track_b_path.stem
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if not args.no_demucs:
                stems_a_dir = run_demucs(track_a_path, temp_path, device=device)
                stems_b_dir = run_demucs(track_b_path, temp_path, device=device)
                
                if stems_a_dir is None or stems_b_dir is None:
                    print("Stem separation failed")
                    sys.exit(1)
            else:
                print("Skipping demucs, using original audio as 'other' stem")
                # Create fake stems dir
                stems_a_dir = temp_path / "track_a"
                stems_b_dir = temp_path / "track_b"
                stems_a_dir.mkdir()
                stems_b_dir.mkdir()
                
                # Copy original as 'other', create silent other stems
                import shutil
                shutil.copy(track_a_path, stems_a_dir / "other.mp3")
                shutil.copy(track_b_path, stems_b_dir / "other.mp3")
    
    # Load stems
    print("Loading stems...")
    sample_rate = config.audio.sample_rate
    
    stems_a = load_stems(stems_a_dir, sample_rate)
    stems_b = load_stems(stems_b_dir, sample_rate)
    
    # Determine transition length
    trans_samples = int(args.duration * sample_rate)
    
    # Align stems to transition length
    def align_stems(stems: Dict[str, torch.Tensor], length: int) -> Dict[str, torch.Tensor]:
        aligned = {}
        for name, audio in stems.items():
            if len(audio) < length:
                audio = F.pad(audio, (0, length - len(audio)))
            else:
                audio = audio[:length]
            aligned[name] = audio
        return aligned
    
    # Take end of track A, start of track B
    stems_a = {name: s[-trans_samples:] for name, s in stems_a.items()}
    stems_b = {name: s[:trans_samples] for name, s in stems_b.items()}
    
    stems_a = align_stems(stems_a, trans_samples)
    stems_b = align_stems(stems_b, trans_samples)
    
    # Analyze tracks
    print("Analyzing tracks...")
    analysis_a = analyze_track(track_a_path, sample_rate)
    analysis_b = analyze_track(track_b_path, sample_rate)
    
    print(f"Track A: {analysis_a.get('bpm', '?')} BPM, {analysis_a.get('key', {}).get('key', '?')} {analysis_a.get('key', {}).get('scale', '')}")
    print(f"Track B: {analysis_b.get('bpm', '?')} BPM, {analysis_b.get('key', {}).get('key', '?')} {analysis_b.get('key', {}).get('scale', '')}")
    
    # Build features
    features_a = build_track_features(
        stems_a, analysis_a,
        n_frames=config.model.n_frames,
        config=config,
        device=device,
    )
    features_b = build_track_features(
        stems_b, analysis_b,
        n_frames=config.model.n_frames,
        config=config,
        device=device,
    )
    
    # Generate transition
    print(f"Generating transition ({args.duration}s, {args.steps} steps)...")
    
    mixed_audio, params = generate_transition(
        model=model,
        mixer=mixer,
        track_a_stems=stems_a,
        track_b_stems=stems_b,
        track_a_features=features_a,
        track_b_features=features_b,
        n_frames=config.model.n_frames,
        device=device,
        guidance_scale=args.guidance_scale,
        n_inference_steps=args.steps,
    )
    
    # Save output
    print(f"Saving to {output_path}")
    torchaudio.save(
        str(output_path),
        mixed_audio.unsqueeze(0),
        sample_rate,
    )
    
    # Save params
    if args.save_params:
        params_path = Path(args.save_params)
        params_data = {
            "track_a": str(track_a_path),
            "track_b": str(track_b_path),
            "duration": args.duration,
            "n_frames": config.model.n_frames,
            "n_params": params.shape[1],
            "params": params.tolist(),
            "param_names": config.effects.param_names,
        }
        with open(params_path, "w") as f:
            json.dump(params_data, f, indent=2)
        print(f"Parameters saved to {params_path}")
    
    # Visualization
    if args.visualize:
        viz_path = output_path.with_suffix(".png")
        try:
            from .visualize import plot_parameters
            plot_parameters(params, output_path=viz_path, config=config)
            print(f"Visualization saved to {viz_path}")
        except ImportError:
            print("Visualization requires matplotlib. Run: pip install matplotlib")
    
    print("Done!")
    
    return {
        "output": str(output_path),
        "params_shape": list(params.shape),
        "duration": args.duration,
    }


def run_inference(
    track_a: str,
    track_b: str,
    output_path: str = "output/transition.wav",
    model_path: Optional[str] = None,
    duration: float = 30.0,
    guidance_scale: float = 2.0,
    n_steps: int = 50,
    device: str = "auto",
    visualize: bool = False,
) -> Dict:
    """
    Run inference from Python (called by CLI).
    
    Args:
        track_a: Path to first (outgoing) track
        track_b: Path to second (incoming) track
        output_path: Output path for mixed audio
        model_path: Path to trained model checkpoint
        duration: Transition duration in seconds
        guidance_scale: Classifier-free guidance scale
        n_steps: Number of diffusion steps
        device: Device to use ("auto", "cpu", "cuda", "mps")
        visualize: Generate parameter visualization
    
    Returns:
        Dict with output info
    """
    if device == "auto":
        device = get_device()
    
    print(f"Using device: {device}")
    
    track_a_path = Path(track_a)
    track_b_path = Path(track_b)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = get_inference_config()
    
    # Load or create model
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path}")
        model = load_model(model_path, device=device)
    else:
        print("No model checkpoint provided, using random initialization")
        print("(Results will be random - train a model first!)")
        from .model import create_model
        model = create_model(config)
        model.to(device)
    
    # Create mixer
    mixer_config = MixerConfig(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
    )
    mixer = DifferentiableDJMixer(mixer_config)
    mixer.to(device)
    
    # Stem separation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("Running stem separation...")
        stems_a_dir = run_demucs(track_a_path, temp_path, device=device)
        stems_b_dir = run_demucs(track_b_path, temp_path, device=device)
        
        if stems_a_dir is None or stems_b_dir is None:
            print("Stem separation failed")
            raise RuntimeError("Demucs stem separation failed")
        
        # Load stems
        print("Loading stems...")
        sample_rate = config.audio.sample_rate
        
        stems_a = load_stems(stems_a_dir, sample_rate)
        stems_b = load_stems(stems_b_dir, sample_rate)
        
        # Determine transition length
        trans_samples = int(duration * sample_rate)
        
        # Align stems to transition length
        def align_stems(stems: Dict[str, torch.Tensor], length: int) -> Dict[str, torch.Tensor]:
            aligned = {}
            for name, audio in stems.items():
                if len(audio) < length:
                    audio = F.pad(audio, (0, length - len(audio)))
                else:
                    audio = audio[:length]
                aligned[name] = audio
            return aligned
        
        # Take end of track A, start of track B
        stems_a = {name: s[-trans_samples:] for name, s in stems_a.items()}
        stems_b = {name: s[:trans_samples] for name, s in stems_b.items()}
        
        stems_a = align_stems(stems_a, trans_samples)
        stems_b = align_stems(stems_b, trans_samples)
        
        # Analyze tracks
        print("Analyzing tracks...")
        analysis_a = analyze_track(track_a_path, sample_rate)
        analysis_b = analyze_track(track_b_path, sample_rate)
        
        print(f"Track A: {analysis_a.get('bpm', '?')} BPM, {analysis_a.get('key', {}).get('key', '?')} {analysis_a.get('key', {}).get('scale', '')}")
        print(f"Track B: {analysis_b.get('bpm', '?')} BPM, {analysis_b.get('key', {}).get('key', '?')} {analysis_b.get('key', {}).get('scale', '')}")
        
        # Build features
        features_a = build_track_features(
            stems_a, analysis_a,
            n_frames=config.model.n_frames,
            config=config,
            device=device,
        )
        features_b = build_track_features(
            stems_b, analysis_b,
            n_frames=config.model.n_frames,
            config=config,
            device=device,
        )
        
        # Generate transition
        print(f"Generating transition ({duration}s, {n_steps} steps)...")
        
        mixed_audio, params = generate_transition(
            model=model,
            mixer=mixer,
            track_a_stems=stems_a,
            track_b_stems=stems_b,
            track_a_features=features_a,
            track_b_features=features_b,
            n_frames=config.model.n_frames,
            device=device,
            guidance_scale=guidance_scale,
            n_inference_steps=n_steps,
        )
    
    # Save output
    print(f"Saving to {output_file}")
    torchaudio.save(
        str(output_file),
        mixed_audio.unsqueeze(0),
        sample_rate,
    )
    
    # Visualization
    if visualize:
        viz_path = output_file.with_suffix(".png")
        try:
            from .visualize import plot_parameters
            plot_parameters(params, output_path=viz_path, config=config)
            print(f"Visualization saved to {viz_path}")
        except ImportError:
            print("Visualization requires matplotlib. Run: pip install matplotlib")
    
    print("Done!")
    
    return {
        "output": str(output_file),
        "params_shape": list(params.shape),
        "duration": duration,
    }


if __name__ == "__main__":
    main()
