#!/usr/bin/env python3
"""
DJTransGAN v2 Dataset

PyTorch Dataset class for DJ transition training data.
Loads pre-processed track pairs (stems + analysis) and real DJ transitions.

Supports automatic download from Google Cloud Storage if local data is missing.
"""

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import gcsfs
    HAS_GCSFS = True
except ImportError:
    HAS_GCSFS = False

from .model.config import Config, AudioConfig
from .model.conditioning import TrackFeatures

logger = logging.getLogger(__name__)


def ensure_gcs_data(
    local_dir: Union[str, Path],
    gcs_path: str,
    use_gsutil: bool = True,
) -> bool:
    """
    Ensure data exists locally, downloading from GCS if needed.
    
    Args:
        local_dir: Local directory path for data
        gcs_path: GCS path (gs://bucket/path)
        use_gsutil: Use gsutil CLI instead of gcsfs (faster for large dirs)
        
    Returns:
        True if data is available, False otherwise
    """
    local_dir = Path(local_dir)
    
    # Check if local data already exists and has content
    if local_dir.exists():
        # Check if it has actual data (at least one subdirectory with analysis.json)
        subdirs = list(local_dir.iterdir()) if local_dir.is_dir() else []
        has_data = any(
            (d / "analysis.json").exists() 
            for d in subdirs 
            if d.is_dir() and not d.name.startswith('_')
        )
        if has_data:
            logger.info(f"Local data found at {local_dir}, skipping download")
            return True
    
    logger.info(f"Local data not found at {local_dir}, downloading from {gcs_path}")
    
    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    if use_gsutil:
        # Use gsutil for faster parallel downloads
        try:
            cmd = ["gsutil", "-m", "rsync", "-r", gcs_path.rstrip('/') + '/', str(local_dir) + '/']
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"gsutil failed: {result.stderr}")
                return False
            logger.info(f"Successfully downloaded data from {gcs_path}")
            return True
        except subprocess.TimeoutExpired:
            logger.error("gsutil download timed out after 1 hour")
            return False
        except FileNotFoundError:
            logger.warning("gsutil not found, falling back to gcsfs")
    
    # Fallback to gcsfs
    if not HAS_GCSFS:
        logger.error("Neither gsutil nor gcsfs available for GCS download")
        return False
    
    try:
        fs = gcsfs.GCSFileSystem()
        
        # Parse GCS path
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[5:]
        
        # List and download all files
        files = fs.glob(f"{gcs_path}/**/*")
        for remote_file in files:
            if fs.isfile(remote_file):
                rel_path = remote_file.replace(gcs_path.rstrip('/') + '/', '')
                local_file = local_dir / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.debug(f"Downloading {remote_file} -> {local_file}")
                fs.get(remote_file, str(local_file))
        
        logger.info(f"Successfully downloaded data from gs://{gcs_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        return False


# Constants
STEM_NAMES = ['drums', 'bass', 'vocals', 'other']
STRUCTURE_LABELS = ['intro', 'verse', 'chorus', 'drop', 'bridge', 'outro', 'breakdown', 'body']


@dataclass
class TransitionSample:
    """A single training sample containing track pair and transition."""
    pair_id: str
    track_a_stems: Dict[str, torch.Tensor]  # stem_name -> [samples]
    track_b_stems: Dict[str, torch.Tensor]
    track_a_features: TrackFeatures
    track_b_features: TrackFeatures
    transition_audio: torch.Tensor  # [samples] - the real DJ transition
    analysis: Dict  # Raw analysis data


def load_audio(path: Union[str, Path], target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Load audio file and resample if needed."""
    path = Path(path)
    
    if torchaudio is not None:
        waveform, sr = torchaudio.load(str(path))
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), target_sr
    elif librosa is not None:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return torch.from_numpy(y).float(), target_sr
    else:
        raise ImportError("Need either torchaudio or librosa installed")


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> torch.Tensor:
    """Compute mel spectrogram from audio."""
    if torchaudio is not None:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel = mel_transform(audio.unsqueeze(0)).squeeze(0)
        # Convert to dB and normalize
        mel = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
        mel = (mel + 40) / 40  # Normalize to roughly [-1, 1]
        return mel.clamp(-1, 1)
    elif librosa is not None:
        mel = librosa.feature.melspectrogram(
            y=audio.numpy(),
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel + 40) / 40
        return torch.from_numpy(mel).float().clamp(-1, 1)
    else:
        raise ImportError("Need either torchaudio or librosa")


def structure_to_onehot(
    structure_segments: List[Dict],
    total_frames: int,
    hop_length: int,
    sample_rate: int,
) -> torch.Tensor:
    """Convert structure segments to one-hot encoding per frame."""
    n_labels = len(STRUCTURE_LABELS)
    result = torch.zeros(total_frames, n_labels)
    
    for seg in structure_segments:
        seg_type = seg.get('type', 'body')
        if seg_type not in STRUCTURE_LABELS:
            seg_type = 'body'
        label_idx = STRUCTURE_LABELS.index(seg_type)
        
        start_sec = seg.get('start', 0)
        end_sec = seg.get('end', 0)
        
        start_frame = int(start_sec * sample_rate / hop_length)
        end_frame = int(end_sec * sample_rate / hop_length)
        
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))
        
        if end_frame > start_frame:
            result[start_frame:end_frame, label_idx] = 1.0
    
    # Default to 'body' where nothing is specified
    no_label = result.sum(dim=1) == 0
    body_idx = STRUCTURE_LABELS.index('body')
    result[no_label, body_idx] = 1.0
    
    return result


def beats_to_tensor(
    beat_times: List[float],
    total_frames: int,
    hop_length: int,
    sample_rate: int,
) -> torch.Tensor:
    """Convert beat times to per-frame binary tensor."""
    result = torch.zeros(total_frames)
    
    for bt in beat_times:
        frame_idx = int(bt * sample_rate / hop_length)
        if 0 <= frame_idx < total_frames:
            result[frame_idx] = 1.0
    
    return result


def key_to_onehot(key_info: Dict, n_keys: int = 24) -> torch.Tensor:
    """Convert key info to one-hot encoding."""
    result = torch.zeros(n_keys)
    
    key_name = key_info.get('key', 'C')
    scale = key_info.get('scale', 'major')
    
    key_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
               'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
               'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    key_idx = key_map.get(key_name, 0)
    if scale == 'minor':
        key_idx += 12
    
    if 0 <= key_idx < n_keys:
        result[key_idx] = 1.0
    else:
        result[0] = 1.0  # Default to C major
    
    return result


class DJTransitionDataset(Dataset):
    """
    Dataset for DJ transitions.
    
    Expected directory structure:
    processed/
      {pair_id}/
        track_a_stems/
          drums.wav
          bass.wav
          vocals.wav
          other.wav
        track_b_stems/
          drums.wav
          bass.wav
          vocals.wav  
          other.wav
        transition.wav
        analysis.json
    
    analysis.json contains:
    {
        "track_a": {bpm, beat_times, key, structure, ...},
        "track_b": {bpm, beat_times, key, structure, ...},
        "transition_duration": float,
        ...
    }
    
    Supports automatic download from GCS if local data is missing.
    Set gcs_bucket to enable (e.g., "gs://clawd139/automix-data/processed")
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        n_frames: int = 128,
        augment: bool = True,
        max_samples: Optional[int] = None,
        gcs_bucket: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config or Config()
        self.n_frames = n_frames
        self.augment = augment
        self.gcs_bucket = gcs_bucket
        
        self.sample_rate = self.config.audio.sample_rate
        self.hop_length = self.config.audio.hop_length
        self.n_mels = self.config.audio.n_mels
        
        # Download from GCS if needed
        if gcs_bucket:
            if not ensure_gcs_data(self.data_dir, gcs_bucket):
                logger.warning(f"Failed to download data from {gcs_bucket}")
        
        # Find all valid samples
        self.samples = self._scan_samples()
        
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"DJTransitionDataset: Found {len(self.samples)} samples in {data_dir}")
    
    def _scan_samples(self) -> List[Path]:
        """Scan data directory for valid samples."""
        samples = []
        
        if not self.data_dir.exists():
            return samples
        
        for pair_dir in self.data_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            
            # Check required files exist
            required = [
                pair_dir / "track_a_stems",
                pair_dir / "track_b_stems", 
                pair_dir / "transition.wav",
                pair_dir / "analysis.json",
            ]
            
            # Also accept .mp3 for transition
            if not (pair_dir / "transition.wav").exists():
                if (pair_dir / "transition.mp3").exists():
                    required[2] = pair_dir / "transition.mp3"
                else:
                    continue
            
            stems_a = pair_dir / "track_a_stems"
            stems_b = pair_dir / "track_b_stems"
            
            # Check stems exist
            stem_files_ok = True
            for stem_name in STEM_NAMES:
                for stems_dir in [stems_a, stems_b]:
                    stem_path = stems_dir / f"{stem_name}.wav"
                    if not stem_path.exists():
                        stem_path = stems_dir / f"{stem_name}.mp3"
                        if not stem_path.exists():
                            stem_files_ok = False
                            break
                if not stem_files_ok:
                    break
            
            if stem_files_ok and (pair_dir / "analysis.json").exists():
                samples.append(pair_dir)
        
        return sorted(samples)
    
    def _load_stems(self, stems_dir: Path) -> Dict[str, torch.Tensor]:
        """Load all stems from a directory."""
        stems = {}
        for stem_name in STEM_NAMES:
            stem_path = stems_dir / f"{stem_name}.wav"
            if not stem_path.exists():
                stem_path = stems_dir / f"{stem_name}.mp3"
            
            audio, _ = load_audio(stem_path, self.sample_rate)
            stems[stem_name] = audio
        
        return stems
    
    def _align_stems(
        self,
        stems: Dict[str, torch.Tensor],
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Align all stems to same length."""
        aligned = {}
        for name, audio in stems.items():
            if len(audio) < target_length:
                # Pad with zeros
                pad = target_length - len(audio)
                audio = F.pad(audio, (0, pad))
            elif len(audio) > target_length:
                # Truncate
                audio = audio[:target_length]
            aligned[name] = audio
        return aligned
    
    def _features_from_analysis(
        self,
        analysis: Dict,
        stems: Dict[str, torch.Tensor],
    ) -> TrackFeatures:
        """Convert analysis dict to TrackFeatures."""
        # Compute mel spectrograms for each stem
        stem_mels = []
        for stem_name in STEM_NAMES:
            mel = compute_mel_spectrogram(
                stems[stem_name],
                self.sample_rate,
                self.config.audio.n_fft,
                self.hop_length,
                self.n_mels,
            )
            stem_mels.append(mel)
        
        # Stack stems: [n_stems, n_mels, n_frames_raw]
        stem_mels = torch.stack(stem_mels, dim=0)
        n_frames_raw = stem_mels.shape[-1]
        
        # Get structure, beats, etc.
        structure = structure_to_onehot(
            analysis.get('structure', []),
            n_frames_raw,
            self.hop_length,
            self.sample_rate,
        )
        
        beats = beats_to_tensor(
            analysis.get('beat_times', []),
            n_frames_raw,
            self.hop_length,
            self.sample_rate,
        )
        
        # Approximate downbeats as every 4th beat
        beat_times = analysis.get('beat_times', [])
        downbeat_times = beat_times[::4] if beat_times else []
        downbeats = beats_to_tensor(
            downbeat_times,
            n_frames_raw,
            self.hop_length,
            self.sample_rate,
        )
        
        # BPM
        bpm = torch.tensor([analysis.get('bpm', 128.0)])
        
        # Key
        key_info = analysis.get('key', {'key': 'C', 'scale': 'major'})
        key = key_to_onehot(key_info, self.config.audio.n_keys)
        
        # Energy curve (compute from audio)
        total_audio = sum(stems.values())
        energy = torch.from_numpy(
            librosa.feature.rms(
                y=total_audio.numpy(),
                frame_length=self.config.audio.n_fft,
                hop_length=self.hop_length,
            )[0]
        ).float() if librosa else torch.zeros(n_frames_raw)
        
        # Interpolate all to target n_frames
        stem_mels = F.interpolate(
            stem_mels.unsqueeze(0),
            size=self.n_frames,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        
        structure = F.interpolate(
            structure.T.unsqueeze(0),
            size=self.n_frames,
            mode='nearest',
        ).squeeze(0).T
        
        beats = F.interpolate(
            beats.unsqueeze(0).unsqueeze(0),
            size=self.n_frames,
            mode='nearest',
        ).squeeze()
        
        downbeats = F.interpolate(
            downbeats.unsqueeze(0).unsqueeze(0),
            size=self.n_frames,
            mode='nearest',
        ).squeeze()
        
        energy = F.interpolate(
            energy.unsqueeze(0).unsqueeze(0),
            size=self.n_frames,
            mode='linear',
            align_corners=False,
        ).squeeze()
        
        return TrackFeatures(
            stem_mels=stem_mels,
            structure=structure,
            beats=beats,
            downbeats=downbeats,
            bpm=bpm,
            key=key,
            energy=energy,
        )
    
    def _augment_pitch_shift(
        self,
        stems_a: Dict[str, torch.Tensor],
        stems_b: Dict[str, torch.Tensor],
        semitones: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply pitch shift augmentation to both tracks."""
        if torchaudio is None or semitones == 0:
            return stems_a, stems_b
        
        # torchaudio pitch shift
        def shift_stems(stems):
            shifted = {}
            for name, audio in stems.items():
                shifted[name] = torchaudio.functional.pitch_shift(
                    audio.unsqueeze(0),
                    self.sample_rate,
                    semitones,
                ).squeeze(0)
            return shifted
        
        return shift_stems(stems_a), shift_stems(stems_b)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        pair_dir = self.samples[idx]
        
        # Load analysis
        with open(pair_dir / "analysis.json") as f:
            analysis = json.load(f)
        
        # Load stems
        stems_a = self._load_stems(pair_dir / "track_a_stems")
        stems_b = self._load_stems(pair_dir / "track_b_stems")
        
        # Load transition audio
        trans_path = pair_dir / "transition.wav"
        if not trans_path.exists():
            trans_path = pair_dir / "transition.mp3"
        transition_audio, _ = load_audio(trans_path, self.sample_rate)
        
        # Determine target length (from transition)
        target_length = len(transition_audio)
        
        # Align stems to transition length
        stems_a = self._align_stems(stems_a, target_length)
        stems_b = self._align_stems(stems_b, target_length)
        
        # Apply augmentation
        if self.augment and random.random() < 0.3:
            semitones = random.randint(-3, 3)
            if semitones != 0:
                stems_a, stems_b = self._augment_pitch_shift(stems_a, stems_b, semitones)
        
        # Build features
        track_a_analysis = analysis.get('track_a', analysis)
        track_b_analysis = analysis.get('track_b', analysis)
        
        features_a = self._features_from_analysis(track_a_analysis, stems_a)
        features_b = self._features_from_analysis(track_b_analysis, stems_b)
        
        return {
            'pair_id': pair_dir.name,
            'track_a_stems': stems_a,
            'track_b_stems': stems_b,
            'track_a_features': features_a,
            'track_b_features': features_b,
            'transition_audio': transition_audio,
        }


class SyntheticTransitionDataset(Dataset):
    """
    Dataset that generates synthetic training data from individual tracks.
    
    Uses simple crossfade transitions as pseudo ground truth,
    useful for pre-training or when real DJ transitions aren't available.
    """
    
    def __init__(
        self,
        tracks_dir: Union[str, Path],
        stems_dir: Union[str, Path],
        config: Optional[Config] = None,
        n_frames: int = 128,
        transition_duration: float = 30.0,
        max_pairs: Optional[int] = None,
    ):
        self.tracks_dir = Path(tracks_dir)
        self.stems_dir = Path(stems_dir)
        self.config = config or Config()
        self.n_frames = n_frames
        self.transition_duration = transition_duration
        
        self.sample_rate = self.config.audio.sample_rate
        self.hop_length = self.config.audio.hop_length
        
        # Find all tracks with stems
        self.tracks = self._scan_tracks()
        
        # Generate random pairs
        self.pairs = self._generate_pairs(max_pairs)
        
        print(f"SyntheticTransitionDataset: {len(self.tracks)} tracks, {len(self.pairs)} pairs")
    
    def _scan_tracks(self) -> List[Path]:
        """Find tracks that have stem separation."""
        tracks = []
        
        for track_path in self.tracks_dir.glob("*.mp3"):
            stem_dir = self.stems_dir / track_path.stem
            if stem_dir.exists():
                # Check all stems exist
                has_all = all(
                    (stem_dir / f"{s}.wav").exists() or (stem_dir / f"{s}.mp3").exists()
                    for s in STEM_NAMES
                )
                if has_all:
                    tracks.append(track_path)
        
        return tracks
    
    def _generate_pairs(self, max_pairs: Optional[int]) -> List[Tuple[Path, Path]]:
        """Generate random track pairs."""
        if len(self.tracks) < 2:
            return []
        
        pairs = []
        n_pairs = max_pairs or len(self.tracks) * 2
        
        for _ in range(n_pairs):
            a, b = random.sample(self.tracks, 2)
            pairs.append((a, b))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        track_a_path, track_b_path = self.pairs[idx]
        
        # Load stems
        def load_track_stems(track_path: Path) -> Dict[str, torch.Tensor]:
            stem_dir = self.stems_dir / track_path.stem
            stems = {}
            for stem_name in STEM_NAMES:
                stem_path = stem_dir / f"{stem_name}.wav"
                if not stem_path.exists():
                    stem_path = stem_dir / f"{stem_name}.mp3"
                audio, _ = load_audio(stem_path, self.sample_rate)
                stems[stem_name] = audio
            return stems
        
        stems_a = load_track_stems(track_a_path)
        stems_b = load_track_stems(track_b_path)
        
        # Create synthetic transition (simple crossfade)
        trans_samples = int(self.transition_duration * self.sample_rate)
        
        # Align stems
        for stems in [stems_a, stems_b]:
            min_len = min(len(s) for s in stems.values())
            for name in stems:
                stems[name] = stems[name][:min_len]
        
        # Extract transition region from each track
        # Use end of track A and start of track B
        min_len_a = min(len(s) for s in stems_a.values())
        min_len_b = min(len(s) for s in stems_b.values())
        
        if min_len_a >= trans_samples and min_len_b >= trans_samples:
            # Take last part of A, first part of B
            for name in stems_a:
                stems_a[name] = stems_a[name][-trans_samples:]
            for name in stems_b:
                stems_b[name] = stems_b[name][:trans_samples]
            
            # Create crossfade
            t = torch.linspace(0, 1, trans_samples)
            fade_out = torch.cos(t * np.pi / 2)
            fade_in = torch.sin(t * np.pi / 2)
            
            transition_audio = torch.zeros(trans_samples)
            for name in STEM_NAMES:
                transition_audio += stems_a[name] * fade_out
                transition_audio += stems_b[name] * fade_in
        else:
            # Tracks too short, just use zeros
            for name in STEM_NAMES:
                stems_a[name] = torch.zeros(trans_samples)
                stems_b[name] = torch.zeros(trans_samples)
            transition_audio = torch.zeros(trans_samples)
        
        # Build simple analysis
        analysis = {
            'bpm': 128.0,
            'beat_times': list(np.arange(0, self.transition_duration, 60/128)),
            'key': {'key': 'C', 'scale': 'major'},
            'structure': [{'type': 'body', 'start': 0, 'end': self.transition_duration}],
        }
        
        # Build features (simplified)
        features_a = self._make_features(stems_a, analysis)
        features_b = self._make_features(stems_b, analysis)
        
        return {
            'pair_id': f"{track_a_path.stem}___{track_b_path.stem}",
            'track_a_stems': stems_a,
            'track_b_stems': stems_b,
            'track_a_features': features_a,
            'track_b_features': features_b,
            'transition_audio': transition_audio,
        }
    
    def _make_features(self, stems: Dict[str, torch.Tensor], analysis: Dict) -> TrackFeatures:
        """Build TrackFeatures from stems and analysis."""
        # Compute mels
        stem_mels = []
        for stem_name in STEM_NAMES:
            mel = compute_mel_spectrogram(
                stems[stem_name],
                self.sample_rate,
                self.config.audio.n_fft,
                self.hop_length,
                self.config.audio.n_mels,
            )
            stem_mels.append(mel)
        
        stem_mels = torch.stack(stem_mels, dim=0)
        n_frames_raw = stem_mels.shape[-1]
        
        # Interpolate to target frames
        stem_mels = F.interpolate(
            stem_mels.unsqueeze(0),
            size=self.n_frames,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        
        # Simple features
        structure = torch.zeros(self.n_frames, len(STRUCTURE_LABELS))
        structure[:, STRUCTURE_LABELS.index('body')] = 1.0
        
        beats = torch.zeros(self.n_frames)
        downbeats = torch.zeros(self.n_frames)
        
        bpm = torch.tensor([analysis.get('bpm', 128.0)])
        key = key_to_onehot(analysis.get('key', {}), self.config.audio.n_keys)
        energy = torch.ones(self.n_frames) * 0.5
        
        return TrackFeatures(
            stem_mels=stem_mels,
            structure=structure,
            beats=beats,
            downbeats=downbeats,
            bpm=bpm,
            key=key,
            energy=energy,
        )


def collate_transitions(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    
    def stack_features(features_list: List[TrackFeatures]) -> TrackFeatures:
        return TrackFeatures(
            stem_mels=torch.stack([f.stem_mels for f in features_list]),
            structure=torch.stack([f.structure for f in features_list]),
            beats=torch.stack([f.beats for f in features_list]),
            downbeats=torch.stack([f.downbeats for f in features_list]),
            bpm=torch.stack([f.bpm for f in features_list]),
            key=torch.stack([f.key for f in features_list]),
            energy=torch.stack([f.energy for f in features_list]),
        )
    
    def stack_stems(stems_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Stack stems to [batch, n_stems, samples]."""
        # Find min length
        min_len = min(
            min(s.shape[-1] for s in stems.values())
            for stems in stems_list
        )
        
        batched = []
        for stems in stems_list:
            stem_tensors = [stems[name][:min_len] for name in STEM_NAMES]
            batched.append(torch.stack(stem_tensors, dim=0))
        
        return torch.stack(batched, dim=0)
    
    # Stack transition audio
    min_trans_len = min(b['transition_audio'].shape[-1] for b in batch)
    transition_audio = torch.stack([
        b['transition_audio'][:min_trans_len] for b in batch
    ])
    
    return {
        'pair_ids': [b['pair_id'] for b in batch],
        'track_a_stems': stack_stems([b['track_a_stems'] for b in batch]),
        'track_b_stems': stack_stems([b['track_b_stems'] for b in batch]),
        'track_a_features': stack_features([b['track_a_features'] for b in batch]),
        'track_b_features': stack_features([b['track_b_features'] for b in batch]),
        'transition_audio': transition_audio,
    }


if __name__ == "__main__":
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="processed")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    
    if args.synthetic:
        dataset = SyntheticTransitionDataset(
            tracks_dir="data/tracks",
            stems_dir="data/stems",
            config=config,
            max_pairs=10,
        )
    else:
        dataset = DJTransitionDataset(
            data_dir=args.data_dir,
            config=config,
        )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Transition audio shape: {sample['transition_audio'].shape}")
        print(f"Track A stems: {list(sample['track_a_stems'].keys())}")
        print(f"Track A mels shape: {sample['track_a_features'].stem_mels.shape}")
