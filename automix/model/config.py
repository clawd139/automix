"""
Configuration for DJ Transition Model

Hyperparameters for the conditional diffusion model that generates
effect parameter curves for DJ transitions.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    
    # Stem names from demucs
    stems: List[str] = field(default_factory=lambda: ["drums", "bass", "vocals", "other"])
    
    # Structure labels (must match dataset.py STRUCTURE_LABELS)
    structure_labels: List[str] = field(default_factory=lambda: [
        "intro", "verse", "chorus", "drop", "bridge", "outro", "breakdown", "body"
    ])
    
    # Musical keys (12 major + 12 minor)
    n_keys: int = 24


@dataclass
class EffectConfig:
    """Effect parameter configuration.
    
    Must match effects/mixer.py parameter layout exactly:
    - 2 tracks × (4 stems × 6 params + 3 global) = 54 total
    
    Per-stem params (6): gain, eq_low, eq_mid, eq_high, lpf, hpf
    Global params (3): reverb_send, delay_send, delay_feedback
    """
    # Per-stem parameters
    # - gain: 1 param
    # - eq_low, eq_mid, eq_high: 3 params  
    # - lpf, hpf: 2 params
    params_per_stem: int = 6
    
    # Global effects per track
    # - reverb_send: 1 param
    # - delay_send: 1 param
    # - delay_feedback: 1 param
    global_params_per_track: int = 3
    
    n_stems: int = 4
    n_tracks: int = 2
    
    @property
    def n_params(self) -> int:
        """Total number of effect parameters per frame."""
        # 2 tracks × (4 stems × 6 params + 3 global) = 54
        return self.n_tracks * (self.n_stems * self.params_per_stem + self.global_params_per_track)
    
    # Parameter names for interpretability (must match mixer layout)
    @property
    def param_names(self) -> List[str]:
        """Parameter names matching effects/mixer.py layout exactly."""
        names = []
        for track in ["A", "B"]:
            # Stem params first (4 stems × 6 params = 24)
            for stem in ["drums", "bass", "vocals", "other"]:
                names.extend([
                    f"{track}_{stem}_gain",
                    f"{track}_{stem}_eq_low",
                    f"{track}_{stem}_eq_mid", 
                    f"{track}_{stem}_eq_high",
                    f"{track}_{stem}_lpf",
                    f"{track}_{stem}_hpf",
                ])
            # Global params for this track (3)
            names.extend([
                f"{track}_reverb_send",
                f"{track}_delay_send",
                f"{track}_delay_feedback",
            ])
        return names


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Conditioning encoder
    audio_embed_dim: int = 256
    structure_embed_dim: int = 64
    beat_embed_dim: int = 64
    condition_dim: int = 512  # Combined conditioning dimension
    
    # Diffusion model (U-Net style)
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    
    # Time embedding
    time_embed_dim: int = 128
    
    # Output
    n_frames: int = 128  # Default frames for 32 bars at ~4 frames/beat
    
    @property
    def n_params(self) -> int:
        return EffectConfig().n_params


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # "linear" or "cosine"
    
    # Sampling
    n_inference_steps: int = 50  # DDIM-style fast sampling
    guidance_scale: float = 2.0  # Classifier-free guidance


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Mixed precision
    use_amp: bool = True
    
    # EMA for model weights
    ema_decay: float = 0.9999
    
    # Classifier-free guidance dropout
    condition_dropout: float = 0.1
    
    # Loss weights
    diffusion_loss_weight: float = 1.0
    perceptual_loss_weight: float = 0.1  # Compare mixed audio spectrograms
    smoothness_loss_weight: float = 0.01  # Encourage smooth parameter curves


@dataclass
class Config:
    """Complete configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    effects: EffectConfig = field(default_factory=EffectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    def __post_init__(self):
        # Sync n_params between configs
        self.model.n_params  # This triggers the property


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def get_inference_config() -> Config:
    """Get configuration optimized for inference."""
    config = Config()
    config.diffusion.n_inference_steps = 25  # Faster sampling
    config.training.use_amp = False  # More stable on MPS
    return config
