# DJ Transition Model v2

A conditional diffusion model for generating DJ transition effect parameters. This module learns to create smooth, musical transitions between two tracks by predicting effect parameter curves that control a differentiable mixer.

## Overview

Unlike the original DJtransGAN (ICASSP 2022) which used a GAN, this v2 architecture uses **conditional diffusion** for several advantages:

- **More stable training** — No adversarial dynamics, mode collapse, or training instability
- **Better quality** — Diffusion models produce higher-quality outputs in low-dimensional spaces
- **Diverse outputs** — Can generate multiple valid transitions for the same track pair
- **Classifier-free guidance** — Controllable generation strength

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DJ Transition Diffusion Model                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CONDITIONING ENCODER                             │   │
│  │                                                                      │   │
│  │   Track A                              Track B                       │   │
│  │   ┌─────────┐                          ┌─────────┐                  │   │
│  │   │ Stems   │ ──► MelEncoder ──┐       │ Stems   │ ──► MelEncoder ──┤   │
│  │   │ (4 ch)  │                  │       │ (4 ch)  │                  │   │
│  │   └─────────┘                  │       └─────────┘                  │   │
│  │                                │                                    │   │
│  │   ┌─────────┐                  │       ┌─────────┐                  │   │
│  │   │Structure│ ──┐              │       │Structure│ ──┐              │   │
│  │   │ Beats   │   │              │       │ Beats   │   │              │   │
│  │   │ BPM/Key │ MetadataEncoder ─┤       │ BPM/Key │ MetadataEncoder ─┤   │
│  │   │ Energy  │   │              │       │ Energy  │   │              │   │
│  │   └─────────┘ ──┘              │       └─────────┘ ──┘              │   │
│  │                                ▼                                    │   │
│  │                    ┌─────────────────────────────┐                  │   │
│  │                    │     Cross-Attention         │                  │   │
│  │                    │   (A attends to B & vice)   │                  │   │
│  │                    └─────────────────────────────┘                  │   │
│  │                                │                                    │   │
│  │                                ▼                                    │   │
│  │                    ┌─────────────────────────────┐                  │   │
│  │                    │   Conditioning Embedding    │                  │   │
│  │                    │   [batch, cond_dim, frames] │                  │   │
│  │                    └─────────────────────────────┘                  │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PARAMETER DIFFUSION MODEL                        │   │
│  │                                                                      │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐        │   │
│  │   │ Noisy Params│ ──► │ Input Proj  │ ──► │ + Pos Embed │         │   │
│  │   │ [B,F,P]     │      │             │      │             │         │   │
│  │   └─────────────┘      └─────────────┘      └──────┬──────┘         │   │
│  │                                                     │                │   │
│  │   ┌─────────────┐                                   │                │   │
│  │   │ Timestep t  │ ──► Time Embed ──────────────────┼─┐              │   │
│  │   └─────────────┘                                   │ │              │   │
│  │                                                     ▼ ▼              │   │
│  │                              ┌────────────────────────────┐         │   │
│  │                              │   Transformer Blocks (×6)   │         │   │
│  │                              │                             │         │   │
│  │   Conditioning ─────────────►│  • Self-Attention          │         │   │
│  │                              │  • Cross-Attention (cond)   │         │   │
│  │                              │  • FeedForward              │         │   │
│  │                              │  • AdaLN (time modulation)  │         │   │
│  │                              └────────────────────────────┘         │   │
│  │                                            │                        │   │
│  │                                            ▼                        │   │
│  │                              ┌─────────────────────────────┐        │   │
│  │                              │     Output Projection        │        │   │
│  │                              │     Predicted Noise          │        │   │
│  │                              │     [batch, frames, params]  │        │   │
│  │                              └─────────────────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Diffusion Process

```
Training (forward diffusion):
   x₀ (clean params) ──► add noise at timestep t ──► xₜ (noisy params)
   
   Model learns: xₜ, t, conditioning ──► ε̂ (predicted noise)
   
   Loss: MSE(ε̂, ε)

Inference (reverse diffusion):
   xₜ (pure noise) ──► iteratively denoise ──► x₀ (clean params)
   
   For each step: xₜ ──► predict noise ──► estimate x₀ ──► xₜ₋₁
```

## Effect Parameters

The model outputs 52 parameters per frame:

```
Per Track (×2 tracks):
  Per Stem (×4 stems: drums, bass, vocals, other):
    - gain:       Volume level [0, 1] → [-inf, 0] dB
    - low_eq:     Low frequency EQ [0, 1] → [-12, +12] dB
    - mid_eq:     Mid frequency EQ [0, 1] → [-12, +12] dB
    - high_eq:    High frequency EQ [0, 1] → [-12, +12] dB
    - highpass:   Highpass cutoff [0, 1] → [20, 2000] Hz
    - lowpass:    Lowpass cutoff [0, 1] → [1000, 20000] Hz
    
Global:
  - crossfader:   Track blend [0, 1] (0=A, 1=B)
  - master_gain:  Master volume [0, 1]
  - reverb_wet:   Reverb amount [0, 1]
  - reverb_decay: Reverb decay time [0, 1]

Total: 2 × 4 × 6 + 4 = 52 parameters
```

## Input Features

### Per Track:
| Feature | Shape | Description |
|---------|-------|-------------|
| `stem_mels` | `[4, 128, T]` | Mel spectrograms per stem (from Demucs) |
| `structure` | `[T, 6]` | One-hot segment labels (intro/verse/chorus/drop/bridge/outro) |
| `beats` | `[T]` | Binary beat positions |
| `downbeats` | `[T]` | Binary downbeat positions |
| `bpm` | `[1]` | Tempo (40-200 BPM) |
| `key` | `[24]` | One-hot key (12 major + 12 minor) |
| `energy` | `[T]` | RMS energy curve |

## Usage

### Quick Start

```python
from model import create_model, get_config, TrackFeatures
import torch

# Create model
config = get_config()
model = create_model(config)
model.to("cuda")  # or "mps" for Mac

# Prepare track features (from your analysis pipeline)
track_a = TrackFeatures(
    stem_mels=torch.randn(1, 4, 128, 512),  # [batch, stems, mels, frames]
    structure=torch.zeros(1, 512, 6),        # One-hot labels
    beats=torch.zeros(1, 512),
    downbeats=torch.zeros(1, 512),
    bpm=torch.tensor([[128.0]]),
    key=torch.zeros(1, 24),
    energy=torch.rand(1, 512),
)
track_b = TrackFeatures(...)  # Same format

# Generate transition
params = model.generate(track_a, track_b, n_frames=128)
# params: [1, 128, 52] — effect curves over 128 frames
```

### Training

```python
from model import train

# With real data
train(
    data_path="./data/transitions/",
    output_dir="./output/",
    config=config,
)

# Or test with dummy data
train(output_dir="./test_output/")
```

### Command Line

```bash
# Train with defaults (dummy data)
python -m model.training --output ./output

# Train with data
python -m model.training --data ./data --output ./output --steps 100000

# Resume training
python -m model.training --resume ./output/checkpoint_50000.pt
```

## Model Dimensions

| Component | Parameters | Notes |
|-----------|------------|-------|
| MelEncoder | ~2M | Shared across stems and tracks |
| MetadataEncoder | ~200K | Per-track metadata |
| CrossAttention | ~500K | Track interaction |
| Denoiser | ~8M | 6 transformer blocks |
| **Total** | **~12M** | Fits on consumer GPU |

## Configuration

Key hyperparameters in `config.py`:

```python
@dataclass
class ModelConfig:
    audio_embed_dim: int = 256      # Audio feature dimension
    condition_dim: int = 512         # Conditioning dimension
    hidden_dim: int = 256           # Transformer hidden dimension
    n_layers: int = 6               # Number of transformer blocks
    n_heads: int = 8                # Attention heads
    n_frames: int = 128             # Default output frames

@dataclass
class DiffusionConfig:
    n_steps: int = 1000             # Training diffusion steps
    n_inference_steps: int = 50     # Fast sampling steps (DDIM)
    guidance_scale: float = 2.0     # Classifier-free guidance
```

## Integration with Mixer

The model outputs parameter curves that feed into the differentiable mixer:

```python
from model import create_model, load_model
from effects import DifferentiableMixer  # From ../effects/

# Load trained model
model = load_model("./best_model.pt")

# Create mixer
mixer = DifferentiableMixer(config)

# Generate and apply
params = model.generate(track_a, track_b)
mixed_audio = mixer(track_a_stems, track_b_stems, params)
```

## Comparison with DJtransGAN v1

| Aspect | v1 (GAN) | v2 (Diffusion) |
|--------|----------|----------------|
| Training stability | Unstable, needs careful tuning | Stable, standard MSE loss |
| Output diversity | Limited (mode collapse) | High (stochastic sampling) |
| Quality control | Fixed | Guidance scale adjustable |
| Inference speed | Fast (single forward pass) | Slower (iterative denoising) |
| Model size | ~5M params | ~12M params |

## References

- [DJtransGAN (ICASSP 2022)](https://arxiv.org/abs/2110.06525) — Original architecture
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM foundation
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG technique
- [Riffusion](https://riffusion.com/) — Diffusion for spectrograms
- [DiT](https://arxiv.org/abs/2212.09748) — Diffusion Transformers

## File Structure

```
model/
├── __init__.py       # Public API
├── config.py         # Hyperparameters
├── conditioning.py   # Audio/metadata encoding
├── model.py          # Diffusion model
├── training.py       # Training loop
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## License

MIT License — see repository root for details.

---

**Part of [automix](https://github.com/clawd139/automix)** — Automatic DJ mixing with neural networks.
