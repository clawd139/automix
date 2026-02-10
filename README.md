# DJTransGAN v2

A conditional diffusion model that generates DJ transition effect parameters. The model learns to create smooth, musically-aware transitions between two tracks by predicting time-varying effect curves that control a differentiable DJ mixer.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       DJTransGAN v2                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Track A ──┬──> Stem Separation (demucs)                        │
│            │         │                                           │
│            │         ▼                                           │
│            │    ┌─────────────────┐                             │
│            ├───>│  Conditioning   │                             │
│            │    │    Encoder      │──┐                          │
│            │    └─────────────────┘  │                          │
│            │                         │    ┌─────────────────┐   │
│  Track B ──┼──> Stem Separation     │    │    Diffusion    │   │
│            │         │               ├───>│      Model      │   │
│            │         ▼               │    │  (Transformer)  │   │
│            │    ┌─────────────────┐  │    └────────┬────────┘   │
│            └───>│  Conditioning   │──┘             │            │
│                 │    Encoder      │                │            │
│                 └─────────────────┘                │            │
│                                                    ▼            │
│                                           Effect Parameters     │
│                                           [n_frames × 54]       │
│                                                    │            │
│  Track A Stems ─────────────────────────┐         │            │
│                                          │         ▼            │
│                                    ┌─────┴─────────────┐        │
│                                    │  Differentiable   │        │
│  Track B Stems ───────────────────>│    DJ Mixer       │        │
│                                    └────────┬──────────┘        │
│                                             │                   │
│                                             ▼                   │
│                                      Mixed Transition           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Parameter Space (54 parameters per frame)

The model outputs 54 effect parameters at each time frame:

### Per Track (27 params × 2 tracks = 54 total)

**Stem Parameters** (4 stems × 6 params = 24 per track):
- `gain` (0-1): Volume fader
- `eq_low` (0-2): Low frequency EQ, 1.0 = unity
- `eq_mid` (0-2): Mid frequency EQ
- `eq_high` (0-2): High frequency EQ
- `lpf` (0-1): Low-pass filter cutoff, 1.0 = fully open
- `hpf` (0-1): High-pass filter cutoff, 0.0 = fully open

**Global Effects** (3 per track):
- `reverb_send` (0-1): Reverb wet/dry mix
- `delay_send` (0-1): Delay wet/dry mix
- `delay_feedback` (0-1): Delay feedback amount

### Stems
- `drums`: Kick, snare, hats, percussion
- `bass`: Bass frequencies
- `vocals`: Vocal content
- `other`: Synths, pads, everything else

## Installation

```bash
# Clone repository
cd djtransgan-v2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install demucs for stem separation
pip install demucs
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- torchaudio
- librosa
- demucs
- matplotlib (for visualization)
- einops

## Quick Start

### 1. Generate a transition (no training required)

```bash
# Using a pre-trained model
python inference.py track_a.mp3 track_b.mp3 -o output/my_transition.wav

# With visualization
python inference.py track_a.mp3 track_b.mp3 -o output/my_transition.wav --visualize

# Adjust transition duration
python inference.py track_a.mp3 track_b.mp3 -o output/my_transition.wav --duration 45
```

### 2. Prepare training data

```bash
# From a track library (creates synthetic crossfade transitions)
python pipeline.py --tracks ~/Music/DJ_Library --output processed --max 100

# From extracted DJ mix transitions
python pipeline.py --transitions data/transitions --output processed
```

### 3. Train the model

```bash
# Basic training
python train.py --data processed --output output/run1

# With custom settings
python train.py --data processed --output output/run1 \
    --batch-size 8 \
    --lr 1e-4 \
    --steps 50000 \
    --wandb
```

### 4. Resume training

```bash
python train.py --data processed --output output/run1 \
    --resume output/run1/checkpoint_10000.pt
```

## Data Preparation

### Option A: From a track library

If you have a collection of DJ tracks, the pipeline can create synthetic training pairs:

```bash
python pipeline.py --tracks ~/Music/DJ_Library --output processed --max 500
```

This will:
1. Pick random track pairs
2. Run demucs stem separation
3. Create synthetic crossfade transitions
4. Analyze BPM, key, structure

### Option B: From real DJ mixes

For higher quality training, use real DJ transitions:

1. **Scrape mix metadata** (from 1001tracklists, YouTube, Mixcloud):
   ```bash
   cd data
   python scrape_1001tracklists.py --genre tech-house --max 50
   ```

2. **Download mixes** (manual or via youtube-dl)

3. **Extract transitions**:
   ```bash
   python extract_transitions.py --max 20
   ```

4. **Process for training**:
   ```bash
   cd ..
   python pipeline.py --transitions data/transitions --output processed
   ```

### Data structure

After processing, data is organized as:
```
processed/
├── pair_001/
│   ├── track_a_stems/
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   ├── vocals.wav
│   │   └── other.wav
│   ├── track_b_stems/
│   │   └── ...
│   ├── transition.wav
│   └── analysis.json
├── pair_002/
│   └── ...
```

## Training

### Basic training command

```bash
python train.py \
    --data processed \
    --output output/experiment1 \
    --batch-size 4 \
    --steps 100000
```

### Configuration options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to processed data directory |
| `--output` | `./output` | Output directory for checkpoints |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--steps` | 100000 | Max training steps |
| `--resume` | None | Resume from checkpoint |
| `--wandb` | False | Enable W&B logging |
| `--synthetic` | False | Use synthetic dataset mode |

### Hardware requirements

| GPU | Batch Size | Memory | Training Time (100k steps) |
|-----|------------|--------|---------------------------|
| A100 80GB | 32 | ~40GB | ~8 hours |
| RTX 4090 | 16 | ~20GB | ~16 hours |
| RTX 3090 | 8 | ~18GB | ~32 hours |
| M2 Max (MPS) | 4 | ~20GB | ~72 hours |

### Training tips

1. **Start small**: Train on 100-500 pairs first to verify the pipeline works
2. **Monitor smoothness loss**: High smoothness loss → jerky parameter curves
3. **Use EMA**: The EMA model usually produces better results
4. **Mixed precision**: Enable AMP on CUDA for 2x speedup

## Inference

### Basic usage

```bash
python inference.py track_a.mp3 track_b.mp3 -o output/transition.wav
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | None | Path to trained model checkpoint |
| `--duration` | 30.0 | Transition duration in seconds |
| `--guidance-scale` | 2.0 | Classifier-free guidance scale |
| `--steps` | 50 | Diffusion sampling steps |
| `--device` | auto | Device (cpu/cuda/mps) |
| `--visualize` | False | Generate parameter plot |
| `--save-params` | None | Save parameters to JSON |

### Examples

```bash
# With custom model and longer transition
python inference.py track_a.mp3 track_b.mp3 \
    --model output/best_model.pt \
    --duration 60 \
    -o output/long_transition.wav

# Lower guidance for more creative transitions
python inference.py track_a.mp3 track_b.mp3 \
    --guidance-scale 1.0 \
    -o output/creative_transition.wav

# Save parameters for analysis
python inference.py track_a.mp3 track_b.mp3 \
    --save-params output/params.json \
    --visualize \
    -o output/transition.wav
```

## Visualization

```bash
# Plot parameter curves
python visualize.py output/params.json -o output/params_plot.png

# Generate stem swap timeline
python visualize.py output/params.json -o output/timeline.png --timeline

# Export text summary
python visualize.py output/params.json --summary output/summary.md
```

## API Usage

```python
from model import load_model, get_inference_config
from effects import DifferentiableDJMixer, MixerConfig
from inference import (
    load_stems,
    analyze_track,
    build_track_features,
    generate_transition,
    run_demucs,
)

# Load model
model = load_model("output/best_model.pt", device="mps")

# Create mixer
mixer = DifferentiableDJMixer()
mixer.to("mps")

# Prepare tracks
stems_a = load_stems("stems/track_a")
stems_b = load_stems("stems/track_b")
features_a = build_track_features(stems_a, analysis_a, ...)
features_b = build_track_features(stems_b, analysis_b, ...)

# Generate
mixed_audio, params = generate_transition(
    model, mixer,
    stems_a, stems_b,
    features_a, features_b,
    n_frames=128,
    device="mps",
)
```

## Project Structure

```
djtransgan-v2/
├── model/                  # Diffusion model
│   ├── config.py          # Hyperparameters
│   ├── model.py           # Main model architecture
│   ├── conditioning.py    # Audio encoding
│   └── training.py        # Training utilities
├── effects/               # Differentiable mixer
│   ├── effects.py         # Individual effects (EQ, filter, reverb)
│   └── mixer.py           # Main mixer class
├── data/                  # Data collection scripts
│   ├── scrape_*.py        # Web scrapers
│   ├── extract_transitions.py
│   └── analyze_tracks.py
├── dataset.py             # PyTorch Dataset
├── pipeline.py            # Data preparation pipeline
├── train.py               # Training script
├── inference.py           # Inference script
├── visualize.py           # Visualization tools
├── pyproject.toml         # Package config
└── README.md
```

## Troubleshooting

### "demucs not found"
```bash
pip install demucs
```

### "MPS out of memory"
Reduce batch size or use CPU:
```bash
python train.py --batch-size 2 --device cpu
```

### "CUDA out of memory"
```bash
python train.py --batch-size 4
```

### "Transition sounds choppy"
- Increase `--steps` during inference
- Check smoothness loss during training
- Try lower guidance scale

### "Model produces silence"
- Model may not be trained enough
- Check that stems are loading correctly
- Verify audio files are not corrupted

## Citation

If you use this code in your research, please cite:

```bibtex
@software{djtransgan2024,
  title = {DJTransGAN v2: Neural DJ Transition Generation},
  author = {Clawd},
  year = {2024},
  url = {https://github.com/clawd139/djtransgan-v2}
}
```

## License

MIT License - see LICENSE file for details.
