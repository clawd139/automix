# AutoMix - Neural DJ Transition Generation

Generate professional-quality DJ transitions between tracks using AI. AutoMix uses conditional diffusion models and differentiable audio effects to create smooth, musically-aware transitions.

## Features

- **One-command interface**: Simple CLI for data preparation, training, and inference
- **Auto device detection**: Works on CUDA GPUs, Apple Silicon (MPS), or CPU
- **Multi-GPU training**: Built-in DDP support for cluster training
- **Differentiable mixing**: End-to-end trainable audio effects chain
- **Stem-aware**: Uses demucs for source separation (drums, bass, vocals, other)

## Quick Start

### Installation

Requires [uv](https://github.com/astral-sh/uv) for Python/dependency management:

```bash
# Clone the repo
git clone https://github.com/clawd139/automix.git
cd automix

# Install with uv (automatically uses Python 3.12)
uv sync

# Verify installation
uv run automix info
```

### Basic Usage

```bash
# Generate a transition between two tracks (no training required)
uv run automix mix track_a.mp3 track_b.mp3 -o transition.wav

# With a trained model
uv run automix mix track_a.mp3 track_b.mp3 --model runs/run1/best.pt -o transition.wav
```

### Training Pipeline

```bash
# 1. Prepare training data from your music library
uv run automix prepare --tracks ~/Music/DJ_Library --output processed --max 100

# 2. Train the model
uv run automix train --data processed --output runs/run1 --steps 100000

# 3. Generate transitions with your trained model
uv run automix mix track_a.mp3 track_b.mp3 --model runs/run1/best.pt -o output.wav
```

## Commands

### `automix info`

Show system information and available capabilities:

```bash
uv run automix info
```

### `automix prepare`

Prepare training data from a track library:

```bash
uv run automix prepare --tracks ~/Music/DJ_Library --output processed --max 100
```

Options:
- `--tracks`, `-t`: Path to track library directory (required)
- `--output`, `-o`: Output directory for processed data (default: `processed`)
- `--max`: Maximum number of track pairs to process
- `--model`: Demucs model to use (default: `htdemucs`)
- `--device`: Device for processing (`auto`, `cpu`, `cuda`, `mps`)

### `automix train`

Train the transition model:

```bash
# Single GPU/MPS
uv run automix train --data processed --output runs/run1 --steps 100000

# Multi-GPU (auto-detected)
uv run automix train --data processed --output runs/run1 --steps 100000

# Specify GPU count
uv run automix train --data processed --output runs/run1 --gpus 4
```

Options:
- `--data`, `-d`: Path to processed data directory (required)
- `--output`, `-o`: Output directory for checkpoints (default: `runs/run1`)
- `--steps`: Maximum training steps (default: 100000)
- `--batch-size`: Batch size (auto-detected if not specified)
- `--lr`: Learning rate
- `--resume`: Resume from checkpoint
- `--gpus`: Number of GPUs (auto-detected)
- `--wandb`: Enable Weights & Biases logging
- `--synthetic`: Use synthetic transition dataset

### `automix mix`

Generate a DJ transition between two tracks:

```bash
uv run automix mix track_a.mp3 track_b.mp3 -o transition.wav
```

Options:
- `--output`, `-o`: Output path (default: `output/transition.wav`)
- `--model`, `-m`: Path to trained model checkpoint
- `--duration`: Transition duration in seconds (default: 30)
- `--guidance`: Classifier-free guidance scale (default: 2.0)
- `--steps`: Number of diffusion steps (default: 50)
- `--visualize`: Generate parameter visualization
- `--device`: Device for inference (`auto`, `cpu`, `cuda`, `mps`)

## Docker (for GPU clusters)

Build and run on Lambda Cloud or any CUDA machine:

```bash
# Build
docker build -t automix .

# Run training
docker run --gpus all -v /path/to/data:/data -v /path/to/output:/output \
    automix train --data /data --output /output --steps 100000

# Run inference
docker run --gpus all -v /path/to/tracks:/tracks -v /path/to/output:/output \
    automix mix /tracks/a.mp3 /tracks/b.mp3 -o /output/transition.wav
```

## Architecture

AutoMix uses a conditional diffusion model architecture:

1. **Track Analysis**: Extract features (BPM, key, structure, mel spectrograms) from both tracks
2. **Stem Separation**: Use demucs to separate drums, bass, vocals, other
3. **Parameter Generation**: Diffusion model predicts effect parameter curves
4. **Differentiable Mixing**: Apply parameters through differentiable DJ effects chain
5. **Audio Output**: Generate the final mixed transition

### Effect Parameters

The model predicts time-varying parameters for:
- Per-stem: gain, 3-band EQ (low/mid/high), low-pass filter, high-pass filter
- Global: crossfader position, reverb send, delay send

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black automix/
uv run isort automix/
```

## License

MIT

## Citation

```bibtex
@software{automix2024,
  author = {Clawd},
  title = {AutoMix: Neural DJ Transition Generation},
  year = {2024},
  url = {https://github.com/clawd139/automix}
}
```
