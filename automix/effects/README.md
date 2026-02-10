# Differentiable DJ Effects Module

A PyTorch-compatible differentiable DJ mixing module designed for neural network-controlled audio transitions. All effects are implemented to support autograd, enabling end-to-end training of models that learn DJ mixing techniques.

## Overview

This module provides the DSP layer that sits between a neural network (which outputs control parameters) and the final mixed audio. It processes stem-separated audio (from tools like Demucs) and applies professional DJ-style effects controllable by learned parameters.

```
Neural Network → Control Parameters → DifferentiableDJMixer → Mixed Audio
                   (54 params/frame)
```

## Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch 2.0+ and torchaudio.

## Quick Start

```python
from effects import DifferentiableDJMixer, SimpleCrossfader, create_mixer

# Create mixer
mixer = create_mixer(sample_rate=44100)

# Prepare stem-separated tracks (dict of tensors)
# Keys: 'drums', 'bass', 'vocals', 'other'
# Values: [batch, samples] or [batch, channels, samples]
track_a = {...}
track_b = {...}

# Generate crossfade parameters (or from your neural network)
crossfader = SimpleCrossfader(fade_type='equal_power')
params = crossfader(n_frames=64, batch_size=1)

# Mix tracks
mixed_audio, intermediates = mixer(
    track_a, track_b, params,
    return_intermediates=True
)
# Output shape: [batch, 2, samples] (stereo)
```

## Parameter Space

The neural network outputs a tensor of shape `[batch, n_frames, 54]` where:
- `n_frames` = number of time frames in the transition
- `54` = total control parameters per frame

### Layout (54 parameters per frame)

| Index Range | Description |
|-------------|-------------|
| 0-23 | Track A stems (4 stems × 6 params) |
| 24-26 | Track A global effects (3 params) |
| 27-50 | Track B stems (4 stems × 6 params) |
| 51-53 | Track B global effects (3 params) |

### Per-Stem Parameters (6 each)

| Param | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | `gain` | 0-1 | Volume fader |
| 1 | `eq_low` | 0-2 | Low band EQ (1=unity) |
| 2 | `eq_mid` | 0-2 | Mid band EQ (1=unity) |
| 3 | `eq_high` | 0-2 | High band EQ (1=unity) |
| 4 | `lpf` | 0-1 | Low-pass filter cutoff (1=open) |
| 5 | `hpf` | 0-1 | High-pass filter cutoff (0=open) |

### Global Parameters (3 each)

| Param | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | `reverb_send` | 0-1 | Reverb wet/dry mix |
| 1 | `delay_send` | 0-1 | Delay/echo wet/dry mix |
| 2 | `delay_feedback` | 0-1 | Delay feedback amount |

### Stem Order

1. `drums` (indices 0-5 for Track A, 27-32 for Track B)
2. `bass` (indices 6-11 for Track A, 33-38 for Track B)
3. `vocals` (indices 12-17 for Track A, 39-44 for Track B)
4. `other` (indices 18-23 for Track A, 45-50 for Track B)

## Effects

### Per-Stem Effects

1. **Gain/Fader** — Time-varying volume control with anti-zipper smoothing
2. **3-Band EQ** — Low/mid/high parametric EQ using frequency-domain soft masks
3. **Low-Pass Filter** — Sweepable filter with exponential frequency mapping
4. **High-Pass Filter** — Sweepable filter with exponential frequency mapping

### Global Effects

5. **Reverb** — FFT-based convolution reverb with algorithmic impulse response
6. **Delay/Echo** — Feedback delay with multiple taps

## Architecture Details

### Frequency Domain Processing

EQ and filters operate in the STFT domain for efficiency and smooth parameter interpolation:

```
Audio → STFT → Apply Frequency Mask → ISTFT → Audio
```

The frequency masks use soft sigmoid transitions for differentiability (no hard cutoffs).

### Time-Varying Parameters

Control parameters are specified per-frame and interpolated to match audio sample rate:

```python
# Neural network outputs: [batch, n_frames, n_params]
# Effect curves interpolated: [batch, n_samples]
```

### Configuration

```python
from effects import MixerConfig, DifferentiableDJMixer

config = MixerConfig(
    sample_rate=44100,
    n_fft=2048,
    hop_length=512,
    reverb_decay_time=1.5,
    reverb_damping=0.5,
    delay_default_ms=375.0,  # 1/8 note at 120 BPM
    delay_max_feedback=0.7,
)

mixer = DifferentiableDJMixer(config)
```

## Training Integration

The mixer is fully differentiable. Here's how to use it in training:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)
        self.param_head = nn.Linear(hidden_dim, 54)  # 54 params per frame
        
    def forward(self, features):
        # features: [batch, n_frames, 128]
        h, _ = self.encoder(features)
        params = torch.sigmoid(self.param_head(h)) * 2  # Scale to param ranges
        return params

# Training loop
model = TransitionModel()
mixer = create_mixer()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    track_a_stems, track_b_stems, features, target_audio = batch
    
    # Generate control parameters
    params = model(features)
    
    # Mix audio
    mixed, _ = mixer(track_a_stems, track_b_stems, params)
    
    # Compute loss (e.g., multi-scale spectral loss)
    loss = compute_spectral_loss(mixed, target_audio)
    
    # Backprop through mixer and model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Crossfade Types

The `SimpleCrossfader` utility generates standard DJ crossfade curves:

```python
from effects import SimpleCrossfader

# Available fade types
crossfader = SimpleCrossfader(fade_type='equal_power')  # Constant perceived loudness
crossfader = SimpleCrossfader(fade_type='linear')       # Linear volume
crossfader = SimpleCrossfader(fade_type='exponential')  # Fast start, slow end
crossfader = SimpleCrossfader(fade_type='logarithmic')  # Slow start, fast end

# Generate with custom cue points
params = crossfader(
    n_frames=64,
    batch_size=1,
    cue_in=0.25,   # Track B starts at 25%
    cue_out=0.75   # Track A finishes at 75%
)
```

## Testing

```bash
# Run all tests with synthetic audio
python test_mixer.py

# Test with real audio files
python test_mixer.py track_a.wav track_b.wav
```

## File Structure

```
effects/
├── __init__.py       # Package exports
├── effects.py        # Individual effect implementations
├── mixer.py          # Main DifferentiableDJMixer class
├── test_mixer.py     # Test suite
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## References

- [DJtransGAN](https://github.com/ChenPaulYu/DJtransGAN) — Original inspiration
- [DDSP](https://magenta.tensorflow.org/ddsp) — Google's Differentiable DSP
- [Pedalboard](https://github.com/spotify/pedalboard) — Spotify's audio effects (API reference)

## License

MIT

## Author

Clawd
