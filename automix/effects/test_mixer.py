#!/usr/bin/env python3
"""
Test script for the Differentiable DJ Mixer

Demonstrates:
1. Basic mixing with default parameters
2. Crossfade generation
3. Gradient computation for training
4. Audio file processing (if files provided)

Usage:
    python test_mixer.py                          # Run with synthetic audio
    python test_mixer.py track_a.wav track_b.wav  # Run with real audio files

Author: Clawd
License: MIT
"""

import sys
import os
import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# Handle both package and standalone execution
# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mixer import (
    DifferentiableDJMixer,
    MixerConfig,
    ParameterParser,
    SimpleCrossfader,
    create_mixer,
    N_TOTAL_PARAMS,
    N_STEMS,
    STEM_NAMES
)


def generate_test_tone(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    batch_size: int = 1
) -> torch.Tensor:
    """Generate a simple sine wave test tone."""
    n_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, n_samples)
    audio = 0.5 * torch.sin(2 * math.pi * frequency * t)
    return audio.unsqueeze(0).expand(batch_size, -1)


def generate_test_stems(
    duration: float,
    sample_rate: int = 44100,
    batch_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic test stems with different characteristics.
    
    - drums: Low frequency pulse
    - bass: Low sine wave
    - vocals: Mid frequency tone
    - other: High frequency shimmer
    """
    n_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, n_samples)
    
    # Drums: Exponential decay pulses at 2Hz (simulating kick)
    pulse_freq = 2
    envelope = torch.exp(-10 * (t % (1/pulse_freq)))
    drums = 0.5 * torch.sin(2 * math.pi * 60 * t) * envelope
    
    # Bass: Low frequency sine with subtle modulation
    bass = 0.4 * torch.sin(2 * math.pi * 80 * t) * (1 + 0.2 * torch.sin(2 * math.pi * 0.5 * t))
    
    # Vocals: Mid frequency with vibrato
    vibrato = torch.sin(2 * math.pi * 5 * t) * 10
    vocals = 0.3 * torch.sin(2 * math.pi * (400 + vibrato) * t)
    
    # Other: High frequency shimmer
    other = 0.2 * (
        torch.sin(2 * math.pi * 2000 * t) +
        0.5 * torch.sin(2 * math.pi * 4000 * t)
    ) * (1 + 0.3 * torch.sin(2 * math.pi * 2 * t))
    
    # Add batch dimension
    stems = {
        'drums': drums.unsqueeze(0).expand(batch_size, -1),
        'bass': bass.unsqueeze(0).expand(batch_size, -1),
        'vocals': vocals.unsqueeze(0).expand(batch_size, -1),
        'other': other.unsqueeze(0).expand(batch_size, -1)
    }
    
    return stems


def test_basic_mixing():
    """Test basic mixing functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Mixing")
    print("="*60)
    
    # Create mixer
    mixer = create_mixer(sample_rate=44100)
    
    # Generate test stems
    duration = 2.0  # 2 seconds
    batch_size = 2
    track_a = generate_test_stems(duration, batch_size=batch_size)
    track_b = generate_test_stems(duration, batch_size=batch_size)
    
    # Modify track B to be different (shift frequencies)
    track_b['drums'] = track_b['drums'] * 0.8
    track_b['vocals'] = track_b['vocals'] * 1.2
    
    # Generate parameters
    n_frames = 64  # 64 time frames for the transition
    params = ParameterParser.get_default_params(n_frames, batch_size)
    
    print(f"Track A stems: {list(track_a.keys())}")
    print(f"Track A drums shape: {track_a['drums'].shape}")
    print(f"Parameters shape: {params.shape}")
    
    # Run mixer
    mixed, intermediates = mixer(
        track_a, track_b, params,
        return_intermediates=True
    )
    
    print(f"Output shape: {mixed.shape}")
    print(f"Output min/max: {mixed.min():.4f} / {mixed.max():.4f}")
    
    # Check intermediates
    if intermediates:
        print(f"Track A final shape: {intermediates['track_a']['final'].shape}")
        print(f"Track B final shape: {intermediates['track_b']['final'].shape}")
    
    print("✓ Basic mixing test passed!")
    return True


def test_crossfader():
    """Test crossfade parameter generation."""
    print("\n" + "="*60)
    print("TEST 2: Crossfade Generation")
    print("="*60)
    
    crossfader = SimpleCrossfader(fade_type='equal_power')
    
    # Generate crossfade parameters
    n_frames = 32
    params = crossfader(n_frames, batch_size=1)
    
    print(f"Generated params shape: {params.shape}")
    print(f"Expected: [1, {n_frames}, {N_TOTAL_PARAMS}]")
    
    # Check that Track A starts loud and ends quiet
    track_a_gain_0 = params[0, 0, 0].item()  # First stem gain at start
    track_a_gain_end = params[0, -1, 0].item()  # First stem gain at end
    
    print(f"Track A gain at start: {track_a_gain_0:.4f}")
    print(f"Track A gain at end: {track_a_gain_end:.4f}")
    
    # Check that Track B starts quiet and ends loud
    track_b_gain_idx = N_TOTAL_PARAMS // 2  # Track B starts halfway
    track_b_gain_0 = params[0, 0, track_b_gain_idx].item()
    track_b_gain_end = params[0, -1, track_b_gain_idx].item()
    
    print(f"Track B gain at start: {track_b_gain_0:.4f}")
    print(f"Track B gain at end: {track_b_gain_end:.4f}")
    
    # Verify crossfade behavior
    assert track_a_gain_0 > 0.9, "Track A should start loud"
    assert track_a_gain_end < 0.1, "Track A should end quiet"
    assert track_b_gain_0 < 0.1, "Track B should start quiet"
    assert track_b_gain_end > 0.9, "Track B should end loud"
    
    print("✓ Crossfade test passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow through the mixer."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Flow")
    print("="*60)
    
    mixer = create_mixer(sample_rate=44100)
    
    # Generate test data
    duration = 1.0
    batch_size = 1
    n_frames = 16
    
    track_a = generate_test_stems(duration, batch_size=batch_size)
    track_b = generate_test_stems(duration, batch_size=batch_size)
    
    # Make parameters require gradients (simulating neural network output)
    params = ParameterParser.get_default_params(n_frames, batch_size)
    params = params.clone().requires_grad_(True)
    
    # Forward pass
    mixed, _ = mixer(track_a, track_b, params)
    
    # Compute a simple loss (mean squared)
    target = torch.zeros_like(mixed)
    loss = F.mse_loss(mixed, target)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Param gradients shape: {params.grad.shape}")
    print(f"Param gradients min/max: {params.grad.min():.6f} / {params.grad.max():.6f}")
    print(f"Param gradients mean: {params.grad.mean():.6f}")
    
    # Verify gradients are not all zero
    assert params.grad is not None, "Gradients should exist"
    assert not torch.all(params.grad == 0), "Gradients should not all be zero"
    
    print("✓ Gradient flow test passed!")
    return True


def test_individual_effects():
    """Test individual effects in isolation."""
    print("\n" + "="*60)
    print("TEST 4: Individual Effects")
    print("="*60)
    
    from effects import (
        DifferentiableGain,
        DifferentiableEQ,
        DifferentiableLowPassFilter,
        DifferentiableHighPassFilter,
        DifferentiableReverb,
        DifferentiableDelay
    )
    
    # Test parameters
    batch_size = 1
    n_samples = 44100  # 1 second
    n_frames = 16
    
    # Generate test audio
    audio = generate_test_tone(440, 1.0, batch_size=batch_size)
    
    # Helper to check shape (effects may add channel dim)
    def check_shape(output, expected_samples, name):
        # Output can be [batch, samples] or [batch, channels, samples]
        assert output.dim() in [2, 3], f"{name} unexpected dims: {output.dim()}"
        out_samples = output.shape[-1]
        assert out_samples == expected_samples, f"{name} sample mismatch: {out_samples} vs {expected_samples}"
    
    # Test Gain
    print("\nTesting DifferentiableGain...")
    gain = DifferentiableGain()
    gain_curve = torch.linspace(1, 0, n_frames).unsqueeze(0)
    output = gain(audio, gain_curve)
    check_shape(output, n_samples, "Gain")
    print(f"  Input max: {audio.abs().max():.4f}, Output max: {output.abs().max():.4f}")
    print("  ✓ Gain OK")
    
    # Test EQ
    print("\nTesting DifferentiableEQ...")
    eq = DifferentiableEQ()
    eq_curve = torch.ones(batch_size, n_frames)
    output = eq(audio, eq_curve, eq_curve, eq_curve)
    check_shape(output, n_samples, "EQ")
    print(f"  Input max: {audio.abs().max():.4f}, Output max: {output.abs().max():.4f}")
    print("  ✓ EQ OK")
    
    # Test Low-pass Filter
    print("\nTesting DifferentiableLowPassFilter...")
    lpf = DifferentiableLowPassFilter()
    cutoff_curve = torch.ones(batch_size, n_frames) * 0.5  # 50% cutoff
    output = lpf(audio, cutoff_curve)
    check_shape(output, n_samples, "LPF")
    print(f"  Input max: {audio.abs().max():.4f}, Output max: {output.abs().max():.4f}")
    print("  ✓ LPF OK")
    
    # Test High-pass Filter
    print("\nTesting DifferentiableHighPassFilter...")
    hpf = DifferentiableHighPassFilter()
    cutoff_curve = torch.ones(batch_size, n_frames) * 0.1  # 10% cutoff
    output = hpf(audio, cutoff_curve)
    check_shape(output, n_samples, "HPF")
    print(f"  Input max: {audio.abs().max():.4f}, Output max: {output.abs().max():.4f}")
    print("  ✓ HPF OK")
    
    # Test Reverb
    print("\nTesting DifferentiableReverb...")
    reverb = DifferentiableReverb()
    wet_dry = torch.ones(batch_size, n_frames) * 0.3
    output = reverb(audio, wet_dry)
    check_shape(output, n_samples, "Reverb")
    print(f"  Input max: {audio.abs().max():.4f}, Output max: {output.abs().max():.4f}")
    print("  ✓ Reverb OK")
    
    # Test Delay
    print("\nTesting DifferentiableDelay...")
    delay = DifferentiableDelay()
    wet_dry = torch.ones(batch_size, n_frames) * 0.3
    output = delay(audio, wet_dry)
    check_shape(output, n_samples, "Delay")
    print(f"  Input max: {audio.abs().max():.4f}, Output max: {output.abs().max():.4f}")
    print("  ✓ Delay OK")
    
    print("\n✓ All individual effects tests passed!")
    return True


def test_with_audio_files(track_a_path: str, track_b_path: str):
    """Test with real audio files (requires torchaudio)."""
    print("\n" + "="*60)
    print("TEST 5: Real Audio Files")
    print("="*60)
    
    try:
        import torchaudio
    except ImportError:
        print("torchaudio not installed, skipping audio file test")
        return True
    
    # Load audio files
    print(f"Loading {track_a_path}...")
    waveform_a, sr_a = torchaudio.load(track_a_path)
    print(f"Loading {track_b_path}...")
    waveform_b, sr_b = torchaudio.load(track_b_path)
    
    # Ensure same sample rate
    sample_rate = 44100
    if sr_a != sample_rate:
        resampler = torchaudio.transforms.Resample(sr_a, sample_rate)
        waveform_a = resampler(waveform_a)
    if sr_b != sample_rate:
        resampler = torchaudio.transforms.Resample(sr_b, sample_rate)
        waveform_b = resampler(waveform_b)
    
    # Convert to mono if stereo
    if waveform_a.shape[0] > 1:
        waveform_a = waveform_a.mean(dim=0, keepdim=True)
    if waveform_b.shape[0] > 1:
        waveform_b = waveform_b.mean(dim=0, keepdim=True)
    
    # Truncate to same length (use shorter one)
    min_length = min(waveform_a.shape[1], waveform_b.shape[1])
    waveform_a = waveform_a[:, :min_length]
    waveform_b = waveform_b[:, :min_length]
    
    print(f"Audio shape: {waveform_a.shape}")
    print(f"Duration: {min_length / sample_rate:.2f} seconds")
    
    # For this test, treat the whole audio as one stem (simplified)
    # In practice, you'd use demucs to separate stems first
    track_a = {
        'drums': waveform_a,
        'bass': torch.zeros_like(waveform_a),
        'vocals': torch.zeros_like(waveform_a),
        'other': torch.zeros_like(waveform_a)
    }
    track_b = {
        'drums': waveform_b,
        'bass': torch.zeros_like(waveform_b),
        'vocals': torch.zeros_like(waveform_b),
        'other': torch.zeros_like(waveform_b)
    }
    
    # Create mixer and crossfader
    mixer = create_mixer(sample_rate=sample_rate)
    crossfader = SimpleCrossfader(fade_type='equal_power')
    
    # Generate crossfade
    n_frames = 64
    params = crossfader(n_frames, batch_size=1)
    
    # Mix
    print("Mixing...")
    mixed, _ = mixer(track_a, track_b, params)
    
    print(f"Output shape: {mixed.shape}")
    print(f"Output min/max: {mixed.min():.4f} / {mixed.max():.4f}")
    
    # Save output
    output_path = "mixed_output.wav"
    # Convert from [batch, channels, samples] to [channels, samples]
    output = mixed.squeeze(0)
    torchaudio.save(output_path, output, sample_rate)
    print(f"Saved mixed output to {output_path}")
    
    print("✓ Audio file test passed!")
    return True


def test_parameter_info():
    """Display parameter space information."""
    print("\n" + "="*60)
    print("PARAMETER SPACE INFO")
    print("="*60)
    
    mixer = create_mixer()
    info = mixer.get_param_info()
    
    print(f"\nTotal parameters per frame: {info['total_params']}")
    print(f"Parameters per track: {info['params_per_track']}")
    print(f"Number of stems: {info['n_stems']}")
    print(f"Stem names: {info['stem_names']}")
    print(f"Parameters per stem: {info['n_stem_params']}")
    print(f"Stem parameter names: {info['stem_param_names']}")
    print(f"Global parameters: {info['n_global_params']}")
    print(f"Global parameter names: {info['global_param_names']}")
    
    print("\nParameter ranges:")
    for name, (min_val, max_val) in info['param_ranges'].items():
        print(f"  {name}: [{min_val}, {max_val}]")
    
    print(info['layout'])
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("DIFFERENTIABLE DJ MIXER TEST SUITE")
    print("="*60)
    
    # Display parameter info
    test_parameter_info()
    
    # Run tests
    tests = [
        ("Basic Mixing", test_basic_mixing),
        ("Crossfade Generation", test_crossfader),
        ("Gradient Flow", test_gradient_flow),
        ("Individual Effects", test_individual_effects),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Test with audio files if provided
    if len(sys.argv) >= 3:
        try:
            passed = test_with_audio_files(sys.argv[1], sys.argv[2])
            results.append(("Audio Files", passed))
        except Exception as e:
            print(f"\n✗ Audio Files FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("Audio Files", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
