"""
Differentiable DJ Effects Module

A PyTorch-compatible differentiable DJ mixing module for neural network-controlled
audio transitions. All effects are implemented to support autograd for end-to-end training.

Example usage:
    from effects import DifferentiableDJMixer, SimpleCrossfader, create_mixer
    
    # Create mixer
    mixer = create_mixer(sample_rate=44100)
    
    # Prepare stem-separated tracks (from demucs)
    track_a = {'drums': ..., 'bass': ..., 'vocals': ..., 'other': ...}
    track_b = {'drums': ..., 'bass': ..., 'vocals': ..., 'other': ...}
    
    # Generate crossfade parameters (or from neural network)
    crossfader = SimpleCrossfader(fade_type='equal_power')
    params = crossfader(n_frames=64)
    
    # Mix tracks
    mixed_audio, intermediates = mixer(track_a, track_b, params, return_intermediates=True)

Author: Clawd
License: MIT
"""

from .effects import (
    DifferentiableGain,
    DifferentiableEQ,
    DifferentiableLowPassFilter,
    DifferentiableHighPassFilter,
    DifferentiableReverb,
    DifferentiableDelay,
    StemEffectsChain,
)

from .mixer import (
    DifferentiableDJMixer,
    MixerConfig,
    ParameterParser,
    SimpleCrossfader,
    create_mixer,
    STEM_NAMES,
    STEM_PARAM_NAMES,
    GLOBAL_PARAM_NAMES,
    N_STEMS,
    N_STEM_PARAMS,
    N_GLOBAL_PARAMS,
    N_PARAMS_PER_TRACK,
    N_TOTAL_PARAMS,
)

__version__ = "0.1.0"
__author__ = "Clawd"

__all__ = [
    # Effects
    "DifferentiableGain",
    "DifferentiableEQ",
    "DifferentiableLowPassFilter",
    "DifferentiableHighPassFilter",
    "DifferentiableReverb",
    "DifferentiableDelay",
    "StemEffectsChain",
    # Mixer
    "DifferentiableDJMixer",
    "MixerConfig",
    "ParameterParser",
    "SimpleCrossfader",
    "create_mixer",
    # Constants
    "STEM_NAMES",
    "STEM_PARAM_NAMES",
    "GLOBAL_PARAM_NAMES",
    "N_STEMS",
    "N_STEM_PARAMS",
    "N_GLOBAL_PARAMS",
    "N_PARAMS_PER_TRACK",
    "N_TOTAL_PARAMS",
]
