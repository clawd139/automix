"""
AutoMix - Neural DJ Transition Generation

A toolkit for generating DJ-style transitions between tracks using
conditional diffusion models and differentiable audio effects.
"""

__version__ = "2.0.0"
__author__ = "Clawd"

from .effects import (
    DifferentiableDJMixer,
    MixerConfig,
    STEM_NAMES,
    N_TOTAL_PARAMS,
)

from .model import (
    Config,
    DJTransitionModel,
    TrackFeatures,
    create_model,
    load_model,
    get_config,
    get_inference_config,
)

__all__ = [
    "__version__",
    # Effects
    "DifferentiableDJMixer",
    "MixerConfig",
    "STEM_NAMES",
    "N_TOTAL_PARAMS",
    # Model
    "Config",
    "DJTransitionModel",
    "TrackFeatures",
    "create_model",
    "load_model",
    "get_config",
    "get_inference_config",
]
