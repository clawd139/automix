"""
DJ Transition Model v2

A conditional diffusion model for generating DJ transition effect parameters.
"""

from .config import (
    Config,
    AudioConfig,
    EffectConfig,
    ModelConfig,
    DiffusionConfig,
    TrainingConfig,
    get_config,
    get_inference_config,
)

from .conditioning import (
    TrackFeatures,
    ConditioningEncoder,
    MelSpectrogramExtractor,
    create_dummy_features,
)

from .model import (
    DJTransitionModel,
    ParameterDiffusionModel,
    DiffusionSchedule,
    create_model,
    load_model,
)

from .training import (
    Trainer,
    train,
    DJTransitionDataset,
    DummyDataset,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "Config",
    "AudioConfig", 
    "EffectConfig",
    "ModelConfig",
    "DiffusionConfig",
    "TrainingConfig",
    "get_config",
    "get_inference_config",
    # Conditioning
    "TrackFeatures",
    "ConditioningEncoder",
    "MelSpectrogramExtractor",
    "create_dummy_features",
    # Model
    "DJTransitionModel",
    "ParameterDiffusionModel",
    "DiffusionSchedule",
    "create_model",
    "load_model",
    # Training
    "Trainer",
    "train",
    "DJTransitionDataset",
    "DummyDataset",
]
