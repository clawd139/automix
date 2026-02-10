"""
Differentiable DJ Mixer Module

Main DifferentiableDJMixer class that processes stem-separated tracks
with neural network-controlled effects for DJ transitions.

Author: Clawd
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

try:
    from .effects import (
        DifferentiableGain,
        DifferentiableEQ,
        DifferentiableLowPassFilter,
        DifferentiableHighPassFilter,
        DifferentiableReverb,
        DifferentiableDelay,
        StemEffectsChain
    )
except ImportError:
    from effects import (
        DifferentiableGain,
        DifferentiableEQ,
        DifferentiableLowPassFilter,
        DifferentiableHighPassFilter,
        DifferentiableReverb,
        DifferentiableDelay,
        StemEffectsChain
    )


# Stem names for reference
STEM_NAMES = ['drums', 'bass', 'vocals', 'other']
N_STEMS = len(STEM_NAMES)

# Effect parameter indices within a stem's parameter block
STEM_PARAM_NAMES = [
    'gain',      # 0: Gain/fader (0-1)
    'eq_low',    # 1: Low EQ (0-2, 1=unity)
    'eq_mid',    # 2: Mid EQ (0-2, 1=unity)
    'eq_high',   # 3: High EQ (0-2, 1=unity)
    'lpf',       # 4: Low-pass cutoff (0-1, 1=fully open)
    'hpf',       # 5: High-pass cutoff (0-1, 0=fully open)
]
N_STEM_PARAMS = len(STEM_PARAM_NAMES)

# Global effect parameter names
GLOBAL_PARAM_NAMES = [
    'reverb_send',  # 0: Reverb wet/dry (0-1)
    'delay_send',   # 1: Delay wet/dry (0-1)
    'delay_feedback',  # 2: Delay feedback (0-1)
]
N_GLOBAL_PARAMS = len(GLOBAL_PARAM_NAMES)

# Total parameters per track: stems + global
N_PARAMS_PER_TRACK = N_STEMS * N_STEM_PARAMS + N_GLOBAL_PARAMS

# Total parameters for both tracks
N_TOTAL_PARAMS = 2 * N_PARAMS_PER_TRACK


@dataclass
class MixerConfig:
    """Configuration for the DifferentiableDJMixer."""
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    
    # Reverb settings
    reverb_decay_time: float = 1.5
    reverb_damping: float = 0.5
    reverb_pre_delay_ms: float = 20.0
    
    # Delay settings
    delay_max_ms: float = 1000.0
    delay_default_ms: float = 375.0  # 1/8 note at 120 BPM
    delay_max_feedback: float = 0.7
    delay_n_taps: int = 4
    
    # Filter settings
    lpf_min_freq: float = 20.0
    lpf_max_freq: float = 20000.0
    hpf_min_freq: float = 20.0
    hpf_max_freq: float = 20000.0
    filter_transition_octaves: float = 0.5


class ParameterParser:
    """
    Utility class to parse flat parameter tensors into structured form.
    
    Parameter layout for each frame:
    [Track A stems (4 × 6), Track A global (3), Track B stems (4 × 6), Track B global (3)]
    Total: 2 × (4 × 6 + 3) = 54 parameters per frame
    """
    
    def __init__(self):
        self.n_stem_params = N_STEM_PARAMS
        self.n_global_params = N_GLOBAL_PARAMS
        self.n_stems = N_STEMS
        self.n_params_per_track = N_PARAMS_PER_TRACK
    
    def parse(
        self,
        params: torch.Tensor
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Parse flat parameter tensor into structured dictionaries.
        
        Args:
            params: [batch, n_frames, n_params] or [n_frames, n_params]
                   where n_params = N_TOTAL_PARAMS (54)
        
        Returns:
            Tuple of (track_a_params, track_b_params) where each is:
            {
                'stems': {
                    'drums': {'gain': tensor, 'eq_low': tensor, ...},
                    'bass': {...},
                    ...
                },
                'global': {
                    'reverb_send': tensor,
                    'delay_send': tensor,
                    'delay_feedback': tensor
                }
            }
        """
        if params.dim() == 2:
            params = params.unsqueeze(0)  # Add batch dimension
        
        batch, n_frames, n_params = params.shape
        
        # Split into track A and track B
        track_a_flat = params[:, :, :self.n_params_per_track]
        track_b_flat = params[:, :, self.n_params_per_track:]
        
        def parse_track(track_params: torch.Tensor) -> Dict:
            """Parse single track parameters."""
            result = {'stems': {}, 'global': {}}
            
            # Parse stem parameters
            stem_end = self.n_stems * self.n_stem_params
            stem_params = track_params[:, :, :stem_end]
            stem_params = stem_params.reshape(batch, n_frames, self.n_stems, self.n_stem_params)
            
            for i, stem_name in enumerate(STEM_NAMES):
                result['stems'][stem_name] = {}
                for j, param_name in enumerate(STEM_PARAM_NAMES):
                    result['stems'][stem_name][param_name] = stem_params[:, :, i, j]
            
            # Parse global parameters
            global_params = track_params[:, :, stem_end:]
            for i, param_name in enumerate(GLOBAL_PARAM_NAMES):
                result['global'][param_name] = global_params[:, :, i]
            
            return result
        
        return parse_track(track_a_flat), parse_track(track_b_flat)
    
    @staticmethod
    def get_default_params(n_frames: int, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Generate default/neutral parameters.
        
        Default values:
        - Gain: 1.0 for track A, 0.0 for track B (simple crossfade start)
        - EQ: 1.0 (unity)
        - LPF: 1.0 (fully open)
        - HPF: 0.0 (fully open)
        - Reverb/Delay: 0.0 (dry)
        """
        params = torch.zeros(batch_size, n_frames, N_TOTAL_PARAMS, device=device)
        
        # Set defaults for track A
        for stem_idx in range(N_STEMS):
            base_idx = stem_idx * N_STEM_PARAMS
            params[:, :, base_idx + 0] = 1.0  # gain
            params[:, :, base_idx + 1] = 1.0  # eq_low
            params[:, :, base_idx + 2] = 1.0  # eq_mid
            params[:, :, base_idx + 3] = 1.0  # eq_high
            params[:, :, base_idx + 4] = 1.0  # lpf (open)
            params[:, :, base_idx + 5] = 0.0  # hpf (open)
        
        # Set defaults for track B (starting quiet)
        track_b_offset = N_PARAMS_PER_TRACK
        for stem_idx in range(N_STEMS):
            base_idx = track_b_offset + stem_idx * N_STEM_PARAMS
            params[:, :, base_idx + 0] = 0.0  # gain (quiet)
            params[:, :, base_idx + 1] = 1.0  # eq_low
            params[:, :, base_idx + 2] = 1.0  # eq_mid
            params[:, :, base_idx + 3] = 1.0  # eq_high
            params[:, :, base_idx + 4] = 1.0  # lpf (open)
            params[:, :, base_idx + 5] = 0.0  # hpf (open)
        
        return params


class DifferentiableDJMixer(nn.Module):
    """
    Differentiable DJ mixer for neural network-controlled transitions.
    
    Takes two stem-separated tracks and control parameters from a neural network,
    applies DJ-style effects (gain, EQ, filters, reverb, delay) to each stem,
    and outputs the mixed result.
    
    All operations are differentiable for end-to-end training.
    
    Input format:
        - track_a_stems: Dict[str, Tensor] with keys 'drums', 'bass', 'vocals', 'other'
          Each tensor is [batch, samples] or [batch, channels, samples]
        - track_b_stems: Same format as track_a
        - params: [batch, n_frames, N_TOTAL_PARAMS] control parameters
    
    Output:
        - Mixed stereo audio [batch, 2, samples]
        - Intermediate outputs dict for visualization/debugging
    
    Args:
        config: MixerConfig with audio and effect settings
    """
    
    def __init__(self, config: Optional[MixerConfig] = None):
        super().__init__()
        
        self.config = config or MixerConfig()
        self.param_parser = ParameterParser()
        
        # Build per-stem effect chains (shared between tracks)
        self.stem_effects = nn.ModuleDict({
            stem: StemEffectsChain(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            for stem in STEM_NAMES
        })
        
        # Global effects (applied to summed mix)
        self.reverb = DifferentiableReverb(
            sample_rate=self.config.sample_rate,
            decay_time=self.config.reverb_decay_time,
            damping=self.config.reverb_damping,
            pre_delay_ms=self.config.reverb_pre_delay_ms
        )
        
        self.delay = DifferentiableDelay(
            sample_rate=self.config.sample_rate,
            max_delay_ms=self.config.delay_max_ms,
            default_delay_ms=self.config.delay_default_ms,
            max_feedback=self.config.delay_max_feedback
        )
    
    def _ensure_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert mono to stereo if needed."""
        if audio.dim() == 2:
            # [batch, samples] -> [batch, 2, samples]
            return audio.unsqueeze(1).expand(-1, 2, -1)
        elif audio.dim() == 3 and audio.shape[1] == 1:
            # [batch, 1, samples] -> [batch, 2, samples]
            return audio.expand(-1, 2, -1)
        elif audio.dim() == 3 and audio.shape[1] == 2:
            return audio
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
    
    def _process_stem(
        self,
        audio: torch.Tensor,
        stem_name: str,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Process a single stem with its effects chain.
        
        Args:
            audio: [batch, samples] or [batch, channels, samples]
            stem_name: Name of stem ('drums', 'bass', 'vocals', 'other')
            params: Dict with 'gain', 'eq_low', 'eq_mid', 'eq_high', 'lpf', 'hpf'
        
        Returns:
            Processed audio [batch, channels, samples]
        """
        return self.stem_effects[stem_name](
            audio,
            gain_curve=params['gain'],
            eq_low=params['eq_low'],
            eq_mid=params['eq_mid'],
            eq_high=params['eq_high'],
            lpf_cutoff=params['lpf'],
            hpf_cutoff=params['hpf']
        )
    
    def _process_track(
        self,
        stems: Dict[str, torch.Tensor],
        parsed_params: Dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process all stems of a track and sum them.
        
        Args:
            stems: Dict mapping stem names to audio tensors
            parsed_params: Parsed parameter dict with 'stems' and 'global'
        
        Returns:
            Tuple of:
            - Summed track audio [batch, channels, samples]
            - Dict of individual processed stems
        """
        processed_stems = {}
        
        for stem_name in STEM_NAMES:
            if stem_name in stems:
                audio = stems[stem_name]
                stem_params = parsed_params['stems'][stem_name]
                processed = self._process_stem(audio, stem_name, stem_params)
                processed_stems[stem_name] = processed
        
        # Sum all processed stems
        stem_list = list(processed_stems.values())
        if len(stem_list) > 0:
            # Ensure all stems have same shape
            summed = stem_list[0]
            for stem in stem_list[1:]:
                summed = summed + stem
        else:
            raise ValueError("No stems provided")
        
        return summed, processed_stems
    
    def _apply_global_effects(
        self,
        audio: torch.Tensor,
        global_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply global effects (reverb, delay) to audio.
        
        Args:
            audio: [batch, channels, samples]
            global_params: Dict with 'reverb_send', 'delay_send', 'delay_feedback'
        
        Returns:
            Processed audio
        """
        # Apply reverb
        audio = self.reverb(audio, global_params['reverb_send'])
        
        # Apply delay
        audio = self.delay(
            audio,
            wet_dry=global_params['delay_send'],
            feedback=global_params['delay_feedback'],
            n_taps=self.config.delay_n_taps
        )
        
        return audio
    
    def forward(
        self,
        track_a_stems: Dict[str, torch.Tensor],
        track_b_stems: Dict[str, torch.Tensor],
        params: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Mix two tracks with neural network-controlled effects.
        
        Args:
            track_a_stems: Dict mapping stem names to audio tensors
                          Each tensor: [batch, samples] or [batch, channels, samples]
            track_b_stems: Same format as track_a_stems
            params: [batch, n_frames, N_TOTAL_PARAMS] control parameters
                   or [n_frames, N_TOTAL_PARAMS] (will add batch dim)
            return_intermediates: If True, return dict with intermediate outputs
        
        Returns:
            Tuple of:
            - Mixed audio [batch, 2, samples]
            - Optional dict with intermediate outputs (if return_intermediates=True)
        """
        # Parse parameters
        params_a, params_b = self.param_parser.parse(params)
        
        # Process each track's stems
        track_a_sum, track_a_processed = self._process_track(track_a_stems, params_a)
        track_b_sum, track_b_processed = self._process_track(track_b_stems, params_b)
        
        # Ensure stereo
        track_a_sum = self._ensure_stereo(track_a_sum)
        track_b_sum = self._ensure_stereo(track_b_sum)
        
        # Apply global effects to each track
        track_a_final = self._apply_global_effects(track_a_sum, params_a['global'])
        track_b_final = self._apply_global_effects(track_b_sum, params_b['global'])
        
        # Sum tracks for final mix
        mixed = track_a_final + track_b_final
        
        # Soft clip to prevent clipping
        mixed = torch.tanh(mixed)
        
        # Prepare output
        intermediates = None
        if return_intermediates:
            intermediates = {
                'track_a': {
                    'stems': track_a_processed,
                    'summed': track_a_sum,
                    'final': track_a_final
                },
                'track_b': {
                    'stems': track_b_processed,
                    'summed': track_b_sum,
                    'final': track_b_final
                },
                'params_a': params_a,
                'params_b': params_b
            }
        
        return mixed, intermediates
    
    def get_param_info(self) -> Dict:
        """
        Get information about the parameter space.
        
        Returns:
            Dict with parameter names, indices, and ranges
        """
        info = {
            'total_params': N_TOTAL_PARAMS,
            'params_per_track': N_PARAMS_PER_TRACK,
            'n_stems': N_STEMS,
            'stem_names': STEM_NAMES,
            'stem_param_names': STEM_PARAM_NAMES,
            'n_stem_params': N_STEM_PARAMS,
            'global_param_names': GLOBAL_PARAM_NAMES,
            'n_global_params': N_GLOBAL_PARAMS,
            'param_ranges': {
                'gain': (0.0, 1.0),
                'eq_low': (0.0, 2.0),
                'eq_mid': (0.0, 2.0),
                'eq_high': (0.0, 2.0),
                'lpf': (0.0, 1.0),  # 0=closed, 1=open
                'hpf': (0.0, 1.0),  # 0=open, 1=closed
                'reverb_send': (0.0, 1.0),
                'delay_send': (0.0, 1.0),
                'delay_feedback': (0.0, 1.0)
            },
            'layout': """
Parameter layout per frame (54 total):
[0-23]  Track A stems (4 stems × 6 params each)
        For each stem: [gain, eq_low, eq_mid, eq_high, lpf, hpf]
        Stem order: drums, bass, vocals, other
[24-26] Track A global: [reverb_send, delay_send, delay_feedback]
[27-50] Track B stems (same layout as Track A)
[51-53] Track B global: [reverb_send, delay_send, delay_feedback]
"""
        }
        return info


class SimpleCrossfader(nn.Module):
    """
    Simple differentiable crossfader for basic A/B transitions.
    
    Generates parameter curves for a smooth crossfade from track A to track B.
    Can be used as a baseline or starting point for neural network outputs.
    
    Args:
        fade_type: 'linear', 'equal_power', 'exponential', or 'logarithmic'
    """
    
    def __init__(self, fade_type: str = 'equal_power'):
        super().__init__()
        self.fade_type = fade_type
    
    def forward(
        self,
        n_frames: int,
        batch_size: int = 1,
        device: str = 'cpu',
        cue_in: float = 0.0,
        cue_out: float = 1.0
    ) -> torch.Tensor:
        """
        Generate crossfade parameters.
        
        Args:
            n_frames: Number of time frames
            batch_size: Batch size
            device: Device for tensors
            cue_in: Normalized position where B starts fading in (0-1)
            cue_out: Normalized position where A finishes fading out (0-1)
        
        Returns:
            [batch, n_frames, N_TOTAL_PARAMS] parameter tensor
        """
        # Create time axis
        t = torch.linspace(0, 1, n_frames, device=device)
        
        # Scale to cue region
        t_scaled = (t - cue_in) / (cue_out - cue_in + 1e-8)
        t_scaled = torch.clamp(t_scaled, 0, 1)
        
        # Calculate fade curves
        if self.fade_type == 'linear':
            fade_out = 1 - t_scaled  # Track A
            fade_in = t_scaled       # Track B
        elif self.fade_type == 'equal_power':
            # Equal power crossfade (constant perceived loudness)
            fade_out = torch.cos(t_scaled * torch.pi / 2)
            fade_in = torch.sin(t_scaled * torch.pi / 2)
        elif self.fade_type == 'exponential':
            fade_out = torch.exp(-3 * t_scaled)
            fade_in = 1 - torch.exp(-3 * t_scaled)
        elif self.fade_type == 'logarithmic':
            fade_out = torch.log10(1 - t_scaled * 0.9 + 0.1) / torch.log10(torch.tensor(0.1))
            fade_in = 1 - fade_out
        else:
            raise ValueError(f"Unknown fade type: {self.fade_type}")
        
        # Start with default parameters
        params = ParameterParser.get_default_params(n_frames, batch_size, device)
        
        # Apply fade curves to all stem gains
        fade_out = fade_out.unsqueeze(0).expand(batch_size, -1)  # [batch, n_frames]
        fade_in = fade_in.unsqueeze(0).expand(batch_size, -1)
        
        # Track A stem gains (indices 0, 6, 12, 18)
        for stem_idx in range(N_STEMS):
            gain_idx = stem_idx * N_STEM_PARAMS
            params[:, :, gain_idx] = fade_out
        
        # Track B stem gains (indices 27, 33, 39, 45)
        for stem_idx in range(N_STEMS):
            gain_idx = N_PARAMS_PER_TRACK + stem_idx * N_STEM_PARAMS
            params[:, :, gain_idx] = fade_in
        
        return params


# Convenience function for quick testing
def create_mixer(
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512
) -> DifferentiableDJMixer:
    """
    Create a DifferentiableDJMixer with common settings.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: STFT hop length
    
    Returns:
        Configured DifferentiableDJMixer instance
    """
    config = MixerConfig(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return DifferentiableDJMixer(config)
