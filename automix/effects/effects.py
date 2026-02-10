"""
Differentiable DJ Effects Module

Individual differentiable effect implementations for neural network-controlled DJ mixing.
All operations are PyTorch autograd compatible.

Author: Clawd
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DifferentiableGain(nn.Module):
    """
    Differentiable gain/fader control.
    
    Applies a time-varying gain curve to audio. The gain curve is interpolated
    from control points to match the audio length.
    
    Args:
        smooth_frames: Number of frames for smoothing transitions (anti-zipper)
    """
    
    def __init__(self, smooth_frames: int = 8):
        super().__init__()
        self.smooth_frames = smooth_frames
    
    def forward(self, audio: torch.Tensor, gain_curve: torch.Tensor) -> torch.Tensor:
        """
        Apply gain curve to audio.
        
        Args:
            audio: [batch, channels, samples] or [batch, samples]
            gain_curve: [batch, n_frames] gain values (0-1)
            
        Returns:
            Gained audio with same shape as input
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        batch, channels, samples = audio.shape
        n_frames = gain_curve.shape[-1]
        
        # Interpolate gain curve to match audio length
        # Use linear interpolation for smooth transitions
        gain_curve = gain_curve.unsqueeze(1)  # [batch, 1, n_frames]
        gain_interp = F.interpolate(
            gain_curve, 
            size=samples, 
            mode='linear', 
            align_corners=True
        )  # [batch, 1, samples]
        
        # Apply optional smoothing to prevent zipper noise
        if self.smooth_frames > 0:
            kernel_size = max(3, self.smooth_frames * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            padding = kernel_size // 2
            gain_interp = F.avg_pool1d(
                gain_interp, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=padding
            )
        
        return audio * gain_interp


class DifferentiableEQ(nn.Module):
    """
    Differentiable 3-band parametric EQ implemented in frequency domain.
    
    Uses STFT for frequency-domain processing with differentiable soft band masks.
    Bands: Low (0-300Hz), Mid (300-4000Hz), High (4000Hz+)
    
    Args:
        sample_rate: Audio sample rate (default 44100)
        n_fft: FFT size (default 2048)
        hop_length: STFT hop length (default 512)
        low_freq: Low band upper cutoff (default 300)
        high_freq: High band lower cutoff (default 4000)
        transition_width: Soft transition width in Hz (default 100)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        low_freq: float = 300.0,
        high_freq: float = 4000.0,
        transition_width: float = 100.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.transition_width = transition_width
        
        # Pre-compute frequency bins
        self.register_buffer(
            'freqs',
            torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        )
        
        # Create soft band masks (differentiable sigmoid transitions)
        self._build_band_masks()
    
    def _build_band_masks(self):
        """Build soft band separation masks using sigmoid transitions."""
        freqs = self.freqs
        tw = self.transition_width
        
        # Low band: sigmoid falloff at low_freq
        low_mask = torch.sigmoid(-(freqs - self.low_freq) / (tw / 4))
        
        # High band: sigmoid rise at high_freq
        high_mask = torch.sigmoid((freqs - self.high_freq) / (tw / 4))
        
        # Mid band: what's left (using soft complement)
        mid_mask = 1.0 - low_mask - high_mask
        mid_mask = F.relu(mid_mask)  # Ensure non-negative
        
        # Normalize so bands sum to 1
        total = low_mask + mid_mask + high_mask + 1e-8
        low_mask = low_mask / total
        mid_mask = mid_mask / total
        high_mask = high_mask / total
        
        self.register_buffer('low_mask', low_mask)
        self.register_buffer('mid_mask', mid_mask)
        self.register_buffer('high_mask', high_mask)
    
    def forward(
        self,
        audio: torch.Tensor,
        low_gain: torch.Tensor,
        mid_gain: torch.Tensor,
        high_gain: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply 3-band EQ to audio.
        
        Args:
            audio: [batch, channels, samples] or [batch, samples]
            low_gain: [batch, n_frames] low band gain (0-2, 1=unity)
            mid_gain: [batch, n_frames] mid band gain (0-2, 1=unity)
            high_gain: [batch, n_frames] high band gain (0-2, 1=unity)
            
        Returns:
            EQ'd audio with same shape as input
        """
        squeeze_output = False
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze_output = True
        
        batch, channels, samples = audio.shape
        
        # Compute STFT
        # Flatten batch and channels for stft
        audio_flat = audio.reshape(batch * channels, samples)
        
        window = torch.hann_window(self.n_fft, device=audio.device, dtype=audio.dtype)
        stft = torch.stft(
            audio_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )  # [batch*channels, freq_bins, time_frames]
        
        _, freq_bins, time_frames = stft.shape
        stft = stft.reshape(batch, channels, freq_bins, time_frames)
        
        # Interpolate gain curves to match STFT time frames
        low_gain_interp = F.interpolate(
            low_gain.unsqueeze(1), size=time_frames, mode='linear', align_corners=True
        ).squeeze(1)  # [batch, time_frames]
        mid_gain_interp = F.interpolate(
            mid_gain.unsqueeze(1), size=time_frames, mode='linear', align_corners=True
        ).squeeze(1)
        high_gain_interp = F.interpolate(
            high_gain.unsqueeze(1), size=time_frames, mode='linear', align_corners=True
        ).squeeze(1)
        
        # Build per-frame gain mask
        # [batch, freq_bins, time_frames]
        low_mask = self.low_mask[:freq_bins].unsqueeze(0).unsqueeze(-1)  # [1, freq, 1]
        mid_mask = self.mid_mask[:freq_bins].unsqueeze(0).unsqueeze(-1)
        high_mask = self.high_mask[:freq_bins].unsqueeze(0).unsqueeze(-1)
        
        low_g = low_gain_interp.unsqueeze(1)  # [batch, 1, time]
        mid_g = mid_gain_interp.unsqueeze(1)
        high_g = high_gain_interp.unsqueeze(1)
        
        # Combined gain per frequency bin per time frame
        gain_mask = (
            low_mask * low_g +
            mid_mask * mid_g +
            high_mask * high_g
        )  # [batch, freq_bins, time_frames]
        
        # Apply gain to STFT
        gain_mask = gain_mask.unsqueeze(1)  # [batch, 1, freq, time]
        stft_eq = stft * gain_mask
        
        # Inverse STFT
        stft_eq_flat = stft_eq.reshape(batch * channels, freq_bins, time_frames)
        audio_eq = torch.istft(
            stft_eq_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=samples
        )
        audio_eq = audio_eq.reshape(batch, channels, samples)
        
        if squeeze_output:
            audio_eq = audio_eq.squeeze(1)
        
        return audio_eq


class DifferentiableLowPassFilter(nn.Module):
    """
    Differentiable low-pass filter with sweepable cutoff frequency.
    
    Implemented in frequency domain using soft sigmoid masks for
    smooth, differentiable filtering.
    
    Args:
        sample_rate: Audio sample rate (default 44100)
        n_fft: FFT size (default 2048)
        hop_length: STFT hop length (default 512)
        min_freq: Minimum cutoff frequency (default 20)
        max_freq: Maximum cutoff frequency (default 20000)
        transition_octaves: Transition width in octaves (default 0.5)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
        transition_octaves: float = 0.5
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.transition_octaves = transition_octaves
        
        self.register_buffer(
            'freqs',
            torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        )
    
    def _get_filter_mask(self, cutoff_freq: torch.Tensor, freq_bins: int) -> torch.Tensor:
        """
        Generate soft low-pass filter mask.
        
        Args:
            cutoff_freq: [batch, time_frames] cutoff frequencies
            freq_bins: Number of frequency bins
            
        Returns:
            [batch, freq_bins, time_frames] filter mask
        """
        freqs = self.freqs[:freq_bins]  # [freq_bins]
        
        # Prevent log of zero
        freqs_safe = torch.clamp(freqs, min=1.0)
        cutoff_safe = torch.clamp(cutoff_freq, min=1.0)
        
        # Work in log-frequency domain for musical response
        log_freqs = torch.log2(freqs_safe)  # [freq_bins]
        log_cutoff = torch.log2(cutoff_safe)  # [batch, time_frames]
        
        # Transition width in log domain
        tw = self.transition_octaves
        
        # Soft sigmoid transition
        # [batch, freq_bins, time_frames]
        log_freqs = log_freqs.unsqueeze(0).unsqueeze(-1)  # [1, freq, 1]
        log_cutoff = log_cutoff.unsqueeze(1)  # [batch, 1, time]
        
        mask = torch.sigmoid(-(log_freqs - log_cutoff) / (tw / 4))
        
        return mask
    
    def forward(
        self,
        audio: torch.Tensor,
        cutoff_curve: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply low-pass filter with time-varying cutoff.
        
        Args:
            audio: [batch, channels, samples] or [batch, samples]
            cutoff_curve: [batch, n_frames] normalized cutoff (0-1)
                         0 = min_freq, 1 = max_freq
            
        Returns:
            Filtered audio with same shape as input
        """
        squeeze_output = False
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze_output = True
        
        batch, channels, samples = audio.shape
        
        # Convert normalized cutoff to frequency
        # Exponential mapping for musical response
        log_min = math.log2(self.min_freq)
        log_max = math.log2(self.max_freq)
        cutoff_freq = 2 ** (log_min + cutoff_curve * (log_max - log_min))
        
        # Compute STFT
        audio_flat = audio.reshape(batch * channels, samples)
        window = torch.hann_window(self.n_fft, device=audio.device, dtype=audio.dtype)
        stft = torch.stft(
            audio_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        
        _, freq_bins, time_frames = stft.shape
        stft = stft.reshape(batch, channels, freq_bins, time_frames)
        
        # Interpolate cutoff curve to match STFT frames
        cutoff_interp = F.interpolate(
            cutoff_freq.unsqueeze(1), size=time_frames, mode='linear', align_corners=True
        ).squeeze(1)
        
        # Get filter mask
        mask = self._get_filter_mask(cutoff_interp, freq_bins)  # [batch, freq, time]
        mask = mask.unsqueeze(1)  # [batch, 1, freq, time]
        
        # Apply filter
        stft_filtered = stft * mask
        
        # Inverse STFT
        stft_filtered_flat = stft_filtered.reshape(batch * channels, freq_bins, time_frames)
        audio_filtered = torch.istft(
            stft_filtered_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=samples
        )
        audio_filtered = audio_filtered.reshape(batch, channels, samples)
        
        if squeeze_output:
            audio_filtered = audio_filtered.squeeze(1)
        
        return audio_filtered


class DifferentiableHighPassFilter(nn.Module):
    """
    Differentiable high-pass filter with sweepable cutoff frequency.
    
    Implemented in frequency domain using soft sigmoid masks.
    
    Args:
        sample_rate: Audio sample rate (default 44100)
        n_fft: FFT size (default 2048)
        hop_length: STFT hop length (default 512)
        min_freq: Minimum cutoff frequency (default 20)
        max_freq: Maximum cutoff frequency (default 20000)
        transition_octaves: Transition width in octaves (default 0.5)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
        transition_octaves: float = 0.5
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.transition_octaves = transition_octaves
        
        self.register_buffer(
            'freqs',
            torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        )
    
    def _get_filter_mask(self, cutoff_freq: torch.Tensor, freq_bins: int) -> torch.Tensor:
        """Generate soft high-pass filter mask."""
        freqs = self.freqs[:freq_bins]
        
        freqs_safe = torch.clamp(freqs, min=1.0)
        cutoff_safe = torch.clamp(cutoff_freq, min=1.0)
        
        log_freqs = torch.log2(freqs_safe)
        log_cutoff = torch.log2(cutoff_safe)
        
        tw = self.transition_octaves
        
        log_freqs = log_freqs.unsqueeze(0).unsqueeze(-1)
        log_cutoff = log_cutoff.unsqueeze(1)
        
        # High-pass: sigmoid rises above cutoff
        mask = torch.sigmoid((log_freqs - log_cutoff) / (tw / 4))
        
        return mask
    
    def forward(
        self,
        audio: torch.Tensor,
        cutoff_curve: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply high-pass filter with time-varying cutoff.
        
        Args:
            audio: [batch, channels, samples] or [batch, samples]
            cutoff_curve: [batch, n_frames] normalized cutoff (0-1)
                         0 = min_freq, 1 = max_freq
            
        Returns:
            Filtered audio with same shape as input
        """
        squeeze_output = False
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze_output = True
        
        batch, channels, samples = audio.shape
        
        log_min = math.log2(self.min_freq)
        log_max = math.log2(self.max_freq)
        cutoff_freq = 2 ** (log_min + cutoff_curve * (log_max - log_min))
        
        audio_flat = audio.reshape(batch * channels, samples)
        window = torch.hann_window(self.n_fft, device=audio.device, dtype=audio.dtype)
        stft = torch.stft(
            audio_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        
        _, freq_bins, time_frames = stft.shape
        stft = stft.reshape(batch, channels, freq_bins, time_frames)
        
        cutoff_interp = F.interpolate(
            cutoff_freq.unsqueeze(1), size=time_frames, mode='linear', align_corners=True
        ).squeeze(1)
        
        mask = self._get_filter_mask(cutoff_interp, freq_bins)
        mask = mask.unsqueeze(1)
        
        stft_filtered = stft * mask
        
        stft_filtered_flat = stft_filtered.reshape(batch * channels, freq_bins, time_frames)
        audio_filtered = torch.istft(
            stft_filtered_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=samples
        )
        audio_filtered = audio_filtered.reshape(batch, channels, samples)
        
        if squeeze_output:
            audio_filtered = audio_filtered.squeeze(1)
        
        return audio_filtered


class DifferentiableReverb(nn.Module):
    """
    Differentiable reverb effect using FFT-based convolution.
    
    Uses a simple algorithmic impulse response that can be generated
    on-the-fly, making it fully differentiable.
    
    Args:
        sample_rate: Audio sample rate (default 44100)
        ir_length: Impulse response length in samples (default 44100 = 1 sec)
        decay_time: RT60 decay time in seconds (default 1.5)
        damping: High frequency damping factor (default 0.5)
        pre_delay_ms: Pre-delay in milliseconds (default 20)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        ir_length: int = 44100,
        decay_time: float = 1.5,
        damping: float = 0.5,
        pre_delay_ms: float = 20.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.ir_length = ir_length
        self.decay_time = decay_time
        self.damping = damping
        self.pre_delay_samples = int(pre_delay_ms * sample_rate / 1000)
        
        # Generate algorithmic IR
        self._generate_ir()
    
    def _generate_ir(self):
        """Generate a simple exponential decay impulse response."""
        t = torch.linspace(0, self.ir_length / self.sample_rate, self.ir_length)
        
        # Exponential decay
        decay = torch.exp(-3.0 * t / self.decay_time)
        
        # Add some early reflections (simplified)
        ir = torch.randn(self.ir_length) * decay
        
        # Apply damping (simple low-pass effect on decay)
        damping_curve = torch.exp(-self.damping * t * 10)
        ir = ir * damping_curve
        
        # Add pre-delay
        if self.pre_delay_samples > 0:
            ir = F.pad(ir, (self.pre_delay_samples, 0))[:self.ir_length]
        
        # Normalize
        ir = ir / (torch.max(torch.abs(ir)) + 1e-8)
        
        self.register_buffer('ir', ir)
    
    def forward(
        self,
        audio: torch.Tensor,
        wet_dry: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply reverb with time-varying wet/dry mix.
        
        Args:
            audio: [batch, channels, samples] or [batch, samples]
            wet_dry: [batch, n_frames] wet amount (0=dry, 1=full wet)
            
        Returns:
            Reverbed audio with same shape as input
        """
        squeeze_output = False
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze_output = True
        
        batch, channels, samples = audio.shape
        
        # FFT convolution for reverb
        # Pad for linear convolution
        fft_size = samples + self.ir_length - 1
        fft_size = 2 ** math.ceil(math.log2(fft_size))  # Next power of 2
        
        audio_flat = audio.reshape(batch * channels, samples)
        
        # FFT of audio
        audio_fft = torch.fft.rfft(audio_flat, n=fft_size)
        
        # FFT of IR (expand to match batch)
        ir_fft = torch.fft.rfft(self.ir, n=fft_size)
        ir_fft = ir_fft.unsqueeze(0).expand(batch * channels, -1)
        
        # Convolution in frequency domain
        reverb_fft = audio_fft * ir_fft
        
        # Inverse FFT
        reverb = torch.fft.irfft(reverb_fft, n=fft_size)
        reverb = reverb[:, :samples]  # Trim to original length
        reverb = reverb.reshape(batch, channels, samples)
        
        # Normalize reverb
        reverb = reverb / (torch.max(torch.abs(reverb)) + 1e-8) * torch.max(torch.abs(audio))
        
        # Interpolate wet/dry curve
        wet_interp = F.interpolate(
            wet_dry.unsqueeze(1), size=samples, mode='linear', align_corners=True
        )  # [batch, 1, samples]
        
        # Mix
        output = audio * (1 - wet_interp) + reverb * wet_interp
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output


class DifferentiableDelay(nn.Module):
    """
    Differentiable delay/echo effect with feedback.
    
    Implements a simple delay line with feedback, suitable for
    echo and slapback effects.
    
    Args:
        sample_rate: Audio sample rate (default 44100)
        max_delay_ms: Maximum delay time in milliseconds (default 1000)
        default_delay_ms: Default delay time (default 375 = 1/8 note at 120bpm)
        max_feedback: Maximum feedback amount (default 0.7)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        max_delay_ms: float = 1000.0,
        default_delay_ms: float = 375.0,
        max_feedback: float = 0.7
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.default_delay_samples = int(default_delay_ms * sample_rate / 1000)
        self.max_feedback = max_feedback
    
    def forward(
        self,
        audio: torch.Tensor,
        wet_dry: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
        delay_time: Optional[torch.Tensor] = None,
        n_taps: int = 4
    ) -> torch.Tensor:
        """
        Apply delay effect with time-varying wet/dry mix.
        
        Args:
            audio: [batch, channels, samples] or [batch, samples]
            wet_dry: [batch, n_frames] wet amount (0=dry, 1=full wet)
            feedback: [batch, n_frames] feedback amount (0-1), optional
            delay_time: [batch, n_frames] delay time normalized (0-1), optional
            n_taps: Number of delay taps for feedback simulation
            
        Returns:
            Delayed audio with same shape as input
        """
        squeeze_output = False
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze_output = True
        
        batch, channels, samples = audio.shape
        
        # Default feedback if not provided
        if feedback is None:
            feedback = torch.full((batch, wet_dry.shape[-1]), 0.5, device=audio.device)
        
        # Default delay time if not provided
        delay_samples = self.default_delay_samples
        
        # Create delayed versions with feedback decay
        # This is a simplified differentiable approximation
        delayed = torch.zeros_like(audio)
        
        # Average feedback over time for simplicity (fully time-varying is expensive)
        avg_feedback = feedback.mean(dim=-1, keepdim=True)  # [batch, 1]
        avg_feedback = avg_feedback * self.max_feedback
        
        for tap in range(n_taps):
            tap_delay = delay_samples * (tap + 1)
            if tap_delay >= samples:
                break
            
            tap_gain = avg_feedback ** (tap + 1)  # [batch, 1]
            tap_gain = tap_gain.unsqueeze(-1)  # [batch, 1, 1]
            
            # Shift audio by delay amount
            shifted = F.pad(audio, (tap_delay, 0))[:, :, :samples]
            delayed = delayed + shifted * tap_gain
        
        # Interpolate wet/dry curve
        wet_interp = F.interpolate(
            wet_dry.unsqueeze(1), size=samples, mode='linear', align_corners=True
        )
        
        # Mix
        output = audio + delayed * wet_interp
        
        # Soft clip to prevent explosion
        output = torch.tanh(output)
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output


class StemEffectsChain(nn.Module):
    """
    Complete effects chain for a single audio stem.
    
    Applies: Gain → EQ → Low-pass → High-pass in series.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size for frequency-domain effects
        hop_length: STFT hop length
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        super().__init__()
        
        self.gain = DifferentiableGain()
        self.eq = DifferentiableEQ(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.lpf = DifferentiableLowPassFilter(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        self.hpf = DifferentiableHighPassFilter(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        gain_curve: torch.Tensor,
        eq_low: torch.Tensor,
        eq_mid: torch.Tensor,
        eq_high: torch.Tensor,
        lpf_cutoff: torch.Tensor,
        hpf_cutoff: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply full effects chain to audio.
        
        Args:
            audio: [batch, samples] or [batch, channels, samples]
            gain_curve: [batch, n_frames] gain (0-1)
            eq_low: [batch, n_frames] low band gain (0-2)
            eq_mid: [batch, n_frames] mid band gain (0-2)
            eq_high: [batch, n_frames] high band gain (0-2)
            lpf_cutoff: [batch, n_frames] LP cutoff (0-1)
            hpf_cutoff: [batch, n_frames] HP cutoff (0-1)
            
        Returns:
            Processed audio
        """
        # Apply effects in chain
        x = self.gain(audio, gain_curve)
        x = self.eq(x, eq_low, eq_mid, eq_high)
        x = self.lpf(x, lpf_cutoff)
        x = self.hpf(x, hpf_cutoff)
        
        return x
