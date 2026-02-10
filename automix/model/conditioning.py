"""
Conditioning Encoder for DJ Transition Model

Encodes audio features and metadata from both tracks into a unified
conditioning representation for the diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import Config, AudioConfig, ModelConfig


@dataclass
class TrackFeatures:
    """Features extracted from a single track."""
    # Per-stem mel spectrograms: [batch, n_stems, n_mels, n_frames]
    stem_mels: torch.Tensor
    
    # Structure labels: [batch, n_frames, n_labels] (one-hot)
    structure: torch.Tensor
    
    # Beat positions: [batch, n_frames] (1 at beats, 0 otherwise)
    beats: torch.Tensor
    
    # Downbeat positions: [batch, n_frames] (1 at downbeats, 0 otherwise)
    downbeats: torch.Tensor
    
    # BPM: [batch, 1]
    bpm: torch.Tensor
    
    # Key: [batch, n_keys] (one-hot)
    key: torch.Tensor
    
    # Energy curve: [batch, n_frames]
    energy: torch.Tensor


class MelEncoder(nn.Module):
    """
    Encodes mel spectrograms using a convolutional network.
    Processes each stem independently then combines.
    """
    
    def __init__(self, config: ModelConfig, n_mels: int = 128, n_stems: int = 4):
        super().__init__()
        self.n_stems = n_stems
        embed_dim = config.audio_embed_dim
        
        # Per-stem encoder (shared weights)
        self.stem_encoder = nn.Sequential(
            # [batch, 1, n_mels, n_frames] -> [batch, 32, n_mels/2, n_frames/2]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            
            # -> [batch, 64, n_mels/4, n_frames/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            
            # -> [batch, 128, n_mels/8, n_frames/8]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            
            # -> [batch, embed_dim, n_mels/16, n_frames/16]
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
        )
        
        # Combine stems
        self.stem_combiner = nn.Sequential(
            nn.Conv1d(embed_dim * n_stems, embed_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
        )
        
    def forward(self, stem_mels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stem_mels: [batch, n_stems, n_mels, n_frames]
            
        Returns:
            [batch, embed_dim, n_frames_compressed]
        """
        batch, n_stems, n_mels, n_frames = stem_mels.shape
        
        # Encode each stem
        stem_features = []
        for i in range(n_stems):
            mel = stem_mels[:, i:i+1, :, :]  # [batch, 1, n_mels, n_frames]
            feat = self.stem_encoder(mel)  # [batch, embed_dim, h, w]
            # Global average pool over frequency
            feat = feat.mean(dim=2)  # [batch, embed_dim, w]
            stem_features.append(feat)
        
        # Concatenate and combine
        combined = torch.cat(stem_features, dim=1)  # [batch, embed_dim*n_stems, w]
        output = self.stem_combiner(combined)  # [batch, embed_dim, w]
        
        return output


class MetadataEncoder(nn.Module):
    """
    Encodes musical metadata (structure, beats, BPM, key, energy).
    """
    
    def __init__(self, config: ModelConfig, audio_config: AudioConfig):
        super().__init__()
        
        # Structure embedding
        n_labels = len(audio_config.structure_labels)
        self.structure_embed = nn.Sequential(
            nn.Linear(n_labels, config.structure_embed_dim),
            nn.GELU(),
            nn.Linear(config.structure_embed_dim, config.structure_embed_dim),
        )
        
        # Beat/downbeat embedding
        self.beat_embed = nn.Sequential(
            nn.Linear(2, config.beat_embed_dim),  # beats + downbeats
            nn.GELU(),
            nn.Linear(config.beat_embed_dim, config.beat_embed_dim),
        )
        
        # BPM embedding (continuous value)
        self.bpm_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        
        # Key embedding
        self.key_embed = nn.Sequential(
            nn.Linear(audio_config.n_keys, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        
        # Energy embedding
        self.energy_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        
        # Combine all metadata
        meta_dim = config.structure_embed_dim + config.beat_embed_dim + 32 + 32 + 32
        self.combiner = nn.Sequential(
            nn.Linear(meta_dim, config.condition_dim // 2),
            nn.GELU(),
            nn.Linear(config.condition_dim // 2, config.condition_dim // 2),
        )
        
    def forward(
        self,
        structure: torch.Tensor,
        beats: torch.Tensor,
        downbeats: torch.Tensor,
        bpm: torch.Tensor,
        key: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            structure: [batch, n_frames, n_labels]
            beats: [batch, n_frames]
            downbeats: [batch, n_frames]
            bpm: [batch, 1]
            key: [batch, n_keys]
            energy: [batch, n_frames]
            
        Returns:
            [batch, condition_dim//2, n_frames]
        """
        batch, n_frames, _ = structure.shape
        
        # Encode structure (per-frame)
        struct_feat = self.structure_embed(structure)  # [batch, n_frames, struct_dim]
        
        # Encode beats (per-frame)
        beat_input = torch.stack([beats, downbeats], dim=-1)  # [batch, n_frames, 2]
        beat_feat = self.beat_embed(beat_input)  # [batch, n_frames, beat_dim]
        
        # Encode BPM (broadcast to all frames)
        bpm_feat = self.bpm_embed(bpm)  # [batch, 32]
        bpm_feat = bpm_feat.unsqueeze(1).expand(-1, n_frames, -1)  # [batch, n_frames, 32]
        
        # Encode key (broadcast to all frames)
        key_feat = self.key_embed(key)  # [batch, 32]
        key_feat = key_feat.unsqueeze(1).expand(-1, n_frames, -1)  # [batch, n_frames, 32]
        
        # Encode energy (per-frame)
        energy_feat = self.energy_embed(energy.unsqueeze(-1))  # [batch, n_frames, 32]
        
        # Combine
        combined = torch.cat([struct_feat, beat_feat, bpm_feat, key_feat, energy_feat], dim=-1)
        output = self.combiner(combined)  # [batch, n_frames, condition_dim//2]
        
        return output.permute(0, 2, 1)  # [batch, condition_dim//2, n_frames]


class ConditioningEncoder(nn.Module):
    """
    Full conditioning encoder for both tracks.
    
    Combines audio embeddings and metadata from track A and track B
    into a unified conditioning signal for the diffusion model.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Audio encoders (one shared encoder for both tracks)
        self.mel_encoder = MelEncoder(
            config.model,
            n_mels=config.audio.n_mels,
            n_stems=len(config.audio.stems)
        )
        
        # Metadata encoder (shared)
        self.metadata_encoder = MetadataEncoder(config.model, config.audio)
        
        # Cross-track attention to model interaction between tracks
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.model.condition_dim,
            num_heads=config.model.n_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(config.model.condition_dim * 2, config.model.condition_dim),
            nn.GELU(),
            nn.Linear(config.model.condition_dim, config.model.condition_dim),
        )
        
        # Learnable null embedding for classifier-free guidance
        self.null_embedding = nn.Parameter(torch.randn(1, config.model.condition_dim, 1))
        
    def encode_track(
        self,
        features: TrackFeatures,
        target_frames: int,
    ) -> torch.Tensor:
        """Encode a single track's features."""
        # Encode audio
        audio_feat = self.mel_encoder(features.stem_mels)  # [batch, audio_dim, frames_compressed]
        
        # Interpolate to target frames
        audio_feat = F.interpolate(audio_feat, size=target_frames, mode='linear', align_corners=False)
        
        # Encode metadata
        # First interpolate metadata to target frames
        batch = features.structure.shape[0]
        structure = F.interpolate(
            features.structure.permute(0, 2, 1),
            size=target_frames,
            mode='nearest'
        ).permute(0, 2, 1)
        beats = F.interpolate(
            features.beats.unsqueeze(1),
            size=target_frames,
            mode='nearest'
        ).squeeze(1)
        downbeats = F.interpolate(
            features.downbeats.unsqueeze(1),
            size=target_frames,
            mode='nearest'
        ).squeeze(1)
        energy = F.interpolate(
            features.energy.unsqueeze(1),
            size=target_frames,
            mode='nearest'
        ).squeeze(1)
        
        meta_feat = self.metadata_encoder(
            structure, beats, downbeats,
            features.bpm, features.key, energy
        )  # [batch, condition_dim//2, target_frames]
        
        # Combine audio and metadata
        combined = torch.cat([audio_feat, meta_feat], dim=1)  # [batch, condition_dim, frames]
        
        return combined
        
    def forward(
        self,
        track_a: TrackFeatures,
        track_b: TrackFeatures,
        n_frames: Optional[int] = None,
        drop_condition: bool = False,
    ) -> torch.Tensor:
        """
        Encode both tracks into conditioning signal.
        
        Args:
            track_a: Features from outgoing track
            track_b: Features from incoming track
            n_frames: Number of output frames (default from config)
            drop_condition: If True, return null embedding (for CFG training)
            
        Returns:
            conditioning: [batch, condition_dim, n_frames]
        """
        if n_frames is None:
            n_frames = self.config.model.n_frames
            
        batch = track_a.stem_mels.shape[0]
        
        # Handle classifier-free guidance
        if drop_condition:
            return self.null_embedding.expand(batch, -1, n_frames)
        
        # Encode both tracks
        feat_a = self.encode_track(track_a, n_frames)  # [batch, cond_dim, frames]
        feat_b = self.encode_track(track_b, n_frames)
        
        # Cross-attention: track A attends to track B and vice versa
        feat_a_t = feat_a.permute(0, 2, 1)  # [batch, frames, cond_dim]
        feat_b_t = feat_b.permute(0, 2, 1)
        
        # A attends to B
        feat_a_cross, _ = self.cross_attention(feat_a_t, feat_b_t, feat_b_t)
        # B attends to A
        feat_b_cross, _ = self.cross_attention(feat_b_t, feat_a_t, feat_a_t)
        
        # Combine with residual
        feat_a_final = feat_a_t + feat_a_cross
        feat_b_final = feat_b_t + feat_b_cross
        
        # Concatenate and project
        combined = torch.cat([feat_a_final, feat_b_final], dim=-1)  # [batch, frames, cond_dim*2]
        output = self.final_proj(combined)  # [batch, frames, cond_dim]
        
        return output.permute(0, 2, 1)  # [batch, cond_dim, frames]


class MelSpectrogramExtractor(nn.Module):
    """
    Utility to extract mel spectrograms from audio.
    Used during inference when raw audio is provided.
    """
    
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [batch, n_samples] or [batch, n_stems, n_samples]
            
        Returns:
            mel: [batch, n_mels, n_frames] or [batch, n_stems, n_mels, n_frames]
        """
        if audio.dim() == 2:
            mel = self.mel_spec(audio)
            mel = self.amplitude_to_db(mel)
        else:
            # Process each stem
            batch, n_stems, n_samples = audio.shape
            mels = []
            for i in range(n_stems):
                mel = self.mel_spec(audio[:, i])
                mel = self.amplitude_to_db(mel)
                mels.append(mel)
            mel = torch.stack(mels, dim=1)
        
        # Normalize to [-1, 1]
        mel = (mel + 40) / 40  # Assuming typical range of -80 to 0 dB
        mel = mel.clamp(-1, 1)
        
        return mel


def create_dummy_features(batch_size: int, n_frames: int, config: Config) -> Tuple[TrackFeatures, TrackFeatures]:
    """Create dummy features for testing."""
    n_stems = len(config.audio.stems)
    n_labels = len(config.audio.structure_labels)
    n_keys = config.audio.n_keys
    
    def make_features():
        return TrackFeatures(
            stem_mels=torch.randn(batch_size, n_stems, config.audio.n_mels, n_frames),
            structure=F.one_hot(torch.randint(0, n_labels, (batch_size, n_frames)), n_labels).float(),
            beats=torch.zeros(batch_size, n_frames),
            downbeats=torch.zeros(batch_size, n_frames),
            bpm=torch.full((batch_size, 1), 128.0),
            key=F.one_hot(torch.randint(0, n_keys, (batch_size,)), n_keys).float(),
            energy=torch.rand(batch_size, n_frames),
        )
    
    return make_features(), make_features()
