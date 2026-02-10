"""
DJ Transition Diffusion Model

A conditional diffusion model that generates effect parameter curves
for DJ transitions. The model learns to denoise a 2D "image" of shape
[n_frames, n_params] conditioned on audio features from both tracks.

Architecture inspired by:
- Riffusion (diffusion for spectrograms)
- DiT (Diffusion Transformer)
- DDPM with U-Net
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat

from .config import Config, DiffusionConfig, ModelConfig
from .conditioning import ConditioningEncoder, TrackFeatures


# ==================== Diffusion Schedule ====================

class DiffusionSchedule:
    """
    Manages the noise schedule for the diffusion process.
    Supports linear and cosine schedules.
    """
    
    def __init__(self, config: DiffusionConfig):
        self.n_steps = config.n_steps
        
        if config.beta_schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.n_steps)
        elif config.beta_schedule == "cosine":
            # Cosine schedule from "Improved DDPM"
            steps = torch.arange(config.n_steps + 1, dtype=torch.float32)
            alpha_bar = torch.cos(((steps / config.n_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")
        
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Store all needed values
        self.register_buffer_dict({
            'betas': betas,
            'alphas': alphas,
            'alpha_cumprod': alpha_cumprod,
            'alpha_cumprod_prev': alpha_cumprod_prev,
            'sqrt_alpha_cumprod': torch.sqrt(alpha_cumprod),
            'sqrt_one_minus_alpha_cumprod': torch.sqrt(1 - alpha_cumprod),
            'sqrt_recip_alpha_cumprod': torch.sqrt(1 / alpha_cumprod),
            'sqrt_recip_alpha_cumprod_minus_one': torch.sqrt(1 / alpha_cumprod - 1),
            'posterior_variance': betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod),
            'posterior_log_variance': torch.log(
                torch.clamp(betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod), min=1e-20)
            ),
            'posterior_mean_coef1': betas * torch.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod),
            'posterior_mean_coef2': (1 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1 - alpha_cumprod),
        })
    
    def register_buffer_dict(self, d: Dict[str, torch.Tensor]):
        """Store tensors as attributes."""
        for k, v in d.items():
            setattr(self, k, v)
    
    def to(self, device):
        """Move all tensors to device."""
        for k in ['betas', 'alphas', 'alpha_cumprod', 'alpha_cumprod_prev',
                  'sqrt_alpha_cumprod', 'sqrt_one_minus_alpha_cumprod',
                  'sqrt_recip_alpha_cumprod', 'sqrt_recip_alpha_cumprod_minus_one',
                  'posterior_variance', 'posterior_log_variance',
                  'posterior_mean_coef1', 'posterior_mean_coef2']:
            setattr(self, k, getattr(self, k).to(device))
        return self
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t]
        
        # Reshape for broadcasting: [batch] -> [batch, 1, 1]
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x0 from noisy xt and predicted noise."""
        sqrt_recip = self.sqrt_recip_alpha_cumprod[t]
        sqrt_recip_m1 = self.sqrt_recip_alpha_cumprod_minus_one[t]
        
        while sqrt_recip.dim() < xt.dim():
            sqrt_recip = sqrt_recip.unsqueeze(-1)
            sqrt_recip_m1 = sqrt_recip_m1.unsqueeze(-1)
        
        return sqrt_recip * xt - sqrt_recip_m1 * noise


# ==================== Model Components ====================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position/time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch] timesteps
        Returns:
            [batch, dim] embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning."""
    
    def __init__(self, dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            context: [batch, context_len, context_dim]
        """
        batch, seq_len, _ = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention with relative position bias."""
    
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    Transformer block with:
    - Self-attention
    - Cross-attention to conditioning
    - Feed-forward
    - AdaLN (Adaptive Layer Norm) for time conditioning
    """
    
    def __init__(self, dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, n_heads, dropout)
        
        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, n_heads, dropout)
        
        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        
        # AdaLN modulation (scale and shift for each norm)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            context: [batch, context_len, context_dim]
            time_emb: [batch, dim]
        """
        # Get modulation parameters
        mod = self.adaLN_modulation(time_emb)
        shift1, scale1, shift2, scale2, shift3, scale3 = mod.chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.self_attn(h)
        
        # Cross-attention with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.cross_attn(h, context)
        
        # Feed-forward with AdaLN
        h = self.norm3(x)
        h = h * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = x + self.ff(h)
        
        return x


# ==================== Main Model ====================

class ParameterDiffusionModel(nn.Module):
    """
    Conditional diffusion model for DJ transition parameter curves.
    
    Takes noisy parameter curves and conditioning from both tracks,
    predicts the noise to denoise the curves.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        model_config = config.model
        
        # Dimensions
        self.n_frames = model_config.n_frames
        self.n_params = config.effects.n_params
        self.hidden_dim = model_config.hidden_dim
        
        # Input projection: [batch, n_frames, n_params] -> [batch, n_frames, hidden_dim]
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_params, model_config.hidden_dim),
            nn.GELU(),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_config.time_embed_dim),
            nn.Linear(model_config.time_embed_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
        )
        
        # Position embedding for frames
        self.pos_embed = nn.Parameter(torch.randn(1, model_config.n_frames, model_config.hidden_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=model_config.hidden_dim,
                context_dim=model_config.condition_dim,
                n_heads=model_config.n_heads,
                dropout=model_config.dropout,
            )
            for _ in range(model_config.n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(model_config.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.GELU(),
            nn.Linear(model_config.hidden_dim, self.n_params),
        )
        
        # Initialize output to zero (helps training stability)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from noisy input.
        
        Args:
            x: [batch, n_frames, n_params] noisy parameter curves
            t: [batch] timesteps
            conditioning: [batch, condition_dim, n_frames] from ConditioningEncoder
            
        Returns:
            noise_pred: [batch, n_frames, n_params] predicted noise
        """
        batch = x.shape[0]
        
        # Project input
        h = self.input_proj(x)  # [batch, n_frames, hidden_dim]
        
        # Add position embedding
        h = h + self.pos_embed[:, :h.shape[1], :]
        
        # Get time embedding
        time_emb = self.time_embed(t)  # [batch, hidden_dim]
        
        # Prepare conditioning: [batch, condition_dim, n_frames] -> [batch, n_frames, condition_dim]
        context = conditioning.permute(0, 2, 1)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, context, time_emb)
        
        # Output
        h = self.output_norm(h)
        noise_pred = self.output_proj(h)
        
        return noise_pred


# ==================== Full System ====================

class DJTransitionModel(nn.Module):
    """
    Complete DJ transition model.
    
    Combines:
    - ConditioningEncoder: Encodes audio and metadata from both tracks
    - ParameterDiffusionModel: Generates effect parameter curves via diffusion
    - DiffusionSchedule: Manages noise schedule
    
    Usage:
        model = DJTransitionModel(config)
        
        # Training
        loss = model.training_step(track_a, track_b, target_params)
        
        # Inference
        params = model.generate(track_a, track_b)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Components
        self.conditioning_encoder = ConditioningEncoder(config)
        self.denoiser = ParameterDiffusionModel(config)
        self.schedule = DiffusionSchedule(config.diffusion)
        
    def to(self, device):
        """Move model and schedule to device."""
        super().to(device)
        self.schedule = self.schedule.to(device)
        return self
        
    def forward(
        self,
        track_a: TrackFeatures,
        track_b: TrackFeatures,
        target_params: torch.Tensor,
        drop_condition_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            track_a: Features from outgoing track
            track_b: Features from incoming track
            target_params: [batch, n_frames, n_params] ground truth parameters
            drop_condition_prob: Probability of dropping conditioning (for CFG)
            
        Returns:
            dict with 'loss', 'noise_pred', 'noise'
        """
        batch = target_params.shape[0]
        device = target_params.device
        
        # Encode conditioning (with random dropout for CFG)
        drop_condition = torch.rand(batch, device=device) < drop_condition_prob
        conditioning = self.conditioning_encoder(
            track_a, track_b,
            n_frames=target_params.shape[1],
            drop_condition=drop_condition.any(),  # Simplification: drop all or none
        )
        
        # Sample random timesteps
        t = torch.randint(0, self.config.diffusion.n_steps, (batch,), device=device)
        
        # Sample noise and create noisy input
        noise = torch.randn_like(target_params)
        noisy_params = self.schedule.q_sample(target_params, t, noise)
        
        # Predict noise
        noise_pred = self.denoiser(noisy_params, t, conditioning)
        
        # MSE loss on noise
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }
    
    @torch.no_grad()
    def generate(
        self,
        track_a: TrackFeatures,
        track_b: TrackFeatures,
        n_frames: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate effect parameter curves.
        
        Args:
            track_a: Features from outgoing track
            track_b: Features from incoming track
            n_frames: Number of output frames (default from config)
            guidance_scale: Classifier-free guidance scale (default from config)
            n_steps: Number of sampling steps (default from config)
            return_trajectory: If True, return intermediate steps
            
        Returns:
            params: [batch, n_frames, n_params] in range [0, 1]
        """
        if n_frames is None:
            n_frames = self.config.model.n_frames
        if guidance_scale is None:
            guidance_scale = self.config.diffusion.guidance_scale
        if n_steps is None:
            n_steps = self.config.diffusion.n_inference_steps
            
        batch = track_a.stem_mels.shape[0]
        device = track_a.stem_mels.device
        
        # Encode conditioning
        cond = self.conditioning_encoder(track_a, track_b, n_frames=n_frames)
        uncond = self.conditioning_encoder(track_a, track_b, n_frames=n_frames, drop_condition=True)
        
        # DDIM sampling with classifier-free guidance
        # Use subset of timesteps for faster sampling
        step_size = self.config.diffusion.n_steps // n_steps
        timesteps = torch.arange(0, self.config.diffusion.n_steps, step_size, device=device)
        timesteps = timesteps.flip(0)  # Reverse for denoising
        
        # Start from pure noise
        x = torch.randn(batch, n_frames, self.config.effects.n_params, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch)
            
            # Classifier-free guidance: combine conditional and unconditional predictions
            noise_cond = self.denoiser(x, t_batch, cond)
            noise_uncond = self.denoiser(x, t_batch, uncond)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # DDIM update
            alpha = self.schedule.alpha_cumprod[t]
            alpha_prev = self.schedule.alpha_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            # Predict x0
            x0_pred = self.schedule.predict_x0_from_noise(x, t_batch, noise_pred)
            
            # Clamp x0 prediction to valid range
            x0_pred = x0_pred.clamp(-1, 1)
            
            # DDIM deterministic update
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        # Final prediction and sigmoid to [0, 1]
        params = torch.sigmoid(x)
        
        if return_trajectory:
            return params, trajectory
        return params
    
    def training_step(
        self,
        track_a: TrackFeatures,
        track_b: TrackFeatures,
        target_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Convenience method for training."""
        return self.forward(
            track_a, track_b, target_params,
            drop_condition_prob=self.config.training.condition_dropout,
        )


def create_model(config: Optional[Config] = None) -> DJTransitionModel:
    """Create a new model with the given config."""
    if config is None:
        config = Config()
    return DJTransitionModel(config)


def load_model(path: str, device: str = "cpu") -> DJTransitionModel:
    """Load a trained model from checkpoint."""
    # Use weights_only=False for checkpoints that include config objects
    # This is safe as we trust our own checkpoints
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', Config())
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
