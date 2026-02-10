#!/usr/bin/env python3
"""
DJTransGAN v2 Training Script

Main training script that:
1. Imports the diffusion model from model/
2. Imports the differentiable mixer from effects/
3. Uses the Dataset from dataset.py
4. Implements full training loop with audio loss

Training loop:
a. Model predicts effect parameter curves from track features
b. Mixer applies those parameters to produce mixed audio
c. Compare against real DJ transition (MSE + perceptual loss)
d. Backprop through mixer → model

Supports:
- Resume from checkpoint
- Logging to tensorboard/wandb
- Multi-GPU training (DDP)
- Mixed precision (AMP)
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .model import (
    Config,
    DJTransitionModel,
    TrackFeatures,
    create_model,
    get_config,
)
from .effects import (
    DifferentiableDJMixer,
    MixerConfig,
    N_TOTAL_PARAMS,
    STEM_NAMES,
)
from .dataset import (
    DJTransitionDataset,
    SyntheticTransitionDataset,
    collate_transitions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class EMA:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class AudioLoss(nn.Module):
    """
    Combined loss for audio comparison.
    
    Components:
    - MSE in waveform domain
    - L1 in spectrogram domain (perceptual)
    - Multi-scale spectrogram loss
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft_list: list = [512, 1024, 2048],
        hop_factor: int = 4,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft_list = n_fft_list
        self.hop_factor = hop_factor
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [batch, samples] or [batch, channels, samples]
            target: same shape as pred
        """
        # Ensure same shape
        if pred.dim() == 3:
            pred = pred.mean(dim=1)
        if target.dim() == 3:
            target = target.mean(dim=1)
        
        # Align lengths
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        losses = {}
        
        # Waveform MSE
        losses['waveform_mse'] = F.mse_loss(pred, target)
        
        # Multi-scale spectrogram loss
        spec_loss = 0
        for n_fft in self.n_fft_list:
            hop_length = n_fft // self.hop_factor
            
            window = torch.hann_window(n_fft, device=pred.device)
            
            pred_spec = torch.stft(
                pred, n_fft, hop_length, window=window, return_complex=True
            )
            target_spec = torch.stft(
                target, n_fft, hop_length, window=window, return_complex=True
            )
            
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()
            
            # L1 loss on magnitude
            spec_loss += F.l1_loss(pred_mag, target_mag)
            
            # Log magnitude loss (better for low energy)
            pred_log = torch.log1p(pred_mag)
            target_log = torch.log1p(target_mag)
            spec_loss += F.l1_loss(pred_log, target_log)
        
        losses['spectral'] = spec_loss / len(self.n_fft_list)
        
        # Combined loss
        losses['total'] = losses['waveform_mse'] + 0.5 * losses['spectral']
        
        return losses


class SmoothnessLoss(nn.Module):
    """Penalize rapid parameter changes."""
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params: [batch, n_frames, n_params]
        diff = params[:, 1:, :] - params[:, :-1, :]
        return (diff ** 2).mean()


class DJTransitionTrainer:
    """
    Full training loop for DJ transition model.
    """
    
    def __init__(
        self,
        config: Config,
        data_dir: str,
        output_dir: str = "./output",
        resume_from: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        synthetic_data: bool = False,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Verify parameter counts match
        assert config.effects.n_params == N_TOTAL_PARAMS, \
            f"Model config has {config.effects.n_params} params but mixer expects {N_TOTAL_PARAMS}"
        
        # Create model
        self.model = create_model(config)
        self.model.to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        
        # Create mixer
        mixer_config = MixerConfig(
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
        )
        self.mixer = DifferentiableDJMixer(mixer_config)
        self.mixer.to(self.device)
        
        # Create dataset
        if synthetic_data:
            self.train_dataset = SyntheticTransitionDataset(
                tracks_dir=f"{data_dir}/tracks",
                stems_dir=f"{data_dir}/stems",
                config=config,
                n_frames=config.model.n_frames,
            )
        else:
            self.train_dataset = DJTransitionDataset(
                data_dir=data_dir,
                config=config,
                n_frames=config.model.n_frames,
            )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_transitions,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Learning rate scheduler
        def lr_lambda(step):
            if step < config.training.warmup_steps:
                return step / max(1, config.training.warmup_steps)
            progress = (step - config.training.warmup_steps) / max(
                1, config.training.max_steps - config.training.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss functions
        self.audio_loss = AudioLoss(sample_rate=config.audio.sample_rate)
        self.smoothness_loss = SmoothnessLoss()
        
        # EMA
        self.ema = EMA(self.model, decay=config.training.ema_decay)
        
        # Mixed precision
        self.use_amp = config.training.use_amp and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.use_wandb = use_wandb and HAS_WANDB
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        
        if self.use_wandb:
            wandb.init(
                project="djtransgan-v2",
                config=vars(config),
                dir=str(self.output_dir),
            )
        
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def move_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        def move_features(f: TrackFeatures) -> TrackFeatures:
            return TrackFeatures(
                stem_mels=f.stem_mels.to(self.device),
                structure=f.structure.to(self.device),
                beats=f.beats.to(self.device),
                downbeats=f.downbeats.to(self.device),
                bpm=f.bpm.to(self.device),
                key=f.key.to(self.device),
                energy=f.energy.to(self.device),
            )
        
        return {
            'track_a_features': move_features(batch['track_a_features']),
            'track_b_features': move_features(batch['track_b_features']),
            'track_a_stems': batch['track_a_stems'].to(self.device),
            'track_b_stems': batch['track_b_stems'].to(self.device),
            'transition_audio': batch['transition_audio'].to(self.device),
        }
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        batch = self.move_to_device(batch)
        
        # Convert stems tensor to dict format for mixer
        def stems_tensor_to_dict(stems_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
            # stems_tensor: [batch, n_stems, samples]
            return {
                name: stems_tensor[:, i]
                for i, name in enumerate(STEM_NAMES)
            }
        
        stems_a = stems_tensor_to_dict(batch['track_a_stems'])
        stems_b = stems_tensor_to_dict(batch['track_b_stems'])
        target_audio = batch['transition_audio']
        
        with autocast(enabled=self.use_amp):
            # Generate effect parameters from model
            params = self.model.generate(
                batch['track_a_features'],
                batch['track_b_features'],
                n_frames=self.config.model.n_frames,
                guidance_scale=1.0,  # No guidance during training
                n_steps=25,  # Faster sampling for training
            )
            
            # Scale params from [0,1] to proper ranges
            # Most params are 0-1, EQ params are 0-2
            # The mixer handles this internally based on param index
            
            # Apply mixer to generate audio
            mixed_audio, _ = self.mixer(stems_a, stems_b, params)
            
            # Compute losses
            audio_losses = self.audio_loss(mixed_audio, target_audio)
            smooth_loss = self.smoothness_loss(params)
            
            loss = (
                audio_losses['total'] +
                self.config.training.smoothness_loss_weight * smooth_loss
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.ema.update()
        
        return {
            'loss': loss.item(),
            'audio_loss': audio_losses['total'].item(),
            'waveform_mse': audio_losses['waveform_mse'].item(),
            'spectral': audio_losses['spectral'].item(),
            'smoothness': smooth_loss.item(),
        }
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_{self.global_step}.pt"
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training from step {self.global_step}")
        logger.info(f"Training for {self.config.training.max_steps} steps")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        
        train_iter = iter(self.train_loader)
        pbar = tqdm(
            total=self.config.training.max_steps,
            initial=self.global_step,
            desc="Training",
        )
        
        running_loss = 0
        log_count = 0
        
        while self.global_step < self.config.training.max_steps:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            losses = self.train_step(batch)
            self.global_step += 1
            
            running_loss += losses['loss']
            log_count += 1
            
            # Logging
            if self.global_step % self.config.training.log_interval == 0:
                avg_loss = running_loss / log_count
                lr = self.scheduler.get_last_lr()[0]
                
                log_str = (
                    f"Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )
                logger.info(log_str)
                
                # Tensorboard logging
                if self.use_tensorboard:
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/audio_loss', losses['audio_loss'], self.global_step)
                    self.writer.add_scalar('train/smoothness', losses['smoothness'], self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
                
                # Wandb logging
                if self.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/audio_loss': losses['audio_loss'],
                        'train/smoothness': losses['smoothness'],
                        'train/lr': lr,
                        'step': self.global_step,
                    })
                
                # Track best model
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint(is_best=True)
                
                running_loss = 0
                log_count = 0
                
                pbar.set_postfix({'loss': f'{losses["loss"]:.4f}', 'lr': f'{lr:.2e}'})
            
            # Checkpointing
            if self.global_step % self.config.training.save_interval == 0:
                self.save_checkpoint()
            
            pbar.update(1)
        
        # Final checkpoint
        self.save_checkpoint()
        pbar.close()
        
        if self.use_tensorboard:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Training complete!")


def run_training(
    data_dir: str,
    output_dir: str = "./output",
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    resume_from: Optional[str] = None,
    use_wandb: bool = False,
    synthetic_data: bool = False,
):
    """
    Run training from Python (called by CLI).
    
    Args:
        data_dir: Path to processed data directory
        output_dir: Output directory for checkpoints and logs
        max_steps: Maximum training steps
        batch_size: Batch size (default: from config)
        learning_rate: Learning rate (default: from config)
        resume_from: Path to checkpoint to resume from
        use_wandb: Enable wandb logging
        synthetic_data: Use synthetic transition dataset
    """
    # Load config
    config = get_config()
    
    # Override config with arguments
    if batch_size:
        config.training.batch_size = batch_size
    if learning_rate:
        config.training.learning_rate = learning_rate
    if max_steps:
        config.training.max_steps = max_steps
    
    logger.info("Configuration:")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Effect params: {config.effects.n_params}")
    logger.info(f"  Model frames: {config.model.n_frames}")
    
    # Create trainer
    trainer = DJTransitionTrainer(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        resume_from=resume_from,
        use_wandb=use_wandb,
        use_tensorboard=True,
        synthetic_data=synthetic_data,
    )
    
    # Train
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train DJTransGAN v2 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic transition dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max training steps (default: from config)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable tensorboard logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override device selection",
    )
    
    args = parser.parse_args()
    
    run_training(
        data_dir=args.data,
        output_dir=args.output,
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_from=args.resume,
        use_wandb=args.wandb,
        synthetic_data=args.synthetic,
    )


if __name__ == "__main__":
    main()
