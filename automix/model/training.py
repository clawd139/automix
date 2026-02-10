"""
Training Loop for DJ Transition Model

Implements:
- Training loop with gradient accumulation
- Mixed precision training (AMP)
- EMA (Exponential Moving Average) of model weights
- Multi-loss: diffusion loss + perceptual loss + smoothness loss
- Logging and checkpointing
"""

import os
import math
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

from .config import Config, TrainingConfig
from .model import DJTransitionModel, create_model
from .conditioning import TrackFeatures, create_dummy_features

logger = logging.getLogger(__name__)


# ==================== Loss Functions ====================

class SmoothnessLoss(nn.Module):
    """
    Penalizes rapid changes in parameter curves.
    Encourages smooth, natural-looking transitions.
    """
    
    def __init__(self, order: int = 1):
        super().__init__()
        self.order = order
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: [batch, n_frames, n_params]
        Returns:
            scalar loss
        """
        # First-order difference (velocity)
        diff1 = params[:, 1:, :] - params[:, :-1, :]
        loss = (diff1 ** 2).mean()
        
        if self.order >= 2:
            # Second-order difference (acceleration)
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
            loss = loss + 0.5 * (diff2 ** 2).mean()
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Compares mixed audio spectrograms.
    Requires the differentiable mixer to be available.
    
    This loss ensures the generated parameters produce audio
    that sounds similar to real DJ transitions.
    """
    
    def __init__(self, mixer: Optional[nn.Module] = None):
        super().__init__()
        self.mixer = mixer
        
        # Mel spectrogram for comparison
        # Will be initialized when first called if needed
        self.mel_spec = None
        
    def forward(
        self,
        pred_params: torch.Tensor,
        target_audio: torch.Tensor,
        track_a_stems: torch.Tensor,
        track_b_stems: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_params: [batch, n_frames, n_params] predicted parameters
            target_audio: [batch, n_samples] target mixed audio
            track_a_stems: [batch, n_stems, n_samples] stems from track A
            track_b_stems: [batch, n_stems, n_samples] stems from track B
            
        Returns:
            scalar loss
        """
        if self.mixer is None:
            return torch.tensor(0.0, device=pred_params.device)
        
        # Apply mixer with predicted parameters
        # The mixer takes stems and parameters, outputs mixed audio
        pred_audio = self.mixer(track_a_stems, track_b_stems, pred_params)
        
        # Compare spectrograms
        if self.mel_spec is None:
            import torchaudio
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=44100,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
            ).to(pred_audio.device)
        
        pred_spec = self.mel_spec(pred_audio)
        target_spec = self.mel_spec(target_audio)
        
        # Log-scale comparison
        pred_spec = torch.log1p(pred_spec)
        target_spec = torch.log1p(target_spec)
        
        return F.l1_loss(pred_spec, target_spec)


class CombinedLoss(nn.Module):
    """
    Combines multiple losses with configurable weights.
    """
    
    def __init__(self, config: TrainingConfig, mixer: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.smoothness_loss = SmoothnessLoss(order=2)
        self.perceptual_loss = PerceptualLoss(mixer)
        
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        pred_params: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        track_a_stems: Optional[torch.Tensor] = None,
        track_b_stems: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            model_output: Output from model.forward() with 'loss' key
            pred_params: Predicted parameters (for smoothness loss)
            target_audio: Target audio (for perceptual loss)
            track_a_stems, track_b_stems: Stems (for perceptual loss)
            
        Returns:
            dict with 'loss', 'diffusion_loss', 'smoothness_loss', 'perceptual_loss'
        """
        losses = {}
        
        # Diffusion loss (from model)
        diffusion_loss = model_output['loss']
        losses['diffusion_loss'] = diffusion_loss
        total_loss = self.config.diffusion_loss_weight * diffusion_loss
        
        # Smoothness loss
        if pred_params is not None:
            smoothness = self.smoothness_loss(pred_params)
            losses['smoothness_loss'] = smoothness
            total_loss = total_loss + self.config.smoothness_loss_weight * smoothness
        
        # Perceptual loss
        if target_audio is not None and track_a_stems is not None:
            perceptual = self.perceptual_loss(
                pred_params, target_audio, track_a_stems, track_b_stems
            )
            losses['perceptual_loss'] = perceptual
            total_loss = total_loss + self.config.perceptual_loss_weight * perceptual
        
        losses['loss'] = total_loss
        return losses


# ==================== EMA ====================

class EMA:
    """
    Exponential Moving Average of model weights.
    Improves generation quality at inference time.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ==================== Dataset ====================

class DJTransitionDataset(Dataset):
    """
    Dataset for DJ transitions.
    
    Expected data format:
    - Each item is a dict with:
        - 'track_a': TrackFeatures
        - 'track_b': TrackFeatures  
        - 'params': [n_frames, n_params] target parameter curves
        - Optionally 'audio': mixed audio for perceptual loss
    """
    
    def __init__(self, data_path: str, config: Config):
        self.config = config
        self.data_path = Path(data_path)
        
        # Load index of samples
        index_path = self.data_path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                self.samples = json.load(f)
        else:
            # Scan directory for .pt files
            self.samples = [str(p) for p in self.data_path.glob("*.pt")]
        
        logger.info(f"Loaded dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        if not sample_path.startswith('/'):
            sample_path = self.data_path / sample_path
        
        data = torch.load(sample_path)
        return data


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, size: int, config: Config):
        self.size = size
        self.config = config
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        track_a, track_b = create_dummy_features(1, self.config.model.n_frames, self.config)
        
        # Squeeze batch dimension
        for field in ['stem_mels', 'structure', 'beats', 'downbeats', 'bpm', 'key', 'energy']:
            setattr(track_a, field, getattr(track_a, field).squeeze(0))
            setattr(track_b, field, getattr(track_b, field).squeeze(0))
        
        # Random target parameters
        params = torch.rand(self.config.model.n_frames, self.config.effects.n_params)
        
        return {
            'track_a': track_a,
            'track_b': track_b,
            'params': params,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    # Stack track features
    def stack_features(features_list):
        return TrackFeatures(
            stem_mels=torch.stack([f.stem_mels for f in features_list]),
            structure=torch.stack([f.structure for f in features_list]),
            beats=torch.stack([f.beats for f in features_list]),
            downbeats=torch.stack([f.downbeats for f in features_list]),
            bpm=torch.stack([f.bpm for f in features_list]),
            key=torch.stack([f.key for f in features_list]),
            energy=torch.stack([f.energy for f in features_list]),
        )
    
    return {
        'track_a': stack_features([b['track_a'] for b in batch]),
        'track_b': stack_features([b['track_b'] for b in batch]),
        'params': torch.stack([b['params'] for b in batch]),
        'audio': torch.stack([b['audio'] for b in batch]) if 'audio' in batch[0] else None,
    }


# ==================== Trainer ====================

class Trainer:
    """
    Training loop for the DJ transition model.
    """
    
    def __init__(
        self,
        model: DJTransitionModel,
        config: Config,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        mixer: Optional[nn.Module] = None,
        output_dir: str = "./output",
    ):
        self.model = model
        self.config = config
        self.train_config = config.training
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.device = config.device
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        
        # Learning rate scheduler (cosine with warmup)
        def lr_lambda(step):
            if step < self.train_config.warmup_steps:
                return step / self.train_config.warmup_steps
            progress = (step - self.train_config.warmup_steps) / (
                self.train_config.max_steps - self.train_config.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss function
        self.loss_fn = CombinedLoss(self.train_config, mixer)
        
        # EMA
        self.ema = EMA(model, decay=self.train_config.ema_decay)
        
        # Mixed precision
        self.scaler = GradScaler() if self.train_config.use_amp and self.device == "cuda" else None
        
        # Logging
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def move_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
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
        
        result = {
            'track_a': move_features(batch['track_a']),
            'track_b': move_features(batch['track_b']),
            'params': batch['params'].to(self.device),
        }
        if batch.get('audio') is not None:
            result['audio'] = batch['audio'].to(self.device)
        return result
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        batch = self.move_to_device(batch)
        
        # Forward pass
        with autocast(enabled=self.scaler is not None):
            # Convert params to model space (logit of [0,1])
            target_params = batch['params']
            target_params_logit = torch.logit(target_params.clamp(1e-6, 1-1e-6))
            
            model_output = self.model(
                batch['track_a'],
                batch['track_b'],
                target_params_logit,
                drop_condition_prob=self.train_config.condition_dropout,
            )
            
            # Compute combined loss
            losses = self.loss_fn(model_output, pred_params=target_params)
            loss = losses['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            self.optimizer.step()
        
        self.scheduler.step()
        self.ema.update()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.ema.apply_shadow()
        self.model.eval()
        
        total_losses = {}
        n_batches = 0
        
        for batch in self.val_loader:
            batch = self.move_to_device(batch)
            target_params = batch['params']
            target_params_logit = torch.logit(target_params.clamp(1e-6, 1-1e-6))
            
            model_output = self.model(
                batch['track_a'],
                batch['track_b'],
                target_params_logit,
            )
            
            losses = self.loss_fn(model_output, pred_params=target_params)
            
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            n_batches += 1
        
        self.ema.restore()
        
        return {f'val_{k}': v / n_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_{self.global_step}.pt"
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
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
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting training from step {self.global_step}")
        logger.info(f"Training for {self.train_config.max_steps} steps")
        logger.info(f"Device: {self.device}")
        
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.train_config.max_steps, initial=self.global_step)
        
        while self.global_step < self.train_config.max_steps:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            losses = self.train_step(batch)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.train_config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                log_str = f"Step {self.global_step} | LR {lr:.2e}"
                for k, v in losses.items():
                    log_str += f" | {k}: {v:.4f}"
                logger.info(log_str)
                pbar.set_postfix({'loss': losses['loss'], 'lr': lr})
            
            # Validation
            if self.global_step % self.train_config.eval_interval == 0:
                val_losses = self.validate()
                if val_losses:
                    log_str = f"Validation at step {self.global_step}"
                    for k, v in val_losses.items():
                        log_str += f" | {k}: {v:.4f}"
                    logger.info(log_str)
                    
                    # Track best model
                    if val_losses.get('val_loss', float('inf')) < self.best_val_loss:
                        self.best_val_loss = val_losses['val_loss']
                        self.save_checkpoint(is_best=True)
            
            # Checkpointing
            if self.global_step % self.train_config.save_interval == 0:
                self.save_checkpoint()
            
            pbar.update(1)
        
        # Final checkpoint
        self.save_checkpoint()
        pbar.close()
        logger.info("Training complete!")


def train(
    config: Optional[Config] = None,
    data_path: Optional[str] = None,
    output_dir: str = "./output",
    resume_from: Optional[str] = None,
):
    """
    Main training entry point.
    
    Args:
        config: Model configuration
        data_path: Path to training data (uses dummy data if None)
        output_dir: Output directory for checkpoints
        resume_from: Path to checkpoint to resume from
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    if config is None:
        config = Config()
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset
    if data_path is not None:
        train_dataset = DJTransitionDataset(data_path, config)
        val_dataset = None  # TODO: split or separate val set
    else:
        logger.warning("No data path provided, using dummy dataset for testing")
        train_dataset = DummyDataset(1000, config)
        val_dataset = DummyDataset(100, config)
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
    )
    
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DJ transition model")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, help="Max training steps")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    config = Config()
    if args.steps:
        config.training.max_steps = args.steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    train(
        config=config,
        data_path=args.data,
        output_dir=args.output,
        resume_from=args.resume,
    )
