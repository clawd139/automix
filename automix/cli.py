#!/usr/bin/env python3
"""
AutoMix CLI - One command to prepare, train, and mix.

Usage:
    automix prepare --tracks ~/Music/DJ_Library --output processed --max 100
    automix train --data processed --output runs/run1 --steps 100000
    automix mix track_a.mp3 track_b.mp3 -o transition.wav
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import click
import torch


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


@click.group()
@click.version_option(version="2.0.0", prog_name="automix")
def main():
    """AutoMix - Neural DJ Transition Generation
    
    Generate DJ-style transitions between tracks using AI.
    """
    pass


@main.command()
@click.option("--tracks", "-t", type=click.Path(exists=True), required=True,
              help="Path to track library directory")
@click.option("--output", "-o", type=click.Path(), default="processed",
              help="Output directory for processed data")
@click.option("--max", "max_pairs", type=int, default=None,
              help="Maximum number of track pairs to process")
@click.option("--model", "demucs_model", type=str, default="htdemucs",
              help="Demucs model (htdemucs, htdemucs_ft, mdx_extra)")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto",
              help="Device for processing")
def prepare(tracks: str, output: str, max_pairs: Optional[int], demucs_model: str, device: str):
    """Prepare training data from a track library.
    
    Separates stems using demucs and creates synthetic transition pairs.
    
    Example:
        automix prepare --tracks ~/Music/DJ_Library --output processed --max 100
    """
    from .pipeline import process_track_library
    
    tracks_path = Path(tracks)
    output_path = Path(output)
    
    click.echo(f"📂 Input: {tracks_path}")
    click.echo(f"📁 Output: {output_path}")
    
    if device == "auto":
        device = get_device()
    click.echo(f"🖥️  Device: {device}")
    
    # Run pipeline
    process_track_library(
        tracks_dir=tracks_path,
        output_dir=output_path,
        demucs_model=demucs_model,
        device=device,
        max_pairs=max_pairs,
    )
    
    click.echo("✅ Data preparation complete!")


@main.command()
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Path to processed data directory")
@click.option("--output", "-o", type=click.Path(), default="runs/run1",
              help="Output directory for checkpoints and logs")
@click.option("--steps", type=int, default=100000,
              help="Maximum training steps")
@click.option("--batch-size", type=int, default=None,
              help="Batch size (default: auto based on device)")
@click.option("--lr", type=float, default=None,
              help="Learning rate")
@click.option("--resume", type=click.Path(exists=True), default=None,
              help="Resume from checkpoint")
@click.option("--gpus", type=int, default=None,
              help="Number of GPUs to use (auto-detected if not specified)")
@click.option("--wandb", is_flag=True, help="Enable Weights & Biases logging")
@click.option("--synthetic", is_flag=True, help="Use synthetic transition dataset")
def train(data: str, output: str, steps: int, batch_size: Optional[int],
          lr: Optional[float], resume: Optional[str], gpus: Optional[int],
          wandb: bool, synthetic: bool):
    """Train the transition model.
    
    Auto-detects device and launches multi-GPU training with DDP if available.
    
    Examples:
        # Single GPU/MPS training
        automix train --data processed --output runs/run1 --steps 100000
        
        # Multi-GPU training (auto-detected)
        automix train --data processed --output runs/run1 --steps 100000
        
        # Specify GPU count
        automix train --data processed --output runs/run1 --gpus 4
    """
    from .train import run_training
    
    device = get_device()
    n_gpus = get_gpu_count()
    
    # Determine training mode
    if gpus is not None:
        n_gpus = min(gpus, n_gpus)
    
    click.echo(f"📂 Data: {data}")
    click.echo(f"📁 Output: {output}")
    click.echo(f"🖥️  Device: {device}")
    click.echo(f"🎯 Steps: {steps}")
    
    if device == "cuda" and n_gpus > 1:
        click.echo(f"🚀 Multi-GPU: {n_gpus} GPUs (DDP)")
        # Launch with torchrun for DDP
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            "--master_port", str(29500 + os.getpid() % 1000),
            "-m", "automix.train",
            "--data", data,
            "--output", output,
            "--steps", str(steps),
        ]
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])
        if lr:
            cmd.extend(["--lr", str(lr)])
        if resume:
            cmd.extend(["--resume", resume])
        if wandb:
            cmd.append("--wandb")
        if synthetic:
            cmd.append("--synthetic")
        
        subprocess.run(cmd)
    else:
        if device == "cuda":
            click.echo(f"🖥️  Single GPU training")
        elif device == "mps":
            click.echo(f"🍎 Apple Silicon (MPS) training")
        else:
            click.echo(f"⚠️  CPU training (this will be slow)")
        
        # Single device training
        run_training(
            data_dir=data,
            output_dir=output,
            max_steps=steps,
            batch_size=batch_size,
            learning_rate=lr,
            resume_from=resume,
            use_wandb=wandb,
            synthetic_data=synthetic,
        )
    
    click.echo("✅ Training complete!")


@main.command()
@click.argument("track_a", type=click.Path(exists=True))
@click.argument("track_b", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="output/transition.wav",
              help="Output path for mixed audio")
@click.option("--model", "-m", type=click.Path(exists=True), default=None,
              help="Path to trained model checkpoint")
@click.option("--duration", type=float, default=30.0,
              help="Transition duration in seconds")
@click.option("--guidance", type=float, default=2.0,
              help="Classifier-free guidance scale")
@click.option("--steps", type=int, default=50,
              help="Number of diffusion steps")
@click.option("--visualize", is_flag=True, help="Generate parameter visualization")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto",
              help="Device for inference")
def mix(track_a: str, track_b: str, output: str, model: Optional[str],
        duration: float, guidance: float, steps: int, visualize: bool, device: str):
    """Generate a DJ transition between two tracks.
    
    Uses demucs for stem separation and the trained model to predict
    optimal effect parameters for the transition.
    
    Examples:
        # With a trained model
        automix mix track_a.mp3 track_b.mp3 --model runs/run1/best.pt -o transition.wav
        
        # Without a model (random parameters - for testing)
        automix mix track_a.mp3 track_b.mp3 -o test.wav
    """
    from .inference import run_inference
    
    if device == "auto":
        device = get_device()
    
    click.echo(f"🎵 Track A: {track_a}")
    click.echo(f"🎵 Track B: {track_b}")
    click.echo(f"📁 Output: {output}")
    click.echo(f"🖥️  Device: {device}")
    click.echo(f"⏱️  Duration: {duration}s")
    
    if model:
        click.echo(f"🧠 Model: {model}")
    else:
        click.echo("⚠️  No model specified - using random parameters")
    
    # Run inference
    result = run_inference(
        track_a=track_a,
        track_b=track_b,
        output_path=output,
        model_path=model,
        duration=duration,
        guidance_scale=guidance,
        n_steps=steps,
        device=device,
        visualize=visualize,
    )
    
    click.echo(f"✅ Transition saved to {result['output']}")


@main.command()
def info():
    """Show system information and capabilities."""
    click.echo("=" * 50)
    click.echo("AutoMix System Information")
    click.echo("=" * 50)
    
    # Python
    click.echo(f"\n🐍 Python: {sys.version.split()[0]}")
    
    # PyTorch
    click.echo(f"🔥 PyTorch: {torch.__version__}")
    
    # Device
    device = get_device()
    click.echo(f"\n🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        click.echo(f"   CUDA GPUs: {n_gpus}")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            click.echo(f"   [{i}] {name} ({mem:.1f} GB)")
    
    if torch.backends.mps.is_available():
        click.echo("   Apple MPS: Available")
    
    # Check demucs
    click.echo("\n📦 Dependencies:")
    try:
        import demucs
        click.echo(f"   demucs: ✅ installed")
    except ImportError:
        click.echo(f"   demucs: ❌ not installed")
    
    try:
        import librosa
        click.echo(f"   librosa: ✅ installed")
    except ImportError:
        click.echo(f"   librosa: ❌ not installed")
    
    try:
        import torchaudio
        click.echo(f"   torchaudio: ✅ {torchaudio.__version__}")
    except ImportError:
        click.echo(f"   torchaudio: ❌ not installed")
    
    click.echo("\n" + "=" * 50)


if __name__ == "__main__":
    main()
