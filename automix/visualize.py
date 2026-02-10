#!/usr/bin/env python3
"""
DJTransGAN v2 Visualization Module

Visualize effect parameter curves:
- Plot 54 parameter curves over time
- Show stem-by-stem breakdown
- Generate timeline showing transition plan
- Export to PNG/SVG
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

from model.config import Config, get_config
from effects.mixer import (
    STEM_NAMES,
    STEM_PARAM_NAMES,
    GLOBAL_PARAM_NAMES,
    N_STEMS,
    N_STEM_PARAMS,
    N_GLOBAL_PARAMS,
    N_PARAMS_PER_TRACK,
    N_TOTAL_PARAMS,
)


# Color schemes
TRACK_COLORS = {
    'A': '#1f77b4',  # Blue
    'B': '#ff7f0e',  # Orange
}

STEM_COLORS = {
    'drums': '#e41a1c',   # Red
    'bass': '#377eb8',    # Blue
    'vocals': '#4daf4a',  # Green
    'other': '#984ea3',   # Purple
}

PARAM_COLORS = {
    'gain': '#2c3e50',
    'eq_low': '#e74c3c',
    'eq_mid': '#f39c12',
    'eq_high': '#3498db',
    'lpf': '#9b59b6',
    'hpf': '#1abc9c',
    'reverb_send': '#34495e',
    'delay_send': '#7f8c8d',
    'delay_feedback': '#95a5a6',
}


def parse_params(params: Union[torch.Tensor, np.ndarray, List]) -> Dict:
    """
    Parse flat parameter tensor into structured dict.
    
    Args:
        params: [n_frames, n_params] tensor/array
        
    Returns:
        Nested dict: {track: {stem: {param: values}}}
    """
    if isinstance(params, torch.Tensor):
        params = params.cpu().numpy()
    elif isinstance(params, list):
        params = np.array(params)
    
    n_frames, n_params = params.shape
    
    result = {'A': {'stems': {}, 'global': {}}, 'B': {'stems': {}, 'global': {}}}
    
    for track_idx, track in enumerate(['A', 'B']):
        base = track_idx * N_PARAMS_PER_TRACK
        
        # Parse stem parameters
        for stem_idx, stem in enumerate(STEM_NAMES):
            stem_base = base + stem_idx * N_STEM_PARAMS
            result[track]['stems'][stem] = {}
            
            for param_idx, param in enumerate(STEM_PARAM_NAMES):
                idx = stem_base + param_idx
                result[track]['stems'][stem][param] = params[:, idx]
        
        # Parse global parameters
        global_base = base + N_STEMS * N_STEM_PARAMS
        for param_idx, param in enumerate(GLOBAL_PARAM_NAMES):
            idx = global_base + param_idx
            result[track]['global'][param] = params[:, idx]
    
    return result


def plot_parameters(
    params: Union[torch.Tensor, np.ndarray, Dict],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None,
    title: str = "DJ Transition Parameters",
    duration: float = 30.0,
    figsize: Tuple[int, int] = (16, 20),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Plot all parameter curves with detailed breakdown.
    
    Args:
        params: Parameter tensor [n_frames, n_params] or parsed dict
        output_path: Path to save figure (optional)
        config: Model config (optional)
        title: Plot title
        duration: Duration in seconds (for x-axis)
        figsize: Figure size
        dpi: Resolution
        
    Returns:
        Figure object if matplotlib available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return None
    
    # Parse params if needed
    if isinstance(params, (torch.Tensor, np.ndarray, list)):
        if isinstance(params, list):
            params = np.array(params)
        elif isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        
        n_frames = params.shape[0]
        parsed = parse_params(params)
    else:
        parsed = params
        n_frames = len(next(iter(parsed['A']['stems']['drums'].values())))
    
    time = np.linspace(0, duration, n_frames)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    gs = GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.15)
    
    # Row 0: Crossfade overview (gains only)
    ax_crossfade = fig.add_subplot(gs[0, :])
    ax_crossfade.set_title("Crossfade Overview (Stem Gains)", fontsize=12)
    
    for track in ['A', 'B']:
        color = TRACK_COLORS[track]
        for stem in STEM_NAMES:
            gains = parsed[track]['stems'][stem]['gain']
            ax_crossfade.plot(time, gains, color=color, alpha=0.5,
                            label=f'{track}-{stem}' if track == 'A' else None)
    
    ax_crossfade.set_xlim(0, duration)
    ax_crossfade.set_ylim(-0.1, 1.1)
    ax_crossfade.set_xlabel("Time (s)")
    ax_crossfade.set_ylabel("Gain")
    ax_crossfade.legend(loc='upper right', ncol=4, fontsize=8)
    ax_crossfade.grid(True, alpha=0.3)
    
    # Rows 1-4: Per-stem parameters
    for stem_idx, stem in enumerate(STEM_NAMES):
        row = stem_idx + 1
        
        for col, track in enumerate(['A', 'B']):
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Track {track} - {stem.capitalize()}", fontsize=11)
            
            stem_params = parsed[track]['stems'][stem]
            
            for param in STEM_PARAM_NAMES:
                values = stem_params[param]
                color = PARAM_COLORS.get(param, '#333333')
                
                # Adjust display range for EQ (0-2 range)
                if param.startswith('eq_'):
                    ax.plot(time, values, label=param, color=color, linewidth=1.5)
                else:
                    ax.plot(time, values, label=param, color=color, linewidth=1.5)
            
            ax.set_xlim(0, duration)
            ax.set_ylim(-0.1, 2.1 if 'eq' in str(stem_params.keys()) else 1.1)
            ax.set_xlabel("Time (s)")
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
    
    # Row 5: Global effects
    ax_global = fig.add_subplot(gs[5, :])
    ax_global.set_title("Global Effects (Reverb & Delay)", fontsize=12)
    
    for track in ['A', 'B']:
        color = TRACK_COLORS[track]
        linestyle = '-' if track == 'A' else '--'
        
        global_params = parsed[track]['global']
        for param in GLOBAL_PARAM_NAMES:
            values = global_params[param]
            ax_global.plot(time, values, color=color, linestyle=linestyle,
                          label=f'{track}-{param}', linewidth=1.5)
    
    ax_global.set_xlim(0, duration)
    ax_global.set_ylim(-0.1, 1.1)
    ax_global.set_xlabel("Time (s)")
    ax_global.set_ylabel("Amount")
    ax_global.legend(loc='upper right', ncol=3, fontsize=8)
    ax_global.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    return fig


def plot_stem_swap_timeline(
    params: Union[torch.Tensor, np.ndarray, Dict],
    output_path: Optional[Union[str, Path]] = None,
    duration: float = 30.0,
    swap_threshold: float = 0.5,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Generate a timeline showing when each stem swaps from A to B.
    
    Args:
        params: Parameter tensor or parsed dict
        output_path: Path to save figure
        duration: Duration in seconds
        swap_threshold: Gain threshold to consider a stem "active"
        figsize: Figure size
        dpi: Resolution
    """
    if not HAS_MATPLOTLIB:
        return None
    
    # Parse params
    if isinstance(params, (torch.Tensor, np.ndarray, list)):
        if isinstance(params, list):
            params = np.array(params)
        elif isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        parsed = parse_params(params)
    else:
        parsed = params
    
    n_frames = len(parsed['A']['stems']['drums']['gain'])
    time = np.linspace(0, duration, n_frames)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Stem Swap Timeline", fontsize=14, fontweight='bold')
    
    y_positions = {stem: i for i, stem in enumerate(STEM_NAMES)}
    
    for stem in STEM_NAMES:
        y = y_positions[stem]
        
        gains_a = parsed['A']['stems'][stem]['gain']
        gains_b = parsed['B']['stems'][stem]['gain']
        
        # Determine which track is "active" at each frame
        for i in range(n_frames - 1):
            t_start = time[i]
            t_end = time[i + 1]
            
            g_a = gains_a[i]
            g_b = gains_b[i]
            
            # Color based on which is louder
            if g_a > g_b + 0.1:
                color = TRACK_COLORS['A']
                alpha = min(1.0, g_a)
            elif g_b > g_a + 0.1:
                color = TRACK_COLORS['B']
                alpha = min(1.0, g_b)
            else:
                # Blend
                ratio = g_b / (g_a + g_b + 1e-6)
                color = plt.cm.coolwarm(ratio)
                alpha = max(g_a, g_b)
            
            rect = mpatches.Rectangle(
                (t_start, y - 0.4),
                t_end - t_start,
                0.8,
                facecolor=color,
                alpha=alpha * 0.8,
                edgecolor='none',
            )
            ax.add_patch(rect)
    
    # Add stem labels
    ax.set_yticks(range(len(STEM_NAMES)))
    ax.set_yticklabels([s.capitalize() for s in STEM_NAMES])
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=TRACK_COLORS['A'], label='Track A'),
        mpatches.Patch(color=TRACK_COLORS['B'], label='Track B'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, len(STEM_NAMES) - 0.5)
    ax.set_xlabel("Time (s)")
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved timeline to {output_path}")
    
    return fig


def plot_comparison(
    params_list: List[Union[torch.Tensor, np.ndarray]],
    labels: List[str],
    output_path: Optional[Union[str, Path]] = None,
    duration: float = 30.0,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[plt.Figure]:
    """
    Compare multiple parameter sets side by side.
    
    Useful for comparing different model outputs or checkpoints.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    n_sets = len(params_list)
    
    fig, axes = plt.subplots(n_sets, 1, figsize=figsize, sharex=True)
    if n_sets == 1:
        axes = [axes]
    
    fig.suptitle("Parameter Comparison", fontsize=14, fontweight='bold')
    
    for idx, (params, label) in enumerate(zip(params_list, labels)):
        if isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        elif isinstance(params, list):
            params = np.array(params)
        
        parsed = parse_params(params)
        n_frames = params.shape[0]
        time = np.linspace(0, duration, n_frames)
        
        ax = axes[idx]
        ax.set_title(label, fontsize=11)
        
        # Plot gains for all stems
        for track in ['A', 'B']:
            color = TRACK_COLORS[track]
            for stem in STEM_NAMES:
                gains = parsed[track]['stems'][stem]['gain']
                ax.plot(time, gains, color=color, alpha=0.6)
        
        ax.set_xlim(0, duration)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Gain")
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def export_params_summary(
    params: Union[torch.Tensor, np.ndarray, Dict],
    output_path: Union[str, Path],
    duration: float = 30.0,
):
    """
    Export a text summary of the parameter curves.
    """
    if isinstance(params, (torch.Tensor, np.ndarray, list)):
        if isinstance(params, list):
            params = np.array(params)
        elif isinstance(params, torch.Tensor):
            params = params.cpu().numpy()
        parsed = parse_params(params)
    else:
        parsed = params
    
    lines = ["# DJ Transition Parameter Summary\n"]
    lines.append(f"Duration: {duration}s\n\n")
    
    for track in ['A', 'B']:
        lines.append(f"## Track {track}\n\n")
        
        lines.append("### Stems\n")
        for stem in STEM_NAMES:
            gains = parsed[track]['stems'][stem]['gain']
            start_gain = gains[0]
            end_gain = gains[-1]
            max_gain = np.max(gains)
            
            lines.append(f"- **{stem}**: {start_gain:.2f} → {end_gain:.2f} (max: {max_gain:.2f})\n")
        
        lines.append("\n### Global Effects\n")
        for param in GLOBAL_PARAM_NAMES:
            values = parsed[track]['global'][param]
            mean_val = np.mean(values)
            max_val = np.max(values)
            lines.append(f"- {param}: mean={mean_val:.2f}, max={max_val:.2f}\n")
        
        lines.append("\n")
    
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DJ transition parameters",
    )
    
    parser.add_argument(
        "params",
        type=str,
        help="Path to parameters JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="params_viz.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Generate stem swap timeline instead",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Export text summary to file",
    )
    
    args = parser.parse_args()
    
    # Load params
    params_path = Path(args.params)
    
    if params_path.suffix == '.json':
        with open(params_path) as f:
            data = json.load(f)
        
        if 'params' in data:
            params = np.array(data['params'])
            duration = data.get('duration', args.duration)
        else:
            params = np.array(data)
            duration = args.duration
    elif params_path.suffix == '.pt':
        params = torch.load(params_path)
        if isinstance(params, dict):
            params = params.get('params', params)
        duration = args.duration
    else:
        print(f"Unknown file format: {params_path.suffix}")
        sys.exit(1)
    
    # Generate visualization
    if args.timeline:
        plot_stem_swap_timeline(
            params,
            output_path=args.output,
            duration=duration,
        )
    else:
        plot_parameters(
            params,
            output_path=args.output,
            duration=duration,
        )
    
    # Export summary
    if args.summary:
        export_params_summary(params, args.summary, duration)
    
    if HAS_MATPLOTLIB:
        plt.show()


if __name__ == "__main__":
    main()
