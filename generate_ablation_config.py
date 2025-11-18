#!/usr/bin/env python3
"""
Automated YAML configuration generator for abliteration based on analysis results.

This script analyzes the refusal measurements and automatically generates an optimized
YAML configuration file for ablation, selecting the best layers based on signal quality.

Usage:
    python generate_ablation_config.py -m measurements.refuse -o config.yml --model-path /path/to/model
    
    # With custom thresholds
    python generate_ablation_config.py -m measurements.refuse -o config.yml --model-path /path/to/model \
        --min-quality 0.015 --top-n 10
    
    # Conservative mode (fewer layers)
    python generate_ablation_config.py -m measurements.refuse -o config.yml --model-path /path/to/model \
        --conservative
    
    # Aggressive mode (more layers)
    python generate_ablation_config.py -m measurements.refuse -o config.yml --model-path /path/to/model \
        --aggressive
"""

import argparse
import torch
import yaml
from pathlib import Path
from typing import List, Tuple, Dict


def compute_signal_quality(
    harmful_mean: torch.Tensor,
    harmless_mean: torch.Tensor,
    refusal_dir: torch.Tensor,
) -> Tuple[float, float, float, float]:
    """
    Compute signal quality metrics for a layer.
    
    Returns:
        Tuple of (cosine_similarity, signal_to_noise, signal_quality, refusal_norm)
    """
    # Compute norms
    harmful_norm = torch.norm(harmful_mean).item()
    harmless_norm = torch.norm(harmless_mean).item()
    refusal_norm = torch.norm(refusal_dir).item()
    
    # Cosine similarity between harmful and harmless
    if harmful_norm > 0 and harmless_norm > 0:
        cosine_sim = (harmful_mean @ harmless_mean) / (harmful_norm * harmless_norm)
        cosine_sim = cosine_sim.item()
    else:
        cosine_sim = 1.0
    
    # Signal-to-noise ratio
    avg_norm = (harmful_norm + harmless_norm) / 2
    if avg_norm > 0:
        snr = refusal_norm / avg_norm
    else:
        snr = 0.0
    
    # Signal quality: combines low cosine similarity with high SNR
    signal_quality = (1 - cosine_sim) * snr
    
    return cosine_sim, snr, signal_quality, refusal_norm


def analyze_measurements(measurements_path: str) -> List[Dict]:
    """
    Analyze measurements and return layer statistics.
    
    Returns:
        List of dicts with layer statistics, sorted by signal quality
    """
    print(f"Loading measurements from {measurements_path}...")
    measures = torch.load(measurements_path)
    
    num_layers = measures.get('layers', 0)
    if num_layers == 0:
        raise ValueError("No layer information found in measurements")
    
    print(f"Analyzing {num_layers} layers...")
    
    layer_stats = []
    for layer in range(num_layers):
        harmful_key = f'harmful_{layer}'
        harmless_key = f'harmless_{layer}'
        refuse_key = f'refuse_{layer}'
        
        if harmful_key not in measures or harmless_key not in measures or refuse_key not in measures:
            continue
        
        harmful_mean = measures[harmful_key]
        harmless_mean = measures[harmless_key]
        refusal_dir = measures[refuse_key]
        
        cosine_sim, snr, signal_quality, refusal_norm = compute_signal_quality(
            harmful_mean, harmless_mean, refusal_dir
        )
        
        layer_stats.append({
            'layer': layer,
            'cosine_similarity': cosine_sim,
            'snr': snr,
            'signal_quality': signal_quality,
            'refusal_norm': refusal_norm,
        })
    
    # Sort by signal quality (descending)
    layer_stats.sort(key=lambda x: x['signal_quality'], reverse=True)
    
    return layer_stats


def select_layers(
    layer_stats: List[Dict],
    mode: str = 'balanced',
    min_quality: float = 0.015,
    top_n: int = None,
) -> Tuple[List[int], int]:
    """
    Select layers for ablation based on signal quality.
    
    Args:
        layer_stats: List of layer statistics
        mode: 'conservative', 'balanced', or 'aggressive'
        min_quality: Minimum signal quality threshold
        top_n: Maximum number of layers to select (None = use mode defaults)
    
    Returns:
        Tuple of (selected_layer_indices, best_measurement_layer)
    """
    # Mode-specific defaults
    mode_configs = {
        'conservative': {'min_quality': 0.025, 'top_n': 8},
        'balanced': {'min_quality': 0.015, 'top_n': 12},
        'aggressive': {'min_quality': 0.010, 'top_n': 20},
    }
    
    if mode in mode_configs:
        config = mode_configs[mode]
        if min_quality is None:
            min_quality = config['min_quality']
        if top_n is None:
            top_n = config['top_n']
    
    # Find best measurement layer (highest signal quality)
    best_layer = layer_stats[0]['layer']
    
    # Select layers above quality threshold
    selected = []
    for stat in layer_stats:
        if stat['signal_quality'] >= min_quality:
            selected.append(stat['layer'])
            if top_n and len(selected) >= top_n:
                break
    
    # Sort selected layers by index for cleaner YAML
    selected.sort()
    
    return selected, best_layer


def generate_yaml_config(
    model_path: str,
    measurements_path: str,
    output_path: str,
    selected_layers: List[int],
    best_measurement_layer: int,
    layer_stats: List[Dict],
    scale: float = 1.0,
    sparsity: float = 0.0,
    scale_decay: bool = True,
) -> None:
    """
    Generate YAML configuration file for ablation.
    
    Args:
        model_path: Path to the model
        measurements_path: Path to measurements file
        output_path: Path for abliterated model output
        selected_layers: List of layer indices to ablate
        best_measurement_layer: Layer to use as measurement source
        layer_stats: Layer statistics for reference
        scale: Base ablation scale
        sparsity: Sparsity fraction
        scale_decay: Whether to reduce scale for lower-quality layers
    """
    # Create layer quality lookup
    quality_map = {stat['layer']: stat['signal_quality'] for stat in layer_stats}
    best_quality = quality_map[best_measurement_layer]
    
    # Build ablation entries
    ablate_entries = []
    for layer in selected_layers:
        layer_quality = quality_map.get(layer, 0)
        
        # Optionally reduce scale for lower quality layers
        if scale_decay and layer_quality < best_quality * 0.7:
            layer_scale = scale * 0.8
        elif scale_decay and layer_quality < best_quality * 0.5:
            layer_scale = scale * 0.6
        else:
            layer_scale = scale
        
        ablate_entries.append({
            'layer': int(layer),
            'measurement': int(best_measurement_layer),
            'scale': float(layer_scale),
            'sparsity': float(sparsity),
        })
    
    # Create YAML structure
    config = {
        'model': model_path,
        'measurements': measurements_path,
        'output': output_path,
        'ablate': ablate_entries,
    }
    
    # Write YAML file
    with open(measurements_path.replace('.refuse', '_config.yml'), 'w') as f:
        # Write header comment
        f.write(f"# Auto-generated ablation configuration\n")
        f.write(f"# Generated from: {measurements_path}\n")
        f.write(f"# Best measurement layer: {best_measurement_layer} (quality: {best_quality:.4f})\n")
        f.write(f"# Selected {len(selected_layers)} layers for ablation\n")
        f.write(f"# Layer range: {min(selected_layers)}-{max(selected_layers)}\n")
        f.write(f"#\n")
        f.write(f"# Top 5 layers by signal quality:\n")
        for i, stat in enumerate(layer_stats[:5]):
            f.write(f"#   {i+1}. Layer {stat['layer']}: quality={stat['signal_quality']:.4f}, "
                   f"snr={stat['snr']:.4f}, cosine={stat['cosine_similarity']:.4f}\n")
        f.write(f"\n")
        
        # Write YAML content
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nConfiguration saved to: {measurements_path.replace('.refuse', '_config.yml')}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized ablation YAML configuration from measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--measurements', '-m',
        type=str,
        required=True,
        help='Path to measurements file (e.g., model.refuse)'
    )
    
    parser.add_argument(
        '--model-path', '-p',
        type=str,
        required=True,
        help='Path to the model (local path or HuggingFace ID)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for abliterated model (default: model-path + "-abliterated")'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['conservative', 'balanced', 'aggressive'],
        default='balanced',
        help='Ablation mode: conservative (fewer layers), balanced, or aggressive (more layers)'
    )
    
    parser.add_argument(
        '--conservative',
        action='store_true',
        help='Shortcut for --mode conservative'
    )
    
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Shortcut for --mode aggressive'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        default=None,
        help='Minimum signal quality threshold (default: mode-dependent)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Maximum number of layers to select (default: mode-dependent)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Base ablation scale factor (default: 1.0)'
    )
    
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.0,
        help='Sparsity fraction (0.0-1.0, default: 0.0)'
    )
    
    parser.add_argument(
        '--no-scale-decay',
        action='store_true',
        help='Disable scale reduction for lower-quality layers'
    )
    
    parser.add_argument(
        '--show-analysis',
        action='store_true',
        help='Show detailed analysis of all layers'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    mode = args.mode
    if args.conservative:
        mode = 'conservative'
    elif args.aggressive:
        mode = 'aggressive'
    
    # Set output path
    output_path = args.output
    if output_path is None:
        model_name = Path(args.model_path).name
        output_path = f"{model_name}-abliterated"
    
    # Analyze measurements
    layer_stats = analyze_measurements(args.measurements)
    
    # Show analysis if requested
    if args.show_analysis:
        print("\n" + "="*70)
        print("LAYER ANALYSIS (sorted by signal quality)")
        print("="*70)
        for stat in layer_stats:
            print(f"Layer {stat['layer']:2d}: "
                  f"quality={stat['signal_quality']:.4f}, "
                  f"snr={stat['snr']:.4f}, "
                  f"cosine={stat['cosine_similarity']:.4f}, "
                  f"refusal_norm={stat['refusal_norm']:.2f}")
        print("="*70 + "\n")
    
    # Select layers
    selected_layers, best_layer = select_layers(
        layer_stats,
        mode=mode,
        min_quality=args.min_quality,
        top_n=args.top_n,
    )
    
    print(f"\nMode: {mode}")
    print(f"Selected {len(selected_layers)} layers: {selected_layers}")
    print(f"Best measurement layer: {best_layer}")
    print(f"Quality range: {layer_stats[-1]['signal_quality']:.4f} - {layer_stats[0]['signal_quality']:.4f}")
    
    # Generate YAML
    generate_yaml_config(
        model_path=args.model_path,
        measurements_path=args.measurements,
        output_path=output_path,
        selected_layers=selected_layers,
        best_measurement_layer=best_layer,
        layer_stats=layer_stats,
        scale=args.scale,
        sparsity=args.sparsity,
        scale_decay=not args.no_scale_decay,
    )
    
    print(f"\nNext step: Run ablation with:")
    print(f"  python sharded_ablate.py {args.measurements.replace('.refuse', '_config.yml')}")


if __name__ == "__main__":
    main()