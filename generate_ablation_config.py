#!/usr/bin/env python3
"""
Automated YAML configuration generator for abliteration based on analysis results.

This script analyzes the refusal measurements and automatically generates an optimized
YAML configuration file for ablation, selecting the best layers based on signal quality.

Automatically detects MoE models and applies appropriate settings:
- MoE models: More layers, higher scale (2.0 core, 1.5 extended)
- Dense models: Fewer layers, standard scale (1.0)

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
from transformers import AutoConfig


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


def is_moe_model(model_path: str) -> bool:
    """
    Check if the model is a Mixture of Experts (MoE) model.
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Check for common MoE indicators in config
        if hasattr(config, 'num_experts') and config.num_experts > 1:
            return True
        if hasattr(config, 'n_routed_experts') and config.n_routed_experts > 1:
            return True
        if hasattr(config, 'moe'):
            return True
        # Check architecture name
        arch = getattr(config, 'architectures', [])
        if arch and any('MoE' in a or 'Mixtral' in a for a in arch):
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not load config to check for MoE: {e}")
        return False


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


def detect_signal_peaks(layer_stats: List[Dict]) -> List[List[int]]:
    """
    Detect clusters of high-quality layers (peaks) in the signal quality data.
    
    Returns:
        List of clusters, where each cluster is a list of layer indices
    """
    if not layer_stats:
        return []
    
    # Sort by layer index for peak detection
    sorted_by_layer = sorted(layer_stats, key=lambda x: x['layer'])
    
    # Calculate quality threshold based on distribution
    qualities = [stat['signal_quality'] for stat in sorted_by_layer]
    max_quality = max(qualities)
    
    # Dynamic threshold: 15% of max quality or 0.005, whichever is higher
    quality_threshold = max(max_quality * 0.15, 0.005)
    
    clusters = []
    current_cluster = []
    
    for stat in sorted_by_layer:
        if stat['signal_quality'] >= quality_threshold:
            if not current_cluster:
                current_cluster = [stat['layer']]
            else:
                # Check if this layer is adjacent to the current cluster
                if stat['layer'] == current_cluster[-1] + 1:
                    current_cluster.append(stat['layer'])
                else:
                    # Start a new cluster
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [stat['layer']]
        else:
            # End current cluster if it exists
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
    
    # Add the last cluster if it exists
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def select_layers_optimized(
    layer_stats: List[Dict],
    mode: str = 'balanced',
    min_quality: float = None,
    top_n: int = None,
    is_moe: bool = False,
) -> Tuple[List[int], int]:
    """
    Select layers for ablation using optimized peak detection strategy.
    
    Args:
        layer_stats: List of layer statistics
        mode: 'conservative', 'balanced', or 'aggressive'
        min_quality: Minimum signal quality threshold (overrides auto-detection)
        top_n: Maximum number of layers to select (None = use mode defaults)
        is_moe: Whether this is a MoE model (affects defaults)
    
    Returns:
        Tuple of (selected_layer_indices, best_measurement_layer)
    """
    num_total_layers = len(layer_stats)
    
    # Find best measurement layer (highest signal quality)
    best_layer = layer_stats[0]['layer']
    best_quality = layer_stats[0]['signal_quality']
    
    # Detect signal quality peaks
    clusters = detect_signal_peaks(layer_stats)
    
    # Mode-specific selection strategy
    if mode == 'conservative':
        # Only select the best cluster and immediate neighbors
        max_clusters = 1
        cluster_expansion = 1
    elif mode == 'aggressive':
        # Select more clusters with more expansion
        max_clusters = 4
        cluster_expansion = 3
    else:  # balanced
        max_clusters = 2
        cluster_expansion = 2
    
    # Sort clusters by their maximum quality
    cluster_scores = []
    for cluster in clusters:
        max_cluster_quality = max(
            stat['signal_quality'] for stat in layer_stats
            if stat['layer'] in cluster
        )
        cluster_scores.append((cluster, max_cluster_quality))
    
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top clusters
    selected_layers = set()
    selected_clusters = cluster_scores[:max_clusters]
    
    for cluster, _ in selected_clusters:
        # Add cluster layers with expansion
        cluster_min = min(cluster)
        cluster_max = max(cluster)
        
        # Expand cluster boundaries
        expanded_min = max(0, cluster_min - cluster_expansion)
        expanded_max = min(num_total_layers - 1, cluster_max + cluster_expansion)
        
        # Add all layers in expanded range
        for layer in range(expanded_min, expanded_max + 1):
            selected_layers.add(layer)
    
    # Apply quality threshold if specified
    if min_quality is not None:
        quality_filtered = set()
        for layer in selected_layers:
            layer_quality = next(
                (stat['signal_quality'] for stat in layer_stats if stat['layer'] == layer),
                0
            )
            if layer_quality >= min_quality:
                quality_filtered.add(layer)
        selected_layers = quality_filtered
    
    # Apply top_n limit if specified
    if top_n is not None:
        # Sort selected layers by quality and take top N
        selected_with_quality = []
        for layer in selected_layers:
            layer_quality = next(
                (stat['signal_quality'] for stat in layer_stats if stat['layer'] == layer),
                0
            )
            selected_with_quality.append((layer, layer_quality))
        
        selected_with_quality.sort(key=lambda x: x[1], reverse=True)
        selected_layers = set(layer for layer, _ in selected_with_quality[:top_n])
    
    # Convert to sorted list
    selected = sorted(list(selected_layers))
    
    return selected, best_layer


def select_layers(
    layer_stats: List[Dict],
    mode: str = 'balanced',
    min_quality: float = None,
    top_n: int = None,
    is_moe: bool = False,
) -> Tuple[List[int], int]:
    """
    Select layers for ablation using the optimized peak detection strategy.
    
    This function now delegates to select_layers_optimized for better results.
    """
    return select_layers_optimized(layer_stats, mode, min_quality, top_n, is_moe)


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
    is_moe: bool = False,
) -> None:
    """
    Generate optimized YAML configuration file for ablation.
    
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
    
    # Detect signal peaks for intelligent scale assignment
    clusters = detect_signal_peaks(layer_stats)
    
    # Find the primary peak (contains the best layer)
    primary_peak = None
    for cluster in clusters:
        if best_measurement_layer in cluster:
            primary_peak = cluster
            break
    
    # Build ablation entries with optimized scale assignment
    ablate_entries = []
    for layer in selected_layers:
        layer_quality = quality_map.get(layer, 0)
        
        # Optimized scale assignment based on quality tiers
        if is_moe:
            # MoE models: more aggressive scaling for high-quality layers
            if layer_quality >= best_quality * 0.8:
                # Elite tier: highest quality layers
                layer_scale = scale * 2.5
            elif layer_quality >= best_quality * 0.6:
                # High tier: very good quality layers
                layer_scale = scale * 2.0
            elif layer_quality >= best_quality * 0.3:
                # Medium tier: moderate quality layers
                layer_scale = scale * 1.5
            else:
                # Low tier: minimal quality layers
                layer_scale = scale * 1.2
        else:
            # Dense models: more conservative scaling
            if layer_quality >= best_quality * 0.8:
                layer_scale = scale * 2.0
            elif layer_quality >= best_quality * 0.6:
                layer_scale = scale * 1.5
            elif layer_quality >= best_quality * 0.3:
                layer_scale = scale * 1.2
            else:
                layer_scale = scale * 1.0
        
        # Apply additional scale decay if requested
        if scale_decay and layer_quality < best_quality * 0.2:
            layer_scale *= 0.8
        
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
        f.write(f"# Optimized auto-generated ablation configuration\n")
        f.write(f"# Generated from: {measurements_path}\n")
        f.write(f"# Model type: {'MoE' if is_moe else 'Dense'}\n")
        f.write(f"# Best measurement layer: {best_measurement_layer} (quality: {best_quality:.4f})\n")
        f.write(f"# Selected {len(selected_layers)} layers for ablation\n")
        f.write(f"# Layer range: {min(selected_layers)}-{max(selected_layers)}\n")
        f.write(f"#\n")
        f.write(f"# Optimization strategy:\n")
        f.write(f"#   - Peak detection: {len(clusters)} signal quality clusters detected\n")
        if primary_peak:
            f.write(f"#   - Primary peak: layers {min(primary_peak)}-{max(primary_peak)}\n")
        if is_moe:
            f.write(f"#   - MoE scaling: Elite (2.5x), High (2.0x), Medium (1.5x), Low (1.2x)\n")
        else:
            f.write(f"#   - Dense scaling: Elite (2.0x), High (1.5x), Medium (1.2x), Low (1.0x)\n")
        f.write(f"#   - Focused on high-quality peaks rather than broad coverage\n")
        f.write(f"#\n")
        f.write(f"# Top 5 layers by signal quality:\n")
        for i, stat in enumerate(layer_stats[:5]):
            f.write(f"#   {i+1}. Layer {stat['layer']}: quality={stat['signal_quality']:.4f}, "
                   f"snr={stat['snr']:.4f}, cosine={stat['cosine_similarity']:.4f}\n")
        f.write(f"\n")
        
        # Write YAML content
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nOptimized configuration saved to: {measurements_path.replace('.refuse', '_config.yml')}")


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
    
    # Detect if MoE model
    is_moe = is_moe_model(args.model_path)
    model_type_str = "MoE" if is_moe else "Dense"
    print(f"\nDetected model type: {model_type_str}")
    
    # Select layers using optimized approach
    selected_layers, best_layer = select_layers(
        layer_stats,
        mode=mode,
        min_quality=args.min_quality,
        top_n=args.top_n,
        is_moe=is_moe,
    )
    
    # Create quality lookup for analysis
    quality_map = {stat['layer']: stat['signal_quality'] for stat in layer_stats}
    
    # Detect peaks for analysis
    clusters = detect_signal_peaks(layer_stats)
    
    print(f"\nOptimized Analysis Results:")
    print(f"Mode: {mode}")
    print(f"Model type: {model_type_str}")
    print(f"Selected {len(selected_layers)} layers: {selected_layers}")
    print(f"Best measurement layer: {best_layer}")
    print(f"Quality range: {layer_stats[-1]['signal_quality']:.4f} - {layer_stats[0]['signal_quality']:.4f}")
    print(f"Signal peaks detected: {len(clusters)}")
    
    for i, cluster in enumerate(clusters[:3]):  # Show top 3 clusters
        cluster_qualities = [quality_map[layer] for layer in cluster if layer in quality_map]
        if cluster_qualities:
            max_cluster_quality = max(cluster_qualities)
            print(f"  Peak {i+1}: layers {min(cluster)}-{max(cluster)} (max quality: {max_cluster_quality:.4f})")
    
    if is_moe:
        print(f"\nMoE-optimized settings:")
        print(f"  - Using tiered scaling (2.5x elite, 2.0x high, 1.5x medium, 1.2x low)")
        print(f"  - Focused on high-quality peaks vs broad coverage")
        print(f"  - Peak-based layer selection")
    else:
        print(f"\nDense-optimized settings:")
        print(f"  - Using conservative scaling (2.0x elite, 1.5x high, 1.2x medium, 1.0x low)")
        print(f"  - Targeted peak detection")
    
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
        is_moe=is_moe,
    )
    
    print(f"\nNext step: Run ablation with:")
    print(f"  python sharded_ablate.py {args.measurements.replace('.refuse', '_config.yml')}")


if __name__ == "__main__":
    main()