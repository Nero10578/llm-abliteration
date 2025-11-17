#!/usr/bin/env python3
"""
Automatically generate YAML configuration files for abliteration based on measurement analysis.
This script analyzes the measurement data and creates optimal ablation configurations.
"""

import argparse
import torch
import yaml
import numpy as np
from typing import List, Tuple, Dict, Any


def parse_analysis_output(measurements_file: str) -> Dict[int, Dict[str, float]]:
    """
    Parse the analysis output from measurements file to extract layer metrics.
    """
    print(f"Loading measurements from {measurements_file}...")
    measures = torch.load(measurements_file)
    layers = measures["layers"]
    
    layer_metrics = {}
    
    for layer in range(layers):
        harmful_mean = measures[f'harmful_{layer}']
        harmless_mean = measures[f'harmless_{layer}']
        refusal_dir = measures[f'refuse_{layer}']
        
        # Calculate metrics
        cos_sim = torch.nn.functional.cosine_similarity(
            harmful_mean.float(), harmless_mean.float(), dim=0
        ).item()
        
        harmful_norm = harmful_mean.norm().item()
        harmless_norm = harmless_mean.norm().item()
        refusal_norm = refusal_dir.norm().item()
        
        # Signal-to-noise ratio
        snr = refusal_norm / max(harmful_norm, harmless_norm)
        
        # Signal quality
        quality = snr * (1 - cos_sim)
        
        layer_metrics[layer] = {
            'cosine_similarity': cos_sim,
            'harmful_norm': harmful_norm,
            'harmless_norm': harmless_norm,
            'refusal_norm': refusal_norm,
            'snr': snr,
            'signal_quality': quality
        }
    
    return layer_metrics


def identify_good_layers(layer_metrics: Dict[int, Dict[str, float]], 
                        total_layers: int) -> Tuple[List[int], List[int]]:
    """
    Identify good target layers and measurement layers based on metrics.
    
    Returns:
        target_layers: Layers to ablate (middle layers with good signals)
        measurement_layers: Layers to use as measurement sources (late layers with strongest signals)
    """
    
    # Calculate thresholds based on data distribution
    snr_values = [metrics['snr'] for metrics in layer_metrics.values()]
    quality_values = [metrics['signal_quality'] for metrics in layer_metrics.values()]
    
    snr_threshold = np.percentile(snr_values, 70)  # Top 30% SNR
    quality_threshold = np.percentile(quality_values, 70)  # Top 30% quality
    
    print(f"SNR threshold: {snr_threshold:.4f}")
    print(f"Quality threshold: {quality_threshold:.6f}")
    
    # Identify measurement layers (late layers with strong signals)
    measurement_layers = []
    late_layer_start = int(total_layers * 0.6)  # Start from 60% through model
    
    for layer in range(late_layer_start, total_layers):
        metrics = layer_metrics[layer]
        if (metrics['snr'] > snr_threshold and 
            metrics['signal_quality'] > quality_threshold and
            metrics['cosine_similarity'] < 0.98):  # Not too similar
            measurement_layers.append(layer)
    
    # If no good measurement layers found, use the best late layers
    if not measurement_layers:
        print("No optimal measurement layers found, using best late layers...")
        late_layers = list(range(late_layer_start, total_layers))
        late_layers.sort(key=lambda l: layer_metrics[l]['signal_quality'], reverse=True)
        measurement_layers = late_layers[:3]  # Top 3 late layers
    
    # Identify target layers (middle layers with decent signals)
    target_layers = []
    middle_layer_start = int(total_layers * 0.3)  # Start from 30% through model
    middle_layer_end = int(total_layers * 0.8)    # End at 80% through model
    
    # Lower thresholds for target layers
    target_snr_threshold = np.percentile(snr_values, 50)  # Top 50% SNR
    target_quality_threshold = np.percentile(quality_values, 50)  # Top 50% quality
    
    for layer in range(middle_layer_start, middle_layer_end):
        metrics = layer_metrics[layer]
        if (metrics['snr'] > target_snr_threshold and 
            metrics['signal_quality'] > target_quality_threshold and
            metrics['refusal_norm'] > 100):  # Minimum refusal signal strength
            target_layers.append(layer)
    
    # If no good target layers, use middle range
    if not target_layers:
        print("No optimal target layers found, using middle range...")
        target_layers = list(range(middle_layer_start, middle_layer_end))
    
    print(f"Identified {len(target_layers)} target layers: {target_layers[:5]}...")
    print(f"Identified {len(measurement_layers)} measurement layers: {measurement_layers}")
    
    return target_layers, measurement_layers


def generate_ablation_config(target_layers: List[int], 
                           measurement_layers: List[int],
                           strategy: str = "conservative") -> List[Dict[str, Any]]:
    """
    Generate ablation configuration based on identified layers.
    
    Strategies:
    - "conservative": Use lower scale, some sparsity
    - "standard": Full scale, minimal sparsity  
    - "aggressive": Higher scale, no sparsity
    """
    
    config = []
    
    # Use the best measurement layer for most targets
    primary_measurement = measurement_layers[-1] if measurement_layers else 0
    
    # Strategy parameters
    if strategy == "conservative":
        scale = 0.8
        sparsity = 0.1
    elif strategy == "aggressive":
        scale = 1.2
        sparsity = 0.0
    else:  # standard
        scale = 1.0
        sparsity = 0.0
    
    # Group target layers by measurement sources
    if len(measurement_layers) >= 2:
        # Use different measurement layers for different target ranges
        mid_point = len(target_layers) // 2
        
        # First half uses earlier measurement layer
        for layer in target_layers[:mid_point]:
            config.append({
                'layer': layer,
                'measurement': measurement_layers[0],
                'scale': scale,
                'sparsity': sparsity
            })
        
        # Second half uses later measurement layer
        for layer in target_layers[mid_point:]:
            config.append({
                'layer': layer,
                'measurement': measurement_layers[-1],
                'scale': scale,
                'sparsity': sparsity
            })
    else:
        # Use single measurement layer for all targets
        for layer in target_layers:
            config.append({
                'layer': layer,
                'measurement': primary_measurement,
                'scale': scale,
                'sparsity': sparsity
            })
    
    return config


def create_yaml_config(measurements_file: str, 
                      model_path: str, 
                      output_dir: str,
                      strategy: str = "standard",
                      custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create complete YAML configuration.
    """
    
    # Parse measurements and identify layers
    layer_metrics = parse_analysis_output(measurements_file)
    total_layers = len(layer_metrics)
    
    print(f"\nAnalyzing {total_layers} layers...")
    
    # Print top layers for manual verification
    print("\nTop 10 layers by signal quality:")
    sorted_layers = sorted(layer_metrics.items(), 
                          key=lambda x: x[1]['signal_quality'], 
                          reverse=True)
    for layer, metrics in sorted_layers[:10]:
        print(f"Layer {layer:2d}: SNR={metrics['snr']:.4f}, "
              f"Quality={metrics['signal_quality']:.6f}, "
              f"Cosine={metrics['cosine_similarity']:.4f}")
    
    # Identify good layers
    target_layers, measurement_layers = identify_good_layers(layer_metrics, total_layers)
    
    # Generate ablation config
    ablation_config = generate_ablation_config(target_layers, measurement_layers, strategy)
    
    # Create full YAML config
    yaml_config = {
        'model': model_path,
        'measurements': measurements_file,
        'output': output_dir,
        'ablate': ablation_config
    }
    
    # Add MoE-specific options if detected
    if custom_config:
        yaml_config.update(custom_config)
    
    return yaml_config


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate YAML configuration for abliteration"
    )
    
    parser.add_argument(
        'measurements_file',
        type=str,
        help='Path to measurements .refuse file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model path or HuggingFace model ID'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for abliterated model'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['conservative', 'standard', 'aggressive'],
        default='standard',
        help='Abliteration strategy'
    )
    
    parser.add_argument(
        '--yaml-output',
        type=str,
        help='Output YAML file path (default: auto_generated_config.yml)'
    )
    
    parser.add_argument(
        '--moe-individual-experts',
        action='store_true',
        help='Enable individual expert abliteration for MoE models'
    )
    
    parser.add_argument(
        '--expert-subset',
        type=str,
        help='Comma-separated expert indices for MoE models'
    )
    
    parser.add_argument(
        '--max-target-layers',
        type=int,
        default=20,
        help='Maximum number of target layers to abliterate'
    )
    
    args = parser.parse_args()
    
    # Custom MoE configuration
    custom_config = {}
    if args.moe_individual_experts:
        custom_config['ablate_individual_experts'] = True
    
    if args.expert_subset:
        try:
            expert_subset = [int(x.strip()) for x in args.expert_subset.split(',')]
            custom_config['expert_subset'] = expert_subset
        except ValueError:
            print("Error: Expert subset must be comma-separated integers")
            return
    
    # Generate configuration
    yaml_config = create_yaml_config(
        measurements_file=args.measurements_file,
        model_path=args.model,
        output_dir=args.output,
        strategy=args.strategy,
        custom_config=custom_config
    )
    
    # Limit target layers if specified
    if args.max_target_layers and len(yaml_config['ablate']) > args.max_target_layers:
        print(f"\nLimiting target layers to {args.max_target_layers} (from {len(yaml_config['ablate'])})")
        yaml_config['ablate'] = yaml_config['ablate'][:args.max_target_layers]
    
    # Determine output file path
    if args.yaml_output:
        yaml_file = args.yaml_output
    else:
        yaml_file = "auto_generated_config.yml"
    
    # Save YAML configuration
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
    
    print(f"\n" + "="*60)
    print("AUTO-GENERATED CONFIGURATION")
    print("="*60)
    print(f"Model: {yaml_config['model']}")
    print(f"Measurements: {yaml_config['measurements']}")
    print(f"Output: {yaml_config['output']}")
    print(f"Strategy: {args.strategy}")
    print(f"Target layers: {len(yaml_config['ablate'])}")
    
    if custom_config:
        print("MoE options:")
        for key, value in custom_config.items():
            print(f"  {key}: {value}")
    
    print(f"\nConfiguration saved to: {yaml_file}")
    print("\nTo run abliteration:")
    print(f"python sharded_ablate.py {yaml_file}")


if __name__ == "__main__":
    main()