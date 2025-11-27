#!/usr/bin/env python3
"""
Model Structure Analysis Script

This script analyzes the structure of a model to understand:
1. Parameter shapes and dimensions
2. Layer organization
3. Expert/MoE structure
4. Parameter naming patterns
5. Compatibility with current ablation methods

Usage:
    python analyze_model_structure.py --model <model_path> [--layer <layer_num>] [--detailed]
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.utils import cached_file


def analyze_parameter_shapes(model_path, target_layer=None, detailed=False):
    """Analyze parameter shapes and structure"""
    print(f"Analyzing model structure: {model_path}")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    print(f"Architecture: {type(config).__name__}")
    
    # Get safetensors index
    try:
        index_path = cached_file(model_path, "model.safetensors.index.json")
        model_dir = Path(index_path).parent
        
        with open(index_path) as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        print(f"Total parameters: {len(weight_map)}")
        
        # Group parameters by layer
        layer_groups = {}
        other_params = []
        
        for key in weight_map.keys():
            if ".layers." in key:
                layer_num = key.split(".layers.")[1].split(".")[0]
                if layer_num.isdigit():
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = []
                    layer_groups[layer_num].append(key)
            else:
                other_params.append(key)
        
        print(f"Found {len(layer_groups)} layers")
        print(f"Other parameters: {len(other_params)}")
        
        # Analyze specific layer or all layers
        layers_to_analyze = [target_layer] if target_layer else sorted(layer_groups.keys())[:3]  # First 3 layers
        
        for layer_num in layers_to_analyze:
            if layer_num not in layer_groups:
                print(f"Layer {layer_num} not found!")
                continue
                
            print(f"\n=== LAYER {layer_num} ANALYSIS ===")
            layer_params = layer_groups[layer_num]
            
            # Group by component type
            components = {}
            for param in layer_params:
                component = param.split(f".layers.{layer_num}.")[1].split(".")[0]
                if component not in components:
                    components[component] = []
                components[component].append(param)
            
            # Load shard containing this layer's parameters
            shard_files = set()
            for param in layer_params:
                shard_files.add(weight_map[param])
            
            if len(shard_files) > 1:
                print(f"  Parameters span multiple shards: {shard_files}")
            
            # Load the first shard and analyze shapes
            shard_file = list(shard_files)[0]
            shard_path = model_dir / shard_file
            
            try:
                state_dict = load_file(str(shard_path))
                
                for component, params in components.items():
                    print(f"\n  {component.upper()}:")
                    for param in params:
                        if param in state_dict:
                            shape = state_dict[param].shape
                            print(f"    {param}: {shape}")
                            
                            if detailed:
                                # Additional analysis for expert parameters
                                if "experts" in param:
                                    analyze_expert_parameter(param, shape, detailed)
                                elif "router" in param:
                                    analyze_router_parameter(param, shape, detailed)
                                elif "self_attn" in param:
                                    analyze_attention_parameter(param, shape, detailed)
                        else:
                            print(f"    {param}: NOT FOUND IN SHARD")
                
            except Exception as e:
                print(f"  Error loading shard {shard_file}: {e}")
        
        # Analyze other parameters
        if other_params and detailed:
            print(f"\n=== OTHER PARAMETERS ===")
            for param in other_params[:10]:  # First 10
                print(f"  {param}")
        
        # Summary analysis
        print(f"\n=== STRUCTURE SUMMARY ===")
        print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
        print(f"Total layers: {len(layer_groups)}")
        
        # Check for MoE structure
        has_experts = any("experts" in param for param in weight_map.keys())
        has_router = any("router" in param for param in weight_map.keys())
        
        print(f"MoE Structure: {'Yes' if has_experts else 'No'}")
        print(f"Router Parameters: {'Yes' if has_router else 'No'}")
        
        # Analyze first layer structure for ablation compatibility
        if "0" in layer_groups:
            print(f"\n=== ABLATION COMPATIBILITY ANALYSIS (Layer 0) ===")
            analyze_ablation_compatibility(layer_groups["0"], weight_map, model_dir)
    
    except Exception as e:
        print(f"Error analyzing model: {e}")


def analyze_expert_parameter(param_name, shape, detailed=False):
    """Analyze expert parameter structure"""
    print(f"      Expert Analysis:")
    print(f"        Type: {param_name.split('.')[-1]}")
    print(f"        Shape: {shape}")
    
    if len(shape) >= 2:
        print(f"        Out features: {shape[0] if len(shape) == 2 else 'N/A'}")
        print(f"        In features: {shape[1] if len(shape) == 2 else 'N/A'}")
        
        if len(shape) == 2 and shape[0] == shape[1]:
            print(f"        Note: Square matrix - might be different structure")


def analyze_router_parameter(param_name, shape, detailed=False):
    """Analyze router parameter structure"""
    print(f"      Router Analysis:")
    print(f"        Type: {param_name.split('.')[-1]}")
    print(f"        Shape: {shape}")
    
    if len(shape) == 1:
        print(f"        This is a bias parameter")
    elif len(shape) == 2:
        print(f"        Weight matrix: {shape[0]} experts x {shape[1]} hidden_size")


def analyze_attention_parameter(param_name, shape, detailed=False):
    """Analyze attention parameter structure"""
    print(f"      Attention Analysis:")
    print(f"        Type: {param_name.split('.')[-1]}")
    print(f"        Shape: {shape}")
    
    if "o_proj" in param_name:
        print(f"        Output projection - good ablation target")


def analyze_ablation_compatibility(layer_params, weight_map, model_dir):
    """Analyze compatibility with current ablation methods"""
    # Find key parameters for ablation
    o_proj = None
    down_proj = None
    router = None
    
    for param in layer_params:
        if "self_attn.o_proj.weight" in param:
            o_proj = param
        elif "mlp.experts" in param and "down_proj" in param:
            down_proj = param
        elif "mlp.router" in param and "weight" in param:
            router = param
    
    print(f"Key ablation targets found:")
    print(f"  Attention output: {'Yes' if o_proj else 'No'} ({o_proj or 'N/A'})")
    print(f"  Expert down proj: {'Yes' if down_proj else 'No'} ({down_proj or 'N/A'})")
    print(f"  Router weights: {'Yes' if router else 'No'} ({router or 'N/A'})")
    
    # Load and analyze shapes
    targets = [p for p in [o_proj, down_proj, router] if p is not None]
    if targets:
        shard_files = set(weight_map[p] for p in targets)
        shard_file = list(shard_files)[0]
        shard_path = model_dir / shard_file
        
        try:
            state_dict = load_file(str(shard_path))
            
            for target in targets:
                if target in state_dict:
                    shape = state_dict[target].shape
                    print(f"\n  {target}:")
                    print(f"    Shape: {shape}")
                    print(f"    Dimensions: {len(shape)}")
                    
                    if len(shape) == 2:
                        print(f"    Matrix: {shape[0]} x {shape[1]}")
                        print(f"    Transposed: {shape[1]} x {shape[0]}")
                        
                        # Check compatibility with typical hidden_size
                        if shape[1] in [4096, 5120, 7168, 8192]:  # Common hidden sizes
                            print(f"    Compatible hidden size: {shape[1]}")
                        else:
                            print(f"    Unusual hidden size: {shape[1]}")
                    
                    # Analyze for ablation method compatibility
                    if len(shape) == 2 and shape[0] != shape[1]:
                        print(f"    Standard weight matrix - compatible with current ablation")
                    elif len(shape) == 2 and shape[0] == shape[1]:
                        print(f"    Square matrix - may need special handling")
                    else:
                        print(f"    Unusual shape - needs custom ablation logic")
                        
        except Exception as e:
            print(f"  Error loading parameters: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model structure for ablation compatibility")
    parser.add_argument("--model", type=str, required=True, help="Model path or identifier")
    parser.add_argument("--layer", type=str, help="Specific layer to analyze (number)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    analyze_parameter_shapes(args.model, args.layer, args.detailed)


if __name__ == "__main__":
    main()