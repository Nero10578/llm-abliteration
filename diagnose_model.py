#!/usr/bin/env python3
"""
Diagnostic script to examine the actual weight structure of GLM-4.5-Air
"""

import json
from pathlib import Path
from transformers import AutoConfig
from transformers.utils import cached_file

def diagnose_model_structure(model_path):
    """
    Examine the actual weight structure to understand the MoE layout
    """
    print(f"Diagnosing model structure for: {model_path}")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    print(f"Model type: {config.model_type}")
    print(f"Architecture: {getattr(config, 'architectures', ['Unknown'])}")
    
    # Get safetensors index
    try:
        index_path = cached_file(model_path, "model.safetensors.index.json")
        print(f"Index file: {index_path}")
        
        with open(index_path) as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        print(f"Total weight keys: {len(weight_map)}")
        
        # Analyze weight patterns
        layer_patterns = {}
        mlp_patterns = {}
        expert_patterns = {}
        gate_patterns = {}
        
        for key in weight_map.keys():
            # Categorize by pattern
            if ".layers." in key:
                parts = key.split(".layers.")
                if len(parts) > 1:
                    layer_part = parts[1].split(".")[0]
                    try:
                        layer_num = int(layer_part)
                        rest = ".".join(parts[1].split(".")[1:])
                        
                        if layer_num not in layer_patterns:
                            layer_patterns[layer_num] = []
                        layer_patterns[layer_num].append(rest)
                        
                        # Analyze MLP patterns
                        if "mlp." in rest:
                            if layer_num not in mlp_patterns:
                                mlp_patterns[layer_num] = []
                            mlp_patterns[layer_num].append(rest)
                        
                        # Look for expert patterns
                        if "experts." in rest:
                            if layer_num not in expert_patterns:
                                expert_patterns[layer_num] = []
                            expert_patterns[layer_num].append(rest)
                        
                        # Look for gate patterns
                        if "gate.weight" in rest:
                            if layer_num not in gate_patterns:
                                gate_patterns[layer_num] = []
                            gate_patterns[layer_num].append(rest)
                            
                    except ValueError:
                        continue
        
        # Print analysis
        print(f"\nFound layers: {sorted(layer_patterns.keys())}")
        print(f"Total layers: {len(layer_patterns)}")
        
        # Show MLP patterns for first few layers
        print(f"\nMLP patterns for first 3 layers:")
        for layer in sorted(mlp_patterns.keys())[:3]:
            print(f"  Layer {layer}:")
            for pattern in sorted(mlp_patterns[layer]):
                print(f"    {pattern}")
        
        # Show expert patterns
        if expert_patterns:
            print(f"\nExpert patterns found:")
            for layer in sorted(expert_patterns.keys())[:3]:
                print(f"  Layer {layer}:")
                for pattern in sorted(expert_patterns[layer]):
                    print(f"    {pattern}")
            
            # Count experts
            expert_indices = set()
            for layer_patterns in expert_patterns.values():
                for pattern in layer_patterns:
                    parts = pattern.split("experts.")
                    if len(parts) > 1:
                        expert_part = parts[1].split(".")[0]
                        try:
                            expert_indices.add(int(expert_part))
                        except ValueError:
                            pass
            
            if expert_indices:
                print(f"\nExpert indices found: {sorted(expert_indices)}")
                print(f"Total experts: {max(expert_indices) + 1}")
        else:
            print(f"\nNo expert patterns found!")
        
        # Show gate patterns
        if gate_patterns:
            print(f"\nGate patterns found:")
            for layer in sorted(gate_patterns.keys())[:3]:
                print(f"  Layer {layer}:")
                for pattern in gate_patterns[layer]:
                    print(f"    {pattern}")
        
        # Look for shared experts
        shared_expert_patterns = {}
        for key in weight_map.keys():
            if "shared_experts." in key:
                parts = key.split(".layers.")
                if len(parts) > 1:
                    layer_part = parts[1].split(".")[0]
                    try:
                        layer_num = int(layer_part)
                        rest = ".".join(parts[1].split(".")[1:])
                        
                        if layer_num not in shared_expert_patterns:
                            shared_expert_patterns[layer_num] = []
                        shared_expert_patterns[layer_num].append(rest)
                    except ValueError:
                        continue
        
        if shared_expert_patterns:
            print(f"\nShared expert patterns found:")
            for layer in sorted(shared_expert_patterns.keys())[:3]:
                print(f"  Layer {layer}:")
                for pattern in sorted(shared_expert_patterns[layer]):
                    print(f"    {pattern}")
        
        # Check weight dimensions for problematic patterns
        print(f"\nChecking weight dimensions for key patterns:")
        
        # Get a sample shard to check dimensions
        shard_files = set(weight_map.values())
        if shard_files:
            sample_shard = list(shard_files)[0]
            shard_path = Path(index_path).parent / sample_shard
            
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(shard_path))
                
                # Check dimensions of key patterns
                patterns_to_check = [
                    "self_attn.o_proj.weight",
                    "mlp.gate.weight",
                    "mlp.shared_experts.down_proj.weight"
                ]
                
                for layer in [0, 1, 2]:  # Check first few layers
                    for pattern in patterns_to_check:
                        full_key = f"model.layers.{layer}.{pattern}"
                        if full_key in state_dict:
                            tensor = state_dict[full_key]
                            print(f"  {full_key}: {tensor.shape} ({tensor.dtype})")
                        elif full_key in weight_map:
                            print(f"  {full_key}: in different shard")
                
            except Exception as e:
                print(f"Could not load shard for dimension check: {e}")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python diagnose_model.py /path/to/model")
        sys.exit(1)
    
    model_path = sys.argv[1]
    diagnose_model_structure(model_path)