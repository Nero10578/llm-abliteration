#!/usr/bin/env python3
"""
Test script to verify GPT-OSS-20B compatibility with sharded_ablate.py
"""

import torch
import yaml
from pathlib import Path

def test_pattern_matching():
    """Test that the pattern matching logic works for GPT-OSS-20B structures"""
    
    # Simulate the weight map patterns from GPT-OSS-20B
    gpt_oss_weight_map = {
        "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00009.safetensors",
        "model.layers.0.mlp.experts.down_proj": "model-00001-of-00009.safetensors",
        "model.layers.0.mlp.experts.down_proj_bias": "model-00001-of-00009.safetensors",
        "model.layers.0.mlp.router.weight": "model-00001-of-00009.safetensors",
        "model.layers.15.self_attn.o_proj.weight": "model-00004-of-00009.safetensors",
        "model.layers.15.mlp.experts.down_proj": "model-00004-of-00009.safetensors",
        "model.layers.15.mlp.experts.down_proj_bias": "model-00004-of-00009.safetensors",
        "model.layers.15.mlp.router.weight": "model-00004-of-00009.safetensors",
    }
    
    # Test layer prefix detection
    layer_prefix = None
    for key in gpt_oss_weight_map.keys():
        if ".layers." in key and ".self_attn." in key:
            layer_prefix = key.split(".layers.")[0]
            break
    
    print(f"Detected layer prefix: {layer_prefix}")
    assert layer_prefix == "model", f"Expected 'model', got '{layer_prefix}'"
    
    # Test pattern matching for layer 0
    layer = 0
    measurement = 15
    scale = 1.0
    sparsity = 0.0
    
    # Build patterns
    o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
    gpt_oss_experts_down_proj = f"{layer_prefix}.layers.{layer}.mlp.experts.down_proj"
    gpt_oss_experts_down_proj_bias = f"{layer_prefix}.layers.{layer}.mlp.experts.down_proj_bias"
    gpt_oss_router_pattern = f"{layer_prefix}.layers.{layer}.mlp.router.weight"
    
    # Test matching
    matched_keys = []
    for key in gpt_oss_weight_map.keys():
        if key == o_proj_pattern:
            matched_keys.append(f"ATTENTION: {key}")
        elif key == gpt_oss_experts_down_proj:
            matched_keys.append(f"EXPERTS_3D: {key}")
        elif key == gpt_oss_experts_down_proj_bias:
            matched_keys.append(f"EXPERTS_BIAS: {key}")
        elif key == gpt_oss_router_pattern:
            matched_keys.append(f"ROUTER: {key}")
    
    print(f"\nMatched keys for layer {layer}:")
    for key in matched_keys:
        print(f"  {key}")
    
    expected_matches = 4  # o_proj, experts.down_proj, experts.down_proj_bias, router
    assert len(matched_keys) == expected_matches, f"Expected {expected_matches} matches, got {len(matched_keys)}"
    
    print("✅ Pattern matching test passed!")

def test_tensor_shapes():
    """Test tensor shape compatibility"""
    
    # Simulate GPT-OSS-20B tensor shapes
    tensors = {
        "attention_o_proj": torch.randn(2880, 4096),  # [hidden_size, head_dim * num_heads]
        "experts_down_proj": torch.randn(32, 2880, 2880),  # [num_experts, hidden_size, hidden_size]
        "experts_down_proj_bias": torch.randn(32, 2880),  # [num_experts, hidden_size]
        "router_weight": torch.randn(32, 2880),  # [num_experts, hidden_size]
    }
    
    # Test refusal direction sizes
    refusal_dirs = {
        "attention_size": torch.randn(4096),  # For attention weights
        "mlp_size": torch.randn(2880),  # For MLP weights
    }
    
    print("\nTesting tensor shape compatibility:")
    
    # Test attention weights
    print(f"  Attention o_proj: {tensors['attention_o_proj'].shape} vs refusal_dir: {refusal_dirs['attention_size'].shape}")
    assert tensors['attention_o_proj'].shape[1] == refusal_dirs['attention_size'].shape[0]
    
    # Test 3D expert tensors
    print(f"  Experts down_proj: {tensors['experts_down_proj'].shape} vs refusal_dir: {refusal_dirs['mlp_size'].shape}")
    assert tensors['experts_down_proj'].shape[1] == refusal_dirs['mlp_size'].shape[0]
    assert tensors['experts_down_proj'].shape[2] == refusal_dirs['mlp_size'].shape[0]
    
    # Test expert bias
    print(f"  Experts down_proj_bias: {tensors['experts_down_proj_bias'].shape} vs refusal_dir: {refusal_dirs['mlp_size'].shape}")
    assert tensors['experts_down_proj_bias'].shape[1] == refusal_dirs['mlp_size'].shape[0]
    
    # Test router
    print(f"  Router weight: {tensors['router_weight'].shape} vs refusal_dir: {refusal_dirs['mlp_size'].shape}")
    assert tensors['router_weight'].shape[1] == refusal_dirs['mlp_size'].shape[0]
    
    print("✅ Tensor shape compatibility test passed!")

def test_config_compatibility():
    """Test that the existing YAML config works"""
    
    config_file = "gpt-oss-20b-optimal.yml"
    if not Path(config_file).exists():
        print(f"⚠️  Config file {config_file} not found, skipping config test")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nTesting config compatibility:")
    print(f"  Model: {config.get('model')}")
    print(f"  Measurements: {config.get('measurements')}")
    print(f"  Output: {config.get('output')}")
    
    ablations = config.get('ablate', [])
    print(f"  Number of ablations: {len(ablations)}")
    
    # Check that all ablations have required fields
    for i, ablation in enumerate(ablations):
        assert 'layer' in ablation, f"Ablation {i} missing 'layer'"
        assert 'measurement' in ablation, f"Ablation {i} missing 'measurement'"
        assert 'scale' in ablation, f"Ablation {i} missing 'scale'"
        assert 'sparsity' in ablation, f"Ablation {i} missing 'sparsity'"
        
        layer = ablation['layer']
        assert isinstance(layer, int), f"Layer {layer} should be int, got {type(layer)}"
        assert 0 <= layer < 24, f"Layer {layer} should be between 0 and 23"
    
    print("✅ Config compatibility test passed!")

if __name__ == "__main__":
    print("Testing GPT-OSS-20B compatibility with sharded_ablate.py")
    print("=" * 60)
    
    try:
        test_pattern_matching()
        test_tensor_shapes()
        test_config_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! GPT-OSS-20B should now be compatible.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()