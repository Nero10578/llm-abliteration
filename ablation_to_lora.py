#!/usr/bin/env python3
"""
Convert abliteration modifications into a LoRA adapter.

This script compares an original model with its abliterated version and creates
a LoRA adapter that captures the differences. This allows you to:
1. Keep the original model unchanged
2. Apply/remove abliteration by loading/unloading the LoRA
3. Share smaller files (LoRA is much smaller than full model)
4. Combine with other LoRAs

Usage:
    python ablation_to_lora.py \
        --original /path/to/original/model \
        --abliterated /path/to/abliterated/model \
        --output /path/to/lora/adapter \
        --rank 64

The rank parameter controls LoRA size/quality tradeoff:
    - rank=8: Very small, may lose some abliteration effect
    - rank=16: Small, good for most cases
    - rank=32: Medium, better quality
    - rank=64: Large, high quality (recommended for abliteration)
    - rank=128: Very large, maximum quality
"""

import argparse
import json
import os
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file
from typing import Dict, Tuple


def compute_lora_decomposition(
    W_original: torch.Tensor,
    W_abliterated: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LoRA decomposition of the weight difference.
    
    LoRA represents weight updates as: ΔW = B @ A
    where A is [rank, in_features] and B is [out_features, rank]
    
    We use SVD to find the best low-rank approximation:
    ΔW ≈ U[:, :rank] @ S[:rank] @ V[:rank, :]
    
    Then: A = S[:rank] @ V[:rank, :], B = U[:, :rank]
    """
    # Compute weight difference
    delta_W = W_abliterated - W_original
    
    # Convert to float32 for SVD
    delta_W_float = delta_W.float()
    
    # Perform SVD
    # delta_W = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(delta_W_float, full_matrices=False)
    
    # Take top 'rank' components
    U_r = U[:, :rank]  # [out_features, rank]
    S_r = S[:rank]     # [rank]
    Vt_r = Vt[:rank, :] # [rank, in_features]
    
    # Create LoRA matrices
    # B = U_r (no scaling here, we'll put it in A)
    B = U_r
    
    # A = sqrt(S_r) @ Vt_r (distribute singular values)
    # This balances the magnitude between A and B
    A = torch.diag(torch.sqrt(S_r)) @ Vt_r
    
    # Convert back to original dtype
    A = A.to(delta_W.dtype)
    B = B.to(delta_W.dtype)
    
    return A, B


def create_lora_adapter(
    original_model_path: str,
    abliterated_model_path: str,
    output_path: str,
    rank: int = 64,
    alpha: int = None,
) -> None:
    """
    Create a LoRA adapter from abliteration differences.
    
    Args:
        original_model_path: Path to original model
        abliterated_model_path: Path to abliterated model
        output_path: Path to save LoRA adapter
        rank: LoRA rank (higher = better quality, larger size)
        alpha: LoRA alpha (scaling factor, default = rank)
    """
    if alpha is None:
        alpha = rank
    
    print(f"Creating LoRA adapter with rank={rank}, alpha={alpha}")
    print(f"Original model: {original_model_path}")
    print(f"Abliterated model: {abliterated_model_path}")
    
    # Load configs
    original_config = AutoConfig.from_pretrained(original_model_path)
    
    # Get model index files
    original_index_path = cached_file(original_model_path, "model.safetensors.index.json")
    abliterated_index_path = cached_file(abliterated_model_path, "model.safetensors.index.json")
    
    original_dir = Path(original_index_path).parent
    abliterated_dir = Path(abliterated_index_path).parent
    
    with open(original_index_path) as f:
        original_index = json.load(f)
    with open(abliterated_index_path) as f:
        abliterated_index = json.load(f)
    
    original_weight_map = original_index["weight_map"]
    abliterated_weight_map = abliterated_index["weight_map"]
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # LoRA adapter state dict
    lora_state_dict = {}
    
    # Track which shards we've processed
    processed_shards = set()
    
    # Process each weight
    print("\nComparing models and creating LoRA decompositions...")
    for weight_name in tqdm(sorted(original_weight_map.keys())):
        if weight_name not in abliterated_weight_map:
            continue
        
        original_shard = original_weight_map[weight_name]
        abliterated_shard = abliterated_weight_map[weight_name]
        
        # Load weights if we haven't processed this shard yet
        shard_key = (original_shard, abliterated_shard)
        if shard_key not in processed_shards:
            original_state = load_file(str(original_dir / original_shard))
            abliterated_state = load_file(str(abliterated_dir / abliterated_shard))
            processed_shards.add(shard_key)
        
        if weight_name not in original_state or weight_name not in abliterated_state:
            continue
        
        W_orig = original_state[weight_name]
        W_abl = abliterated_state[weight_name]
        
        # Check if weights are different
        if torch.allclose(W_orig, W_abl, rtol=1e-5, atol=1e-8):
            continue  # No change, skip
        
        # Only create LoRA for 2D weight matrices
        if W_orig.dim() != 2:
            print(f"  Skipping {weight_name}: not a 2D matrix (shape: {W_orig.shape})")
            continue
        
        # Compute LoRA decomposition
        A, B = compute_lora_decomposition(W_orig, W_abl, rank)
        
        # Store LoRA weights with standard naming convention
        # For weight "model.layers.0.self_attn.o_proj.weight"
        # Create "model.layers.0.self_attn.o_proj.lora_A" and "lora_B"
        base_name = weight_name.replace(".weight", "")
        lora_state_dict[f"{base_name}.lora_A.weight"] = A
        lora_state_dict[f"{base_name}.lora_B.weight"] = B
        
        print(f"  Created LoRA for {weight_name}: {W_orig.shape} -> A{A.shape} @ B{B.shape}")
    
    print(f"\nCreated LoRA decompositions for {len(lora_state_dict) // 2} weights")
    
    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {output_path}...")
    save_file(lora_state_dict, f"{output_path}/adapter_model.safetensors")
    
    # Create adapter config
    adapter_config = {
        "base_model_name_or_path": original_model_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": list(set([
            name.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            for name in lora_state_dict.keys()
        ])),
        "task_type": "CAUSAL_LM",
    }
    
    with open(f"{output_path}/adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"Saved adapter config to {output_path}/adapter_config.json")
    
    # Calculate size reduction
    original_size = sum(W_orig.numel() * W_orig.element_size() 
                       for W_orig in original_state.values()) / (1024**3)
    lora_size = sum(w.numel() * w.element_size() 
                   for w in lora_state_dict.values()) / (1024**3)
    
    print(f"\nSize comparison:")
    print(f"  Original model shard: ~{original_size:.2f} GB")
    print(f"  LoRA adapter: {lora_size:.2f} GB")
    print(f"  Reduction: {(1 - lora_size/original_size)*100:.1f}%")
    
    print(f"\nLoRA adapter created successfully!")
    print(f"\nTo use this adapter:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  ")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{original_model_path}')")
    print(f"  model = PeftModel.from_pretrained(model, '{output_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert abliteration into a LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--original", "-o",
        type=str,
        required=True,
        help="Path to original model"
    )
    
    parser.add_argument(
        "--abliterated", "-a",
        type=str,
        required=True,
        help="Path to abliterated model"
    )
    
    parser.add_argument(
        "--output", "-out",
        type=str,
        required=True,
        help="Output path for LoRA adapter"
    )
    
    parser.add_argument(
        "--rank", "-r",
        type=int,
        default=64,
        help="LoRA rank (default: 64). Higher = better quality but larger size."
    )
    
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling factor (default: same as rank)"
    )
    
    args = parser.parse_args()
    
    create_lora_adapter(
        original_model_path=args.original,
        abliterated_model_path=args.abliterated,
        output_path=args.output,
        rank=args.rank,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()