#!/usr/bin/env python3
"""
Create a LoRA adapter directly from abliteration, without saving the full abliterated model.

This is more efficient than the two-step process:
1. Old way: Abliterate → Save full model → Convert to LoRA
2. New way: Abliterate → Create LoRA directly (this script)

Usage:
    python ablate_to_lora_direct.py config.yml --output lora-adapter --rank 64

This saves disk space and time by skipping the intermediate full model.
"""

import argparse
import gc
import json
import os
import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file
from typing import Tuple


def magnitude_sparsify(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep only the top fraction of values by magnitude, zero out the rest."""
    if fraction >= 1.0:
        return tensor
    k = int(tensor.numel() * fraction)
    if k == 0:
        return torch.zeros_like(tensor)
    
    flat = tensor.flatten()
    threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
    mask = tensor.abs() >= threshold
    return tensor * mask


def modify_tensor_norm_preserved(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Compute the abliterated weight (same as sharded_ablate.py).
    Returns the modified weight tensor.
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Check dimension compatibility
        if W_gpu.shape[1] != refusal_dir_gpu.shape[0]:
            if W_gpu.shape[0] == refusal_dir_gpu.shape[0]:
                # Routing gate case
                refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
                W_norm = torch.norm(W_gpu, dim=0, keepdim=True)
                W_direction = torch.nn.functional.normalize(W_gpu, dim=0)
                projection = torch.matmul(refusal_normalized, W_direction)
                W_direction_new = W_direction - scale_factor * torch.outer(refusal_normalized, projection)
                W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=0)
                W_modified = W_norm * W_direction_new
            else:
                # Incompatible - return original
                return W.detach().clone()
        else:
            # Standard case
            refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
            W_norm = torch.norm(W_gpu, dim=1, keepdim=True)
            W_direction = torch.nn.functional.normalize(W_gpu, dim=1)
            projection = torch.matmul(W_direction, refusal_normalized)
            W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
            W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
            W_modified = W_norm * W_direction_new
        
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)
        
        del W_gpu, refusal_dir_gpu, refusal_normalized
        del W_direction, W_direction_new, W_norm, projection, W_modified
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


def compute_lora_from_delta(
    delta_W: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LoRA decomposition from weight delta.
    Returns (A, B) where delta_W ≈ B @ A
    """
    delta_W_float = delta_W.float()
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(delta_W_float, full_matrices=False)
    
    # Take top rank components
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    
    # Create LoRA matrices
    B = U_r
    A = torch.diag(torch.sqrt(S_r)) @ Vt_r
    
    # Convert back to original dtype
    A = A.to(delta_W.dtype)
    B = B.to(delta_W.dtype)
    
    return A, B


def ablate_to_lora_direct(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    rank: int = 64,
    alpha: int = None,
) -> None:
    """
    Create LoRA adapter directly from abliteration without saving full model.
    """
    if alpha is None:
        alpha = rank
    
    print(f"Creating LoRA adapter directly (rank={rank}, alpha={alpha})")
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    
    # Determine precision
    if hasattr(config, "torch_dtype"):
        precision = config.torch_dtype
    elif hasattr(config, "dtype"):
        precision = config.dtype
    else:
        precision = torch.float32
    
    if isinstance(precision, str):
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        precision = precision_map.get(precision, torch.float32)
    
    # Get model index
    index_path = cached_file(model_name, "model.safetensors.index.json")
    model_dir = Path(index_path).parent
    
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find layer prefix
    layer_prefix = None
    for key in weight_map.keys():
        if ".layers." in key and ".self_attn." in key:
            layer_prefix = key.split(".layers.")[0]
            break
    
    if layer_prefix is None:
        raise ValueError("Could not detect layer structure")
    
    # Build modification map (same as sharded_ablate.py)
    shard_modifications = {}
    
    for layer, measurement, scale, sparsity in marching_orders:
        o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
        down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.down_proj.weight"
        experts_down_proj_prefix = f"{layer_prefix}.layers.{layer}.mlp.experts."
        shared_experts_down_proj = f"{layer_prefix}.layers.{layer}.mlp.shared_experts.down_proj.weight"
        gate_pattern = f"{layer_prefix}.layers.{layer}.mlp.gate.weight"
        
        for key, shard_file in weight_map.items():
            if key == o_proj_pattern or key == down_proj_pattern:
                if shard_file not in shard_modifications:
                    shard_modifications[shard_file] = []
                shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
            elif key.startswith(experts_down_proj_prefix) and key.endswith(".down_proj.weight"):
                if shard_file not in shard_modifications:
                    shard_modifications[shard_file] = []
                shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
            elif key == shared_experts_down_proj:
                if shard_file not in shard_modifications:
                    shard_modifications[shard_file] = []
                shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
            elif key == gate_pattern:
                if shard_file not in shard_modifications:
                    shard_modifications[shard_file] = []
                shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
    
    os.makedirs(output_path, exist_ok=True)
    
    # LoRA state dict
    lora_state_dict = {}
    
    # Process each shard
    all_shards = sorted(set(weight_map.values()))
    
    print(f"\nProcessing {len(shard_modifications)} shards with modifications...")
    
    for shard_file in tqdm(all_shards, desc="Creating LoRA"):
        if shard_file not in shard_modifications:
            continue
        
        shard_path = model_dir / shard_file
        state_dict = load_file(str(shard_path))
        
        # Process each weight in this shard
        for key, layer, measurement, scale, sparsity in shard_modifications[shard_file]:
            if key not in state_dict:
                continue
            
            # Get original weight
            W_original = state_dict[key]
            
            # Compute refusal direction
            refusal_dir = measures[f'refuse_{measurement}'].float()
            harmless_dir = measures[f'harmless_{layer}'].float()
            
            harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
            projection_scalar = refusal_dir @ harmless_normalized
            refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
            refusal_dir = refined_refusal_dir.to(precision)
            
            if sparsity > 0.0:
                refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
            
            refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
            
            # Compute abliterated weight
            W_abliterated = modify_tensor_norm_preserved(W_original, refusal_dir, scale)
            
            # Compute delta
            delta_W = W_abliterated - W_original
            
            # Check if there's actually a change
            if torch.allclose(delta_W, torch.zeros_like(delta_W), rtol=1e-5, atol=1e-8):
                continue
            
            # Only create LoRA for 2D matrices
            if delta_W.dim() != 2:
                continue
            
            # Compute LoRA decomposition
            A, B = compute_lora_from_delta(delta_W, rank)
            
            # Store with standard naming
            base_name = key.replace(".weight", "")
            lora_state_dict[f"{base_name}.lora_A.weight"] = A
            lora_state_dict[f"{base_name}.lora_B.weight"] = B
            
            # Cleanup
            del refusal_dir, harmless_dir, harmless_normalized, refined_refusal_dir
            del W_abliterated, delta_W, A, B
            gc.collect()
        
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nCreated LoRA for {len(lora_state_dict) // 2} weights")
    
    # Save LoRA
    print(f"Saving LoRA adapter to {output_path}...")
    save_file(lora_state_dict, f"{output_path}/adapter_model.safetensors")
    
    # Create adapter config
    adapter_config = {
        "base_model_name_or_path": model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(set([
            name.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            for name in lora_state_dict.keys()
        ])),
        "task_type": "CAUSAL_LM",
    }
    
    with open(f"{output_path}/adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    lora_size = sum(w.numel() * w.element_size() for w in lora_state_dict.values()) / (1024**3)
    print(f"\nLoRA adapter size: {lora_size:.2f} GB")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create LoRA adapter directly from abliteration config"
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to YAML configuration file (same as for sharded_ablate.py)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for LoRA adapter'
    )
    
    parser.add_argument(
        '--rank', '-r',
        type=int,
        default=64,
        help='LoRA rank (default: 64)'
    )
    
    parser.add_argument(
        '--alpha',
        type=int,
        default=None,
        help='LoRA alpha (default: same as rank)'
    )
    
    args = parser.parse_args()
    
    # Load YAML config
    with open(args.config_file, 'r') as f:
        ydata = yaml.safe_load(f)
    
    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    ablations = ydata.get("ablate")
    
    print("=" * 60)
    print("DIRECT LORA CREATION FROM ABLITERATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output: {args.output}")
    print(f"Rank: {args.rank}")
    print(f"Number of ablations: {len(ablations)}")
    print("=" * 60)
    
    # Load measurements
    print(f"\nLoading measurements...")
    measures = torch.load(measurement_file)
    
    # Parse ablation orders
    orders = [
        (
            int(item['layer']),
            int(item['measurement']),
            float(item['scale']),
            float(item['sparsity']),
        )
        for item in ablations
    ]
    
    # Create LoRA
    ablate_to_lora_direct(
        model_name=model_name,
        measures=measures,
        marching_orders=orders,
        output_path=args.output,
        rank=args.rank,
        alpha=args.alpha,
    )
    
    print("\n" + "=" * 60)
    print("LORA CREATION COMPLETE")
    print("=" * 60)
    print(f"\nTo use this LoRA:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  ")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(model, '{args.output}')")


if __name__ == "__main__":
    main()