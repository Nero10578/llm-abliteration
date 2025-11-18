import argparse
import gc
import json
import os
import shutil
import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file


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


"""
A warning regarding PyTorch's convention vs. Safetensors storage:

PyTorch nn.Linear layers store weights as [out_features, in_features] - each row is an output neuron's weights
Safetensors (HuggingFace format) stores them as [in_features, out_features] - transposed!
"""


def modify_tensor_norm_preserved(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        # Decompose weight matrix
        # W_gpu is [out_features, in_features]
        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron
    
        # Apply abliteration to the DIRECTIONAL component
        # Compute dot product of each row with refusal direction
        projection = torch.matmul(W_direction, refusal_normalized)  # [in_features]
        
        # Subtract the projection
        W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
    
        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
    
        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new
        
        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, refusal_dir_gpu, refusal_normalized
        del W_direction, W_direction_new, W_norm, projection, W_modified
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


def modify_tensor_norm_preserved_moe(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
    is_expert_gate: bool = False,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Special handling for MoE expert gates.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)
        
        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
        
        # Check dimension compatibility
        if W_gpu.shape[1] != refusal_dir_gpu.shape[0]:
            print(f"Warning: Dimension mismatch - W shape: {W_gpu.shape}, refusal_dir shape: {refusal_dir_gpu.shape}")
            
            if is_expert_gate:
                # For expert gates, we need to handle dimension mismatch differently
                # Expert gates typically project from hidden_size to num_experts
                # We can only modify the input dimension (hidden_size) part
                
                # Option 1: Skip modification for incompatible dimensions
                print("Skipping expert gate modification due to dimension mismatch")
                return W.detach().clone()
                
                # Option 2: Truncate/pad refusal direction (commented out - more risky)
                # min_dim = min(W_gpu.shape[1], refusal_dir_gpu.shape[0])
                # W_gpu_reduced = W_gpu[:, :min_dim]
                # refusal_dir_reduced = refusal_dir_gpu[:min_dim]
                # refusal_normalized = torch.nn.functional.normalize(refusal_dir_reduced, dim=0)
        
        if is_expert_gate:
            # For expert gates, we use a more conservative approach
            # Expert gates determine routing, so we want to be more subtle
            scale_factor = scale_factor * 0.5  # Reduce intensity for gates
            
            # Only proceed if dimensions are compatible
            if W_gpu.shape[1] == refusal_dir_gpu.shape[0]:
                # For gates, we modify the routing probabilities rather than weights
                # Apply a smaller perturbation to avoid breaking expert selection
                W_norm = torch.norm(W_gpu, dim=1, keepdim=True)
                W_direction = torch.nn.functional.normalize(W_gpu, dim=1)
                
                # Compute dot product with refusal direction
                projection = torch.matmul(W_direction, refusal_normalized)
                
                # Apply reduced modification
                W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
                
                # Re-normalize
                W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
                W_modified = W_norm * W_direction_new
            else:
                # Skip modification for incompatible dimensions
                print("Skipping expert gate modification due to dimension mismatch")
                return W.detach().clone()
        else:
            # Standard weight modification for shared experts and attention
            W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
            W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron
            
            # Apply abliteration to the DIRECTIONAL component
            projection = torch.matmul(W_direction, refusal_normalized)  # [in_features]
            
            # Subtract the projection
            W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
            
            # Re-normalize the adjusted direction
            W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
            
            # Recombine: keep original magnitude, use new direction
            W_modified = W_norm * W_direction_new
        
        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)
        
        # Cleanup
        del W_gpu, refusal_dir_gpu, refusal_normalized
        del W_direction, W_direction_new, W_norm, projection, W_modified
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    return result.detach().clone()


def ablate_by_layers_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    ablate_individual_experts: bool = False,
    expert_subset: list = None,
) -> None:
    """
    Memory-efficient ablation for sharded models.
    Handles both local paths and HuggingFace Hub models.
    Loads one shard at a time, applies all modifications, then saves.
    """
    
    # Load config using transformers (handles both local and HF hub)
    print(f"Loading config for {model_name}...")
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
    
    print(f"Model precision: {precision}")
    
    # Get the safetensors index file (handles cache)
    index_path = cached_file(model_name, "model.safetensors.index.json")
    model_dir = Path(index_path).parent
    
    print(f"Model directory: {model_dir}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Detect model architecture and find layer prefix
    layer_prefix = None
    is_moe = False
    expert_count = 0
    
    # Check for MoE architecture first
    for key in weight_map.keys():
        if ".layers." in key and ".self_attn." in key:
            layer_prefix = key.split(".layers.")[0]
            break
    
    if layer_prefix is None:
        raise ValueError("Could not detect layer structure in model weights")
    
    # Now check for MoE architecture
    expert_keys = [k for k in weight_map.keys() if ".mlp.experts." in k]
    shared_expert_keys = [k for k in weight_map.keys() if ".mlp.shared_experts." in k]
    gate_keys = [k for k in weight_map.keys() if ".mlp.gate.weight" in k]
    
    if expert_keys or shared_expert_keys or gate_keys:
        is_moe = True
        
        # Try to extract expert count from expert keys
        if expert_keys:
            # Find all expert indices
            expert_indices = set()
            for key in expert_keys:
                parts = key.split(".mlp.experts.")
                if len(parts) > 1:
                    expert_part = parts[1].split(".")[0]
                    try:
                        expert_indices.add(int(expert_part))
                    except ValueError:
                        pass
            
            if expert_indices:
                expert_count = max(expert_indices) + 1
                print(f"Detected {expert_count} individual experts from expert keys")
        
        # If no individual experts found, check for shared experts
        if expert_count == 0 and shared_expert_keys:
            expert_count = len(shared_expert_keys)  # Approximate
            print(f"Detected shared experts (count estimated: {expert_count})")
        
        # If still no experts found but we have gate, assume MoE with unknown count
        if expert_count == 0 and gate_keys:
            expert_count = 8  # Default assumption
            print(f"Detected expert gate but couldn't count experts, assuming {expert_count}")
    
    print(f"Detected layer prefix: {layer_prefix}")
    print(f"MoE architecture: {is_moe}")
    if is_moe:
        print(f"Total experts per layer: {expert_count}")
    
    # Build a map of which keys in which shards need modification
    shard_modifications = {}  # shard_file -> [(key, layer, measurement, scale, sparsity)]
    
    for layer, measurement, scale, sparsity in marching_orders:
        # Build the key patterns for this layer
        o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
        
        if is_moe:
            # For MoE models, we need to handle multiple down_proj patterns:
            # 1. Shared experts down_proj
            # 2. Individual expert down_proj (optional - usually too many to modify all)
            # 3. Expert routing gate
            shared_down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.shared_experts.down_proj.weight"
            gate_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.gate.weight"  # Expert routing gate
            
            # Find keys that match
            for key, shard_file in weight_map.items():
                if key == o_proj_pattern or key == shared_down_proj_pattern or key == gate_proj_pattern:
                    if shard_file not in shard_modifications:
                        shard_modifications[shard_file] = []
                    shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
            
            # Optionally ablate individual experts
            if ablate_individual_experts and expert_subset is None:
                expert_subset = list(range(expert_count))  # Ablate all experts
            
            if ablate_individual_experts and expert_subset:
                for expert_idx in expert_subset:
                    if expert_idx >= expert_count:
                        print(f"Warning: Expert {expert_idx} not found (max: {expert_count-1})")
                        continue
                    
                    expert_down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.experts.{expert_idx}.down_proj.weight"
                    expert_gate_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"
                    expert_up_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.experts.{expert_idx}.up_proj.weight"
                    
                    for key, shard_file in weight_map.items():
                        if key in [expert_down_proj_pattern, expert_gate_proj_pattern, expert_up_proj_pattern]:
                            if shard_file not in shard_modifications:
                                shard_modifications[shard_file] = []
                            shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
        else:
            # Standard transformer architecture
            down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.down_proj.weight"
            
            # Find keys that match
            for key, shard_file in weight_map.items():
                if key == o_proj_pattern or key == down_proj_pattern:
                    if shard_file not in shard_modifications:
                        shard_modifications[shard_file] = []
                    shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
    
    print(f"\nWill modify {len(shard_modifications)} shards out of {len(set(weight_map.values()))} total")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Process each shard
    all_shards = sorted(set(weight_map.values()))
    
    for shard_file in tqdm(all_shards, desc="Processing shards"):
        shard_path = model_dir / shard_file
        
        if shard_file in shard_modifications:
            print(f"\nLoading and modifying {shard_file}...")
            
            # Load the entire shard
            state_dict = load_file(str(shard_path))
            
            # Apply all modifications for this shard
            for key, layer, measurement, scale, sparsity in shard_modifications[shard_file]:
                if key in state_dict:
                    print(f"  Modifying layer {layer}: {key}")
                    
                    try:
                        # Compute refusal direction on-the-fly
                        refusal_dir = measures[f'refuse_{measurement}'].float()
                        harmless_dir = measures[f'harmless_{layer}'].float()
                        
                        # Debug info
                        print(f"    Refusal dir shape: {refusal_dir.shape}")
                        print(f"    Weight shape: {state_dict[key].shape}")
                        
                        # Normalize harmless direction
                        harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
                        
                        # Project and subtract to refine refusal direction
                        projection_scalar = refusal_dir @ harmless_normalized
                        refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
                        refusal_dir = refined_refusal_dir.to(precision)
                        
                        # Apply sparsity
                        if sparsity > 0.0:
                            refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
                        
                        # Normalize
                        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
                        
                        # Determine if this is an expert gate for MoE handling
                        is_expert_gate = is_moe and ".mlp.gate.weight" in key
                        
                        # Apply modification with MoE awareness
                        if is_moe:
                            modified_weight = modify_tensor_norm_preserved_moe(
                                state_dict[key],
                                refusal_dir,
                                scale,
                                is_expert_gate=is_expert_gate,
                            ).contiguous()
                            
                            # Check if modification was actually applied
                            if modified_weight.shape != state_dict[key].shape:
                                print(f"    Warning: Shape changed during modification!")
                            else:
                                state_dict[key] = modified_weight
                                print(f"    Successfully modified {key}")
                        else:
                            state_dict[key] = modify_tensor_norm_preserved(
                                state_dict[key],
                                refusal_dir,
                                scale,
                            ).contiguous()
                            print(f"    Successfully modified {key}")
                        
                    except Exception as e:
                        print(f"    ERROR modifying {key}: {str(e)}")
                        print(f"    Skipping this weight and continuing...")
                        continue
                    
                    # Clean up
                    del refusal_dir, harmless_dir, harmless_normalized, refined_refusal_dir
                    gc.collect()
            
            # Save modified shard
            print(f"  Saving {shard_file}...")
            save_file(state_dict, f"{output_path}/{shard_file}")
            
            # Clean up
            del state_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        else:
            # Just copy unmodified shards (no need to load)
            shutil.copy(str(shard_path), f"{output_path}/{shard_file}")
    
    # Copy the index file
    print("\nCopying configuration files...")
    shutil.copy(str(index_path), f"{output_path}/model.safetensors.index.json")
    
    # Copy all config files that exist
    config_files = [
        "config.json", 
        "tokenizer_config.json", 
        "tokenizer.json",
        "special_tokens_map.json", 
        "generation_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "preprocessor_config.json",
        "chat_template.json"
    ]
    
    for file in config_files:
        try:
            src_path = cached_file(model_name, file)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, f"{output_path}/{file}")
        except Exception:
            # File doesn't exist, skip it
            pass
    
    print(f"\nModified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient sharded ablation script using YAML configuration."
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to a YAML configuration file',
    )
    
    parser.add_argument(
        '--ablate-individual-experts',
        action='store_true',
        help='Ablate individual experts in MoE models (can be very memory intensive)',
    )
    
    parser.add_argument(
        '--expert-subset',
        type=str,
        help='Comma-separated list of expert indices to ablate (e.g., "0,1,2,3")',
    )
    
    args = parser.parse_args()
    
    # Parse expert subset if provided
    expert_subset = None
    if args.expert_subset:
        try:
            expert_subset = [int(x.strip()) for x in args.expert_subset.split(',')]
        except ValueError:
            print("Error: Expert subset must be comma-separated integers")
            return
    
    # Load YAML configuration
    with open(args.file_path, 'r') as file:
        ydata = yaml.safe_load(file)
    
    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    output_dir = ydata.get("output")
    ablations = ydata.get("ablate")
    
    # MoE-specific options
    ablate_individual_experts = ydata.get("ablate_individual_experts", False)
    if args.ablate_individual_experts:
        ablate_individual_experts = True
    
    expert_subset_config = ydata.get("expert_subset", None)
    if expert_subset_config is not None:
        expert_subset = expert_subset_config
    
    print("=" * 60)
    print("SHARDED ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output directory: {output_dir}")
    print(f"Number of ablations: {len(ablations)}")
    print(f"Ablate individual experts: {ablate_individual_experts}")
    if expert_subset:
        print(f"Expert subset: {expert_subset}")
    print("=" * 60)
    
    # Load measurements
    print(f"\nLoading measurements from {measurement_file}...")
    measures = torch.load(measurement_file)
    print(f"Loaded {len(measures)} measurements")
    
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
    
    print("\nAblation orders:")
    for layer, measurement, scale, sparsity in orders:
        print(f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity}")
    
    # Perform sharded ablation
    print("\n" + "=" * 60)
    print("STARTING ABLATION")
    print("=" * 60)
    ablate_by_layers_sharded(
        model_name=model_name,
        measures=measures,
        marching_orders=orders,
        output_path=output_dir,
        ablate_individual_experts=ablate_individual_experts,
        expert_subset=expert_subset,
    )
    
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
