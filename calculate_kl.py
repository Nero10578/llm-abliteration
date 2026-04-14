"""
Simple KL Divergence Calculator for Abliterated Models

This script loads a YAML configuration file, applies the specified abliteration
to a model, and calculates the KL divergence between the original and ablated model.

Usage:
    python calculate_kl.py --config gemma4-31b.yml --output kl_results.json
"""

import argparse
import gc
import json
import os
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data import load_data
from utils.device import clear_device_cache, get_preferred_device, synchronize_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_ablation_to_model(
    model,
    measures: dict,
    layer: int,
    measurement: int,
    scale: float = 1.0,
    sparsity: float = 0.0,
    norm_preserve: bool = True,
    projected: bool = True,
):
    """
    Apply abliteration to a single layer in the model.
    """
    from sharded_ablate import modify_tensor, modify_tensor_norm_preserved, magnitude_sparsify
    
    # Get the model's layer structure
    if hasattr(model, 'language_model'):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, 'language_model'):
            layer_base = layer_base.language_model
    
    layer_idx = layer
    layer = layer_base.layers[layer_idx]
    
    # Find the device of the layer's weights
    target_device = None
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
        target_device = layer.self_attn.o_proj.weight.device
    elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
        target_device = layer.mlp.down_proj.weight.device
    elif hasattr(layer, 'ffn') and hasattr(layer.ffn, 'down_proj'):
        target_device = layer.ffn.down_proj.weight.device
    
    if target_device is None:
        return
    
    # Get refusal direction for this layer and move to target device
    refusal_dir = measures[f'refuse_{measurement}'].float().to(target_device)
    harmless_dir = measures[f'harmless_{layer_idx}'].float().to(target_device)
    
    if projected:
        # Orthogonalize refusal against harmless direction
        harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
        projection_scalar = refusal_dir @ harmless_normalized
        refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
        refusal_dir = refined_refusal_dir
        del harmless_normalized, refined_refusal_dir
    
    # Apply sparsity
    if sparsity > 0.0:
        refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
    
    # Normalize
    refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
    
    # Modify attention output projection (o_proj)
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
        with torch.no_grad():
            if norm_preserve:
                modified_weight = modify_tensor_norm_preserved(
                    layer.self_attn.o_proj.weight,
                    refusal_dir,
                    scale,
                )
            else:
                modified_weight = modify_tensor(
                    layer.self_attn.o_proj.weight,
                    refusal_dir,
                    scale,
                )
            layer.self_attn.o_proj.weight.copy_(modified_weight)
    
    # Modify MLP output projection (down_proj)
    mlp_block = None
    if hasattr(layer, 'mlp'):
        mlp_block = layer.mlp
    elif hasattr(layer, 'ffn'):
        mlp_block = layer.ffn
    
    if mlp_block is not None:
        if hasattr(mlp_block, 'down_proj'):
            with torch.no_grad():
                if norm_preserve:
                    modified_weight = modify_tensor_norm_preserved(
                        mlp_block.down_proj.weight,
                        refusal_dir,
                        scale,
                    )
                else:
                    modified_weight = modify_tensor(
                        mlp_block.down_proj.weight,
                        refusal_dir,
                        scale,
                    )
                mlp_block.down_proj.weight.copy_(modified_weight)
    
    # Clean up
    del refusal_dir, harmless_dir


def calculate_kl_divergence(
    model,
    tokenizer,
    prompts,
    original_logits,
    batch_size=8,
):
    """
    Calculate KL divergence between original and ablated model.
    """
    model.eval()
    
    total_kl = 0.0
    batch_idx = 0
    num_batches = 0
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Calculating KL divergence"):
        batch = prompts[i:i+batch_size]
        
        # Format as chat
        formatted = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            for prompt in batch
        ]
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        inputs = tokenizer(formatted, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits for the last token
            logits = outputs.logits[:, -1, :]
            
            # Get original logits for this batch
            orig_logits = original_logits[batch_idx].to(model.device)
            
            # Convert original to probabilities, and current to log-probabilities
            p_probs = torch.nn.functional.softmax(orig_logits, dim=-1)
            q_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Calculate KL divergence (batchmean already averages over batch dimension)
            kl = torch.nn.functional.kl_div(q_log_probs, p_probs, reduction='batchmean')
            total_kl += kl.item()
        
        batch_idx += 1
        num_batches += 1
    
    # Average over number of batches (not prompts, since batchmean already averages per batch)
    avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
    return avg_kl


def get_model_state_backup(model, layers):
    """Get a backup of model weights for specified layers."""
    state = {}
    if hasattr(model, 'language_model'):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, 'language_model'):
            layer_base = layer_base.language_model
    
    for idx in layers:
        layer = layer_base.layers[idx]
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            state[f'layer_{idx}_self_attn_o_proj'] = layer.self_attn.o_proj.weight.data.cpu().clone()
        
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
            state[f'layer_{idx}_mlp_down_proj'] = layer.mlp.down_proj.weight.data.cpu().clone()
        elif hasattr(layer, 'ffn') and hasattr(layer.ffn, 'down_proj'):
            state[f'layer_{idx}_ffn_down_proj'] = layer.ffn.down_proj.weight.data.cpu().clone()
    
    return state


def restore_model_state(model, state):
    """Restore model weights from saved state."""
    if hasattr(model, 'language_model'):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, 'language_model'):
            layer_base = layer_base.language_model
    
    for key, weight in state.items():
        if 'self_attn_o_proj' in key:
            layer_idx = int(key.split('_')[1])
            if hasattr(layer_base.layers[layer_idx], 'self_attn') and hasattr(layer_base.layers[layer_idx].self_attn, 'o_proj'):
                with torch.no_grad():
                    layer_base.layers[layer_idx].self_attn.o_proj.weight.copy_(
                        weight.to(layer_base.layers[layer_idx].self_attn.o_proj.weight.device)
                    )
        elif 'mlp_down_proj' in key:
            layer_idx = int(key.split('_')[1])
            if hasattr(layer_base.layers[layer_idx], 'mlp') and hasattr(layer_base.layers[layer_idx].mlp, 'down_proj'):
                with torch.no_grad():
                    layer_base.layers[layer_idx].mlp.down_proj.weight.copy_(
                        weight.to(layer_base.layers[layer_idx].mlp.down_proj.weight.device)
                    )
        elif 'ffn_down_proj' in key:
            layer_idx = int(key.split('_')[1])
            if hasattr(layer_base.layers[layer_idx], 'ffn') and hasattr(layer_base.layers[layer_idx].ffn, 'down_proj'):
                with torch.no_grad():
                    layer_base.layers[layer_idx].ffn.down_proj.weight.copy_(
                        weight.to(layer_base.layers[layer_idx].ffn.down_proj.weight.device)
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate KL divergence for abliterated model using config file"
    )
    
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output", "-o", type=str, default="kl_results.json", help="Output JSON file")
    parser.add_argument("--data-harmless", type=str, default=None, help="Harmless prompts file for KL calculation")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts for KL calculation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for KL calculation")
    parser.add_argument("--normpreserve", action="store_true", default=True, help="Use norm-preserving ablation")
    parser.add_argument("--projected", action="store_true", default=True, help="Use projected ablation")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    model_path = config.get("model")
    measurements_path = config.get("measurements")
    ablations = config.get("ablate", [])
    
    print(f"Model: {model_path}")
    print(f"Measurements: {measurements_path}")
    print(f"Number of ablations: {len(ablations)}")
    
    # Load measurements
    print(f"\nLoading measurements from {measurements_path}...")
    measures = torch.load(measurements_path)
    
    # Load harmless prompts for KL calculation
    if args.data_harmless:
        harmless_prompts = load_data(args.data_harmless)
    else:
        harmless_prompts = load_data("./data/harmless.parquet")
    
    # Limit prompts
    harmless_prompts = harmless_prompts[:args.num_prompts]
    print(f"Using {len(harmless_prompts)} prompts for KL calculation")
    
    # Get device
    device = get_preferred_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding=True)
    print("Model loaded successfully")
    
    # Calculate original logits
    print("\nCalculating original logits...")
    original_logits = []
    
    for i in tqdm(range(0, len(harmless_prompts), args.batch_size), desc="Original logits"):
        batch = harmless_prompts[i:i+args.batch_size]
        
        formatted = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            for prompt in batch
        ]
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        inputs = tokenizer(formatted, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Save logits for the last token
            original_logits.append(outputs.logits[:, -1, :].cpu())
    
    # Backup model state
    layers_to_modify = [ablation['layer'] for ablation in ablations]
    print(f"\nBacking up model state for layers {layers_to_modify}...")
    state = get_model_state_backup(model, layers_to_modify)
    
    # Apply abliteration
    print("\nApplying abliteration...")
    for ablation in ablations:
        layer = ablation['layer']
        measurement = ablation.get('measurement', layer)
        scale = ablation.get('scale', 1.0)
        sparsity = ablation.get('sparsity', 0.0)
        
        print(f"  Ablating layer {layer} using measurement from layer {measurement} (scale={scale}, sparsity={sparsity})")
        apply_ablation_to_model(
            model=model,
            measures=measures,
            layer=layer,
            measurement=measurement,
            scale=scale,
            sparsity=sparsity,
            norm_preserve=args.normpreserve,
            projected=args.projected,
        )
    
    # Calculate KL divergence
    print("\nCalculating KL divergence...")
    kl_divergence = calculate_kl_divergence(
        model=model,
        tokenizer=tokenizer,
        prompts=harmless_prompts,
        original_logits=original_logits,
        batch_size=args.batch_size,
    )
    
    print(f"\n{'='*60}")
    print("KL DIVERGENCE RESULTS")
    print(f"{'='*60}")
    print(f"KL Divergence: {kl_divergence:.6f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "config_file": args.config,
        "model": model_path,
        "measurements": measurements_path,
        "num_prompts": len(harmless_prompts),
        "num_ablations": len(ablations),
        "ablated_layers": layers_to_modify,
        "kl_divergence": kl_divergence,
        "norm_preserve": args.normpreserve,
        "projected": args.projected,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Restore model state
    print("\nRestoring original model state...")
    restore_model_state(model, state)
    print("Model restored")
    
    # Cleanup
    del model, tokenizer, measures
    clear_device_cache()
    gc.collect()


if __name__ == "__main__":
    main()
