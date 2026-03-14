"""
Refusal Circuit Scanner (Fast Version with KL Divergence) - A "brain scanner" for identifying refusal circuits in LLMs.

This version applies abliteration on-the-fly during inference, eliminating the need to save/load
models for each configuration. Much faster than the original version.

It evaluates capability preservation by calculating the KL Divergence of the ablated model
against the original base model using a single forward pass, providing high accuracy and speed.

Supports multi-GPU parallelization for near-linear speedup.

Usage:
    python refusal_circuit_scanner_fast.py -m <model> -o <output_dir> --start <i> --end <j>
    
For a full sweep (generates heatmaps):
    python refusal_circuit_scanner_fast.py -m <model> -o <output_dir> --sweep
    
For multi-GPU sweep:
    python refusal_circuit_scanner_fast.py -m <model> -o <output_dir> --sweep --num-gpus 4
"""

import argparse
import gc
import json
import os
import re
import torch
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data import load_data
from utils.device import clear_device_cache, get_preferred_device, synchronize_device


def get_best_source_layer(measures: dict) -> int:
    """Auto-detect the layer with the best refusal signal quality."""
    best_source = None
    best_quality = -1
    
    for key in measures.keys():
        if key.startswith('refuse_'):
            layer_num = int(key.split('_')[1])
            
            # Get the measurements for this layer
            refusal_dir = measures[f'refuse_{layer_num}']
            harmful_mean = measures.get(f'harmful_{layer_num}')
            harmless_mean = measures.get(f'harmless_{layer_num}')
            
            if harmful_mean is None or harmless_mean is None:
                continue
            
            # Calculate signal quality
            harmful_norm = harmful_mean.norm().item()
            harmless_norm = harmless_mean.norm().item()
            refusal_norm = refusal_dir.norm().item()
            
            # Signal-to-noise ratio
            snr = refusal_norm / max(harmful_norm, harmless_norm)
            
            # Cosine similarity between harmful and harmless
            cos_sim = torch.nn.functional.cosine_similarity(
                harmful_mean.float(), harmless_mean.float(), dim=0
            ).item()
            
            # Refusal purity ratio
            harmless_normalized = harmless_mean / harmless_mean.norm()
            
            # Fast path for CUDA/MPS, fallback to avoid CPU-copy on XPU
            if refusal_dir.device.type == "xpu":
                projection_scalar = torch.sum(refusal_dir * harmless_normalized)
            else:
                projection_scalar = refusal_dir @ harmless_normalized
                
            projection = projection_scalar * harmless_normalized
            refusal_orth = refusal_dir - projection
            if refusal_dir.norm() > 0:
                purity_ratio = refusal_orth.norm() / refusal_dir.norm()
            else:
                purity_ratio = 0
            
            # Signal quality
            quality = snr * (1 - cos_sim) * purity_ratio
            
            if quality > best_quality:
                best_quality = quality
                best_source = layer_num
    
    if best_source is None:
        raise ValueError("No refusal measurements found")
        
    return best_source


def apply_ablation_to_model(
    model,
    measures: dict,
    start_layer: int,
    end_layer: int,
    source_layer: int,
    norm_preserve: bool = True,
    projected: bool = True,
    scale: float = 1.0,
    sparsity: float = 0.0,
    verbose: bool = True,
):
    """
    Apply abliteration to model weights in-place (on-the-fly).
    """
    from sharded_ablate import modify_tensor, modify_tensor_norm_preserved, magnitude_sparsify
    
    if hasattr(model, 'language_model'):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, 'language_model'):
            layer_base = layer_base.language_model
    
    for layer_idx in range(start_layer, end_layer + 1):
        layer = layer_base.layers[layer_idx]
        
        target_device = None
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            target_device = layer.self_attn.o_proj.weight.device
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
            target_device = layer.mlp.down_proj.weight.device
        elif hasattr(layer, 'ffn') and hasattr(layer.ffn, 'down_proj'):
            target_device = layer.ffn.down_proj.weight.device
            
        if target_device is None:
            continue
        
        refusal_dir = measures[f'refuse_{source_layer}'].float().to(target_device)
        harmless_dir = measures[f'harmless_{layer_idx}'].float().to(target_device)
        
        if projected:
            harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
            if refusal_dir.device.type == "xpu":
                projection_scalar = torch.sum(refusal_dir * harmless_normalized)
            else:
                projection_scalar = refusal_dir @ harmless_normalized
                
            refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
            refusal_dir = refined_refusal_dir
            del harmless_normalized, refined_refusal_dir
        
        if sparsity > 0.0:
            refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
        
        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
        
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            with torch.no_grad():
                if norm_preserve:
                    modified_weight = modify_tensor_norm_preserved(
                        layer.self_attn.o_proj.weight, refusal_dir, scale,
                    )
                else:
                    modified_weight = modify_tensor(
                        layer.self_attn.o_proj.weight, refusal_dir, scale,
                    )
                layer.self_attn.o_proj.weight.copy_(modified_weight)
        
        mlp_block = None
        if hasattr(layer, 'mlp'):
            mlp_block = layer.mlp
        elif hasattr(layer, 'ffn'):
            mlp_block = layer.ffn
        
        if mlp_block is not None and hasattr(mlp_block, 'down_proj'):
            with torch.no_grad():
                if norm_preserve:
                    modified_weight = modify_tensor_norm_preserved(
                        mlp_block.down_proj.weight, refusal_dir, scale,
                    )
                else:
                    modified_weight = modify_tensor(
                        mlp_block.down_proj.weight, refusal_dir, scale,
                    )
                mlp_block.down_proj.weight.copy_(modified_weight)
        
        del refusal_dir, harmless_dir
    
    if verbose:
        print(f"Applied abliteration to layers {start_layer}-{end_layer} using measurement from layer {source_layer} with scale {scale}")


def pre_tokenize_prompts(tokenizer, prompts, batch_size=8):
    """Pre-tokenize prompts into batches to save time during evaluation."""
    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in batch
        ]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        inputs = tokenizer(formatted, padding=True, return_tensors="pt")
        batches.append(inputs)
    return batches


def precompute_base_logits(model, harmless_batches):
    """Run once before the sweep to get the 'gold standard' distributions for KL Divergence."""
    model.eval()
    base_logits = []
    
    with torch.no_grad():
        for inputs in harmless_batches:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Only need logits for the final token in the prompt
            next_token_logits = outputs.logits[:, -1, :]
            # Convert to probabilities (targets for KL Div)
            base_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Move to CPU to save VRAM
            base_logits.append(base_probs.cpu()) 
            
    return base_logits


def calculate_refusal_and_kl(model, tokenizer, harmful_batches, harmless_batches, base_harmless_probs, max_tokens=50):
    """
    Calculate refusal rate via generation, and capability via KL Divergence against base model.
    Lower is better for both metrics.
    """
    model.eval()
    refusal_pattern = re.compile(
        r"i cannot|i'm unable|i am unable|i can't|i am not able|i'm not able|cannot|unable|"
        r"not appropriate|inappropriate|against my guidelines|against guidelines|violates|"
        r"harmful|dangerous|illegal|unethical|i don't|i won't|i will not"
    )
    
    # 1. Test harmful prompts for refusal rate (generation required)
    refusals = 0
    total_harmful = sum(batch['input_ids'].shape[0] for batch in harmful_batches)
    
    for inputs in harmful_batches:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        for response in responses:
            if refusal_pattern.search(response.lower()):
                refusals += 1
    
    refusal_rate = (refusals / total_harmful) * 100 if total_harmful > 0 else 0
    
    # 2. Test harmless prompts for KL Divergence (single forward pass only, much faster)
    kl_scores = []
    
    with torch.no_grad():
        for i, inputs in enumerate(harmless_batches):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            ablated_logits = outputs.logits[:, -1, :]
            
            ablated_log_probs = torch.nn.functional.log_softmax(ablated_logits, dim=-1)
            target_probs = base_harmless_probs[i].to(model.device)
            
            kl_div = torch.nn.functional.kl_div(ablated_log_probs, target_probs, reduction='batchmean')
            kl_scores.append(kl_div.item())
    
    mean_kl_div = sum(kl_scores) / len(kl_scores) if kl_scores else 0
    
    return refusal_rate, mean_kl_div


def get_model_state_backup(model, start_layer, end_layer):
    """Get a backup of model weights for the specified layers in CPU memory."""
    state = {}
    if hasattr(model, 'language_model'):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, 'language_model'):
            layer_base = layer_base.language_model
    
    for idx in range(start_layer, end_layer + 1):
        layer = layer_base.layers[idx]
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            state[f'layer_{idx}_self_attn_o_proj'] = layer.self_attn.o_proj.weight.data.cpu().clone()
        if hasattr(layer, 'linear_attn') and hasattr(layer.linear_attn, 'out_proj'):
            state[f'layer_{idx}_linear_attn_out_proj'] = layer.linear_attn.out_proj.weight.data.cpu().clone()
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
            state[f'layer_{idx}_mlp_down_proj'] = layer.mlp.down_proj.weight.data.cpu().clone()
        elif hasattr(layer, 'ffn') and hasattr(layer.ffn, 'down_proj'):
            state[f'layer_{idx}_ffn_down_proj'] = layer.ffn.down_proj.weight.data.cpu().clone()
    
    return state


def restore_model_state(model, state, verbose: bool = True):
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
                    layer_base.layers[layer_idx].self_attn.o_proj.weight.copy_(weight.to(layer_base.layers[layer_idx].self_attn.o_proj.weight.device))
        elif 'linear_attn_out_proj' in key:
            layer_idx = int(key.split('_')[1])
            if hasattr(layer_base.layers[layer_idx], 'linear_attn') and hasattr(layer_base.layers[layer_idx].linear_attn, 'out_proj'):
                with torch.no_grad():
                    layer_base.layers[layer_idx].linear_attn.out_proj.weight.copy_(weight.to(layer_base.layers[layer_idx].linear_attn.out_proj.weight.device))
        elif 'mlp_down_proj' in key:
            layer_idx = int(key.split('_')[1])
            if hasattr(layer_base.layers[layer_idx], 'mlp') and hasattr(layer_base.layers[layer_idx].mlp, 'down_proj'):
                with torch.no_grad():
                    layer_base.layers[layer_idx].mlp.down_proj.weight.copy_(weight.to(layer_base.layers[layer_idx].mlp.down_proj.weight.device))
        elif 'ffn_down_proj' in key:
            layer_idx = int(key.split('_')[1])
            if hasattr(layer_base.layers[layer_idx], 'ffn') and hasattr(layer_base.layers[layer_idx].ffn, 'down_proj'):
                with torch.no_grad():
                    layer_base.layers[layer_idx].ffn.down_proj.weight.copy_(weight.to(layer_base.layers[layer_idx].ffn.down_proj.weight.device))
    
    if verbose:
        print("Restored model to original state")


def run_single_scan(
    model,
    tokenizer,
    measures: dict,
    start_layer: int,
    end_layer: int,
    harmful_batches: list,
    harmless_batches: list,
    norm_preserve: bool = True,
    projected: bool = True,
    max_tokens: int = 50,
    scale: float = 1.0,
    source_layer: int = None,
) -> dict:
    """Run a single scan configuration and return results."""
    print(f"\n=== Scanning configuration ({start_layer}, {end_layer}) ===")

    print("Pre-computing base model logits...")
    base_harmless_probs = precompute_base_logits(model, harmless_batches)

    apply_ablation_to_model(
        model=model,
        measures=measures,
        start_layer=start_layer,
        end_layer=end_layer,
        norm_preserve=norm_preserve,
        projected=projected,
        scale=scale,
        source_layer=source_layer,
        verbose=True
    )

    print("Evaluating refusal removal and KL Divergence...")
    refusal_rate, kl_divergence = calculate_refusal_and_kl(
        model, tokenizer, harmful_batches, harmless_batches, base_harmless_probs, max_tokens=max_tokens
    )

    print(f"Refusal rate: {refusal_rate:.2f}%")
    print(f"KL Divergence: {kl_divergence:.4f}")

    combined_score = refusal_rate + (kl_divergence * 100)

    return {
        "start_layer": start_layer,
        "end_layer": end_layer,
        "refusal_rate": refusal_rate,
        "kl_divergence": kl_divergence,
        "combined_score": combined_score,
    }


def run_config_batch_worker(args):
    """Worker function to run a batch of configurations on a specific GPU."""
    (config_batch, model_path, measurements_path, harmful_batches,
     harmless_batches, gpu_id, norm_preserve, projected, max_tokens,
     flash_attn, scale, source_layer) = args
    
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    import transformers
    transformers.logging.set_verbosity_error()
    
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_type = "xpu"
        device = f"xpu:{gpu_id}"
        torch.xpu.set_device(gpu_id)
    else:
        device_type = "cuda"
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    
    measures = torch.load(measurements_path, map_location=device)
    attn_impl = "flash_attention_2" if flash_attn else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding=True)
    
    # Pre-compute baseline for this worker GPU
    base_harmless_probs = precompute_base_logits(model, harmless_batches)
    
    results = []
    
    with tqdm(config_batch, desc=f"GPU {gpu_id}", position=gpu_id, leave=True) as pbar:
        for start, end in pbar:
            state = get_model_state_backup(model, start, end)
            
            apply_ablation_to_model(
                model=model, measures=measures, start_layer=start, end_layer=end,
                norm_preserve=norm_preserve, projected=projected, scale=scale,
                source_layer=source_layer, verbose=False
            )
            
            refusal_rate, kl_divergence = calculate_refusal_and_kl(
                model, tokenizer, harmful_batches, harmless_batches, base_harmless_probs, max_tokens=max_tokens
            )
            
            combined_score = refusal_rate + (kl_divergence * 100)
            
            log_msg = f"[GPU {gpu_id}] Abliterated layers {start:>2}-{end:<2} | Refusal: {refusal_rate:>5.1f}% | KL Div: {kl_divergence:.4f}"
            tqdm.write(log_msg)
            
            pbar.set_postfix({
                "cfg": f"{start}-{end}",
                "refusal": f"{refusal_rate:.1f}%",
                "kl": f"{kl_divergence:.4f}"
            })
            
            results.append({
                "start_layer": start,
                "end_layer": end,
                "refusal_rate": refusal_rate,
                "kl_divergence": kl_divergence,
                "combined_score": combined_score,
            })
            
            restore_model_state(model, state, verbose=False)
    
    del model, tokenizer, measures, base_harmless_probs
    if device_type == "xpu":
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()
    
    return results


def run_parallel_sweep(
    model_path: str,
    measurements_path: str,
    harmful_batches: list,
    harmless_batches: list,
    output_dir: str,
    num_layers: int,
    num_gpus: int,
    norm_preserve: bool = True,
    projected: bool = True,
    max_tokens: int = 50,
    flash_attn: bool = False,
    scale: float = 1.0,
    source_layer: int = None,
) -> dict:
    """Run a full sweep across multiple GPUs in parallel."""
    all_configs = []
    for start in range(num_layers):
        for end in range(start + 1, num_layers):
            all_configs.append((start, end))
    
    total_configs = len(all_configs)
    print(f"Total configurations: {total_configs}")
    print(f"Distributing across {num_gpus} GPUs...\n")
    
    configs_per_gpu = (total_configs + num_gpus - 1) // num_gpus
    gpu_configs = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * configs_per_gpu
        end_idx = min(start_idx + configs_per_gpu, total_configs)
        gpu_configs.append(all_configs[start_idx:end_idx])
    
    worker_args = [
        (
            gpu_configs[gpu_id], model_path, measurements_path, harmful_batches, harmless_batches,
            gpu_id, norm_preserve, projected, max_tokens, flash_attn, scale, source_layer,
        )
        for gpu_id in range(num_gpus)
    ]
    
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=num_gpus) as pool:
        all_results = pool.map(run_config_batch_worker, worker_args)
    
    print("\n" * num_gpus)
    
    results = {}
    for gpu_results in all_results:
        for result in gpu_results:
            results[(result["start_layer"], result["end_layer"])] = result
    
    json_results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
    with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nCompleted {total_configs} configurations")
    return results


def run_full_sweep(
    model,
    tokenizer,
    measures: dict,
    harmful_batches: list,
    harmless_batches: list,
    output_dir: str,
    num_layers: int,
    norm_preserve: bool = True,
    projected: bool = True,
    max_tokens: int = 50,
    scale: float = 1.0,
    source_layer: int = None,
) -> dict:
    """Run a full sweep of all layer configurations."""
    results = {}
    
    print("Pre-computing base model logits for KL Divergence...")
    base_harmless_probs = precompute_base_logits(model, harmless_batches)
    
    total_configs = num_layers * (num_layers - 1) // 2
    print(f"Running full sweep: {total_configs} configurations\n")
    
    with tqdm(total=total_configs, desc="Full Sweep") as pbar:
        for start in range(num_layers):
            for end in range(start + 1, num_layers):
                state = get_model_state_backup(model, start, end)
                
                apply_ablation_to_model(
                    model=model, measures=measures, start_layer=start, end_layer=end,
                    norm_preserve=norm_preserve, projected=projected, scale=scale,
                    source_layer=source_layer, verbose=False
                )
                
                refusal_rate, kl_divergence = calculate_refusal_and_kl(
                    model, tokenizer, harmful_batches, harmless_batches, base_harmless_probs, max_tokens=max_tokens
                )
                
                combined_score = refusal_rate + (kl_divergence * 100)
                
                log_msg = f"Abliterated layers {start:>2}-{end:<2} | Refusal: {refusal_rate:>5.1f}% | KL Div: {kl_divergence:.4f}"
                tqdm.write(log_msg)
                
                results[(start, end)] = {
                    "start_layer": start,
                    "end_layer": end,
                    "refusal_rate": refusal_rate,
                    "kl_divergence": kl_divergence,
                    "combined_score": combined_score,
                }
                
                restore_model_state(model, state, verbose=False)
                
                pbar.set_postfix({
                    "cfg": f"{start}-{end}",
                    "refusal": f"{refusal_rate:.1f}%",
                    "kl": f"{kl_divergence:.4f}"
                })
                pbar.update(1)
                
                json_results = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
                with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
                    json.dump(json_results, f, indent=2)
    
    return results


def generate_heatmap_visualization(results: dict, output_dir: str, num_layers: int):
    """Generate heatmap visualization from sweep results."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    refusal_matrix = np.full((num_layers, num_layers), np.nan)
    kl_matrix = np.full((num_layers, num_layers), np.nan)
    combined_matrix = np.full((num_layers, num_layers), np.nan)
    
    for (start, end), result in results.items():
        refusal_matrix[start, end] = result["refusal_rate"]
        kl_matrix[start, end] = result["kl_divergence"]
        combined_matrix[start, end] = result["combined_score"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Refusal rate heatmap (Lower is better)
    im1 = axes[0].imshow(refusal_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_title('Refusal Rate (Lower is Better)')
    axes[0].set_xlabel('End Layer (j)')
    axes[0].set_ylabel('Start Layer (i)')
    plt.colorbar(im1, ax=axes[0])
    
    # KL Divergence heatmap (Lower is better -> reverse colormap compared to old score)
    im2 = axes[1].imshow(kl_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_title('KL Divergence (Lower is Better)')
    axes[1].set_xlabel('End Layer (j)')
    axes[1].set_ylabel('Start Layer (i)')
    plt.colorbar(im2, ax=axes[1])
    
    # Combined score heatmap (Lower is better)
    im3 = axes[2].imshow(combined_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[2].set_title('Combined Score (Lower is Better)')
    axes[2].set_xlabel('End Layer (j)')
    axes[2].set_ylabel('Start Layer (i)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "refusal_circuit_heatmap.png"), dpi=300)
    print(f"Saved heatmap to {os.path.join(output_dir, 'refusal_circuit_heatmap.png')}")
    
    # Find optimal configuration
    best_config = min(results.items(), key=lambda x: x[1]["combined_score"])
    print(f"\n{'='*60}")
    print(f"OPTIMAL CONFIGURATION FOUND")
    print(f"{'='*60}")
    print(f"Layers: {best_config[0][0]} to {best_config[0][1]}")
    print(f"Refusal rate: {best_config[1]['refusal_rate']:.2f}%")
    print(f"KL Divergence: {best_config[1]['kl_divergence']:.4f}")
    print(f"Combined score: {best_config[1]['combined_score']:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Refusal Circuit Scanner (Fast) - Identify refusal circuits in LLMs using KL Divergence"
    )
    
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--measurements", type=str, required=True, help="Path to measurements file from measure.py")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for results")
    parser.add_argument("--data-harmful", type=str, default=None, help="Harmful prompts file")
    parser.add_argument("--data-harmless", type=str, default=None, help="Harmless prompts file")
    parser.add_argument("--start", type=int, default=None, help="Start layer for single scan")
    parser.add_argument("--end", type=int, default=None, help="End layer for single scan")
    parser.add_argument("--sweep", action="store_true", help="Run full sweep of all layer configurations")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers in model")
    parser.add_argument("--normpreserve", action="store_true", default=True, help="Use norm-preserving ablation")
    parser.add_argument("--projected", action="store_true", default=True, help="Use projected ablation")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate for refusal test")
    parser.add_argument("--flash-attn", action="store_true", default=False, help="Use Flash Attention 2")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for parallel sweep (default: 1)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for ablation (default: 1.0)")
    parser.add_argument("--source-layer", type=int, default=None, help="Layer to use as refusal direction source (default: auto-detect highest layer)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading measurements from {args.measurements}...")
    measures = torch.load(args.measurements)
    num_layers = measures.get("layers", args.num_layers)
    
    if num_layers is None:
        raise ValueError("Could not determine number of layers. Specify --num-layers")
    
    print(f"Model has {num_layers} layers")
    
    if args.data_harmful:
        harmful_prompts = load_data(args.data_harmful)
    else:
        harmful_prompts = load_data("./data/harmful.parquet")
    
    if args.data_harmless:
        harmless_prompts = load_data(args.data_harmless)
    else:
        harmless_prompts = load_data("./data/harmless.parquet")
    
    harmful_prompts = harmful_prompts[:100]
    harmless_prompts = harmless_prompts[:100]
    
    print(f"Using {len(harmful_prompts)} harmful prompts and {len(harmless_prompts)} harmless prompts")
    
    device = get_preferred_device()
    print(f"Using device: {device}")
    
    attn_impl = "flash_attention_2" if args.flash_attn and device == "cuda" else None
    if attn_impl:
        print("Using Flash Attention 2 for faster inference")
    
    print(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding=True)
    
    print("Pre-tokenizing prompts...")
    harmful_batches = pre_tokenize_prompts(tokenizer, harmful_prompts, args.batch_size)
    harmless_batches = pre_tokenize_prompts(tokenizer, harmless_prompts, args.batch_size)
    
    if args.source_layer is not None:
        source_layer = args.source_layer
        print(f"Using specified source layer: {source_layer}")
    else:
        print("Auto-detecting best source layer...")
        source_layer = get_best_source_layer(measures)
        print(f"Auto-detected best source layer: {source_layer}")
    
    if args.sweep and args.num_gpus > 1:
        print(f"\n{'='*60}")
        print("STARTING FULL SWEEP")
        print(f"{'='*60}")
        
        results = run_parallel_sweep(
            model_path=args.model,
            measurements_path=args.measurements,
            harmful_batches=harmful_batches,
            harmless_batches=harmless_batches,
            output_dir=args.output,
            num_layers=num_layers,
            num_gpus=args.num_gpus,
            norm_preserve=args.normpreserve,
            projected=args.projected,
            max_tokens=args.max_tokens,
            flash_attn=args.flash_attn,
            scale=args.scale,
            source_layer=source_layer,
        )
        
        generate_heatmap_visualization(results, args.output, num_layers)
    else:
        print(f"Loading model {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation=attn_impl,
        )
        print("Model loaded successfully")
        
        if args.sweep:
            print(f"\n{'='*60}")
            print("STARTING FULL SWEEP")
            print(f"{'='*60}")
            
            results = run_full_sweep(
                model=model,
                tokenizer=tokenizer,
                measures=measures,
                harmful_batches=harmful_batches,
                harmless_batches=harmless_batches,
                output_dir=args.output,
                num_layers=num_layers,
                norm_preserve=args.normpreserve,
                projected=args.projected,
                max_tokens=args.max_tokens,
                scale=args.scale,
                source_layer=source_layer,
            )
            
            generate_heatmap_visualization(results, args.output, num_layers)
            
        elif args.start is not None and args.end is not None:
            result = run_single_scan(
                model=model,
                tokenizer=tokenizer,
                measures=measures,
                start_layer=args.start,
                end_layer=args.end,
                harmful_batches=harmful_batches,
                harmless_batches=harmless_batches,
                norm_preserve=args.normpreserve,
                projected=args.projected,
                max_tokens=args.max_tokens,
                scale=args.scale,
                source_layer=source_layer,
            )
            
            print(f"\n{'='*60}")
            print("SCAN RESULTS")
            print(f"{'='*60}")
            print(f"Layers: {result['start_layer']} to {result['end_layer']}")
            print(f"Refusal rate: {result['refusal_rate']:.2f}%")
            print(f"KL Divergence: {result['kl_divergence']:.4f}")
            print(f"Combined score: {result['combined_score']:.2f}")
            print(f"{'='*60}")
            
            with open(os.path.join(args.output, "scan_result.json"), "w") as f:
                json.dump(result, f, indent=2)
        
        else:
            print("Error: Specify either --sweep or both --start and --end")
            parser.print_help()


if __name__ == "__main__":
    main()