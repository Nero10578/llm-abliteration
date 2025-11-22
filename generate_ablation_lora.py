import argparse
import torch
import os
from safetensors.torch import save_file
import json
import yaml
from tqdm import tqdm

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

def calculate_lora_weights(
    weight_name: str,
    original_weight_shape: tuple,
    refusal_dir: torch.Tensor,
    scale: float,
    lora_rank: int = 1
):
    """
    Calculate LoRA A and B matrices to approximate the ablation.
    Ablation: W_new = W - scale * outer(projection, refusal)
    LoRA: W_new = W + A @ B
    
    We want A @ B = -scale * outer(projection, refusal)
    
    Note: The 'projection' depends on the original weight W.
    Since we don't want to load the full model weights here, we make a simplification:
    The ablation logic in sharded_ablate.py computes projection = W_direction @ refusal.
    
    If we want to create a LoRA adapter *without* access to the base model weights,
    we strictly speaking cannot, because the ablation direction depends on W.
    
    However, if we assume the user provides the 'refusal' direction which is in the 
    activation space (hidden_size), we can construct a LoRA that projects *out* 
    this direction.
    
    Actually, the standard ablation method (Soups/Abliteration) defines the refusal direction
    in the *residual stream* (activation space).
    
    For a linear layer W (in_features -> out_features):
    y = xW^T
    
    We want to remove the refusal component from the output y.
    y_new = y - (y @ refusal) * refusal^T
    
    This is equivalent to modifying W:
    W_new = W - W @ outer(refusal, refusal)
    
    This is a rank-1 update:
    W_new = W + (W @ refusal) @ (-refusal^T)
    
    Here:
    B = -refusal^T  (shape: 1 x in_features)
    A = W @ refusal (shape: out_features x 1)
    
    CRITICAL ISSUE: We need W to compute A.
    Without loading the base model weights, we cannot compute the exact ablation LoRA.
    
    Workaround:
    1. The user must run this script on a machine with access to the base model weights (or at least the specific layers).
    2. Or, we accept that we need to load the weights.
    
    Given the user's environment has the model (implied by sharded_ablate.py usage), 
    we will assume we can load the necessary shards.
    """
    pass

def main():
    parser = argparse.ArgumentParser(description="Generate LoRA adapter for model abliteration.")
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to base model (for calculating projections)')
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    measurements_path = config['measurements']
    output_dir = config.get('output', 'ablation_lora')
    ablations = config['ablate']
    
    print(f"Loading measurements from {measurements_path}...")
    measures = torch.load(measurements_path)
    
    # Prepare LoRA config
    lora_config = {
        "peft_type": "LORA",
        "r": 1,  # Rank 1 is sufficient for single-direction ablation
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [],
        "modules_to_save": None,
    }
    
    from transformers import AutoModelForCausalLM
    # We only need to load weights, not the full model structure if we are careful, 
    # but using AutoModel is safest to get correct layer mapping.
    # To save memory, we can try to load layer by layer or use safetensors directly if we know the mapping.
    # Given the complexity, let's use safetensors directly with the index.
    
    from transformers.utils import cached_file
    index_path = cached_file(args.model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    
    # Identify target layers
    # We target: self_attn.o_proj, mlp.down_proj (if exists), mlp.gate (if exists)
    # Note: For MoE, mlp.experts...down_proj is hard to target with standard LoRA in vLLM.
    # We will focus on o_proj and gate (router) which are standard Linear layers.
    
    targets = set()
    for item in ablations:
        layer_idx = item['layer']
        targets.add(f"model.layers.{layer_idx}.self_attn.o_proj")
        # targets.add(f"model.layers.{layer_idx}.mlp.gate") # Router - NOT SUPPORTED by vLLM LoRA
        # Add shared experts down_proj if possible?
        # Usually shared experts are standard MLP, so targetable.
        targets.add(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj")
        
        # Add all expert down_proj layers
        # We don't know exactly how many experts there are without loading config,
        # but we can infer from the weight map later.
        # For now, we add a prefix to check against.
        
    # Filter targets that actually exist in the weight map
    # We need to find which shard contains which weight
    
    lora_weights = {}
    target_modules_list = set()
    
    from safetensors.torch import load_file
    
    # Group targets by shard to minimize loading
    shard_to_targets = {}
    for weight_name, shard in weight_map.items():
        # Check if this weight corresponds to a target module
        # weight_name looks like: model.layers.0.self_attn.o_proj.weight
        module_name = weight_name.replace(".weight", "")
        
        is_target = False
        if module_name in targets:
            is_target = True
        else:
            # Check for expert down_proj
            # Pattern: model.layers.X.mlp.experts.Y.down_proj
            parts = module_name.split('.')
            if len(parts) >= 6 and parts[3] == 'mlp' and parts[4] == 'experts' and parts[-1] == 'down_proj':
                try:
                    layer_idx = int(parts[2])
                    # Check if this layer is in our ablation list
                    if any(x['layer'] == layer_idx for x in ablations):
                        is_target = True
                except:
                    pass
        
        if is_target:
            if shard not in shard_to_targets:
                shard_to_targets[shard] = []
            shard_to_targets[shard].append(module_name)
            
    print(f"Found {len(shard_to_targets)} shards containing target weights.")
    
    for shard, modules in tqdm(shard_to_targets.items(), desc="Processing shards"):
        shard_path = os.path.join(args.model_path, shard)
        state_dict = load_file(shard_path)
        
        for module_name in modules:
            weight_key = f"{module_name}.weight"
            if weight_key not in state_dict:
                continue
                
            # Find ablation config for this layer
            # module_name: model.layers.X.yyy
            try:
                layer_idx = int(module_name.split('.')[2])
            except:
                continue
                
            ablation_cfg = next((x for x in ablations if x['layer'] == layer_idx), None)
            if not ablation_cfg:
                continue
                
            # Get measurements
            refusal_dir = measures[f"refuse_{ablation_cfg['measurement']}"].float()
            harmless_dir = measures[f"harmless_{layer_idx}"].float()
            
            # Orthogonalize
            harmless_norm = torch.nn.functional.normalize(harmless_dir, dim=0)
            proj = refusal_dir @ harmless_norm
            refusal_dir = refusal_dir - proj * harmless_norm
            
            # Sparsify
            if ablation_cfg['sparsity'] > 0:
                refusal_dir = magnitude_sparsify(refusal_dir, ablation_cfg['sparsity'])
                
            refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
            
            # Calculate LoRA weights
            # W_new = W - scale * outer(projection, refusal)
            # projection = W @ refusal
            # Update = -scale * (W @ refusal) @ refusal^T
            # LoRA = A @ B
            # Let B = refusal^T (1, in_features)
            # Let A = -scale * (W @ refusal) (out_features, 1)
            
            W = state_dict[weight_key].float()
            # W shape: [out_features, in_features]
            # refusal shape: [in_features] (or [hidden_size])
            
            # Check shapes
            scale = float(ablation_cfg['scale'])
            
            if W.shape[1] == refusal_dir.shape[0]:
                # Case 1: Input dimension matches refusal (e.g. Gate, Up, Q, K, V)
                # We remove refusal from the input projection.
                # W_new = W - scale * (W @ u) @ u.T
                # A = -scale * (W @ u)
                # B = u.T
                
                A = -scale * (W @ refusal_dir) # [out_features]
                A = A.unsqueeze(1) # [out_features, 1]
                B = refusal_dir.unsqueeze(0) # [1, in_features]
                
            elif W.shape[0] == refusal_dir.shape[0]:
                # Case 2: Output dimension matches refusal (e.g. Down, O_proj)
                # We remove refusal from the output generation.
                # W_new = W - scale * u @ (u.T @ W)
                # A = -scale * u
                # B = u.T @ W
                
                A = -scale * refusal_dir.unsqueeze(1) # [out_features, 1]
                B = refusal_dir.unsqueeze(0) @ W # [1, in_features]
                
            else:
                # Mismatch
                print(f"Skipping {module_name}: shape mismatch {W.shape} vs {refusal_dir.shape}")
                continue
            
            # Save to lora_weights
            # PEFT naming convention:
            # base_model.model.layers.X.self_attn.o_proj.lora_A.weight
            # base_model.model.layers.X.self_attn.o_proj.lora_B.weight
            
            suffix = module_name.replace("model.", "") # layers.X...
            target_modules_list.add(module_name.split('.')[-1])
            
            lora_weights[f"base_model.model.{suffix}.lora_A.weight"] = A.contiguous()
            lora_weights[f"base_model.model.{suffix}.lora_B.weight"] = B.contiguous()
            
            del W, A, B
            
        del state_dict
        torch.cuda.empty_cache()

    # Save LoRA adapter
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config
    lora_config['target_modules'] = list(target_modules_list)
    
    print(f"Saving LoRA adapter to {output_dir}...")
    save_file(lora_weights, os.path.join(output_dir, "adapter_model.safetensors"))
    
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(lora_config, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()