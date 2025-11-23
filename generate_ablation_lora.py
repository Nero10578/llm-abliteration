import argparse
import torch
import os
import gc
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

def modify_tensor_norm_preserved(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    
    Handles both standard weights (where in_features matches refusal_dir size)
    and routing gates (where in_features doesn't match).
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
        
        # Check if this is a routing gate (different dimensions)
        # W_gpu is [out_features, in_features]
        # For routing gates: in_features = hidden_size, out_features = num_experts
        # For standard weights: in_features = hidden_size
        if W_gpu.shape[1] != refusal_dir_gpu.shape[0]:
            # This is likely a routing gate or other incompatible weight
            # We can only ablate along the input dimension (hidden_size)
            # which is the first dimension after transpose
            if W_gpu.shape[0] == refusal_dir_gpu.shape[0]:
                # Ablate along the first dimension (rows in transposed view)
                refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
                
                # For each output (expert), compute projection and subtract
                W_norm = torch.norm(W_gpu, dim=0, keepdim=True)  # [1, out_features]
                W_direction = torch.nn.functional.normalize(W_gpu, dim=0)  # normalized per output
                
                # Compute projection for each output
                projection = torch.matmul(refusal_normalized, W_direction)  # [out_features]
                
                # Subtract the projection
                W_direction_new = W_direction - scale_factor * torch.outer(refusal_normalized, projection)
                
                # Re-normalize
                W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=0)
                
                # Recombine
                W_modified = W_norm * W_direction_new
            else:
                # Cannot ablate this weight - dimensions don't match
                # Return original weight unchanged
                print(f"    Warning: Skipping weight with incompatible shape {W.shape} (refusal_dir: {refusal_dir.shape})")
                return W.detach().clone()
        else:
            # Standard ablation for compatible weights
            # Normalize refusal direction
            refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

            # Decompose weight matrix
            # W_gpu is [out_features, in_features]
            W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
            W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron
        
            # Apply abliteration to the DIRECTIONAL component
            # Compute dot product of each row with refusal direction
            projection = torch.matmul(W_direction, refusal_normalized)  # [out_features]
            
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
    lora_rank = 32
    lora_config = {
        "peft_type": "LORA",
        "r": lora_rank,
        "lora_alpha": lora_rank, # Usually alpha = rank for full rank updates
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
        # targets.add(f"model.layers.{layer_idx}.mlp.gate") # Router - Now supported via ReplicatedLinear
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
            
            # Calculate LoRA weights using SVD on the delta
            # This ensures we capture the exact transformation (including norm preservation)
            
            W = state_dict[weight_key]
            scale = float(ablation_cfg['scale'])
            
            # Compute modified weight
            W_modified = modify_tensor_norm_preserved(W, refusal_dir, scale)
            
            # Compute delta
            # W_new = W + A @ B
            # Delta = W_new - W = A @ B
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            Delta = (W_modified - W).to(device, dtype=torch.float32)
            
            # SVD decomposition
            # Use svd_lowrank for speed since we only need top k components
            k = lora_rank
            try:
                # svd_lowrank returns U, S, V (not Vh)
                # U: (m, k), S: (k,), V: (n, k)
                # We need Vh: (k, n) -> V.T
                U, S, V = torch.svd_lowrank(Delta, q=k + 10, niter=2)
                Vh = V.T
            except RuntimeError:
                # Fallback to cpu if cuda fails (OOM)
                print(f"  SVD OOM on GPU, falling back to CPU for {module_name}")
                U, S, V = torch.svd_lowrank(Delta.cpu(), q=k + 10, niter=2)
                Vh = V.T
                U = U.to(device)
                S = S.to(device)
                Vh = Vh.to(device)
            
            # Top k components
            # svd_lowrank returns exactly q components, but we want k
            # It might return slightly more or less depending on implementation details/convergence
            # But usually it returns q.
            
            # Ensure we don't exceed available dimensions
            actual_k = min(k, U.shape[1], Vh.shape[0])
            
            u = U[:, :actual_k]
            s = S[:actual_k]
            v = Vh[:actual_k, :]
            
            # Distribute sigma
            # A = u * sqrt(s)
            # B = v * sqrt(s)
            sqrt_s = torch.sqrt(s)
            
            # A: [out, k]
            # B: [k, in]
            A = u * sqrt_s.unsqueeze(0)
            B = v * sqrt_s.unsqueeze(1)
            
            # Transpose B to match LoRA shape [k, in]?
            # Vh is [in, in] (transposed V).
            # v = Vh[:k, :] is [k, in].
            # B should be [k, in].
            # v * sqrt_s.unsqueeze(1) -> [k, in] * [k, 1] -> [k, in]. Correct.
            
            # Save to lora_weights
            suffix = module_name.replace("model.", "") # layers.X...
            target_modules_list.add(module_name.split('.')[-1])
            
            # Use bfloat16 to match model dtype (assuming model is bf16)
            # This avoids the "self and mat2 must have the same dtype" error in vLLM
            dtype = torch.bfloat16
            
            lora_weights[f"base_model.model.{suffix}.lora_A.weight"] = A.to(dtype).contiguous()
            lora_weights[f"base_model.model.{suffix}.lora_B.weight"] = B.to(dtype).contiguous()
            
            del W, W_modified, Delta, U, S, Vh, A, B
            
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