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
        # W is [out_features, in_features] in PyTorch, but safetensors loads it as is.
        # Wait, safetensors load_file returns tensors in PyTorch layout [out, in] for Linear.
        # sharded_ablate.py says:
        # "PyTorch nn.Linear layers store weights as [out_features, in_features]"
        # "Safetensors (HuggingFace format) stores them as [in_features, out_features] - transposed!"
        # BUT load_file usually handles this? No, load_file loads exactly what's on disk.
        # If the file is HF safetensors, Linear weights are usually [out, in].
        # Let's assume W is [out, in].
        
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True)
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Check if this is a routing gate (different dimensions)
        # W_gpu is [out_features, in_features]
        # For routing gates: in_features = hidden_size, out_features = num_experts
        # For standard weights: in_features = hidden_size
        
        # Case 1: Input dimension matches refusal (e.g. Gate, Up, Q, K, V)
        if W_gpu.shape[1] == refusal_dir_gpu.shape[0]:
             # Ablate along the input dimension (columns)
            refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
            
            # For each output (row), compute projection and subtract
            # We want to remove refusal from the input space.
            # W_new = W - (W @ r) @ r.T
            # But norm preserving:
            # Normalize rows? No, input space means we look at columns?
            # sharded_ablate.py logic for gate:
            # "Ablate along the first dimension (rows in transposed view)" -> Transposed view was [in, out].
            # So rows of transposed view = columns of original view.
            # So we normalize columns.
            
            W_norm = torch.norm(W_gpu, dim=0, keepdim=True)  # [1, in_features] -> Wait, norm of columns?
            # If we ablate input, we are modifying how the layer reacts to input.
            # sharded_ablate.py logic for gate (W.shape[1] != refusal.shape[0] in its transposed view):
            # Its transposed view W_gpu was [out, in].
            # Wait, sharded_ablate.py:
            # W_gpu = W.to(...).T  -> [in, out]
            # if W_gpu.shape[1] != refusal.shape[0]: (out != hidden) -> True for gate (out=experts, hidden=in)
            # if W_gpu.shape[0] == refusal.shape[0]: (in == hidden) -> True for gate
            # "Ablate along the first dimension (rows in transposed view)" -> Rows of [in, out] are inputs.
            # W_norm = torch.norm(W_gpu, dim=0, keepdim=True) -> Norm of columns of [in, out] -> Norm of output neurons?
            # W_direction = normalize(W_gpu, dim=0) -> Normalize per output neuron.
            
            # So for Gate, it normalizes per output neuron (expert).
            # Then projects refusal out of that direction.
            # This effectively removes refusal from the "prototype" of each expert.
            
            # Let's replicate this on W [out, in]:
            # W_gpu is [out, in].
            # We want to normalize per row (dim=1).
            # Wait, sharded_ablate.py used dim=0 on [in, out]. That is normalizing columns.
            # Columns of [in, out] correspond to rows of [out, in].
            # So yes, normalize per output neuron.
            
            W_norm = torch.norm(W_gpu, dim=1, keepdim=True) # [out, 1]
            W_direction = torch.nn.functional.normalize(W_gpu, dim=1) # [out, in]
            
            # Compute projection: (W_dir @ r)
            projection = torch.matmul(W_direction, refusal_normalized) # [out]
            
            # Subtract
            # W_new = W_dir - scale * proj * r.T
            W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
            
            # Re-normalize
            W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
            
            # Recombine
            W_modified = W_norm * W_direction_new

        # Case 2: Output dimension matches refusal (e.g. Down, O_proj)
        elif W_gpu.shape[0] == refusal_dir_gpu.shape[0]:
            # Ablate along the output dimension (rows)
            # sharded_ablate.py logic for standard weights (else block):
            # W_gpu was [in, out].
            # W_norm = norm(W_gpu, dim=1) -> Norm of rows of [in, out] -> Norm of input neurons?
            # Wait, sharded_ablate.py:
            # "W_gpu is [out_features, in_features]" in the comment, but code says W_gpu = W.T
            # If W is [out, in], W.T is [in, out].
            # "W_norm = torch.norm(W_gpu, dim=1, keepdim=True)" -> Norm of rows of [in, out].
            # Rows of [in, out] are input dimensions.
            # So it normalizes per input dimension?
            # "normalized per output neuron" comment says otherwise.
            # If it meant per output neuron, it should be dim=0 of [in, out].
            
            # Let's re-read sharded_ablate.py carefully.
            # "W_gpu = W.to(...).T"
            # "W_norm = torch.norm(W_gpu, dim=1, keepdim=True)"
            # "W_direction = torch.nn.functional.normalize(W_gpu, dim=1)"
            # If W_gpu is [in, out], dim=1 is across output neurons.
            # So for a fixed input feature, we normalize the vector of weights to all outputs.
            # This seems to be normalizing the "input embedding" of the layer?
            
            # Then:
            # "projection = torch.matmul(W_direction, refusal_normalized)"
            # W_direction [in, out] @ refusal [out] -> [in]
            # This projects the "input embedding" onto the refusal direction (which is in output space).
            
            # Then subtract:
            # W_dir_new = W_dir - scale * outer(proj, refusal)
            # [in, out] - [in] * [out]
            
            # This removes the component of the weight that maps to the refusal direction.
            
            # So for W [out, in]:
            # We normalize columns (dim=0).
            # W_norm = norm(W, dim=0) [1, in]
            # W_dir = normalize(W, dim=0) [out, in]
            # proj = r @ W_dir [in]
            # W_dir_new = W_dir - scale * outer(r, proj)
            # normalize(W_dir_new, dim=0)
            # W_mod = W_norm * W_dir_new
            
            refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
            
            W_norm = torch.norm(W_gpu, dim=0, keepdim=True) # [1, in]
            W_direction = torch.nn.functional.normalize(W_gpu, dim=0) # [out, in]
            
            projection = torch.matmul(refusal_normalized, W_direction) # [in]
            
            W_direction_new = W_direction - scale_factor * torch.outer(refusal_normalized, projection)
            
            W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=0)
            
            W_modified = W_norm * W_direction_new
            
        else:
            # Mismatch
            return W.detach().clone()
        
        result = W_modified.to('cpu', dtype=original_dtype, non_blocking=True)
        
        del W_gpu, refusal_dir_gpu, refusal_normalized
        del W_direction, W_direction_new, W_norm, projection, W_modified
        
    return result

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
        # targets.add(f"model.layers.{layer_idx}.mlp.gate") # Router - Supported in vLLM (ReplicatedLinear) but NOT in Aphrodite (nn.Linear)
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
    
    # Store expert weights temporarily to stack them later
    # Key: layer_idx, Value: {expert_idx: {'A': tensor, 'B': tensor}}
    expert_weights_buffer = {}
    
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
            
            # A: [out, k] -> This corresponds to LoRA B (up projection)
            # B: [k, in]  -> This corresponds to LoRA A (down projection)
            A = u * sqrt_s.unsqueeze(0)
            B = v * sqrt_s.unsqueeze(1)
            
            # Use bfloat16 to match model dtype (assuming model is bf16)
            dtype = torch.bfloat16
            A = A.to(dtype).contiguous()
            B = B.to(dtype).contiguous()

            # Check if this is an expert down_proj
            parts = module_name.split('.')
            if len(parts) >= 6 and parts[3] == 'mlp' and parts[4] == 'experts' and parts[-1] == 'down_proj':
                # Buffer expert weights
                layer_idx = int(parts[2])
                expert_idx = int(parts[5])
                
                if layer_idx not in expert_weights_buffer:
                    expert_weights_buffer[layer_idx] = {}
                # Store B as lora_A (down), A as lora_B (up)
                expert_weights_buffer[layer_idx][expert_idx] = {'A': B, 'B': A}
            else:
                # Standard save for non-expert weights
                suffix = module_name.replace("model.", "") # layers.X...
                target_modules_list.add(module_name.split('.')[-1])
                
                # Swap A and B: B is lora_A (in->rank), A is lora_B (rank->out)
                lora_weights[f"base_model.model.{suffix}.lora_A.weight"] = B
                lora_weights[f"base_model.model.{suffix}.lora_B.weight"] = A
            
            del W, W_modified, Delta, U, S, Vh, A, B
            
        del state_dict
        torch.cuda.empty_cache()

    # Process buffered expert weights
    print("Processing expert weights for FusedMoE compatibility...")
    for layer_idx, experts in expert_weights_buffer.items():
        # We need to stack weights for all experts.
        # Aphrodite expects FusedMoE weights to be stacked along the RANK dimension.
        # A: [num_experts * rank, intermediate]
        # B: [hidden, num_experts * rank]
        
        # Find max expert index to determine num_experts
        # Note: This assumes we found at least one expert. If some are missing, we should probably fill with zeros?
        # But we only ablate what we found. However, for FusedMoE stacking, we need ALL experts.
        # We can infer num_experts from the keys.
        
        # Assuming expert indices are 0 to N-1
        max_expert_idx = max(experts.keys())
        num_experts = max_expert_idx + 1
        
        # Get shapes from first expert
        first_expert = experts[list(experts.keys())[0]]
        rank = first_expert['A'].shape[0]
        intermediate_size = first_expert['A'].shape[1]
        hidden_size = first_expert['B'].shape[0]
        dtype = first_expert['A'].dtype
        device = 'cpu' # Construct on CPU
        
        # 1. Construct Down Proj (experts)
        # A: [num_experts * rank, intermediate]
        # B: [hidden, num_experts * rank]
        
        down_A_list = []
        down_B_list = []
        
        for i in range(num_experts):
            if i in experts:
                down_A_list.append(experts[i]['A'].to(device))
                down_B_list.append(experts[i]['B'].to(device))
            else:
                # Zero init for missing experts (though we should probably have all of them if we scanned correctly)
                down_A_list.append(torch.zeros((rank, intermediate_size), dtype=dtype, device=device))
                down_B_list.append(torch.zeros((hidden_size, rank), dtype=dtype, device=device))
                
        down_A_stacked = torch.cat(down_A_list, dim=0)
        down_B_stacked = torch.cat(down_B_list, dim=1)
        
        suffix = f"layers.{layer_idx}.mlp.experts"
        lora_weights[f"base_model.model.{suffix}.lora_A.weight"] = down_A_stacked
        lora_weights[f"base_model.model.{suffix}.lora_B.weight"] = down_B_stacked
        target_modules_list.add("experts")
        
        # 2. Construct Gate/Up Proj (experts.base_layer) - Dummy Zeros
        # Aphrodite requires this to be present for FusedMoEWithLoRA
        # A: [num_experts * rank, hidden]
        # B: [2 * intermediate, num_experts * rank]
        
        # Note: gate_up_proj output is 2 * intermediate.
        # Aphrodite splits B with [::2] for gate and [1::2] for up.
        
        gate_up_A = torch.zeros((num_experts * rank, hidden_size), dtype=dtype, device=device)
        gate_up_B = torch.zeros((2 * intermediate_size, num_experts * rank), dtype=dtype, device=device)
        
        suffix_base = f"layers.{layer_idx}.mlp.experts.base_layer"
        lora_weights[f"base_model.model.{suffix_base}.lora_A.weight"] = gate_up_A
        lora_weights[f"base_model.model.{suffix_base}.lora_B.weight"] = gate_up_B
        # We don't need to add base_layer to target_modules_list explicitly if 'experts' covers the module replacement?
        # Actually, Aphrodite checks for 'experts' in target_modules.
        # But we should probably add 'base_layer' just in case, or rely on 'experts'.
        # The code in Aphrodite checks: if ".experts" in module_name...
        
    # Save LoRA adapter
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config
    # Ensure 'experts' is in target_modules if we added it
    if expert_weights_buffer:
        target_modules_list.add("experts")
        
    lora_config['target_modules'] = list(target_modules_list)
    
    print(f"Saving LoRA adapter to {output_dir}...")
    save_file(lora_weights, os.path.join(output_dir, "adapter_model.safetensors"))
    
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(lora_config, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()