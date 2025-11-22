import argparse
import torch
from safetensors.torch import load_file
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Verify LoRA adapter structure against base model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter directory")
    args = parser.parse_args()

    print(f"Verifying adapter at {args.adapter_path} against model at {args.model_path}")

    # Load adapter weights
    adapter_file = os.path.join(args.adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        print(f"Error: Adapter file not found at {adapter_file}")
        return

    adapter_weights = load_file(adapter_file)
    print(f"Loaded {len(adapter_weights)} adapter tensors.")

    # Load model index to find weight shapes
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: Model index not found at {index_path}")
        return

    with open(index_path, 'r') as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Helper to get weight shape from shards
    # We cache loaded shards to avoid reloading
    loaded_shards = {}

    def get_base_weight_shape(module_name):
        weight_name = f"{module_name}.weight"
        if weight_name not in weight_map:
            return None
        
        shard_file = weight_map[weight_name]
        if shard_file not in loaded_shards:
            print(f"Loading shard {shard_file}...")
            loaded_shards[shard_file] = load_file(os.path.join(args.model_path, shard_file))
        
        return loaded_shards[shard_file][weight_name].shape

    # Verify each LoRA module
    # LoRA keys: base_model.model.{module}.lora_{A/B}.weight
    
    modules_checked = set()
    
    for key, tensor in adapter_weights.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
            
        # Extract module name
        # key: base_model.model.layers.0.self_attn.o_proj.lora_A.weight
        # module: layers.0.self_attn.o_proj
        parts = key.split('.')
        if parts[0] == "base_model" and parts[1] == "model":
            module_parts = parts[2:-2] # remove base_model.model and lora_X.weight
            module_name = "model." + ".".join(module_parts)
        else:
            print(f"Skipping unknown key format: {key}")
            continue
            
        if module_name in modules_checked:
            continue
            
        # Check if it's a standard layer or FusedMoE expert
        is_expert = "experts" in module_name
        
        if is_expert:
            print(f"\nChecking FusedMoE module: {module_name}")
            # For FusedMoE, we expect stacked weights.
            # module_name might be: model.layers.X.mlp.experts or model.layers.X.mlp.experts.base_layer
            
            # Find corresponding base model layer to check dimensions
            # The base model has 'mlp.experts.weight' (if fused) or 'mlp.experts.0.down_proj.weight' (if not fused on disk)
            # GLM-4-Air usually has separate experts on disk?
            # Let's check one expert to verify dimensions.
            
            # Assuming module_name is 'model.layers.X.mlp.experts' (down_proj)
            # We need to find 'model.layers.X.mlp.experts.0.down_proj.weight'
            
            layer_idx = module_name.split('.')[2]
            
            if "base_layer" in module_name:
                # Gate/Up proj
                # Check 'model.layers.X.mlp.gate_up_proj' ? No, experts have their own gate_up?
                # Or 'model.layers.X.mlp.experts.0.gate_up_proj'?
                # Let's try to find a matching weight in the map
                search_prefix = f"model.layers.{layer_idx}.mlp.experts.0"
                # Try to find a weight starting with this
                found = False
                for w_name in weight_map:
                    if w_name.startswith(search_prefix):
                        # e.g. model.layers.0.mlp.experts.0.gate_proj.weight
                        # or gate_up_proj
                        found = True
                        break
                if not found:
                    print(f"  Warning: Could not find base weights for {module_name} to verify dimensions.")
                    continue
            else:
                # Down proj
                # Check 'model.layers.X.mlp.experts.0.down_proj.weight'
                base_weight_name = f"model.layers.{layer_idx}.mlp.experts.0.down_proj"
                shape = get_base_weight_shape(base_weight_name)
                if shape:
                    # shape is [hidden, intermediate] (PyTorch: out, in)
                    hidden, intermediate = shape
                    print(f"  Base expert down_proj shape: {shape} (hidden={hidden}, intermediate={intermediate})")
                    
                    # lora_A should be [num_experts * rank, intermediate]
                    # lora_B should be [hidden, num_experts * rank]
                    
                    lora_A = adapter_weights[f"{key.replace('lora_B', 'lora_A')}"] if "lora_B" in key else tensor
                    lora_B = adapter_weights[f"{key.replace('lora_A', 'lora_B')}"] if "lora_A" in key else tensor
                    
                    print(f"  lora_A shape: {lora_A.shape}")
                    print(f"  lora_B shape: {lora_B.shape}")
                    
                    rank = 32 # assumed
                    # Check intermediate
                    if lora_A.shape[1] != intermediate:
                        print(f"  ERROR: lora_A input dim {lora_A.shape[1]} != base intermediate {intermediate}")
                    else:
                        print(f"  OK: lora_A input dim matches intermediate.")
                        
                    # Check hidden
                    if lora_B.shape[0] != hidden:
                        print(f"  ERROR: lora_B output dim {lora_B.shape[0]} != base hidden {hidden}")
                    else:
                        print(f"  OK: lora_B output dim matches hidden.")
                        
                    # Check stacking
                    num_experts = lora_A.shape[0] // rank
                    print(f"  Inferred num_experts: {num_experts}")
                    
                else:
                    print(f"  Warning: Could not find base weight {base_weight_name}")

        else:
            print(f"\nChecking Standard module: {module_name}")
            shape = get_base_weight_shape(module_name)
            if shape:
                # shape is [out, in]
                out_dim, in_dim = shape
                print(f"  Base weight shape: {shape} (out={out_dim}, in={in_dim})")
                
                # lora_A: [rank, in]
                # lora_B: [out, rank]
                
                lora_A_key = key.replace("lora_B", "lora_A") if "lora_B" in key else key
                lora_B_key = key.replace("lora_A", "lora_B") if "lora_A" in key else key
                
                lora_A = adapter_weights[lora_A_key]
                lora_B = adapter_weights[lora_B_key]
                
                print(f"  lora_A shape: {lora_A.shape}")
                print(f"  lora_B shape: {lora_B.shape}")
                
                if lora_A.shape[1] != in_dim:
                    print(f"  ERROR: lora_A input dim {lora_A.shape[1]} != base in dim {in_dim}")
                else:
                    print(f"  OK: lora_A input dim matches.")
                    
                if lora_B.shape[0] != out_dim:
                    print(f"  ERROR: lora_B output dim {lora_B.shape[0]} != base out dim {out_dim}")
                else:
                    print(f"  OK: lora_B output dim matches.")
            else:
                print(f"  Warning: Could not find base weight for {module_name}")
                
        modules_checked.add(module_name)

    print("\nVerification complete.")

if __name__ == "__main__":
    main()