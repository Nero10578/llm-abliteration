import torch
import yaml
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    BitsAndBytesConfig
)
# Import the math logic from your existing script
from sharded_ablate import modify_tensor_norm_preserved, magnitude_sparsify

def apply_ablation(model, config_path):
    print(f"Loading ablation config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    measurements_path = config.get("measurements")
    print(f"Loading measurements from {measurements_path}...")
    measures = torch.load(measurements_path)

    # Locate the layers in the loaded model
    # This logic mirrors measure.py to support various architectures (Llama, Qwen, GLM, etc.)
    layer_base = model.model
    if hasattr(layer_base, "language_model"):
        layer_base = layer_base.language_model
    
    if not hasattr(layer_base, "layers"):
        raise ValueError(f"Could not find layers in model structure. Model base: {type(layer_base)}")
    
    layers = layer_base.layers
    
    print("Applying ablation to model weights in memory...")
    
    # Iterate through the ablation instructions
    for item in config.get("ablate", []):
        layer_idx = int(item['layer'])
        measurement_idx = int(item['measurement'])
        scale = float(item['scale'])
        sparsity = float(item.get('sparsity', 0.0))

        # Get directions
        refusal_dir = measures[f'refuse_{measurement_idx}'].float()
        harmless_dir = measures[f'harmless_{layer_idx}'].float()

        # Refine refusal direction (Project out harmless direction)
        harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
        projection_scalar = refusal_dir @ harmless_normalized
        refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
        
        # Apply sparsity if configured
        if sparsity > 0.0:
            refined_refusal_dir = magnitude_sparsify(refined_refusal_dir, fraction=sparsity)
            
        # Final normalization
        final_refusal_dir = torch.nn.functional.normalize(refined_refusal_dir, dim=-1)

        # Identify target modules to modify in this layer
        # We target modules that write to the residual stream: o_proj and down_proj
        layer_module = layers[layer_idx]
        targets = []

        # Standard Attention Output
        if hasattr(layer_module.self_attn, "o_proj"):
            targets.append(layer_module.self_attn.o_proj)
        
        # Standard MLP Down Projection
        if hasattr(layer_module.mlp, "down_proj"):
            targets.append(layer_module.mlp.down_proj)
            
        # MoE: Experts Down Projection
        if hasattr(layer_module.mlp, "experts"):
            # Iterate through experts (assuming ModuleList)
            for expert in layer_module.mlp.experts:
                if hasattr(expert, "down_proj"):
                    targets.append(expert.down_proj)
                    
        # MoE: Shared Experts
        if hasattr(layer_module.mlp, "shared_experts"):
             if hasattr(layer_module.mlp.shared_experts, "down_proj"):
                targets.append(layer_module.mlp.shared_experts.down_proj)

        # MoE: Gating
        if hasattr(layer_module.mlp, "gate"):
             targets.append(layer_module.mlp.gate)

        # Apply modification to identified targets
        for module in targets:
            # modify_tensor_norm_preserved expects a tensor and returns a tensor
            # We need to update the module's weight parameter
            with torch.no_grad():
                original_device = module.weight.device
                original_dtype = module.weight.dtype
                
                # Apply the math
                new_weight = modify_tensor_norm_preserved(
                    module.weight.data, 
                    final_refusal_dir, 
                    scale_factor=scale
                )
                
                # Update the parameter in-place
                module.weight.data = new_weight.to(original_device).to(original_dtype)

    print("Ablation applied successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to model directory")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to ablation YAML config")
    parser.add_argument("--precision", "-p", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--max-new-tokens", "-n", type=int, default=256, help="Max new tokens to generate")
    
    args = parser.parse_args()

    # 1. Setup Precision and Quantization
    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        precision = torch.float32

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    # 2. Load Model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=precision,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 3. Apply Ablation In-Memory
    apply_ablation(model, args.config)

    # 4. Start Chat Loop
    streamer = TextStreamer(tokenizer)
    conversation = []
    
    print("\nModel ready. Type /clear to clear history, /exit to quit.")
    while True:
        try:
            prompt = input("User> ")
        except EOFError:
            break
            
        if prompt.strip() == "/exit":
            break
        if prompt.strip() == "/clear":
            conversation = []
            print("History cleared.")
            continue
        if not prompt.strip():
            continue

        conversation.append({"role": "user", "content": prompt})
        
        inputs = tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)

        gen = model.generate(
            inputs, 
            streamer=streamer, 
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        
        # Decode and append response to conversation history
        # The streamer prints it, but we need it in history for context
        decoded = tokenizer.batch_decode(
            gen[0][len(inputs[0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": "".join(decoded)})