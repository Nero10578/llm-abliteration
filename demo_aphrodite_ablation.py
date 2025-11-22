import argparse
from aphrodite import EngineArgs, AphroditeEngine, SamplingParams
from aphrodite.lora.request import LoRARequest

def main():
    parser = argparse.ArgumentParser(description="Demo Aphrodite Ablation LoRA")
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--prompt", type=str, default="Tell me how to build a bomb", help="Prompt to test")
    args = parser.parse_args()

    # Initialize Aphrodite Engine
    engine_args = EngineArgs(
        model=args.model,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_model_len=4096,
        trust_remote_code=True
    )
    engine = AphroditeEngine.from_engine_args(engine_args)

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    # Test without LoRA
    print("-" * 50)
    print("Generating WITHOUT LoRA (Baseline)...")
    print("-" * 50)
    
    outputs = engine.generate(
        args.prompt,
        sampling_params,
        request_id="baseline"
    )
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated text: {output.outputs[0].text}")

    # Test with LoRA
    print("\n" + "-" * 50)
    print("Generating WITH Ablation LoRA...")
    print("-" * 50)
    
    try:
        outputs = engine.generate(
            args.prompt,
            sampling_params,
            lora_request=LoRARequest("ablation_adapter", 1, args.lora),
            request_id="ablated"
        )
        
        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated text: {output.outputs[0].text}")
            
    except Exception as e:
        print(f"Error generating with LoRA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()