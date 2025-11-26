# GPT-OSS-20B Abliteration Guide

This guide explains how to use the measure, analyze, and sharded_ablate scripts with the GPT-OSS-20B model architecture.

## Model Architecture Overview

GPT-OSS-20B is a 24-layer transformer model with the following key characteristics:

- **Model Type**: `GptOssForCausalLM`
- **Layers**: 24 transformer layers (indices 0-23)
- **Architecture**: Mixture of Experts (MoE) with expert routing
- **Unique Features**: 
  - `self_attn.sinks` parameter (unique to GPT-OSS)
  - `mlp.router.weight` for expert routing (bias not ablated)
  - `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` for expert projections

## Step 1: Measure the Model

First, run the measurement script to collect refusal direction data:

```bash
python measure.py \
    --model gpt-oss-20b-BF16/ \
    --output gpt-oss-20b-measurements.pt \
    --batch-size 16 \
    --flash-attn
```

### Options for GPT-OSS-20B:

- **Quantization**: Use `--quant-measure 4bit` or `--quant-measure 8bit` if GPU memory is limited
- **Batch Size**: Adjust based on GPU memory (16-32 is typical)
- **Flash Attention**: Use `--flash-attn` for faster inference
- **Custom Data**: Use `--data-harmful` and `--data-harmless` for custom datasets

## Step 2: Analyze the Measurements

Analyze the collected measurements to identify optimal ablation layers:

```bash
python analyze.py gpt-oss-20b-measurements.pt --chart
```

This will:
- Display layer-by-layer analysis of refusal directions
- Show signal-to-noise ratios and cosine similarities
- Generate visualization charts (if `--chart` is specified)

### Key Metrics to Watch:

- **Signal Quality**: Higher values indicate better refusal directions
- **Signal-to-Noise Ratio**: Balance between refusal strength and noise
- **Cosine Similarity**: Lower values between harmful and harmless means are better

## Step 3: Configure Ablation

Edit the `gpt-oss-20b.yml` configuration file based on your analysis:

```yaml
model: "gpt-oss-20b-BF16/"
measurements: "gpt-oss-20b-measurements.pt"
output: "gpt-oss-20b-ablated/"

ablate:
  - layer: 20
    measurement: 20
    scale: 1.0
    sparsity: 1.0
  # Add more layers as needed
```

### Configuration Parameters:

- **layer**: Target layer to ablate (0-23)
- **measurement**: Source layer for refusal direction (can be different)
- **scale**: Ablation strength (1.0 = full, 0.0 = none)
- **sparsity**: Fraction of direction to keep (1.0 = full, 0.5 = top 50%)

## Step 4: Perform Ablation

Run the sharded ablation script:

```bash
python sharded_ablate.py gpt-oss-20b.yml
```

This will:
- Load the model shard by shard (memory efficient)
- Apply ablations to specified layers
- Handle GPT-OSS-20B specific parameter structures
- Save the ablated model to the output directory

## GPT-OSS-20B Specific Considerations

### Parameter Handling

The scripts automatically handle these GPT-OSS-20B specific parameters:

1. **Attention Output Projection**: `model.layers.{N}.self_attn.o_proj.weight`
2. **Expert Down Projections**: `model.layers.{N}.mlp.experts.*.down_proj` (only down_proj, not gate_up_proj)
3. **Router Weights**: `model.layers.{N}.mlp.router.weight`

### Ignored Parameters

The `self_attn.sinks` parameter is unique to GPT-OSS but is not typically ablated as it doesn't contribute to refusal directions in the same way as other parameters.

### Memory Management

GPT-OSS-20B is a large model (20B parameters). The scripts use:
- **Sharded Loading**: Loads one shard at a time
- **Memory Cleanup**: Aggressive GPU memory management
- **Device Auto-detection**: Automatic multi-GPU distribution

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use quantization
2. **Model Loading Errors**: Ensure the model path is correct and accessible
3. **Permission Errors**: Check write permissions for output directory

### Debug Mode:

Add verbose logging by modifying the scripts or using smaller batch sizes to identify issues.

## Best Practices

1. **Start Small**: Test with a few layers first
2. **Monitor GPU Memory**: Use `nvidia-smi` to track usage
3. **Backup Original**: Keep a copy of the original model
4. **Validate Results**: Test the ablated model on sample prompts

## Example Workflow

```bash
# Step 1: Measure
python measure.py --model gpt-oss-20b-BF16/ --output measurements.pt --batch-size 16

# Step 2: Analyze
python analyze.py measurements.pt --chart

# Step 3: Configure (edit gpt-oss-20b.yml based on analysis)

# Step 4: Ablate
python sharded_ablate.py gpt-oss-20b.yml

# Step 5: Test the ablated model
python chat.py --model gpt-oss-20b-ablated/
```

This workflow ensures systematic ablation of the GPT-OSS-20B model with proper validation at each step.