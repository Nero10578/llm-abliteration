# Refusal Circuit Scanner

A "brain scanner" for identifying refusal circuits in LLMs, inspired by David Noel Ng's "LLM Neuroanatomy" article.

## Two Versions Available

### 1. `refusal_circuit_scanner.py` (Original)
- Saves model to disk for each configuration
- Slower but more straightforward
- Useful for debugging or when you want to keep abliterated models

### 2. `refusal_circuit_scanner_fast.py` (Recommended ⚡)
- **Applies abliteration on-the-fly during inference**
- **Much faster** - no disk I/O for saving/loading models
- Loads model once, modifies weights in-memory, evaluates, resets
- **Recommended for most use cases**

**Performance Comparison:**
- Original: ~5-10 minutes per configuration (includes save/load time)
- Fast: ~2-5 minutes per configuration (no save/load overhead)
- For 2,016 configurations: Original ~7-14 days vs Fast ~3-7 days

## Overview

Just as the article discovered that LLMs have a "reasoning cortex" in middle layers organized into functional circuits, this tool helps identify the **"refusal circuit"** - the contiguous block of layers that, when ablated, most effectively removes refusals while preserving model capabilities.

## Key Concepts

### What is a Refusal Circuit?

Based on the article's findings about LLM neuroanatomy:
- **Early layers (0-20)**: Encode input into abstract representations
- **Middle layers (20-50)**: Reasoning cortex - organized into functional circuits
- **Late layers (50-64)**: Decode abstract representations back to output

The refusal circuit is a specific functional circuit within the middle layers that handles safety/refusal mechanisms. Like reasoning circuits, it's **indivisible** - you must ablate the entire circuit for effective refusal removal.

### How It Works

Instead of duplicating layers (as in the article), this tool **ablates layers** to identify which ranges most effectively remove refusals:

1. **Systematic ablation**: Test all possible layer ranges (i, j)
2. **Dual evaluation**: Measure both refusal removal AND capability preservation
3. **Heatmap visualization**: Identify optimal layer ranges visually

## Installation

No additional dependencies needed - uses the same requirements as the main repository.

```shell
pip install -r requirements.txt
```

## How the Fast Version Works

The fast version (`refusal_circuit_scanner_fast.py`) uses a completely different approach:

### Original Version (Slow)
```
For each configuration:
  1. Ablate layers → Save model to disk (~2-3 min)
  2. Load model from disk (~1-2 min)
  3. Evaluate model (~2-5 min)
  4. Delete model (~30 sec)
Total: ~5-10 minutes per configuration
```

### Fast Version (Recommended)
```
Load model ONCE at startup (~2-3 min)

For each configuration:
  1. Save original weights to memory (~1 sec)
  2. Apply abliteration in-place (~10-30 sec)
  3. Evaluate model (~2-5 min)
  4. Restore original weights (~1 sec)
Total: ~2-5 minutes per configuration
```

### Key Optimizations

1. **Single model load**: Model is loaded once at startup, not for each configuration
2. **In-place modification**: Weights are modified directly in memory using `torch.nn.Parameter`
3. **State save/restore**: Original weights are saved once and restored between configurations
4. **No disk I/O**: No saving/loading of model files during scanning

### Technical Details

The fast version uses these key functions:

- **`apply_ablation_to_model()`**: Modifies model weights in-place using PyTorch's `torch.nn.Parameter`
- **`save_model_state()`**: Saves original weights to a temporary file (only once)
- **`restore_model_state()`**: Restores original weights before each new configuration
- **`run_single_scan()`**: Applies abliteration, evaluates, and returns results without disk I/O

## Usage

### Fast Version (Recommended ⚡)

#### Single Configuration Scan
```shell
python refusal_circuit_scanner_fast.py \
    -m <model_path> \
    --measurements <measurements_file> \
    -o <output_dir> \
    --start 30 \
    --end 40
```

#### Full Sweep (Generate Heatmaps)
```shell
python refusal_circuit_scanner_fast.py \
    -m <model_path> \
    --measurements <measurements_file> \
    -o <output_dir> \
    --sweep \
    --num-layers 64
```

### Original Version (Slower)

#### Single Configuration Scan
```shell
python refusal_circuit_scanner.py \
    -m <model_path> \
    --measurements <measurements_file> \
    -o <output_dir> \
    --start 30 \
    --end 40
```

#### Full Sweep (Generate Heatmaps)
```shell
python refusal_circuit_scanner.py \
    -m <model_path> \
    --measurements <measurements_file> \
    -o <output_dir> \
    --sweep \
    --num-layers 64
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model path or HuggingFace ID | Required |
| `--measurements` | Path to measurements from measure.py | Required |
| `-o, --output` | Output directory for results | Required |
| `--data-harmful` | Custom harmful prompts file | ./data/harmful.parquet |
| `--data-harmless` | Custom harmless prompts file | ./data/harmless.parquet |
| `--start` | Start layer for single scan | None |
| `--end` | End layer for single scan | None |
| `--sweep` | Run full sweep of all configurations | False |
| `--num-layers` | Number of layers in model | Auto-detected |
| `--normpreserve` | Use norm-preserving ablation | True |
| `--projected` | Use projected ablation | True |
| `--batch-size` | Batch size for evaluation | 8 |

## Understanding the Output

### Heatmap Visualization

The sweep generates three heatmaps:

1. **Refusal Rate (Lower is Better)**
   - Green regions: Effective refusal removal
   - Red regions: Poor refusal removal

2. **Capability Score (Higher is Better)**
   - Green regions: Preserved model capabilities
   - Red regions: Degraded capabilities

3. **Combined Score (Lower is Better)**
   - Balances refusal removal and capability preservation
   - Green regions: Optimal configurations

### Interpreting the Heatmaps

Based on the article's findings, expect to see:

- **Early layers (0-20)**: Poor refusal removal (these encode input)
- **Middle layers (20-50)**: Complex patterns - look for contiguous green regions
- **Late layers (50-64)**: Mixed results (these decode output)

The optimal configuration will be a **contiguous block** in the middle layers - this is the refusal circuit.

### Example Output

```
============================================================
OPTIMAL CONFIGURATION FOUND
============================================================
Layers: 32 to 38
Refusal rate: 5.23%
Capability score: 0.92
Combined score: -36.77
============================================================
```

This means ablating layers 32-38 (inclusive) provides the best balance of refusal removal and capability preservation.

## Workflow Integration

The refusal circuit scanner fits into the existing abliteration workflow:

### Traditional Workflow
1. Measure directions → 2. Analyze → 3. Guess layers → 4. Abliterate → 5. Test

### Scanner-Enhanced Workflow
1. Measure directions → 2. **Run scanner** → 3. Use optimal layers → 4. Abliterate → 5. Test

### Example Complete Workflow

```shell
# Step 1: Measure directions
python measure.py -m Qwen/Qwen2.5-27B-Instruct -o measurements.pt --projected

# Step 2: Run scanner to find optimal layers (using fast version)
python refusal_circuit_scanner_fast.py \
    -m Qwen/Qwen2.5-27B-Instruct \
    --measurements measurements.pt \
    -o scanner_results \
    --sweep \
    --num-layers 64

# Step 3: Create YAML config using optimal layers from scanner
# (e.g., layers 32-38 from scanner output)

# Step 4: Abliterate using optimal layers
python sharded_ablate.py config.yml --normpreserve --projected

# Step 5: Test the model
python chat.py -m abliterated_model
```

## Expected Results Based on Article

Based on the "LLM Neuroanatomy" article:

1. **Circuit boundaries**: The refusal circuit will have clear boundaries
   - Too few layers: Incomplete circuit, poor refusal removal
   - Too many layers: Includes neighboring circuits, degraded capabilities

2. **Contiguous requirement**: Only contiguous layer ranges work well
   - Non-contiguous ablation: Poor results (like duplicating single layers in the article)

3. **Model variation**: Different models have different neuroanatomy
   - Smaller models: More entangled, less clear circuits
   - Larger models: More differentiated, clearer circuits

## Performance Considerations

### Full Sweep Time

A full sweep of all layer configurations is computationally expensive:

- **64-layer model**: ~2,016 configurations
- **Per configuration (fast version)**: ~2-5 minutes (depends on model size and hardware)
- **Per configuration (original version)**: ~5-10 minutes (includes save/load overhead)
- **Total time (fast version)**: ~3-7 days on single GPU
- **Total time (original version)**: ~7-14 days on single GPU

**The fast version is approximately 2x faster!**

### Optimization Strategies

1. **Coarse sweep first**: Test every 5th layer, then refine
2. **Limited prompts**: Use fewer prompts for faster scanning
3. **Parallel processing**: Run multiple configurations on different GPUs
4. **Incremental approach**: Start with promising ranges from analyze.py

### Recommended Approach

Instead of a full sweep, use a **targeted approach**:

```shell
# Use analyze.py to identify promising regions
python analyze.py measurements.pt -c

# Run scanner on promising regions only (using fast version)
python refusal_circuit_scanner_fast.py \
    -m <model> \
    --measurements measurements.pt \
    -o scanner_results \
    --start 25 \
    --end 45
```

## Comparison with Article's Method

| Aspect | Article (RYS) | Refusal Circuit Scanner |
|--------|---------------|------------------------|
| **Operation** | Duplicate layers (i, j) | Ablate layers (i, j) |
| **Goal** | Improve reasoning | Remove refusals |
| **Probe 1** | Hard math | Refusal removal effectiveness |
| **Probe 2** | Emotional intelligence | Capability preservation |
| **Heatmap Meaning** | Red = better performance | Red = more refusals removed |
| **Circuit Discovery** | Find reasoning circuits | Find refusal circuits |
| **Expected Pattern** | Middle layers improve tasks | Middle layers remove refusals |

## Troubleshooting

### Scanner shows no good configurations

- **Issue**: Model may not have clear refusal circuits
- **Solution**: Try different measurement sources or adjust probe prompts

### Heatmaps are noisy

- **Issue**: Too few prompts or high variance
- **Solution**: Increase prompt count or run multiple times

### Capability score is always low

- **Issue**: Ablation too aggressive or wrong layers
- **Solution**: Try smaller layer ranges or different measurement source

### Scanner takes too long

- **Issue**: Full sweep on large model
- **Solution**: Use targeted approach or reduce prompt count

## Future Enhancements

Potential improvements to the scanner:

1. **Adaptive sampling**: Focus on promising regions
2. **Multi-objective optimization**: Pareto frontier of refusal vs capability
3. **Circuit boundary detection**: Automatic identification of circuit edges
4. **Transfer learning**: Use results from similar models
5. **Real-time visualization**: Interactive heatmap exploration

## Credits

Inspired by:
- [David Noel Ng's "LLM Neuroanatomy" article](https://davidnoelng.com/llm-neuroanatomy)
- Original abliteration research and implementations

## License

Same as the main llm-abliteration repository.
