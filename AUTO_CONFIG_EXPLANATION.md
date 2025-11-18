# How Auto Config Generation Works

## The Selection Process

The script doesn't arbitrarily pick 26 layers. Here's what actually happens:

### Step 1: Analyze All Layers
```python
# Computes signal quality for ALL layers (0-45 for GLM-4.5-Air)
for layer in range(num_layers):
    signal_quality = (1 - cosine_similarity) * signal_to_noise_ratio
```

### Step 2: Rank by Quality
```python
# Sorts all layers by signal quality (best first)
layer_stats.sort(key=lambda x: x['signal_quality'], reverse=True)
```

### Step 3: Apply Selection Criteria

**For MoE models (balanced mode):**
```python
min_quality = 0.012  # Minimum threshold
top_n = 26           # Maximum layers to select

selected = []
for layer in sorted_layers:
    if layer.signal_quality >= 0.012:  # Must meet quality threshold
        selected.append(layer)
        if len(selected) >= 26:  # Stop at maximum
            break
```

**For Dense models (balanced mode):**
```python
min_quality = 0.015  # Higher threshold (more selective)
top_n = 12           # Fewer layers needed
```

## Why Different Numbers?

### The Quality Threshold

**MoE models: min_quality = 0.012**
- Lower threshold because refusal is distributed
- Captures more layers with moderate signal
- Example: Layers with quality 0.012-0.043 all selected

**Dense models: min_quality = 0.015**
- Higher threshold because refusal is concentrated
- Only captures strongest signal layers
- Example: Only layers with quality 0.015+ selected

### The Maximum Count (top_n)

**MoE models: top_n = 26**
- Even if 30 layers meet the quality threshold, only take top 26
- Prevents over-ablation
- Based on empirical success (your GLM-4.5-Air worked with 26)

**Dense models: top_n = 12**
- Even if 15 layers meet threshold, only take top 12
- Sufficient for concentrated refusal patterns

## Example with Your GLM-4.5-Air

Your analysis showed:
```
Layer 0:  quality = 0.0000  ❌ Below threshold (0.012)
Layer 1:  quality = 0.0061  ❌ Below threshold
...
Layer 10: quality = 0.0067  ❌ Below threshold
Layer 11: quality = 0.0063  ❌ Below threshold
Layer 12: quality = 0.0089  ❌ Below threshold
Layer 13: quality = 0.0105  ❌ Below threshold
Layer 14: quality = 0.0119  ❌ Below threshold
Layer 15: quality = 0.0185  ✅ Above threshold - SELECTED
Layer 16: quality = 0.0208  ✅ Above threshold - SELECTED
Layer 17: quality = 0.0233  ✅ Above threshold - SELECTED
Layer 18: quality = 0.0431  ✅ Above threshold - SELECTED (BEST)
Layer 19: quality = 0.0369  ✅ Above threshold - SELECTED
Layer 20: quality = 0.0380  ✅ Above threshold - SELECTED
...
Layer 35: quality = 0.0068  ❌ Below threshold (probably)
...
```

The script would:
1. Find all layers with quality >= 0.012
2. Take the top 26 of those
3. Result: Layers 15-35 (or similar, depending on actual quality values)

## It's NOT Arbitrary!

The "26" is:
1. **A maximum limit** - won't select more than 26 even if 30 qualify
2. **Based on analysis** - only selects layers meeting quality threshold
3. **Empirically validated** - worked for your GLM-4.5-Air
4. **Adjustable** - you can override with `--top-n` parameter

## Modes Explained

### Conservative Mode (MoE)
```python
min_quality = 0.020  # Higher threshold - more selective
top_n = 16           # Fewer layers
```
Result: Only the very best layers (e.g., 16-25)

### Balanced Mode (MoE) - Default
```python
min_quality = 0.012  # Moderate threshold
top_n = 26           # More layers
```
Result: All good layers (e.g., 10-35)

### Aggressive Mode (MoE)
```python
min_quality = 0.008  # Lower threshold - less selective
top_n = 36           # Many layers
```
Result: Even moderate-quality layers (e.g., 5-40)

## Custom Override

You can always override:
```bash
# Select exactly 20 layers with quality >= 0.015
python generate_ablation_config.py -m GLM.refuse -p model --min-quality 0.015 --top-n 20

# Select all layers with quality >= 0.010 (no limit)
python generate_ablation_config.py -m GLM.refuse -p model --min-quality 0.010
```

## Summary

The "26 layers" is NOT arbitrary:
- ✅ Based on signal quality analysis
- ✅ Filtered by minimum quality threshold (0.012)
- ✅ Limited to top 26 to prevent over-ablation
- ✅ Empirically validated on your model
- ✅ Adjustable via parameters

It's a **data-driven default** that worked for GLM-4.5-Air and should work for similar MoE models!