# Converting Abliteration to LoRA Adapters

## Overview

Instead of creating a full abliterated model, you can convert the abliteration into a **LoRA (Low-Rank Adaptation) adapter**. This offers several advantages:

### Benefits of LoRA Adapters

1. **Much Smaller Size**
   - Full model: 50+ GB
   - LoRA adapter: 100-500 MB (depending on rank)
   - **99% size reduction!**

2. **Reversible**
   - Keep original model unchanged
   - Load/unload abliteration as needed
   - Easy to experiment with different settings

3. **Shareable**
   - Easy to distribute (small file size)
   - Can be uploaded to HuggingFace
   - Others can use without re-abliterating

4. **Combinable**
   - Can combine with other LoRAs
   - Stack multiple adaptations
   - More flexible than full model replacement

5. **Faster Iteration**
   - Quick to create from abliterated model
   - Easy to test different ranks
   - No need to re-abliterate for experiments

---

## How It Works

### The Math

Abliteration modifies weights: `W_new = W_original + ΔW`

LoRA approximates this change as a low-rank decomposition:
```
ΔW ≈ B @ A
```

Where:
- `A` is `[rank, in_features]`
- `B` is `[out_features, rank]`
- `rank` << `min(in_features, out_features)`

### Example

For a weight matrix of shape `[4096, 4096]`:
- **Full change**: 4096 × 4096 = 16.7M parameters
- **LoRA (rank=64)**: (4096 × 64) + (64 × 4096) = 524K parameters
- **Compression**: 97% reduction per weight!

---

## Usage

### Step 1: Create Abliterated Model (as usual)

```bash
# Measure refusal directions
python measure.py -m /path/to/model -o measurements.refuse

# Abliterate the model
python sharded_ablate.py config.yml
```

This creates the abliterated model in the output directory.

### Step 2: Convert to LoRA

```bash
python ablation_to_lora.py \
    --original /path/to/original/model \
    --abliterated /path/to/abliterated/model \
    --output /path/to/lora/adapter \
    --rank 64
```

### Step 3: Use the LoRA Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load original model
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/original/model",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapter (applies abliteration)
model = PeftModel.from_pretrained(
    model,
    "/path/to/lora/adapter"
)

# Use normally
tokenizer = AutoTokenizer.from_pretrained("/path/to/original/model")
inputs = tokenizer("Your prompt here", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
```

---

## Rank Selection Guide

The `rank` parameter controls the quality/size tradeoff:

| Rank | Size | Quality | Use Case |
|------|------|---------|----------|
| 8 | Tiny (~50 MB) | Low | Testing, very limited abliteration |
| 16 | Small (~100 MB) | Moderate | Quick experiments, mild abliteration |
| 32 | Medium (~200 MB) | Good | Balanced approach |
| **64** | Large (~400 MB) | **High** | **Recommended for abliteration** |
| 128 | Very Large (~800 MB) | Very High | Maximum quality, if size isn't a concern |
| 256 | Huge (~1.6 GB) | Near-perfect | Overkill for most cases |

### Recommendation

**Start with rank=64** - it provides excellent quality while keeping size manageable.

If abliteration effect is too weak, try rank=128.
If you need smaller files, try rank=32.

---

## Advanced Usage

### Combining Multiple LoRAs

You can combine abliteration with other LoRAs:

```python
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("base-model")

# Load abliteration LoRA
model = PeftModel.from_pretrained(model, "abliteration-lora")

# Load another LoRA (e.g., instruction tuning)
model.load_adapter("instruction-lora", adapter_name="instruction")

# Use both adapters
model.set_adapter(["default", "instruction"])
```

### Adjusting LoRA Strength

You can scale the LoRA effect:

```python
# Stronger abliteration (1.5x)
model.set_adapter("default", alpha=96)  # 64 * 1.5

# Weaker abliteration (0.5x)
model.set_adapter("default", alpha=32)  # 64 * 0.5
```

### Merging LoRA Back to Full Model

If you want to create a permanent abliterated model from the LoRA:

```python
from peft import PeftModel

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained("original-model")
model = PeftModel.from_pretrained(model, "abliteration-lora")

# Merge LoRA into base model
model = model.merge_and_unload()

# Save as full model
model.save_pretrained("merged-abliterated-model")
```

---

## Comparison: Full Model vs LoRA

### Full Abliterated Model

**Pros:**
- ✅ Exact abliteration (no approximation)
- ✅ Slightly faster inference (no adapter overhead)
- ✅ Works with any inference framework

**Cons:**
- ❌ Very large file size (50+ GB)
- ❌ Requires full model storage
- ❌ Hard to share/distribute
- ❌ Can't easily revert to original

### LoRA Adapter

**Pros:**
- ✅ Tiny file size (100-500 MB)
- ✅ Easy to share/distribute
- ✅ Reversible (keep original model)
- ✅ Combinable with other LoRAs
- ✅ Quick to create and experiment

**Cons:**
- ❌ Slight approximation (usually negligible with rank=64)
- ❌ Tiny inference overhead (usually <1%)
- ❌ Requires PEFT library

---

## Quality Verification

To verify the LoRA quality, compare outputs:

```python
# Test with original model
original_model = AutoModelForCausalLM.from_pretrained("original")
original_output = original_model.generate(...)

# Test with full abliterated model
abliterated_model = AutoModelForCausalLM.from_pretrained("abliterated")
abliterated_output = abliterated_model.generate(...)

# Test with LoRA adapter
lora_model = PeftModel.from_pretrained(original_model, "lora-adapter")
lora_output = lora_model.generate(...)

# Compare: abliterated_output should ≈ lora_output
```

With rank=64, the outputs should be nearly identical.

---

## Troubleshooting

### "LoRA effect is too weak"

**Solution 1:** Increase rank
```bash
python ablation_to_lora.py ... --rank 128
```

**Solution 2:** Increase alpha
```python
model.set_adapter("default", alpha=128)  # 2x strength
```

### "LoRA file is too large"

**Solution:** Decrease rank
```bash
python ablation_to_lora.py ... --rank 32
```

Note: This may reduce abliteration effectiveness.

### "Out of memory when creating LoRA"

**Solution:** The script processes shards one at a time, but if you still run out of memory:
1. Close other applications
2. Use a machine with more RAM
3. Process on CPU instead of GPU

---

## Example Workflow

### Complete Example: GLM-4.5-Air

```bash
# 1. Measure refusal directions
python measure.py \
    -m /home/arli/models/GLM-4.5-Air \
    -o GLM.refuse

# 2. Abliterate the model
python sharded_ablate.py glm4-air-optimized.yml

# 3. Convert to LoRA (much smaller!)
python ablation_to_lora.py \
    --original /home/arli/models/GLM-4.5-Air \
    --abliterated glm4-air-abliterated \
    --output glm4-air-abliteration-lora \
    --rank 64

# 4. Use the LoRA
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    '/home/arli/models/GLM-4.5-Air',
    device_map='auto'
)
model = PeftModel.from_pretrained(model, 'glm4-air-abliteration-lora')
print('Abliteration LoRA loaded successfully!')
"
```

### Size Comparison

- **Original model**: ~50 GB
- **Abliterated model**: ~50 GB
- **LoRA adapter (rank=64)**: ~400 MB

You can delete the abliterated model and just keep the LoRA!

---

## Sharing Your LoRA

### Upload to HuggingFace

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
huggingface-cli upload your-username/glm4-air-abliteration-lora ./glm4-air-abliteration-lora
```

### Others Can Use It

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("THUDM/GLM-4.5-Air")

# Load your LoRA from HuggingFace
model = PeftModel.from_pretrained(model, "your-username/glm4-air-abliteration-lora")
```

---

## Summary

**LoRA conversion is highly recommended** for abliteration because:

1. **99% smaller** than full model
2. **Reversible** - keep original model
3. **Easy to share** - upload to HuggingFace
4. **Flexible** - combine with other LoRAs
5. **Nearly identical quality** with rank=64

The only downside is a tiny approximation, which is negligible for rank ≥ 64.