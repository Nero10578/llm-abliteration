# MoE Ablation: What Gets Modified and What Doesn't

## GLM-4.5-Air Architecture Overview

Your model has **46 layers** (layers 0-45), each containing:

### Standard Components (in ALL layers):
- **Attention mechanism**:
  - `q_proj.weight` + `q_proj.bias` (query projection)
  - `k_proj.weight` + `k_proj.bias` (key projection)
  - `v_proj.weight` + `v_proj.bias` (value projection)
  - `o_proj.weight` (output projection) ← **ABLATED**

- **Layer normalization**:
  - `input_layernorm.weight`
  - `post_attention_layernorm.weight`

### MoE-Specific Components (in layers with MoE):
- **Routing gate**: `mlp.gate.weight` ← **ABLATED** (NEW!)
- **Standard MLP** (if present):
  - `mlp.gate_proj.weight`
  - `mlp.up_proj.weight`
  - `mlp.down_proj.weight` ← **ABLATED**

- **Expert networks** (multiple experts per layer):
  - `mlp.experts.[0-N].gate_proj.weight`
  - `mlp.experts.[0-N].up_proj.weight`
  - `mlp.experts.[0-N].down_proj.weight` ← **ABLATED**

- **Shared experts**:
  - `mlp.shared_experts.gate_proj.weight`
  - `mlp.shared_experts.up_proj.weight`
  - `mlp.shared_experts.down_proj.weight` ← **ABLATED**

---

## What Gets Ablated (Modified)

Based on your YAML configuration (e.g., `glm4-air-optimized.yml`), the code ablates:

### For Each Layer Listed in Your YAML:

**Example: If your YAML specifies layers 15-25:**

#### Layer 15:
- ✅ `model.layers.15.self_attn.o_proj.weight`
- ✅ `model.layers.15.mlp.gate.weight` (routing gate)
- ✅ `model.layers.15.mlp.down_proj.weight` (if exists)
- ✅ `model.layers.15.mlp.experts.0.down_proj.weight`
- ✅ `model.layers.15.mlp.experts.1.down_proj.weight`
- ✅ ... (all expert down projections)
- ✅ `model.layers.15.mlp.experts.N.down_proj.weight`
- ✅ `model.layers.15.mlp.shared_experts.down_proj.weight`

#### Layer 16-25:
Same pattern as Layer 15

---

## What Does NOT Get Ablated

### 1. Layers NOT in Your YAML Configuration
If your YAML only specifies layers 15-25, then:
- ❌ Layers 0-14: **NOT ablated** (completely untouched)
- ❌ Layers 26-45: **NOT ablated** (completely untouched)

### 2. Weight Types Never Ablated (in ANY layer)
Even in ablated layers, these weights are **NEVER** modified:
- ❌ Query projections (`q_proj.weight`, `q_proj.bias`)
- ❌ Key projections (`k_proj.weight`, `k_proj.bias`)
- ❌ Value projections (`v_proj.weight`, `v_proj.bias`)
- ❌ Layer normalization (`input_layernorm.weight`, `post_attention_layernorm.weight`)
- ❌ Expert gate/up projections (`mlp.experts.[i].gate_proj.weight`, `mlp.experts.[i].up_proj.weight`)
- ❌ Shared expert gate/up projections (`mlp.shared_experts.gate_proj.weight`, `mlp.shared_experts.up_proj.weight`)
- ❌ Embedding layer (`model.embed_tokens.weight`)
- ❌ Output head (`lm_head.weight`)
- ❌ Final layer norm (`model.norm.weight`)

---

## Your Specific Configuration

Based on `glm4-air-optimized.yml`, you're ablating **layers 15-25** (11 layers total).

### Ablated Layers: 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25

For each of these 11 layers, the following weights are modified:
1. Attention output projection
2. MoE routing gate (**critical for MoE**)
3. Standard MLP down projection (if exists)
4. All expert down projections (~125 experts per layer based on your architecture)
5. Shared expert down projections

### NOT Ablated: Layers 0-14 and 26-45 (35 layers remain untouched)

---

## Why This Selective Approach?

Your analysis showed that **layers 18-22 have the strongest refusal signal**:
- Layer 18: Signal quality = 0.0431 (highest)
- Layer 20: Signal quality = 0.0380
- Layer 19: Signal quality = 0.0369

By focusing on layers 15-25 (centered around the peak), we:
1. Target the layers where refusal is most concentrated
2. Preserve the model's general capabilities in other layers
3. Avoid over-ablation which could degrade performance

---

## Summary Table

| Component | Layers 0-14 | Layers 15-25 | Layers 26-45 |
|-----------|-------------|--------------|--------------|
| Attention Q/K/V | ❌ Not ablated | ❌ Not ablated | ❌ Not ablated |
| Attention Output | ❌ Not ablated | ✅ **ABLATED** | ❌ Not ablated |
| MoE Routing Gate | ❌ Not ablated | ✅ **ABLATED** | ❌ Not ablated |
| Expert Down Proj | ❌ Not ablated | ✅ **ABLATED** | ❌ Not ablated |
| Shared Expert Down | ❌ Not ablated | ✅ **ABLATED** | ❌ Not ablated |
| Expert Gate/Up | ❌ Not ablated | ❌ Not ablated | ❌ Not ablated |
| Layer Norms | ❌ Not ablated | ❌ Not ablated | ❌ Not ablated |

---

## How to Verify What Was Ablated

After running ablation, you can check the logs. You should see output like:

```
Modifying layer 15: model.layers.15.self_attn.o_proj.weight
Modifying layer 15: model.layers.15.mlp.gate.weight
Modifying layer 15: model.layers.15.mlp.experts.0.down_proj.weight
Modifying layer 15: model.layers.15.mlp.experts.1.down_proj.weight
...
Modifying layer 15: model.layers.15.mlp.shared_experts.down_proj.weight
```

This confirms exactly which weights were modified.