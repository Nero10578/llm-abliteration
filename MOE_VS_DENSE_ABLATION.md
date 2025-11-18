# MoE vs Dense Models: Ablation Differences

## Architecture Comparison

### Dense Models (e.g., Llama, Mistral, Qwen)

```
Layer Structure:
├── Self-Attention
│   ├── q_proj (query)
│   ├── k_proj (key)
│   ├── v_proj (value)
│   └── o_proj (output) ← ABLATED
│
└── MLP (Feed-Forward)
    ├── gate_proj (gating)
    ├── up_proj (expansion)
    └── down_proj (compression) ← ABLATED
```

**Key characteristics:**
- Single MLP per layer
- All tokens processed through the same MLP
- Simple, deterministic computation path

### MoE Models (e.g., GLM-4.5-Air, Mixtral, DeepSeek-V2)

```
Layer Structure:
├── Self-Attention
│   ├── q_proj (query)
│   ├── k_proj (key)
│   ├── v_proj (value)
│   └── o_proj (output) ← ABLATED
│
└── MoE MLP
    ├── Routing Gate ← ABLATED (NEW!)
    │   └── Decides which experts to use
    │
    ├── Expert 0
    │   ├── gate_proj
    │   ├── up_proj
    │   └── down_proj ← ABLATED
    │
    ├── Expert 1
    │   ├── gate_proj
    │   ├── up_proj
    │   └── down_proj ← ABLATED
    │
    ├── ... (many more experts)
    │
    ├── Expert N
    │   ├── gate_proj
    │   ├── up_proj
    │   └── down_proj ← ABLATED
    │
    └── Shared Experts (optional)
        ├── gate_proj
        ├── up_proj
        └── down_proj ← ABLATED
```

**Key characteristics:**
- Multiple expert MLPs per layer (e.g., 125 experts in GLM-4.5-Air)
- Routing gate selects which experts to activate
- Different tokens can use different experts
- More complex, dynamic computation path

---

## Ablation Differences

### Dense Model Ablation (Traditional)

**What gets ablated per layer:**
1. Attention output projection (`o_proj.weight`)
2. MLP down projection (`down_proj.weight`)

**Total: 2 weight matrices per layer**

**Example for Llama-3.2 (32 layers):**
- If ablating layers 10-25 (16 layers)
- Total weights modified: **16 layers × 2 weights = 32 weight matrices**

### MoE Model Ablation (Enhanced)

**What gets ablated per layer:**
1. Attention output projection (`o_proj.weight`)
2. **Routing gate** (`gate.weight`) ← **NEW & CRITICAL**
3. MLP down projection (`down_proj.weight`, if exists)
4. **Expert 0 down projection** (`experts.0.down_proj.weight`) ← **NEW**
5. **Expert 1 down projection** (`experts.1.down_proj.weight`) ← **NEW**
6. ... (all experts)
7. **Expert N down projection** (`experts.N.down_proj.weight`) ← **NEW**
8. **Shared expert down projection** (`shared_experts.down_proj.weight`) ← **NEW**

**Total: 3+ weight matrices per layer (depends on number of experts)**

**Example for GLM-4.5-Air (46 layers, ~125 experts per layer):**
- If ablating layers 15-25 (11 layers)
- Per layer: 1 (o_proj) + 1 (gate) + 125 (experts) + 1 (shared) = **~128 weights**
- Total weights modified: **11 layers × 128 weights = ~1,408 weight matrices**

---

## Why MoE Requires Special Handling

### 1. The Routing Problem

**Dense models:**
```
Input → MLP → Output
```
Simple path, ablate the MLP output.

**MoE models:**
```
Input → Routing Gate → Select Experts → Combine Outputs → Output
              ↓
        [Expert 0, Expert 5, Expert 12]
```

**Problem:** Even if we ablate all expert weights, the routing gate could still:
- Route "harmful" inputs to specific expert combinations
- Learn refusal patterns in the routing logic itself
- Preserve censorship through expert selection

**Solution:** Ablate the routing gate to prevent refusal-based routing.

### 2. Expert Specialization

MoE models can develop **specialized experts**:
- Some experts might specialize in refusals
- Some experts might handle sensitive topics
- The routing gate learns to activate "safe" experts for potentially harmful queries

**Example scenario:**
```
User: "How to make a bomb?"

Without gate ablation:
Gate → Routes to [Expert 42, Expert 87] (refusal specialists)
Result: "I cannot help with that."

With gate ablation:
Gate → Routes more randomly/uniformly
Result: More likely to provide information
```

### 3. Shared vs. Routed Experts

Some MoE architectures (like GLM-4.5-Air) have:
- **Routed experts:** Conditionally activated based on input
- **Shared experts:** Always activated for all inputs

Both need ablation because:
- Routed experts can specialize in refusals
- Shared experts provide a "baseline" that might include refusal behavior

---

## Code Implementation Differences

### Dense Model Code (Original)

```python
# Only need to match 2 patterns
o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.down_proj.weight"

if key == o_proj_pattern or key == down_proj_pattern:
    # Ablate this weight
```

### MoE Model Code (Enhanced)

```python
# Need to match 5+ patterns
o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.down_proj.weight"
gate_pattern = f"{layer_prefix}.layers.{layer}.mlp.gate.weight"  # NEW
experts_prefix = f"{layer_prefix}.layers.{layer}.mlp.experts."  # NEW
shared_pattern = f"{layer_prefix}.layers.{layer}.mlp.shared_experts.down_proj.weight"  # NEW

if key == o_proj_pattern or key == down_proj_pattern:
    # Ablate standard weights
elif key == gate_pattern:
    # Ablate routing gate (CRITICAL for MoE)
elif key.startswith(experts_prefix) and key.endswith(".down_proj.weight"):
    # Ablate all expert down projections
elif key == shared_pattern:
    # Ablate shared expert down projection
```

---

## Measurement Phase (Same for Both)

**Good news:** The measurement phase is identical for both architectures!

```python
# Works for both dense and MoE models
hidden_states = model.generate(..., output_hidden_states=True)
layer_activations = hidden_states[layer_idx][:, -1, :]
```

**Why it works:**
- We measure at the **layer output** level
- MoE routing happens internally within the layer
- The refusal direction is captured in the final layer activations
- We don't need to know which experts were activated

---

## Performance Implications

### Dense Models
- **Ablation time:** Fast (fewer weights to modify)
- **Memory usage:** Lower (smaller weight matrices)
- **Disk space:** Smaller model size

### MoE Models
- **Ablation time:** Slower (many more weights to modify)
- **Memory usage:** Higher (larger weight matrices, but sharded)
- **Disk space:** Larger model size
- **Benefit:** Better quality/parameter ratio (experts provide specialization)

**Example timing (approximate):**
- Dense model (7B params): ~5-10 minutes
- MoE model (50B params, 8 experts): ~20-40 minutes

---

## Summary Table

| Aspect | Dense Models | MoE Models |
|--------|--------------|------------|
| **Weights per layer** | 2 (o_proj, down_proj) | 3+ (o_proj, gate, experts, shared) |
| **Routing** | None | Dynamic expert selection |
| **Ablation targets** | Output projections only | Output projections + routing gate |
| **Complexity** | Simple | Complex (expert specialization) |
| **Critical addition** | N/A | **Routing gate ablation** |
| **Measurement** | Same | Same |
| **Ablation time** | Faster | Slower |
| **Effectiveness** | Good | Better (if gate is ablated) |

---

## Key Takeaway

**For MoE models, ablating the routing gate is just as important as ablating the expert weights themselves.**

Without gate ablation, the model can preserve refusal behavior through expert selection, even if all expert weights are modified. This is why your initial ablation didn't work - the gate was still routing to refusal patterns!

The updated code now handles this correctly by ablating:
1. Expert outputs (what they compute)
2. **Routing gate (which experts get selected)** ← Critical difference
3. Shared expert outputs (baseline computation)