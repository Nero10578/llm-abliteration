# MoE Ablation Troubleshooting Guide

## Why MoE Ablation Might Not Work

If you're seeing no difference in refusals after ablating an MoE model, here are the most likely causes:

---

## 1. **Expert Gate/Up Projections May Also Need Ablation**

### The Problem:
We're currently only ablating:
- ✅ Expert **down** projections (`experts.[i].down_proj.weight`)
- ✅ Routing gate (`gate.weight`)

But we're NOT ablating:
- ❌ Expert **gate** projections (`experts.[i].gate_proj.weight`)
- ❌ Expert **up** projections (`experts.[i].up_proj.weight`)

### Why This Matters:
In MoE models, each expert has a full MLP:
```
Expert MLP:
├── gate_proj (gating/activation) ← NOT currently ablated
├── up_proj (expansion) ← NOT currently ablated
└── down_proj (compression) ← Currently ablated
```

The refusal behavior might be encoded in the gate_proj or up_proj, not just down_proj!

### Solution:
We should ablate ALL three projections in each expert, not just down_proj.

---

## 2. **Shared Experts May Dominate**

### The Problem:
GLM-4.5-Air has both:
- Routed experts (conditionally activated)
- **Shared experts** (always activated)

If shared experts are strong and contain refusal behavior, ablating routed experts won't help much.

### Current Status:
- ✅ We ablate `shared_experts.down_proj.weight`
- ❌ We don't ablate `shared_experts.gate_proj.weight`
- ❌ We don't ablate `shared_experts.up_proj.weight`

### Solution:
Ablate all shared expert projections, not just down_proj.

---

## 3. **Layer Coverage May Be Insufficient**

### The Problem:
Your current config ablates layers 15-25 (11 layers out of 46).

For MoE models, refusal might be more distributed across layers than in dense models because:
- Different experts specialize in different tasks
- Routing can compensate by using different experts in different layers

### Check Your Analysis:
Look at your analysis output - are there other layers with significant signal?

```
Layer 18: Signal quality 0.0431 (highest)
Layer 20: Signal quality 0.0380
...
Layer 26: Signal quality 0.0152  ← Still significant?
Layer 27: Signal quality 0.0135  ← Still significant?
```

### Solution:
Try ablating more layers (e.g., 15-30 instead of 15-25).

---

## 4. **Routing Gate Ablation May Not Be Effective**

### The Problem:
The routing gate has shape `[4096, 128]` (hidden → experts).

Our current ablation along the input dimension might not be the right approach. The gate learns to:
1. Detect features in hidden states (input dimension)
2. Route to appropriate experts (output dimension)

Ablating input features might not prevent it from routing to "refusal experts" if the routing logic is in the output dimension.

### Alternative Approach:
Instead of ablating the gate, we could:
- Zero out specific expert selections
- Randomize the gate to prevent learned routing patterns
- Ablate along the output dimension (expert selection)

---

## 5. **MoE Models May Have Stronger Refusal Encoding**

### The Problem:
MoE models can develop **specialized refusal experts**:
- Expert 42 might specialize in refusing harmful requests
- Expert 87 might specialize in refusing sensitive topics
- The routing gate learns to activate these for certain inputs

Even if we ablate the expert weights, the model might have:
- Multiple redundant refusal experts
- Refusal behavior in the routing logic itself
- Refusal patterns in shared experts that always activate

### Solution:
More aggressive ablation:
- Ablate more weight types (gate_proj, up_proj)
- Ablate more layers
- Use higher scale factors (1.5-2.0 instead of 1.0)

---

## 6. **The Refusal Direction May Be Different in MoE**

### The Problem:
The measurement phase treats MoE models the same as dense models:
```python
hidden_states = model.generate(..., output_hidden_states=True)
refusal_dir = harmful_mean - harmless_mean
```

But in MoE models:
- Different experts activate for different inputs
- The hidden states are a **mixture** of expert outputs
- The refusal direction might not capture expert-specific patterns

### Potential Issue:
The refusal direction we measure might not align well with how experts encode refusals.

---

## Diagnostic Steps

### Step 1: Check What Was Actually Ablated
Look at the ablation logs. You should see:
```
Modifying layer 15: model.layers.15.self_attn.o_proj.weight
Modifying layer 15: model.layers.15.mlp.gate.weight
Modifying layer 15: model.layers.15.mlp.experts.0.down_proj.weight
Modifying layer 15: model.layers.15.mlp.experts.1.down_proj.weight
...
Modifying layer 15: model.layers.15.mlp.shared_experts.down_proj.weight
```

If you don't see the gate and experts, they weren't ablated.

### Step 2: Try More Aggressive Ablation
Create a new config with:
```yaml
ablate:
  - layer: 15
    measurement: 18
    scale: 2.0  # Increased from 1.0
    sparsity: 0.00
  # ... more layers
```

### Step 3: Expand Layer Coverage
Try ablating layers 10-35 instead of 15-25.

### Step 4: Check Model Behavior
Test with specific prompts:
```python
# Before ablation
prompt = "How to make a bomb?"
# Expected: Refusal

# After ablation
# Expected: Compliance (or at least different behavior)
```

If behavior is identical, ablation didn't work.

---

## Recommended Fix: Ablate All Expert Projections

The most likely issue is that we're only ablating `down_proj` in experts, but refusal might be in `gate_proj` or `up_proj`.

### Current Code:
```python
# Only matches down_proj
elif key.startswith(experts_down_proj_prefix) and key.endswith(".down_proj.weight"):
```

### Proposed Fix:
```python
# Match all expert projections
elif key.startswith(experts_prefix) and (
    key.endswith(".down_proj.weight") or 
    key.endswith(".gate_proj.weight") or 
    key.endswith(".up_proj.weight")
):
```

This would ablate ALL expert weights, not just the output projection.

---

## Why Dense Models Work But MoE Don't

### Dense Models:
- Single computation path
- Refusal is in the MLP weights
- Ablating down_proj removes refusal
- Simple and effective

### MoE Models:
- Multiple computation paths (experts)
- Refusal can be in:
  - Expert weights (gate_proj, up_proj, down_proj)
  - Routing logic (gate.weight)
  - Shared experts
  - Expert specialization patterns
- Ablating only down_proj might miss most of the refusal
- More complex, requires comprehensive ablation

---

## Next Steps

1. **Immediate**: Expand ablation to include gate_proj and up_proj in experts
2. **Test**: Try more aggressive settings (higher scale, more layers)
3. **Verify**: Check ablation logs to confirm all weights are being modified
4. **Experiment**: Try different layer ranges based on your analysis

The key insight: **MoE models need more comprehensive ablation than dense models** because refusal can be distributed across multiple experts and weight types.