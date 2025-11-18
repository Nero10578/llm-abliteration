# MoE Ablation: Why It Might Not Work (Deeper Analysis)

## Key Question:
**If dense models work with only down_proj ablation, why doesn't MoE?**

You're right to question this! Let's think more carefully.

---

## Dense Models (What Works)

```
Standard MLP:
Input (4096) 
  → gate_proj (4096 → 14336) [NOT ablated]
  → SiLU activation
  → up_proj (4096 → 14336) [NOT ablated]
  → element-wise multiply
  → down_proj (14336 → 4096) [ABLATED] ✓
  → Output (4096)
```

**Why only down_proj works:**
- The down projection is the **final output** of the MLP
- It aggregates all the information from gate and up projections
- Ablating it removes the refusal direction from the final output
- Gate and up projections don't matter if the output is ablated

---

## MoE Models (Current Hypothesis - May Be Wrong)

```
MoE MLP:
Input (4096)
  → Routing Gate (4096 → 128) [ABLATED]
  → Select top-k experts (e.g., top-2)
  
  For each selected expert:
    → gate_proj (4096 → 10944) [NOW ablated]
    → SiLU activation  
    → up_proj (4096 → 10944) [NOW ablated]
    → element-wise multiply
    → down_proj (10944 → 4096) [ABLATED]
  
  → Weighted combination of expert outputs
  → Output (4096)
```

**Question: Should we ablate gate_proj and up_proj?**

### Argument FOR (current approach):
- Each expert is independent
- Refusal might be in any expert projection
- More comprehensive = more effective

### Argument AGAINST (your point):
- Dense models only need down_proj
- Each expert is just a mini-MLP
- Should work the same way
- We might be over-ablating

---

## The Real Issue Might Be Different

Let me reconsider what could actually be wrong:

### 1. **Routing Gate Ablation Method**

The routing gate has shape `[4096, 128]`:
- Input: hidden states (4096)
- Output: expert scores (128)

**Current ablation:**
- We ablate along the input dimension
- This removes refusal features from the hidden state processing

**Problem:**
- The routing might not use refusal features directly
- It might use other features that correlate with refusals
- Ablating input features might not prevent routing to refusal experts

**Alternative:**
Maybe we shouldn't ablate the gate at all, or ablate it differently?

### 2. **Expert Combination Weights**

After experts compute their outputs, they're combined:
```python
output = sum(weight[i] * expert[i].output for i in selected_experts)
```

The weights come from the routing gate. Even if we ablate expert outputs, the **combination weights** might preserve refusal patterns.

### 3. **Shared Experts Always Active**

GLM-4.5-Air has shared experts that are ALWAYS active:
```python
output = routed_expert_output + shared_expert_output
```

If shared experts contain strong refusal behavior, ablating routed experts won't help much.

**Current status:**
- ✅ We ablate shared expert down_proj
- ❓ Is this enough?

### 4. **Layer Coverage**

Your analysis showed:
- Layer 18: Best signal (0.0431)
- Layers 15-25: Good signal

But what about:
- Layers 0-14: Weak signal, but still present
- Layers 26-45: Declining signal, but not zero

**MoE hypothesis:**
Maybe refusal is more distributed in MoE models. Dense models concentrate refusal in specific layers, but MoE spreads it across more layers through expert specialization.

**Test:** Try ablating MORE layers (e.g., 10-35 instead of 15-25).

### 5. **Scale Factor Too Low**

Dense models use scale=1.0 and it works.

**MoE hypothesis:**
Maybe MoE models need stronger ablation (scale=1.5 or 2.0) because:
- Refusal is distributed across multiple experts
- Each expert contributes partially
- Weak ablation doesn't overcome the combined effect

**Test:** Try scale=2.0 instead of 1.0.

---

## Recommended Experiments

### Experiment 1: Revert to Down_Proj Only (Like Dense Models)
```python
# Only ablate down_proj in experts (like dense models)
elif key.startswith(experts_prefix) and key.endswith(".down_proj.weight"):
```

**Rationale:** If dense models work this way, MoE should too.

### Experiment 2: More Layers
```yaml
# Ablate layers 10-35 instead of 15-25
ablate:
  - layer: 10
    measurement: 18
    scale: 1.0
  # ... through layer 35
```

**Rationale:** MoE might distribute refusal more broadly.

### Experiment 3: Higher Scale
```yaml
ablate:
  - layer: 15
    measurement: 18
    scale: 2.0  # Instead of 1.0
```

**Rationale:** Stronger ablation might be needed for MoE.

### Experiment 4: Don't Ablate Routing Gate
```python
# Comment out gate ablation
# elif key == gate_pattern:
#     ...
```

**Rationale:** Gate ablation might be counterproductive.

### Experiment 5: Verify Measurements Are Correct

Check if the refusal direction is actually meaningful:
```python
# After loading measurements
refusal_dir = measures['refuse_18']
print(f"Refusal direction norm: {torch.norm(refusal_dir)}")
print(f"Refusal direction stats: min={refusal_dir.min()}, max={refusal_dir.max()}")
```

If the refusal direction is all zeros or very small, measurements failed.

---

## My Best Guess

The most likely issues, in order:

1. **Layer coverage too narrow** - Try ablating 10-35 instead of 15-25
2. **Scale too low** - Try scale=2.0 instead of 1.0
3. **Shared experts dominate** - They're always active and might override routed experts
4. **Routing gate ablation is wrong** - Our method might not work for gates

**Least likely:**
- Need to ablate gate_proj/up_proj (your intuition is probably right - down_proj should be enough)

---

## Recommendation

**Step 1:** Revert to down_proj only (like dense models)
**Step 2:** Try more layers (10-35)
**Step 3:** Try higher scale (2.0)
**Step 4:** If still doesn't work, then try comprehensive ablation

This follows Occam's Razor - start with the simplest approach that works for dense models.