# What Gets Ablated in MoE Models?

## Overview

For Mixture of Experts (MoE) models like GLM-4.5-Air, abliteration targets the **output/down projection layers** where the refusal behavior is encoded. This is similar to standard models, but MoE architectures have multiple parallel expert pathways that all need to be ablated.

## Standard (Non-MoE) Models

In regular transformer models, abliteration modifies:

```
Layer N:
├── Self-Attention
│   └── o_proj.weight ✓ ABLATED (attention output projection)
└── MLP
    ├── gate_proj.weight (not ablated)
    ├── up_proj.weight (not ablated)
    └── down_proj.weight ✓ ABLATED (MLP output projection)
```

## MoE Models (e.g., GLM-4.5-Air)

In MoE models, the MLP is replaced with multiple expert pathways plus a routing mechanism:

```
Layer N:
├── Self-Attention
│   └── o_proj.weight ✓ ABLATED (attention output projection)
│
└── MLP (MoE Structure)
    ├── gate.weight (routing gate - NOT ablated)
    │
    ├── Standard MLP (if present)
    │   ├── gate_proj.weight (not ablated)
    │   ├── up_proj.weight (not ablated)
    │   └── down_proj.weight ✓ ABLATED
    │
    ├── Experts (e.g., 125 experts per layer in GLM-4.5-Air)
    │   ├── experts.0
    │   │   ├── gate_proj.weight (not ablated)
    │   │   ├── up_proj.weight (not ablated)
    │   │   └── down_proj.weight ✓ ABLATED
    │   ├── experts.1
    │   │   ├── gate_proj.weight (not ablated)
    │   │   ├── up_proj.weight (not ablated)
    │   │   └── down_proj.weight ✓ ABLATED
    │   ├── experts.2
    │   │   └── down_proj.weight ✓ ABLATED
    │   └── ... (all 125 experts)
    │       └── down_proj.weight ✓ ABLATED
    │
    └── Shared Experts (if present)
        ├── gate_proj.weight (not ablated)
        ├── up_proj.weight (not ablated)
        └── down_proj.weight ✓ ABLATED
```

## Why These Specific Weights?

### 1. Attention Output Projection (`o_proj.weight`)
- **What it does**: Combines attention heads and projects back to model dimension
- **Why ablate**: Refusal signals can be encoded in how attention outputs are combined
- **Impact**: Removes refusal direction from attention mechanism

### 2. MLP Down Projection (`down_proj.weight`)
- **What it does**: Projects from intermediate dimension back to model dimension
- **Why ablate**: Primary location where refusal behavior is encoded
- **Impact**: Removes refusal direction from feed-forward processing

### 3. Expert Down Projections (`experts.[i].down_proj.weight`)
- **What it does**: Each expert's output projection (same role as standard down_proj)
- **Why ablate**: Each expert can independently encode refusal behavior
- **Impact**: Ensures refusal is removed from ALL expert pathways
- **Critical for MoE**: If we only ablated one expert, the router could still send refusal-triggering inputs to non-ablated experts

### 4. Shared Expert Down Projections (`shared_experts.down_proj.weight`)
- **What it does**: Output projection for experts that process all inputs
- **Why ablate**: Shared experts see all tokens and can encode refusal
- **Impact**: Removes refusal from the always-active expert pathway

## GLM-4.5-Air Specific Numbers

Based on your model architecture:
- **46 layers** total
- **~125 experts per layer** (5760 total experts / 46 layers)
- **1 shared expert pathway** per layer (45 total)

For each layer you ablate, the code will modify:
- 1 × `self_attn.o_proj.weight`
- 1 × `mlp.down_proj.weight` (if present)
- ~125 × `mlp.experts.[0-124].down_proj.weight`
- 1 × `mlp.shared_experts.down_proj.weight`

**Total per layer: ~127 weight tensors ablated**

## What Does NOT Get Ablated?

### Never Ablated:
- **Input projections** (`q_proj`, `k_proj`, `v_proj`) - these encode the input representation
- **Gate/Up projections** (`gate_proj`, `up_proj`) - these expand to intermediate dimension
- **Routing gates** (`mlp.gate.weight`) - these decide which experts to use
- **Layer norms** - these normalize activations
- **Embeddings** - these convert tokens to vectors

### Why Not Ablate These?
- **Input projections**: Ablating these would corrupt the model's ability to understand inputs
- **Gate/Up projections**: These create the intermediate representation; ablating them would break the MLP
- **Routing gates**: These are learned routing decisions; ablating them would break expert selection
- **The refusal direction is primarily encoded in OUTPUT projections**, not input transformations

## How Ablation Works

For each targeted weight tensor:

1. **Extract refusal direction** from measurements (harmful - harmless activations)
2. **Decompose weight matrix** into magnitude and direction components
3. **Remove refusal component** from the directional part
4. **Renormalize** the adjusted direction
5. **Recombine** with original magnitude (norm-preserving)

This ensures:
- ✓ Refusal behavior is removed
- ✓ Model's general capabilities are preserved
- ✓ Weight magnitudes remain stable (no gradient explosion/vanishing)

## Practical Impact

When you ablate a layer in an MoE model:
- **All expert pathways** are modified to remove refusal
- **No matter which expert** the router selects, refusal is suppressed
- **Shared experts** (always active) also have refusal removed
- **Attention mechanism** no longer encodes refusal in its outputs

This comprehensive approach ensures that refusal cannot "hide" in any single expert pathway.