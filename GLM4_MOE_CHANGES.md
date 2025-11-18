# Changes Required for GLM-4.5-Air MoE Support

## Architecture Overview
- **Model Type**: `Glm4MoeForCausalLM`
- **Layers**: 46 layers
- **Hidden Size**: 4096
- **Key Difference**: MoE architecture with expert routing in MLP layers

## Critical MoE-Specific Components

### 1. MLP Structure Differences
The GLM-4.5-Air uses a complex MoE MLP structure:
```
model.layers.[i].mlp.experts.[expert_idx].gate_proj.weight  (5760 experts total)
model.layers.[i].mlp.experts.[expert_idx].up_proj.weight
model.layers.[i].mlp.experts.[expert_idx].down_proj.weight
model.layers.[i].mlp.gate.weight  (routing gate, 45 layers)
model.layers.[i].mlp.shared_experts.gate_proj.weight  (45 layers)
model.layers.[i].mlp.shared_experts.up_proj.weight
model.layers.[i].mlp.shared_experts.down_proj.weight
```

Plus standard MLP weights:
```
model.layers.[i].mlp.gate_proj.weight  (1 instance)
model.layers.[i].mlp.up_proj.weight
model.layers.[i].mlp.down_proj.weight
```

### 2. Attention Structure (Similar to Standard)
```
model.layers.[i].self_attn.q_proj.weight (with bias)
model.layers.[i].self_attn.k_proj.weight (with bias)
model.layers.[i].self_attn.v_proj.weight (with bias)
model.layers.[i].self_attn.o_proj.weight
```

## Required Code Changes

### 1. `sharded_ablate.py` - CRITICAL CHANGES

**Current Issue**: The code only targets:
- `self_attn.o_proj.weight`
- `mlp.down_proj.weight`

**Required Changes**:
1. Add support for MoE expert weights
2. Handle routing gate weights (optional, may not need ablation)
3. Handle shared expert weights
4. Update weight pattern matching to include:
   - `mlp.experts.[expert_idx].down_proj.weight`
   - `mlp.shared_experts.down_proj.weight`

**Lines to Modify**: 153-162 (weight pattern matching)

### 2. `measure.py` - MODERATE CHANGES

**Current Issue**: May not properly detect GLM model structure

**Required Changes**:
1. Ensure `model.model` base detection works for GLM models
2. Verify layer access pattern: `model.model.layers[i]`
3. No special handling needed for hidden states extraction (works at model level)

**Lines to Check**: 142-145, 371-373

### 3. `utils/models.py` - MINOR CHANGES

**Current Issue**: No GLM family detection for tied weights

**Required Changes**:
1. Add GLM model family to tied weights detection (if needed)
2. GLM models may not use tied weights, need to verify

**Lines to Modify**: 24-27

## Implementation Priority

### HIGH PRIORITY (Required for Basic Functionality)
1. ✅ Update `sharded_ablate.py` to handle MoE down_proj patterns
2. ✅ Test layer detection in `measure.py`

### MEDIUM PRIORITY (For Complete Support)
3. ✅ Add expert weight ablation support
4. ✅ Handle shared expert weights

### LOW PRIORITY (Optional Enhancements)
5. ⚠️ Consider routing gate ablation (may not be beneficial)
6. ⚠️ Add GLM-specific optimizations

## Key Architectural Considerations

### Expert Routing
- The model has **5760 expert weights** across layers
- Each layer appears to have multiple experts (5760/46 ≈ 125 experts per layer)
- Routing is handled by `mlp.gate.weight`

### Ablation Strategy
For MoE models, we should ablate:
1. **Attention output projection** (`o_proj`) - Same as current
2. **Expert down projections** (`experts.[i].down_proj`) - NEW
3. **Shared expert down projections** (`shared_experts.down_proj`) - NEW
4. **Routing gate weights** (`mlp.gate.weight`) - **CRITICAL for MoE!**
5. **Standard MLP down projection** (`mlp.down_proj`) - Current (if exists)

**Why the routing gate is critical:**
The routing gate determines which experts are activated for each token. Even if we ablate all expert weights, the gate could still route tokens to "refusal patterns" learned in the routing logic itself. By ablating the gate weights, we prevent the model from selectively activating experts based on refusal-related features.

### Weight Pattern Matching Strategy
Use exact patterns to match:
- `.*\.mlp\.down_proj\.weight$` (standard MLP)
- `.*\.mlp\.experts\.\d+\.down_proj\.weight$` (expert down projections)
- `.*\.mlp\.shared_experts\.down_proj\.weight$` (shared expert down projections)
- `.*\.mlp\.gate\.weight$` (routing gate - **CRITICAL**)

## Testing Checklist
- [ ] Verify model loads correctly
- [ ] Verify layer structure detection
- [ ] Verify weight pattern matching finds all relevant weights
- [ ] Test measurement phase completes
- [ ] Test ablation phase completes
- [ ] Verify output model structure matches input