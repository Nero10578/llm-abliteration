# MoE Compatibility Guide for GLM-4.5-Air

This document explains the modifications made to support Mixture of Experts (MoE) models, specifically the GLM-4.5-Air architecture.

## Architecture Differences

### Standard Transformer vs MoE

**Standard Transformer (e.g., Llama, Mistral):**
```
model.layers.{N}.self_attn.o_proj.weight
model.layers.{N}.mlp.down_proj.weight
```

**GLM-4.5-Air MoE Architecture:**
```
model.layers.{N}.self_attn.o_proj.weight
model.layers.{N}.mlp.shared_experts.down_proj.weight
model.layers.{N}.mlp.gate.weight  # Expert routing gate
model.layers.{N}.mlp.experts.{E}.down_proj.weight  # Individual experts
model.layers.{N}.mlp.experts.{E}.gate_proj.weight
model.layers.{N}.mlp.experts.{E}.up_proj.weight
```

## Key Modifications Made

### 1. Architecture Detection (`sharded_ablate.py` lines 216-245)

```python
# Detect MoE architecture by looking for expert patterns
expert_keys = [k for k in weight_map.keys() if ".mlp.experts." in k]
if expert_keys:
    is_moe = True
    # Extract expert count from key patterns
```

### 2. Weight Pattern Matching (`sharded_ablate.py` lines 254-287)

**For MoE Models:**
- **Attention Output**: `self_attn.o_proj.weight` (same as standard)
- **Shared Experts**: `mlp.shared_experts.down_proj.weight`
- **Expert Routing Gate**: `mlp.gate.weight`
- **Individual Experts**: Optional `mlp.experts.{E}.*.weight`

**For Standard Models:**
- **Attention Output**: `self_attn.o_proj.weight`
- **MLP Down Projection**: `mlp.down_proj.weight`

### 3. MoE-Aware Weight Modification (`sharded_ablate.py` lines 93-166)

New function `modify_tensor_norm_preserved_moe()` with special handling:

**Expert Gate Handling:**
- Reduced intensity (50% scale factor)
- More conservative modifications to preserve routing behavior
- Avoids breaking expert selection logic

**Shared Expert Handling:**
- Standard abliteration approach
- Full intensity modifications

**Individual Expert Handling:**
- Optional per-expert abliteration
- Memory intensive but allows fine-grained control

### 4. Command Line Interface Extensions

New parameters added:
```bash
--ablate-individual-experts    # Enable individual expert abliteration
--expert-subset "0,1,2,3"      # Specify which experts to target
```

### 5. YAML Configuration Extensions

New MoE-specific options:
```yaml
ablate_individual_experts: false  # Enable individual expert abliteration
expert_subset: null               # Optional: [0, 1, 2, 3] for specific experts
```

## Usage Instructions

### Basic MoE Abliteration

1. **Run Measurement Phase:**
```bash
python measure.py -m /home/arli/models/GLM-4.5-Air -o glm4_measurements.refuse
```

2. **Analyze Measurements:**
```bash
python analyze.py glm4_measurements.refuse -c
```

3. **Run Abliteration:**
```bash
python sharded_ablate.py glm4-5-air-moe.yml
```

### Advanced MoE Abliteration

**Target Specific Experts:**
```bash
python sharded_ablate.py glm4-5-air-moe.yml --expert-subset "0,1,2,3"
```

**Abliterate All Individual Experts:**
```bash
python sharded_ablate.py glm4-5-air-moe.yml --ablate-individual-experts
```

## Memory Considerations

### Standard MoE Abliteration
- **VRAM Usage**: Similar to standard models
- **Targets**: Attention output + shared experts + routing gate
- **Memory Impact**: Moderate

### Individual Expert Abliteration
- **VRAM Usage**: Significantly higher (experts × memory multiplier)
- **Targets**: All individual expert weights
- **Memory Impact**: High (use with caution)

**Recommendation**: Start with standard MoE abliteration, only use individual expert abliteration if needed.

## GLM-4.5-Air Specific Considerations

### Model Specifications
- **Layers**: 46 transformer layers
- **Experts**: Multiple experts per layer (detected automatically)
- **Hidden Size**: 4096
- **Architecture**: Glm4MoeForCausalLM

### Abliteration Strategy
1. **Middle Layers (20-35)**: Often contain strong refusal signals
2. **Late Layers (36-45)**: Final refusal decisions
3. **Measurement Layers**: Use later layers (35-40) as measurement sources

### Expected Behavior
- **Shared Experts**: Modified to reduce refusal tendencies
- **Expert Routing**: Subtly influenced to avoid refusal-biased routing
- **Individual Experts**: Optionally modified for fine-grained control

## Troubleshooting

### Common Issues

1. **"Could not detect layer structure"**
   - Verify model path is correct
   - Check if model uses standard naming conventions

2. **"Expert X not found"**
   - Expert indices are 0-based
   - Check actual expert count in model

3. **Memory Issues**
   - Use 4-bit quantization during measurement phase
   - Avoid individual expert abliteration for large models
   - Reduce batch sizes

### Validation Steps

1. **Check Architecture Detection:**
```bash
# Look for these messages in output:
"Detected layer prefix: model"
"MoE architecture: True"
"Detected X experts per layer"
```

2. **Verify Weight Patterns:**
```bash
# Should show modifications for:
# - self_attn.o_proj.weight
# - mlp.shared_experts.down_proj.weight  
# - mlp.gate.weight
```

3. **Test Abliterated Model:**
```bash
python chat.py -m /path/to/abliterated/model
```

## Performance Notes

### Advantages of MoE Abliteration
- **Targeted**: Only affects relevant components
- **Preserves Expertise**: Maintains model capabilities
- **Flexible**: Granular control over abliteration scope

### Limitations
- **Memory Intensive**: Especially for individual expert abliteration
- **Complex**: More failure modes than standard abliteration
- **Model Specific**: Requires MoE architecture support

## Future Enhancements

Potential improvements:
1. **Adaptive Expert Selection**: Automatically choose most relevant experts
2. **Hierarchical Abliteration**: Different strategies for different expert types
3. **Memory Optimization**: More efficient individual expert processing
4. **Validation Metrics**: MoE-specific abliteration quality measures

## Example Configuration Files

### Conservative MoE Abliteration
```yaml
model: "/path/to/GLM-4.5-Air"
measurements: measurements.refuse
output: glm4-conservative
ablate_individual_experts: false
expert_subset: null
ablate:
  - layer: 30
    measurement: 40
    scale: 0.8  # Reduced intensity
    sparsity: 0.1
```

### Aggressive Individual Expert Abliteration
```yaml
model: "/path/to/GLM-4.5-Air"
measurements: measurements.refuse
output: glm4-aggressive
ablate_individual_experts: true
expert_subset: [0, 1, 2, 3]  # First 4 experts only
ablate:
  - layer: 25
    measurement: 35
    scale: 1.2  # Increased intensity
    sparsity: 0.0
```

This guide provides comprehensive information for using the MoE-compatible abliteration system with GLM-4.5-Air and similar MoE architectures.