# Complete Guide: How Model Ablation Works

## Table of Contents
1. [What is Ablation?](#what-is-ablation)
2. [The Three-Phase Process](#the-three-phase-process)
3. [Mathematical Details](#mathematical-details)
4. [Dense vs MoE Models](#dense-vs-moe-models)
5. [Code Walkthrough](#code-walkthrough)

---

## What is Ablation?

**Ablation** is the process of removing specific behaviors from a neural network by identifying and eliminating the directions in activation space that encode those behaviors.

### The Core Idea

Language models represent concepts as **directions in high-dimensional space**. For example:
- "Refusal" has a direction in the model's hidden states
- When the model wants to refuse, activations move in this direction
- By removing this direction from the model's weights, we prevent refusals

### Analogy

Think of it like removing a specific color filter from a camera:
- The camera (model) can produce many colors (behaviors)
- One filter (refusal direction) makes everything look red (causes refusals)
- Remove that filter → camera can't produce red anymore (can't refuse)

---

## The Three-Phase Process

### Phase 1: Measurement (measure.py)

**Goal:** Find the "refusal direction" in the model's activation space.

#### Step 1.1: Prepare Datasets
```
Harmful prompts:  ["How to make a bomb?", "How to hack a bank?", ...]
Harmless prompts: ["How to make a cake?", "How to use a computer?", ...]
```

#### Step 1.2: Generate Activations
For each prompt, run the model and capture hidden states at each layer:

```python
# Pseudocode
for prompt in harmful_prompts:
    output = model.generate(prompt, output_hidden_states=True)
    for layer in range(num_layers):
        harmful_activations[layer].append(output.hidden_states[layer][:, -1, :])
        
# Same for harmless prompts
```

**Key point:** We capture the hidden state at the **last token position** before generation starts. This is where the model "decides" whether to refuse.

#### Step 1.3: Compute Mean Activations
```python
# For each layer
harmful_mean[layer] = mean(harmful_activations[layer])    # Average of all harmful prompts
harmless_mean[layer] = mean(harmless_activations[layer])  # Average of all harmless prompts
```

#### Step 1.4: Compute Refusal Direction
```python
# The refusal direction is the difference
refusal_direction[layer] = harmful_mean[layer] - harmless_mean[layer]
```

**Optional refinement** (biprojection):
```python
# Remove harmless component from refusal direction
harmless_normalized = normalize(harmless_mean[layer])
projection = dot(refusal_direction[layer], harmless_normalized)
refusal_direction[layer] = refusal_direction[layer] - projection * harmless_normalized
```

This ensures we only remove the "refusal" part, not general capabilities.

#### Output
A file (e.g., `model.refuse`) containing:
```python
{
    'layers': 46,
    'harmful_0': tensor([...]),   # Harmful mean for layer 0
    'harmless_0': tensor([...]),  # Harmless mean for layer 0
    'refuse_0': tensor([...]),    # Refusal direction for layer 0
    'harmful_1': tensor([...]),
    ...
}
```

---

### Phase 2: Analysis (analyze.py)

**Goal:** Determine which layers have the strongest refusal signal.

#### Metrics Computed

For each layer:

1. **Cosine Similarity** between harmful and harmless means:
   ```python
   cosine_sim = dot(harmful_mean, harmless_mean) / (norm(harmful_mean) * norm(harmless_mean))
   ```
   - Low similarity (< 0.95) = activations are different = good for ablation
   - High similarity (> 0.98) = activations are similar = weak signal

2. **Signal-to-Noise Ratio**:
   ```python
   snr = norm(refusal_direction) / mean(norm(harmful_mean), norm(harmless_mean))
   ```
   - High SNR = strong refusal signal relative to base activations

3. **Signal Quality** (combined metric):
   ```python
   signal_quality = (1 - cosine_sim) * snr
   ```
   - Higher = better candidate for ablation

#### Output
```
Layer 18: Signal quality 0.0431 (BEST)
Layer 20: Signal quality 0.0380
Layer 19: Signal quality 0.0369
...
```

**Decision:** Ablate layers with highest signal quality (typically middle layers).

---

### Phase 3: Ablation (sharded_ablate.py)

**Goal:** Modify model weights to remove the refusal direction.

#### Step 3.1: Load Configuration
```yaml
model: /path/to/model
measurements: model.refuse
output: model-abliterated
ablate:
  - layer: 18
    measurement: 18  # Use layer 18's refusal direction
    scale: 1.0       # Ablation strength
    sparsity: 0.0    # Keep all components
```

#### Step 3.2: Identify Weights to Modify

For each layer specified in the config, find these weight matrices:

**Dense models:**
- `layers.{i}.self_attn.o_proj.weight` (attention output)
- `layers.{i}.mlp.down_proj.weight` (MLP output)

**MoE models (additional):**
- `layers.{i}.mlp.gate.weight` (routing gate)
- `layers.{i}.mlp.experts.{j}.down_proj.weight` (each expert's output)
- `layers.{i}.mlp.shared_experts.down_proj.weight` (shared expert output)

#### Step 3.3: Apply Norm-Preserving Ablation

For each weight matrix `W`:

**Input:**
- `W`: Weight matrix, shape `[in_features, out_features]` (safetensors format)
- `refusal_dir`: Refusal direction, shape `[in_features]`
- `scale`: Ablation strength (typically 1.0)

**Process:**

1. **Transpose** (convert to PyTorch convention):
   ```python
   W = W.T  # Now [out_features, in_features]
   ```

2. **Decompose into magnitude and direction**:
   ```python
   W_norm = norm(W, dim=1, keepdim=True)        # [out_features, 1]
   W_direction = normalize(W, dim=1)             # [out_features, in_features]
   ```
   
   Each row of `W` represents one output neuron. We separate:
   - **Magnitude**: How strong the neuron is
   - **Direction**: What features it responds to

3. **Project onto refusal direction**:
   ```python
   refusal_normalized = normalize(refusal_dir)
   projection = matmul(W_direction, refusal_normalized)  # [out_features]
   ```
   
   This tells us how much each output neuron aligns with the refusal direction.

4. **Remove refusal component**:
   ```python
   W_direction_new = W_direction - scale * outer(projection, refusal_normalized)
   ```
   
   Subtract the refusal direction from each neuron's direction vector.

5. **Re-normalize direction**:
   ```python
   W_direction_new = normalize(W_direction_new, dim=1)
   ```
   
   Ensure each neuron still has unit direction (only direction changed, not magnitude).

6. **Recombine**:
   ```python
   W_modified = W_norm * W_direction_new
   ```
   
   Multiply back the original magnitudes. This preserves the strength of each neuron while changing what it responds to.

7. **Transpose back**:
   ```python
   W_modified = W_modified.T  # Back to safetensors format
   ```

**Why norm-preserving?**
- Preserves the overall "strength" of the network
- Prevents degradation of general capabilities
- Only changes the direction, not the magnitude

#### Step 3.4: Save Modified Model

Write the modified weights back to safetensors files, preserving the original model structure.

---

## Mathematical Details

### Why This Works

The key insight is that neural network weights can be viewed as **linear transformations** that move activations in specific directions.

#### Before Ablation
```
Input activation: x = [features about the prompt]
Weight matrix: W = [neurons that detect various patterns]
Output: y = W @ x

If x contains refusal features, and W has neurons aligned with refusal direction,
then y will have strong refusal signal.
```

#### After Ablation
```
Modified weight: W' = W with refusal direction removed
Output: y' = W' @ x

Even if x contains refusal features, W' doesn't respond to them,
so y' has no refusal signal.
```

### The Projection Formula

Given:
- Weight row: `w` (what one neuron responds to)
- Refusal direction: `r` (normalized)

The component of `w` along `r` is:
```
component = (w · r) * r
```

Removing it:
```
w_new = w - scale * (w · r) * r
```

This is exactly what we do for each row of the weight matrix.

### Why Only Output Projections?

In a transformer layer:
```
Input → Attention → Add & Norm → MLP → Add & Norm → Output
                                  ↓
                            [gate, up, down]
```

The **output projections** (attention `o_proj` and MLP `down_proj`) are where information flows back to the residual stream. By ablating these, we prevent refusal information from propagating forward.

We don't need to ablate `gate_proj` or `up_proj` because:
- They're internal to the MLP
- Their output gets processed by `down_proj`
- Ablating `down_proj` is sufficient to block refusal

---

## Dense vs MoE Models

### Dense Model Architecture

```
Layer i:
├── Self-Attention
│   ├── q_proj, k_proj, v_proj (compute attention)
│   └── o_proj ← ABLATED (attention output)
│
└── MLP
    ├── gate_proj (gating)
    ├── up_proj (expansion)
    └── down_proj ← ABLATED (MLP output)
```

**Ablation targets:** 2 weights per layer
- `o_proj.weight`
- `down_proj.weight`

**Why it works:**
- Single computation path
- Refusal encoded in the MLP weights
- Ablating output removes refusal from residual stream

---

### MoE Model Architecture

```
Layer i:
├── Self-Attention
│   └── o_proj ← ABLATED
│
└── MoE MLP
    ├── Routing Gate ← ABLATED
    │   └── Selects which experts to use
    │
    ├── Expert 0
    │   ├── gate_proj
    │   ├── up_proj
    │   └── down_proj ← ABLATED
    │
    ├── Expert 1
    │   └── down_proj ← ABLATED
    │
    ├── ... (many experts)
    │
    ├── Expert N
    │   └── down_proj ← ABLATED
    │
    └── Shared Experts
        └── down_proj ← ABLATED
```

**Ablation targets:** 3+ weights per layer (depends on number of experts)
- `o_proj.weight` (1)
- `gate.weight` (1) - routing gate
- `experts.{j}.down_proj.weight` (N experts)
- `shared_experts.down_proj.weight` (1)

**Why it's different:**

1. **Multiple Computation Paths**
   - Dense: Input → MLP → Output
   - MoE: Input → Gate → [Expert 0, Expert 5, Expert 12] → Combine → Output
   
   Must ablate ALL expert outputs, not just one.

2. **Routing Gate**
   - Determines which experts activate
   - Can route to "refusal experts" even if expert weights are ablated
   - Must ablate gate to prevent refusal-based routing

3. **Expert Specialization**
   - Different experts may specialize in different tasks
   - Some experts might be "refusal specialists"
   - Need to ablate all of them

4. **Shared Experts**
   - Always active (not routed)
   - Provide baseline computation
   - Must ablate to prevent refusal in baseline

---

## Code Walkthrough

### Key Functions

#### 1. `modify_tensor_norm_preserved()` (sharded_ablate.py)

```python
def modify_tensor_norm_preserved(W, refusal_dir, scale_factor=1.0):
    """
    Core ablation function.
    
    Args:
        W: Weight matrix [in_features, out_features] (safetensors format)
        refusal_dir: Refusal direction [in_features]
        scale_factor: Ablation strength (1.0 = full ablation)
    
    Returns:
        Modified weight matrix with refusal direction removed
    """
    # 1. Transpose to PyTorch convention
    W_gpu = W.T  # [out_features, in_features]
    
    # 2. Decompose into magnitude and direction
    W_norm = torch.norm(W_gpu, dim=1, keepdim=True)
    W_direction = torch.nn.functional.normalize(W_gpu, dim=1)
    
    # 3. Normalize refusal direction
    refusal_normalized = torch.nn.functional.normalize(refusal_dir, dim=0)
    
    # 4. Compute projection onto refusal direction
    projection = torch.matmul(W_direction, refusal_normalized)
    
    # 5. Remove refusal component
    W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
    
    # 6. Re-normalize
    W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
    
    # 7. Recombine with original magnitude
    W_modified = W_norm * W_direction_new
    
    # 8. Transpose back to safetensors format
    return W_modified.T
```

**Special case for routing gates:**
```python
# Routing gate has shape [hidden_size, num_experts]
# Different from other weights [hidden_size, hidden_size]
if W_gpu.shape[1] != refusal_dir.shape[0]:
    # Ablate along input dimension instead
    # (prevents routing based on refusal features in hidden states)
```

#### 2. `ablate_by_layers_sharded()` (sharded_ablate.py)

```python
def ablate_by_layers_sharded(model_name, measures, marching_orders, output_path):
    """
    Main ablation function. Processes model shard by shard.
    
    Args:
        model_name: Path to model
        measures: Dict of refusal directions from measurement phase
        marching_orders: List of (layer, measurement, scale, sparsity) tuples
        output_path: Where to save abliterated model
    """
    # 1. Load model config and index
    config = AutoConfig.from_pretrained(model_name)
    index = load_safetensors_index(model_name)
    
    # 2. Build map of which weights to modify
    for layer, measurement, scale, sparsity in marching_orders:
        # Find all relevant weights for this layer
        o_proj_pattern = f"layers.{layer}.self_attn.o_proj.weight"
        down_proj_pattern = f"layers.{layer}.mlp.down_proj.weight"
        
        # MoE-specific
        gate_pattern = f"layers.{layer}.mlp.gate.weight"
        expert_pattern = f"layers.{layer}.mlp.experts.*.down_proj.weight"
        shared_pattern = f"layers.{layer}.mlp.shared_experts.down_proj.weight"
        
        # Add to modification list
        shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
    
    # 3. Process each shard
    for shard_file in all_shards:
        # Load shard
        state_dict = load_file(shard_path)
        
        # Apply modifications
        for key, layer, measurement, scale, sparsity in shard_modifications[shard_file]:
            # Get refusal direction
            refusal_dir = measures[f'refuse_{measurement}']
            
            # Optionally apply sparsity
            if sparsity > 0:
                refusal_dir = magnitude_sparsify(refusal_dir, sparsity)
            
            # Ablate
            state_dict[key] = modify_tensor_norm_preserved(
                state_dict[key], refusal_dir, scale
            )
        
        # Save modified shard
        save_file(state_dict, output_path / shard_file)
```

---

## Summary: Dense vs MoE

| Aspect | Dense Models | MoE Models |
|--------|--------------|------------|
| **Measurement** | Same | Same |
| **Analysis** | Same | Same |
| **Weights ablated** | 2 per layer | 3+ per layer |
| **Complexity** | Simple | Complex |
| **Key difference** | Single MLP | Multiple experts + routing |
| **Critical addition** | N/A | Routing gate ablation |
| **Why different** | One path | Multiple paths + routing |

### The Core Difference

**Dense models:**
```
Input → MLP → Output
         ↑
    Ablate here (down_proj)
```

**MoE models:**
```
Input → Gate → Expert 0 → ↓
             → Expert 1 → ↓ → Combine → Output
             → Expert N → ↓      ↑
                                Ablate all expert outputs
                                AND the gate
```

The routing gate is critical because it can preserve refusal behavior through expert selection, even if expert weights are ablated.

---

## Conclusion

Ablation works by:
1. **Measuring** where refusal lives in activation space
2. **Analyzing** which layers have the strongest signal
3. **Modifying** weights to remove the refusal direction while preserving capabilities

For MoE models, the key insight is that refusal can be distributed across multiple experts and preserved through routing, requiring more comprehensive ablation than dense models.