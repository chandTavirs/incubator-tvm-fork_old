# Phase 1 Implementation: Single Integrated Module with Dynamic Weight Quantization

## Overview

This phase implements a revised architecture for OFA weight derivation + quantization:

**Old Approach (Split):**
```
mod_deriv (pool vars → derive weights) ← two separate runtimes, data marshalling
    ↓
mod_infer (derived weights → inference)
```

**New Approach (Integrated):**
```
Single mod_integrated:
  input: (image, pool_w_0, pool_w_1, ..., transform_tm_0, ...)
         ↓
  [weight derivation ops: slice + optional dense transform]
         ↓
  [conv layers using derived weights]
         ↓
  output: logits
```

## Why Single Integrated Module?

1. **Simpler VTA compilation**: One relay.build() call, not two separate runtimes
2. **Better dataflow visibility**: VTA compiler sees full weight derivation → conv pipeline
3. **Unified parameter dict**: No need to marshal derived weights between modules
4. **Easier debugging**: Single Relay IR to inspect, not two
5. **Runtime efficiency**: No round-trip through host for weight computation

## Key Files

### New Module: `quantize_dynamic_weights.py`

Provides the new quantization API that preserves weight variables:

#### Main Function: `quantize_with_dynamic_weights()`

```python
mod_q = quantize_with_dynamic_weights(
    mod=mod_integrated,
    params=all_params,
    dynamic_weight_var_names=["pool_w_0", "pool_w_1", ...],
    scale_table=None,  # optional per-op scales
    dataset=None,      # calibration data
)
```

**What it does:**
- Skips binding parameters that are in `dynamic_weight_var_names` 
- Prevents FoldConstant from materializing pool vars as constants
- Allows BN/bias params to be folded normally
- Returns quantized module with weight vars still present as Var nodes

**Strategy (Option A: Skip Bind):**
```
1. Filter out protected vars from params dict before binding
2. Run prerequisite_optimize with filtered params
3. FoldConstant can't fold what isn't bound → weight derivation stays as ops
4. Result: quantized graph with weight vars intact
```

#### Helper Function: `merge_derivation_and_inference_modules()`

Merges `mod_deriv` and `mod_infer` back into a single module:

```python
mod_merged, all_params = merge_derivation_and_inference_modules(
    mod_deriv, deriv_params, 
    mod_infer, infer_params,
    derived_weight_names=["derived_w_0", "derived_w_1", ...]
)
```

**What it does:**
- Takes the tuple of derived weights from mod_deriv
- Extracts each weight via TupleGetItem
- Substitutes them into mod_infer's body
- Combines free variables (pool vars + non-weight params + input)
- Returns single integrated module with clean signature

### Updated File: `step3_relay_build_poc.py`

#### Modified `build_first_step_relay()`

```python
# Step 1: Build full graph with weight derivations
mod_full, tvm_params_full = build_relay_with_ofa_pool_vars(...)

# Step 2: Split (temporary, for compatibility)
mod_deriv, deriv_params, mod_infer, infer_params, derived_weight_names = (
    split_derivation_and_inference_modules(mod_full, tvm_params_full)
)

# Step 3: Merge back into single integrated module
mod_integrated, all_params = merge_derivation_and_inference_modules(
    mod_deriv, deriv_params, mod_infer, infer_params, derived_weight_names
)

# Return integrated module
return {
    "mod_full": mod_integrated,  # ← now the single integrated module
    "tvm_params_full": all_params,
    "derived_weight_names": derived_weight_names,
    ...
}
```

#### Updated `step3b_vta_compile()`

```python
# Use new quantize_with_dynamic_weights API
with relay.quantize.qconfig(global_scale=GLOBAL_SCALE, ...):
    mod_q = quantize_with_dynamic_weights(
        split_artifacts["mod_full"],  # single integrated module
        split_artifacts["tvm_params_full"],
        dynamic_weight_var_names=split_artifacts["derived_weight_names"],
    )

# Then graph_pack and relay.build as normal
relay_prog = graph_pack(mod_q["main"], ...)
graph, lib, built_params = relay.build(relay_prog, target=env.target, ...)
```

## Implementation Choices Made

### 1. Per-Op Scale Table (Option B)

We use **static per-op scales** (computed offline from calibration data) rather than:
- Option A (global scale): Less flexible for layers with different weight ranges
- Option C (runtime histogram): Complex pipeline, dataset dependency

**Current approach:** Uses global_scale from qconfig. Pre-computed per-op table can be stored as JSON metadata and applied later.

### 2. Weight-Var Preservation (Option B: Skip Bind)

We skip binding protected parameters before prerequisite_optimize:

```python
# Only bind non-protected params
filtered_params = {k: v for k, v in params.items() 
                   if k not in protected_var_names}
mod["main"] = _bind_params(mod["main"], filtered_params)
```

Why this over alternatives:
- **Option A (attribute marking)**: Requires patching FoldConstant pass (invasive)
- **Option C (selective restoration)**: Complex, error-prone state management
- **Option B (skip bind)**: Simple, non-invasive, works because FoldConstant can't fold unbound vars

### 3. Single vs Split Module (Integration point)

We integrate after `split_derivation_and_inference_modules()` but before quantization:

```
build_relay_with_ofa_pool_vars()
    ↓
split_derivation_and_inference_modules()  [temporary split]
    ↓
merge_derivation_and_inference_modules()  [merge back]
    ↓
quantize_with_dynamic_weights()           [quantize integrated]
    ↓
graph_pack()
    ↓
relay.build()
```

Why merge before quantization:
- Quantize needs to see the full dataflow to make informed folding decisions
- VTA compiler sees optimal weight derivation → conv graph
- Avoids passing derived weights as separate params through quantization

## Execution Flow (CPU Validation)

```python
# Step 3A: CPU Validation
# 1. Materialize derived weights on CPU (run mod_deriv)
#    derived_w_i = cpu_eval(mod_deriv, deriv_params)[i]
#
# 2. Build + run integrated module with materialized weights
#    out_pool = cpu_eval(mod_integrated, {..., derived_w_0, derived_w_1, ..., ...})
#
# 3. Compare against PyTorch reference
#    assert top1(out_pool) == top1(pytorch_out)
```

```python
# Step 3B: VTA Compile
# 1. Quantize integrated module (weights preserved as vars)
#    mod_q = quantize_with_dynamic_weights(mod_integrated, ...)
#
# 2. graph_pack (packs conv + pool layers)
#    relay_prog = graph_pack(mod_q, ...)
#
# 3. relay.build for VTA target
#    graph, lib, params = relay.build(relay_prog, target=env.target, ...)
```

```python
# Step 3C: VTA Inference
# 1. Upload compiled library to device
#    remote.upload(lib_tar)
#
# 2. Create runtime
#    m = graph_runtime.create(graph, lib_tar, ctx)
#
# 3. Set all inputs (pool vars + activation)
#    m.set_input(**all_params)  # pool_w_0, pool_w_1, ...
#    m.set_input(input_name, input_tensor)
#
# 4. Run inference
#    m.run()
#    output = m.get_output(0)
```

## Expected Outcomes

### CPU Validation (3A)
- ✓ Derived weights materialize correctly (max_diff < 1e-4 vs PyTorch)
- ✓ Integrated module executes on CPU without errors
- ✓ Top-1 class matches PyTorch reference

### VTA Compile (3B)
- ✓ Quantization preserves weight vars (not materialized as constants)
- ✓ graph_pack compiles successfully
- ✓ relay.build produces valid VTA library

### VTA Inference (3C)
- ✓ Library loads on device
- ✓ Inference completes (within quantization tolerance)
- ✓ Top-1 class matches float32 reference (or within tolerance)

## Testing & Validation

Run the full pipeline:
```bash
cd /path/to/phase_b
python step3_relay_build_poc.py \
    --n 25 --lambda 4 --seed 0 --num-subnets 2 \
    --skip-vta  # for quick CPU-only test
```

Output artifacts:
- `step3_results/relay_ir_{subnet_id}.txt`: Quantized module IR
- `step3_results/step3_summary.json`: Results for all subnets

Expected console output:
```
[1] Loading OFA model...
[2] Loading OFA weight pool...
[3] Loading candidate set...
[4] Setting up VTA environment...

Processing subnet_001:
  3A CPU Validation: subnet_001
    Materializing derived weights from pool variables...
    ✓ Materialized 52 derived weight tensors
    Max  |diff|:  9.32e-05
    Mean |diff|:  2.14e-06
    Top-1 match:  ✓
  3A Status: ✓ PASSED

  3B VTA Compile: subnet_001
    Applying quantization with dynamic weight preservation...
    ✓ Quantization done in 12.3s
    Applying graph_pack...
    ✓ graph_pack done in 5.2s
    Running relay.build for VTA target...
    ✓ relay.build done in 45.1s

  3C VTA Inference: subnet_001
    Running VTA inference...
    ✓ Inference done in 125.3ms
    Top-1 match:  ✓
```

## Future Optimizations

1. **Per-op quantization scales**: Store scales in JSON, apply during quantize
2. **Hardware-aware scheduling**: Use VTA's schedule hints for weight derivation ops
3. **Pooled weight caching**: Reuse derived weights across inference calls
4. **Multi-model support**: Extend to other OFA architectures (MobileNetV3, etc.)

## Debugging Tips

- **Module inspection**: Print `mod["main"]` to see full Relay IR
- **Free variables**: Use `relay.analysis.free_vars(func.body)` to verify inputs
- **Quantization issues**: Check that `mod_q["main"]` still has weight vars as free vars
- **Graph pack problems**: Verify `PACK_DICT[MODEL_NAME]` markers exist in IR
- **Build failures**: Check schedule logs for tuning compatibility

## References

- Previous docs: PHASE1_IMPLEMENTATION_SUMMARY.md
- Builder code: ofa_relay_graph_builder.py
- Extractor code: ofa_derivation_extractor.py
- Quantize module: tvm/python/tvm/relay/quantize/quantize.py

