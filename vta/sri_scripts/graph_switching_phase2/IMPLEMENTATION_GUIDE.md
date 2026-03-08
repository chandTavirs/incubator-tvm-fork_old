# Phase 2 Implementation Guide

## Overview

This guide explains how the Phase 2 Quick Win implementation works and how to use it.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         OFA Superset                            │
│                   (61 parameters on device)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Runtime 0    │    │  Runtime 1    │    │  Runtime 24   │
│  (18 layers)  │    │  (30 layers)  │    │  (28 layers)  │
│  Uses: p0-p34 │    │  Uses: p0-p52 │    │  Uses: p0-p54 │
└───────────────┘    └───────────────┘    └───────────────┘
        ↓                     ↓                     ↓
    Zero-copy            Zero-copy            Zero-copy
    references           references           references
```

### Key Components

1. **SharedParamManager**: Manages the unified parameter pool
   - Uploads OFA params once to device
   - Tracks parameter usage across models
   - Reports memory savings

2. **MultiRuntimeExecutor**: Manages multiple graph runtimes
   - Creates separate runtime for each model
   - Links them all to shared parameters
   - Provides fast model switching API

3. **Phase2Builder**: Compiles models and sets up infrastructure
   - Integrates with existing compilation pipeline
   - Extracts parameter requirements
   - Builds and configures executor

---

## How It Works

### Step 1: Model Compilation

```python
# Compile each model from candidate set
for arch_map in candidate_set:
    compiled = compile_model(arch_map, ofa_model, ...)
    
    # Extract:
    # - graph_json (computation graph)
    # - lib (compiled operators)
    # - params (parameter dictionary)
    # - param_names (list of param names needed)
```

### Step 2: Shared Parameter Upload

```python
# Upload all OFA parameters ONCE to device
param_manager = SharedParamManager(ctx)
param_manager.upload_params(ofa_params)

# Result: All 61 parameters now on device memory
# Memory used: ~25 MB (instead of 25 × 25 MB = 625 MB)
```

### Step 3: Runtime Creation with Zero-Copy Linking

```python
for model_id, compiled in compiled_models.items():
    # Create runtime for this model
    runtime = graph_runtime.create(
        compiled.graph_json,
        compiled.lib,
        ctx
    )
    
    # Link to shared parameters (zero-copy!)
    for param_name in compiled.param_names:
        idx = runtime.get_input_index(param_name)
        runtime.set_input_zero_copy(
            idx,
            param_manager.shared_params[param_name]
        )
    
    # This runtime now references shared device memory
    # No data duplication!
```

### Step 4: Fast Model Switching

```python
# Switch models by just changing active runtime
executor.set_active_model("model_5")  # <1ms
executor.set_input("data", input_tensor)
executor.run()
output = executor.get_output(0)

# No parameter upload needed!
# Just execute different computation graph
```

---

## Memory Savings Breakdown

### Without Sharing (Baseline)

```
Model 0: 25 MB params
Model 1: 25 MB params
...
Model 24: 25 MB params
─────────────────────
Total: 625 MB
```

### With Sharing (Phase 2)

```
Shared pool: 25 MB params (uploaded once)
Runtime 0 overhead: ~100 KB
Runtime 1 overhead: ~100 KB
...
Runtime 24 overhead: ~100 KB
─────────────────────
Total: ~27.5 MB (96% savings!)
```

---

## API Usage

### Basic Usage

```python
from graph_switching_phase2 import Phase2Builder
from execute_candidate_set_refactored import Config

# Setup
config = Config()
config.experiment_name = "sa_lam_2.0"
config.device_host = "10.42.0.188"

# Build executor
builder = Phase2Builder(config)
executor = builder.build_from_experiment(
    "sa_lam_2.0",
    num_models=25
)

# Run inference on different models
for model_id in executor.model_ids:
    executor.set_active_model(model_id)
    executor.set_input("data", input_data)
    executor.run()
    output = executor.get_output(0)
    
    # Process output...
```

### Advanced Usage

```python
# Manual construction
from graph_switching_phase2 import (
    SharedParamManager,
    MultiRuntimeExecutor,
    CompiledModelInfo
)

# Create components
ctx = tvm.ext_dev(0)  # VTA device
param_manager = SharedParamManager(ctx)

# Upload shared params
param_manager.upload_params(ofa_params)

# Create executor
executor = MultiRuntimeExecutor(ctx)
executor.param_manager = param_manager

# Add models
for compiled in compiled_models:
    info = CompiledModelInfo(
        model_id=compiled.arch,
        graph_json=compiled.graph,
        lib=compiled.lib,
        param_names=list(compiled.params.keys()),
        params=compiled.params
    )
    executor.add_model(info)

# Link all parameters
executor.link_all_params()

# Ready to use!
```

### Benchmarking

```python
# Memory stats
mem_stats = executor.get_memory_stats()
print(f"Memory savings: {mem_stats['memory_savings']['savings_percent']:.1f}%")

# Model switching performance
switch_stats = executor.benchmark_model_switching(num_switches=1000)
print(f"Avg switch time: {switch_stats['mean_ms']:.3f} ms")

# Print comprehensive summary
executor.print_summary()
```

---

## Performance Characteristics

### Expected Performance (Based on Phase 1 Analysis)

| Metric | Value |
|--------|-------|
| **Memory Savings** | ~88% (216 MB → 26 MB) |
| **Model Switch Time** | <1 ms |
| **Inference Time** | Same as baseline |
| **Compilation Time** | Same as baseline |
| **Setup Time** | +2-3 seconds (param upload) |

### Scaling

- **Memory**: O(1) - constant regardless of model count
- **Switch time**: O(1) - just pointer swap
- **Inference**: O(n) - same per model
- **Build time**: O(n) - linear in model count

---

## Troubleshooting

### Issue: Shape Mismatch Error

```
TVMError: Shape mismatch in set_input_zero_copy
```

**Cause**: Parameter shape differs between OFA and compiled model

**Solutions**:
1. Verify OFA checkpoint path is correct
2. Check model compilation succeeded
3. Ensure quantization settings match
4. Validate weight loading in `load_static_resnet_from_arch`

### Issue: Missing Parameters

```
⚠️  Missing from shared pool: ['weight_5', 'weight_10']
```

**Cause**: Compiled model requires params not in OFA dict

**Solutions**:
1. Verify OFA model loaded correctly
2. Check if these are bias terms (may need separate handling)
3. Review parameter extraction in compilation

### Issue: High Memory Usage

```
Total memory: 200 MB (expected ~25 MB)
```

**Cause**: Parameters being duplicated instead of shared

**Solutions**:
1. Verify `set_input_zero_copy()` is being used (not `set_input()`)
2. Check that shared_params are NDArrays on device
3. Ensure ctx is consistent across all operations

### Issue: Slow Model Switching

```
Avg switch time: 50 ms (expected <1 ms)
```

**Cause**: Unnecessary work during switch

**Solutions**:
1. Check if parameters are being re-uploaded
2. Verify no graph recompilation happening
3. Profile to find bottleneck
4. Ensure warm-up run completed

---

## Testing Strategy

### Unit Tests
```bash
# Test individual components
python -c "from shared_param_manager import SharedParamManager; # test..."
```

### Integration Test (Quick)
```bash
# 3 models, ~3 minutes
./run_phase2_quick_test.sh
```

### Full System Test
```bash
# All 25 models, ~20 minutes
./run_phase2_full.sh
```

### Validation Checklist

- [ ] All models compile successfully
- [ ] Parameters uploaded to device
- [ ] Zero-copy linking works
- [ ] Memory savings ≥ 80%
- [ ] Model switching < 1ms
- [ ] Inference accuracy unchanged
- [ ] No memory leaks
- [ ] VTA execution works

---

## Comparison: Phase 2 vs Alternatives

### vs. Separate Runtimes (Current)

| Aspect | Separate | Phase 2 |
|--------|----------|---------|
| Memory | 625 MB | 27 MB |
| Switch time | N/A | <1 ms |
| Complexity | Low | Medium |
| Maintenance | Easy | Easy |

### vs. C++ MultiGraphRuntime (Future)

| Aspect | C++ Runtime | Phase 2 |
|--------|-------------|---------|
| Memory | 25 MB | 27 MB |
| Switch time | <0.1 ms | <1 ms |
| Development | 2-3 weeks | Done |
| C++ changes | Required | None |
| Risk | Medium | Low |

**Recommendation**: Use Phase 2 now, migrate to C++ if needed later.

---

## Next Steps

### Immediate (This Week)

1. ✅ Run quick test (3 models)
2. ⏳ Validate memory savings
3. ⏳ Benchmark switching performance
4. ⏳ Test on VTA hardware

### Short Term (Next 2 Weeks)

5. ⏳ Run full test (25 models)
6. ⏳ Validate all models execute correctly
7. ⏳ Measure power consumption
8. ⏳ Document any issues

### Medium Term (Next Month)

9. ⏳ Integrate into main execution pipeline
10. ⏳ Add automated testing
11. ⏳ Optimize further if needed
12. ⏳ Consider C++ migration if critical

---

## Code Organization

```
graph_switching_phase2/
├── __init__.py                   # Package exports
├── README.md                     # Overview documentation
├── IMPLEMENTATION_GUIDE.md       # This file
│
├── shared_param_manager.py       # Core: shared parameter pool
├── multi_runtime_executor.py     # Core: multi-runtime management
├── phase2_builder.py             # Core: build from candidate set
│
├── test_phase2_quick.py          # Test: quick validation (3 models)
├── test_phase2_full.py           # Test: full validation (25 models)
│
├── run_phase2_quick_test.sh      # Script: run quick test
├── run_phase2_full.sh            # Script: run full test
│
└── phase2_results/               # Output: test results
    ├── phase2_full_test_results.json
    └── phase2_full_test.log
```

---

## References

- **Phase 1 Analysis**: `../phase1_results/PHASE1_ANALYSIS_REPORT.md`
- **Design Doc**: `/home/srchand/Desktop/research/TVM_Intel_Fork/MULTI_GRAPH_RUNTIME_DESIGN.md`
- **Quick Reference**: `/home/srchand/Desktop/research/TVM_Intel_Fork/PHASE1_QUICK_REFERENCE.md`

---

**Last Updated**: March 7, 2026

