# Explicit Task Tuning - Quick Summary

## What Was Created

A new script `tune_relay_arm_explicit_tasks.py` that allows explicit specification of AutoTVM tasks to tune.

## Tasks Currently Configured

| # | Task Name | Operation | Input Shape | Weight Shape | Config Space |
|---|-----------|-----------|-------------|--------------|--------------|
| 1 | dense_nopack.x86 | Dense | (1, 512) | (10, 512) | 40 configs |
| 2 | dense_pack.x86 | Dense | (1, 512) | (10, 512) | 90 configs |
| 3 | dense_nopack.x86 | Dense | (1, 256) | (10, 256) | 36 configs |
| 4 | dense_pack.x86 | Dense | (1, 256) | (10, 256) | 81 configs |
| 5 | conv2d_nchw_spatial_pack.arm_cpu | Conv2D | (1, 3, 224, 224) | (128, 3, 7, 7) | ~128K configs |

### Conv2D Parameters
- Stride: (2, 2)
- Padding: (3, 3, 3, 3)
- Dilation: (1, 1)

## Quick Start

1. **Start RPC Tracker:**
   ```bash
   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
   ```

2. **Register Device (on target device):**
   ```bash
   python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=zcu104
   ```

3. **Run Tuning:**
   ```bash
   cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
   python tune_relay_arm_explicit_tasks.py
   ```

## Output Location

```
/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/arm_cpu_explicit_tasks_zcu104.log
```

## Key Features

✅ **No network required** - Tune individual operations directly
✅ **Easy customization** - Simple function to add/modify tasks
✅ **Proper task registration** - Uses `autotvm.task.create()` API
✅ **Transfer learning** - Shares knowledge between tasks
✅ **XGBoost tuner** - Smart configuration search

## Differences from Original Script

### Original (`tune_relay_arm_candidate_set.py`)
- Extracts tasks from ResNet models
- Requires full model definition
- Tunes all layers automatically
- 100+ tasks from complete networks

### New (`tune_relay_arm_explicit_tasks.py`)
- Manually specify each task
- No model needed
- Tune only what you specify
- 5 tasks (customizable)

## How to Add More Tasks

Edit the `create_explicit_tasks()` function:

```python
def create_explicit_tasks(target):
    tasks = []
    
    # Add your task here:
    my_args = (
        ('TENSOR', (batch, in_features), 'float32'),
        ('TENSOR', (out_features, in_features), 'float32'),
        None,
        'float32'
    )
    tasks.append(autotvm.task.create(
        'dense_pack.x86',  # or 'dense_nopack.x86'
        args=my_args,
        target=target
    ))
    
    return tasks
```

## Verification

All tasks have been tested and verified to create successfully:

```
✓ dense_nopack.x86 (1, 512) → (10, 512)
✓ dense_pack.x86 (1, 512) → (10, 512)
✓ dense_nopack.x86 (1, 256) → (10, 256)
✓ dense_pack.x86 (1, 256) → (10, 256)
✓ conv2d_nchw_spatial_pack.arm_cpu (1,3,224,224) with (128,3,7,7)
```

## Estimated Tuning Time

With 1000 trials per task and 5 measurements per config:

- Dense tasks (4): ~30-60 minutes total
- Conv2D task (1): ~3-6 hours (large config space)
- **Total: ~4-7 hours**

Adjust `n_trial` and `early_stopping` in the script to reduce time.

## Common Task Names Reference

### Dense Operations
- `dense_nopack.x86` - Unpacked dense (simpler)
- `dense_pack.x86` - Packed dense (cache-friendly)

### Conv2D Operations
- `conv2d_nchw_spatial_pack.arm_cpu` - ARM spatial packing
- `conv2d_NCHWc.x86` - x86 with channel packing
- `depthwise_conv2d_nchw.arm_cpu` - Depthwise convolution

### Other Operations
- `conv2d_transpose_nchw.x86` - Transposed convolution
- `batch_matmul.x86` - Batch matrix multiplication
- `pool.x86` - Pooling operations

## Next Steps

1. ✅ Script created and tested
2. ✅ Task creation verified
3. ⏳ Run tuning (requires RPC setup)
4. ⏳ Collect results
5. ⏳ Use in compilation with `autotvm.apply_history_best()`

---
**Created:** March 1, 2026
**Status:** Ready to use

