# Explicit Task Tuning Script

## Overview

The `tune_relay_arm_explicit_tasks.py` script allows you to tune specific AutoTVM tasks without extracting them from a neural network. This is useful when you want to optimize specific operations with particular shapes.

## Features

- **Explicit Task Definition**: Directly specify the tasks you want to tune
- **No Network Required**: No need to define a full neural network
- **Easy Customization**: Simple to add, remove, or modify tasks
- **Full AutoTVM Integration**: Uses the same tuning infrastructure as network-based tuning

## Currently Configured Tasks

The script is configured to tune the following tasks:

### Dense Operations

1. **dense_nopack.x86** for shape (1, 512) → (10, 512)
2. **dense_pack.x86** for shape (1, 512) → (10, 512)
3. **dense_nopack.x86** for shape (1, 256) → (10, 256)
4. **dense_pack.x86** for shape (1, 256) → (10, 256)

### Convolution Operations

5. **conv2d_nchw_spatial_pack.arm_cpu**
   - Input shape: (1, 3, 224, 224)
   - Weight shape: (128, 3, 7, 7)
   - Stride: (2, 2)
   - Padding: (3, 3, 3, 3)
   - Dilation: (1, 1)

## Configuration Space Sizes

- Dense tasks: 36-90 configurations each
- Conv2d task: ~128,000 configurations

## Prerequisites

1. **RPC Tracker**: Must be running on port 9190
   ```bash
   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
   ```

2. **Target Device**: Must be registered with the tracker (default: "zcu104")
   ```bash
   python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=zcu104
   ```

3. **Python Environment**: TVM must be built and accessible

## Usage

### Basic Usage

```bash
python tune_relay_arm_explicit_tasks.py
```

### Customizing Tasks

To add or modify tasks, edit the `create_explicit_tasks()` function:

```python
def create_explicit_tasks(target):
    tasks = []
    
    # Example: Add a dense task for shape (1, 1024) → (10, 1024)
    dense_1024_args = (
        ('TENSOR', (1, 1024), 'float32'),
        ('TENSOR', (10, 1024), 'float32'),
        None,
        'float32'
    )
    tasks.append(autotvm.task.create(
        'dense_nopack.x86',
        args=dense_1024_args,
        target=target
    ))
    
    # Example: Add a conv2d task
    conv2d_args = (
        ('TENSOR', (1, 64, 56, 56), 'float32'),  # input
        ('TENSOR', (64, 64, 3, 3), 'float32'),   # weight
        (1, 1),      # stride
        (1, 1, 1, 1), # padding
        (1, 1),      # dilation
        'float32'
    )
    tasks.append(autotvm.task.create(
        'conv2d_nchw_spatial_pack.arm_cpu',
        args=conv2d_args,
        target=target
    ))
    
    return tasks
```

### Available Task Names

Common task names for different operations:

- **Dense**: `dense_nopack.x86`, `dense_pack.x86`
- **Conv2D (ARM)**: `conv2d_nchw_spatial_pack.arm_cpu`
- **Conv2D (x86)**: `conv2d_NCHWc.x86`
- **Depthwise Conv2D**: `depthwise_conv2d_nchw.arm_cpu`

### Configuration Options

Edit the configuration section at the bottom of the script:

```python
# Target configuration
target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")

# Device key for RPC tracker
device_key = "zcu104"

# Tuning options
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",           # XGBoost tuner
    "n_trial": 1000,           # Number of trials per task
    "early_stopping": 800,     # Stop if no improvement after N trials
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"),
        runner=autotvm.RPCRunner(
            device_key,
            host="127.0.0.1",
            port=9190,
            number=5,          # Number of measurements per config
            timeout=10,        # Timeout in seconds
        ),
    ),
}
```

## Output

Tuning results are saved to:
```
/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/arm_cpu_explicit_tasks_zcu104.log
```

## Tuning Process

For each task, the script will:

1. Create the task with specified shape parameters
2. Initialize the configuration space
3. Run the XGBoost tuner for up to 1000 trials
4. Measure performance on the target device via RPC
5. Save the best configuration to the log file

## Using Tuned Results

After tuning, use the log file when compiling:

```python
with autotvm.apply_history_best(log_file):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
```

## Troubleshooting

### "Could not find a registered function for the task"

Make sure `from tvm import topi` is imported. This registers all task functions.

### RPC Connection Errors

- Verify the RPC tracker is running
- Check that the device is registered with the correct key
- Ensure firewall allows connections on port 9190

### Out of Memory Errors

Reduce `n_trial` or the batch size (`number` parameter in RPCRunner).

## Comparison with Network-based Tuning

**Network-based tuning** (e.g., `tune_relay_arm_candidate_set.py`):
- Extracts tasks from a full network
- Tunes all operations in the network
- Requires model definition

**Explicit task tuning** (this script):
- Manually specify individual tasks
- Focus on specific operation shapes
- No network required
- More control over what gets tuned

## Example Output

```
Creating explicit tasks...

Found 5 tasks to tune:
  1. dense_nopack.x86
     Workload: dense_nopack.x86
     Input shape: (1, 512), Weight shape: (10, 512)

  2. dense_pack.x86
     Workload: dense_pack.x86
     Input shape: (1, 512), Weight shape: (10, 512)

  ...

Starting tuning process...
================================================================================
Tuning Task 1/5: Task(func_name=dense_nopack.x86, ...)
================================================================================
Tuning with 40 trials (config space size: 40)
[Task 1/5]  Current/Best:   10.45/  12.34 GFLOPS | Progress: (40/40) | 45.23 s
...
```

## Notes

- Tuning can take several hours depending on the number of tasks and trials
- The script uses transfer learning between tasks by default
- Results are incrementally saved during tuning
- If interrupted, you can resume by keeping the `.tmp` log file

