# VTA Candidate Set Benchmarking Suite

## Overview

This suite benchmarks candidate set models on VTA hardware, measuring mean and standard deviation of execution time for each model. Processing is done in chunks for stability and resume capability.

## Files

1. **`benchmark_candidate_set_chunked.py`** - Main benchmarking script
2. **`run_benchmark_chunks.sh`** - Bash wrapper for running all chunks
3. **`aggregate_benchmark_results.py`** - Aggregates chunk results into final outputs

## Features

- ✅ **Chunked Processing**: Process models in chunks of 10 (configurable)
- ✅ **Resume Capability**: Resume from any chunk using `--chunk_id`
- ✅ **Model Filtering**: Excludes unsupported patterns (same as `extract_workloads_chunked.py`)
- ✅ **Precise Timing**: Uses `number=4, repeat=3` for reliable measurements
- ✅ **Dual Output**: Both JSON (JSONL) and CSV formats
- ✅ **Error Logging**: Detailed logs for failed models
- ✅ **Progress Tracking**: Per-chunk and master logs

## Configuration

### Default Settings

```python
# Timing (same as execute_candidate_set_refactored.py)
run_num = 4      # Number of measurements per repeat
run_rep = 3      # Number of repeats

# Chunking
chunk_size = 10  # Models per chunk

# Device
device_host = "10.42.0.188"
device_port = "9091"
```

### Paths

All paths are configurable via command-line arguments:

- Architecture config: `/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json`
- Architecture meta: `/home/srchand/Desktop/research/OFA_Obfs/transferability_matrix_try_final_remaining/transfer_meta.jsonl`
- Schedule logs: `/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/*.log`
- Output directory: `benchmark_results/`

## Usage

### Option 1: Run All Chunks (Automated)

```bash
# Run chunks 0-49 (500 models with chunk_size=10)
./run_benchmark_chunks.sh 0 50 10

# Custom range
./run_benchmark_chunks.sh 10 20 10  # Chunks 10-19
```

### Option 2: Run Individual Chunks (Manual)

```bash
# Run chunk 0
python benchmark_candidate_set_chunked.py --chunk_id 0 --chunk_size 10

# Run chunk 1
python benchmark_candidate_set_chunked.py --chunk_id 1 --chunk_size 10

# Custom configuration
python benchmark_candidate_set_chunked.py \
    --chunk_id 5 \
    --chunk_size 10 \
    --device_host 10.42.0.188 \
    --device_port 9091 \
    --output_dir my_results
```

### Option 3: Resume from Specific Chunk

If chunk 15 failed:

```bash
# Re-run just chunk 15
python benchmark_candidate_set_chunked.py --chunk_id 15 --chunk_size 10

# Or use bash script to continue from chunk 15
./run_benchmark_chunks.sh 15 50 10
```

### Aggregating Results

After all chunks complete:

```bash
python aggregate_benchmark_results.py \
    --input_dir benchmark_results \
    --output_dir benchmark_results
```

## Output Files

### Per-Chunk Outputs

Each chunk generates:

```
benchmark_results/
├── benchmark_chunk_0.jsonl      # JSONL format (like transfer_meta.jsonl)
├── benchmark_chunk_0.csv        # CSV format
├── benchmark_chunk_1.jsonl
├── benchmark_chunk_1.csv
├── ...
├── errors_chunk_0.log           # Only if chunk had failures
├── logs/
│   ├── chunk_0.log              # Full console output
│   ├── chunk_1.log
│   └── ...
```

### JSONL Format (per chunk)

```jsonl
{"model_id": "model_123", "mean_exec_time_ms": 45.23, "std_exec_time_ms": 2.15, "compile_time_s": 12.34, "success": true}
{"model_id": "model_456", "mean_exec_time_ms": 52.18, "std_exec_time_ms": 1.98, "compile_time_s": 11.87, "success": true}
{"model_id": "model_789", "mean_exec_time_ms": 0.0, "std_exec_time_ms": 0.0, "compile_time_s": 0.0, "success": false, "error": "RuntimeError: ..."}
```

### CSV Format (per chunk)

```csv
model_id,mean_exec_time_ms,std_exec_time_ms,compile_time_s,success,error
model_123,45.2300,2.1500,12.34,True,
model_456,52.1800,1.9800,11.87,True,
model_789,N/A,N/A,0.00,False,RuntimeError: ...
```

### Aggregated Outputs

After running `aggregate_benchmark_results.py`:

```
benchmark_results/
├── benchmark_results_all.jsonl      # All models combined
├── benchmark_results_all.csv        # All models combined (CSV)
├── benchmark_successful.jsonl       # Successful models only
├── benchmark_failed.jsonl           # Failed models only
├── benchmark_statistics.json        # Statistics summary
```

### Statistics File

```json
{
  "total_models": 500,
  "successful_models": 485,
  "failed_models": 15,
  "success_rate": 0.97,
  "chunks_processed": 50,
  "execution_time_stats": {
    "min_ms": 35.12,
    "max_ms": 78.45,
    "avg_ms": 48.23,
    "median_ms": 46.89
  },
  "std_time_stats": {
    "min_ms": 0.85,
    "max_ms": 3.21,
    "avg_ms": 1.92
  },
  "compile_time_stats": {
    "min_s": 8.45,
    "max_s": 18.32,
    "avg_s": 12.15,
    "total_s": 5892.75
  },
  "chunk_statistics": [...]
}
```

## Model Filtering

Models are filtered using the same logic as `extract_workloads_chunked.py`:

**Excluded**: Models where `out_channel_setting_list[0] == 0` AND `decomp_type_list[0]` contains any residual with `decomp_type == 2`

**Reason**: These patterns are not supported by Relay frontend

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Architectures                       │
│              (arch_config + arch_meta)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Filter Unsupported Models                      │
│     (skip decomp_type==2 in certain positions)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Divide into Chunks (size=10)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴──────────┬──────────────┬──────────────┐
         ▼                      ▼              ▼              ▼
    ┌────────┐            ┌────────┐     ┌────────┐    ┌────────┐
    │Chunk 0 │            │Chunk 1 │ ... │Chunk N-1│   │Chunk N │
    │(10 mod)│            │(10 mod)│     │(10 mod) │   │(<= 10) │
    └───┬────┘            └───┬────┘     └───┬─────┘   └───┬────┘
        │                     │              │             │
        ▼                     ▼              ▼             ▼
   For each model:       (same process)  (same)       (same)
   ├─ Load from OFA
   ├─ Convert to Relay
   ├─ Quantize + Pack
   ├─ Compile for VTA
   ├─ Upload to device
   ├─ Run timer (4x3)
   ├─ Log mean & std
   └─ Save results
        │
        ▼
   ┌──────────────────────┐
   │  benchmark_chunk_X   │
   │  .jsonl + .csv       │
   └──────────────────────┘
        │
        └──────────────────────────────────────────┐
                                                   │
                     After all chunks              │
                            ▼                      ▼
                    ┌─────────────────────────────────┐
                    │   aggregate_benchmark_results   │
                    │                                 │
                    │  Combines all chunk files into: │
                    │  • benchmark_results_all.jsonl  │
                    │  • benchmark_results_all.csv    │
                    │  • benchmark_statistics.json    │
                    └─────────────────────────────────┘
```

## Example Session

```bash
# Terminal 1: Start RPC server on VTA device
# (on ZCU104)
python -m tvm.exec.rpc_server --host 0.0.0.0 --port 9091

# Terminal 2: Run benchmarks
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts

# Test single chunk first
python benchmark_candidate_set_chunked.py --chunk_id 0 --chunk_size 10

# If successful, run all chunks
./run_benchmark_chunks.sh 0 50 10

# Monitor progress
tail -f benchmark_results/logs/chunk_0.log

# After completion, aggregate results
python aggregate_benchmark_results.py --input_dir benchmark_results
```

## Troubleshooting

### Chunk Fails

1. Check error log: `benchmark_results/errors_chunk_X.log`
2. Check full log: `benchmark_results/logs/chunk_X.log`
3. Re-run that specific chunk:
   ```bash
   python benchmark_candidate_set_chunked.py --chunk_id X --chunk_size 10
   ```

### RPC Connection Issues

- Verify RPC server is running on VTA device
- Check network connectivity: `ping 10.42.0.188`
- Verify port is correct: `--device_port 9091`

### Out of Memory

- Reduce chunk size: `--chunk_size 5`
- Ensure VTA device is programmed correctly
- Check for leaked resources (restart RPC server)

### Model Compilation Failures

- Check if model has unsupported pattern (should be filtered)
- Verify schedule logs are available
- Check architecture JSON format

## Performance Expectations

### Timing Per Model

- Compile: 8-18 seconds
- Upload: 1-2 seconds
- Benchmark: 5-10 seconds (with 4x3 timing)
- **Total per model**: ~15-30 seconds

### Timing Per Chunk (10 models)

- **Average**: 3-5 minutes
- **With failures**: May be longer

### Full Run (500 models, 50 chunks)

- **Estimated time**: 2.5-4 hours
- **Compile time alone**: ~1.5-2 hours
- **Benchmark time**: ~1 hour

## Comparison with extract_workloads_chunked.py

| Feature | extract_workloads_chunked | benchmark_candidate_set_chunked |
|---------|---------------------------|--------------------------------|
| Purpose | Extract conv2d workloads | Measure execution time |
| Execution | PyTorch only | Full VTA execution |
| Output | Workload specs | Timing measurements |
| Per model | <1 second | ~20 seconds |
| Filtering | Yes (same logic) | Yes (same logic) |
| Chunking | Yes | Yes |
| Resume | Yes | Yes |

## Advanced Usage

### Custom Timing Configuration

Edit `benchmark_candidate_set_chunked.py`:

```python
# Line ~60
run_num: int = 10  # Increase for more stable measurements
run_rep: int = 5   # Increase for better statistics
```

### Different Device

```bash
python benchmark_candidate_set_chunked.py \
    --chunk_id 0 \
    --device_host 192.168.1.100 \
    --device_port 9090
```

### Process Specific Models

Modify the filtering logic or manually specify model IDs by editing `chunk_model_ids` in the script.

## Notes

- Results are saved after each model completes (no loss on failure)
- Failed models are logged but don't stop chunk execution
- Aggregation can be run multiple times (idempotent)
- JSONL format matches `transfer_meta.jsonl` for easy integration

## Related Scripts

- `extract_workloads_chunked.py` - Extract workloads in chunks
- `execute_candidate_set_refactored.py` - Single model execution (basis for this script)
- `tune_relay_arm_candidate_set.py` - AutoTVM tuning for candidate set

---

**Created**: March 1, 2026  
**Status**: Ready to use (do not run yet per user request)

