# VTA Candidate Set Benchmarking Suite - Complete

## ✅ All Scripts Created Successfully

### Created Files

1. **benchmark_candidate_set_chunked.py** (24 KB)
   - Main benchmarking script
   - Processes models in chunks
   - Measures mean and std execution time
   - Uses VTA execution functions from execute_candidate_set_refactored.py
   - Same filtering logic as extract_workloads_chunked.py

2. **run_benchmark_chunks.sh** (618 B)
   - Bash wrapper for running multiple chunks
   - Handles logging and progress tracking
   - Calls aggregation after completion

3. **aggregate_benchmark_results.py** (9.3 KB)
   - Combines results from all chunks
   - Generates final JSONL and CSV files
   - Calculates statistics (min, max, avg, median)

4. **BENCHMARK_CANDIDATE_SET_README.md** (13 KB)
   - Comprehensive documentation
   - Usage examples
   - Troubleshooting guide
   - Performance estimates

5. **BENCHMARK_QUICK_START.txt** (9.3 KB)
   - Quick reference guide
   - Command examples
   - Output format specifications

## Summary of Implementation

### What the Scripts Do

1. **Load architectures** from arch_config JSON and meta JSONL
2. **Filter models** using the same exclusion logic as extract_workloads_chunked.py:
   - Skip models with `out_channel_setting_list[0] == 0` AND `decomp_type == 2`
3. **Divide into chunks** of 10 models each
4. **For each model in chunk**:
   - Load from OFA checkpoint
   - Convert PyTorch → Relay IR
   - Quantize and graph pack for VTA
   - Compile with AutoTVM schedule logs
   - Upload to VTA device via RPC
   - Create runtime and load parameters
   - Run timer with `number=4, repeat=3`
   - Calculate mean and std execution time (ms)
   - Save to JSONL and CSV
5. **After all chunks**: Aggregate results into final outputs

### Key Features Implemented

✅ **Chunk size**: 10 models (configurable)
✅ **Timing settings**: `number=4, repeat=3` (same as execute_candidate_set_refactored.py)
✅ **Output formats**: Both JSON (JSONL) and CSV
✅ **JSON format**: Matches transfer_meta.jsonl style
✅ **Resume capability**: `--chunk_id` argument to start from specific chunk
✅ **Progress saving**: Results saved after each chunk automatically
✅ **Exclusion patterns**: Same as extract_workloads_chunked.py
✅ **Error handling**: Detailed error logs for failed models
✅ **Statistics**: Min, max, avg, median execution times

### Configuration

```python
# Default settings
chunk_size = 10
run_num = 4      # Same as execute_candidate_set_refactored.py
run_rep = 3      # Same as execute_candidate_set_refactored.py
device_host = "10.42.0.188"
device_port = "9091"
```

### Usage Examples

```bash
# Run single chunk (test)
python benchmark_candidate_set_chunked.py --chunk_id 0 --chunk_size 10

# Run all chunks
./run_benchmark_chunks.sh 0 50 10

# Resume from chunk 15
./run_benchmark_chunks.sh 15 50 10

# Aggregate results
python aggregate_benchmark_results.py --input_dir benchmark_results
```

### Output Files

**Per chunk:**
- `benchmark_chunk_X.jsonl` - JSONL format (like transfer_meta.jsonl)
- `benchmark_chunk_X.csv` - CSV format
- `errors_chunk_X.log` - Error details (if failures)
- `logs/chunk_X.log` - Full console output

**Aggregated:**
- `benchmark_results_all.jsonl` - All models
- `benchmark_results_all.csv` - All models (CSV)
- `benchmark_successful.jsonl` - Successful only
- `benchmark_failed.jsonl` - Failed only
- `benchmark_statistics.json` - Statistics

### JSONL Format (matches transfer_meta.jsonl)

```jsonl
{"model_id": "model_123", "mean_exec_time_ms": 45.23, "std_exec_time_ms": 2.15, "compile_time_s": 12.34, "success": true}
{"model_id": "model_456", "mean_exec_time_ms": 52.18, "std_exec_time_ms": 1.98, "compile_time_s": 11.87, "success": true}
{"model_id": "model_789", "mean_exec_time_ms": 0.0, "std_exec_time_ms": 0.0, "compile_time_s": 0.0, "success": false, "error": "..."}
```

### Performance Estimates

| Metric | Time |
|--------|------|
| Per model | 15-30 seconds |
| Per chunk (10 models) | 3-5 minutes |
| Full run (500 models) | 2.5-4 hours |

## Verification

All files created and verified:
- ✅ benchmark_candidate_set_chunked.py (24 KB)
- ✅ run_benchmark_chunks.sh (618 B, executable)
- ✅ aggregate_benchmark_results.py (9.3 KB)
- ✅ BENCHMARK_CANDIDATE_SET_README.md (13 KB)
- ✅ BENCHMARK_QUICK_START.txt (9.3 KB)

## Next Steps (When Ready to Run)

1. **Test single chunk**:
   ```bash
   python benchmark_candidate_set_chunked.py --chunk_id 0 --chunk_size 10
   ```

2. **Run all chunks** (if test successful):
   ```bash
   ./run_benchmark_chunks.sh 0 50 10
   ```

3. **Aggregate results**:
   ```bash
   python aggregate_benchmark_results.py --input_dir benchmark_results
   ```

## Questions Answered

1. ✅ **Chunk size**: 10 models per chunk
2. ✅ **Timing configuration**: `number=4, repeat=3` (same as execute_candidate_set_refactored.py)
3. ✅ **Output format**: Both JSON (JSONL) and CSV
4. ✅ **JSON format**: Uses transfer_meta.jsonl as reference
5. ✅ **Resume mechanism**: 
   - Saves progress after each chunk automatically
   - Allows `--chunk_id` argument to resume from specific chunk
6. ✅ **Exclusion patterns**: Same as extract_workloads_chunked.py

## Technical Details

### Functions Leveraged from execute_candidate_set_refactored.py

- `load_ofa_model()` - Load OFA checkpoint
- `load_static_resnet_from_arch()` - Create static ResNet
- `setup_rpc_connection()` - Connect to VTA device
- `pytorch_to_relay()` - Convert to Relay IR
- `apply_quantization_and_packing()` - Quantize for VTA
- `build_relay_graph()` - Compile for VTA

### Model Filtering Logic (from extract_workloads_chunked.py)

```python
# Skip if:
if out_channel_setting_list[0] == 0:
    if any decomp_type == 2 in decomp_type_list[0]:
        skip = True
```

**Reason**: Relay frontend doesn't support this pattern

### Timing Measurement

```python
timer = m.module.time_evaluator("run", ctx, number=4, repeat=3)
tcost = timer()
mean_ms = tcost.mean * 1000
std_ms = np.std(tcost.results) * 1000
```

## Status

🟢 **READY TO USE** (when user is ready)

⚠️ **NOT EXECUTED YET** per user request - user is currently running something else

---

**Created**: March 1, 2026  
**Author**: AI Assistant  
**Purpose**: Benchmark candidate set models on VTA hardware in chunks

