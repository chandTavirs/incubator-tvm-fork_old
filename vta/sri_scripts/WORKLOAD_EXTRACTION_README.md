# Chunked Workload Extraction System

This system allows you to extract workloads from a large set of OFA networks in manageable chunks, avoiding memory issues that occur when processing all networks at once.

## Key Features

- **Chunked Processing**: Process networks in batches to avoid memory issues
- **Automatic Filtering**: Excludes 179 known workloads from candidate_set_wkls
- **Robust Error Handling**: Continues on failures, logs all errors with full details
- **Detailed Error Logging**: Failed models are logged with architecture configs for debugging
- **Progress Tracking**: Real-time feedback on chunk and model-level progress

## Files

1. **extract_workloads_chunked.py** - Main extraction script that processes networks in chunks
2. **aggregate_workloads.py** - Aggregates workloads from all chunk files and removes duplicates
3. **run_workload_extraction_batch.sh** - Batch runner that executes extraction and aggregation

## Usage

### Quick Start (Mock Run)

Test with 2 iterations, 2 networks per chunk (4 networks total):

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
bash run_workload_extraction_batch.sh 2 2 /home/srchand/anaconda3/envs/tvm-build-il-2/bin/python
```

### Full Run (1000 Networks)

Process all ~826 valid networks with 50 iterations, 20 networks per chunk:

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
bash run_workload_extraction_batch.sh 50 20 /home/srchand/anaconda3/envs/tvm-build-il-2/bin/python
```

This will process approximately 1000 networks (50 × 20) in 50 separate runs.

### Custom Configuration

```bash
bash run_workload_extraction_batch.sh <num_iterations> <chunk_size> <python_binary>
```

**Parameters:**
- `num_iterations`: Number of chunks to process (default: 50)
- `chunk_size`: Number of networks per chunk (default: 20)
- `python_binary`: Python executable to use (default: python)

**Example:**
```bash
# Process 100 networks with 10 chunks of 10 networks each
bash run_workload_extraction_batch.sh 10 10 /home/srchand/anaconda3/envs/tvm-build-il-2/bin/python
```

## Output

### Directory Structure

```
wkl_extraction/
├── workloads_chunk_0.txt          # Workloads from chunk 0
├── workloads_chunk_1.txt          # Workloads from chunk 1
├── ...
├── workloads_chunk_N.txt          # Workloads from chunk N
└── aggregated_workloads_final.txt # Final aggregated unique workloads
```

### Aggregated Output Format

The final output file contains unique workloads excluding the existing `candidate_set_wkls`:

```python
("workload_0", Workload(1, 14, 14, 256, 128, 3, 3, 1, 1, 2, 2)),
("workload_1", Workload(1, 14, 14, 64, 512, 5, 5, 2, 2, 2, 2)),
...
```

## Individual Script Usage

### Run Single Chunk

```bash
python extract_workloads_chunked.py --chunk_id 0 --chunk_size 20 --output_dir wkl_extraction
```

**Options:**
- `--chunk_id`: Chunk ID (0-indexed)
- `--chunk_size`: Number of networks per chunk (default: 20)
- `--arch_config`: Path to architecture config JSON
- `--arch_meta`: Path to architecture meta JSONL
- `--output_dir`: Output directory (default: wkl_extraction)

### Run Aggregator Manually

```bash
python aggregate_workloads.py --input_dir wkl_extraction --output_file aggregated_workloads.txt
```

**Options:**
- `--input_dir`: Directory containing chunk files (default: wkl_extraction)
- `--output_file`: Output file path (default: aggregated_workloads.txt)

## Features

### Automatic Filtering

The aggregator automatically excludes 179 existing workloads from `candidate_set_wkls` that are already known.

### Memory Management

Processing networks in chunks prevents the script from being killed due to memory issues. Each chunk:
1. Loads the OFA model
2. Processes N networks
3. Saves workloads to a file
4. Exits (freeing memory)

### Error Handling

- The batch script continues processing even if individual chunks fail
- Each chunk's output is preserved separately
- Failed chunks are logged but don't stop the entire process

## Example Output

```
===============================================
Workload Extraction Batch Runner
===============================================
Number of iterations: 2
Chunk size: 2 networks
Output directory: .../wkl_extraction
Python binary: /home/srchand/anaconda3/envs/tvm-build-il-2/bin/python
===============================================

Starting workload extraction...

----------------------------------------
Running chunk 0 of 2...
----------------------------------------
...
Chunk 0 complete!
  Total workloads extracted: 518
  Saved to: .../workloads_chunk_0.txt

----------------------------------------
Running chunk 1 of 2...
----------------------------------------
...
Chunk 1 complete!
  Total workloads extracted: 276
  Saved to: .../workloads_chunk_1.txt

===============================================
All chunks processed. Running aggregator...
===============================================

Found 2 chunk files
  workloads_chunk_0.txt: 518 workloads
  workloads_chunk_1.txt: 276 workloads

Total workloads collected: 794
Unique workloads: 57
Candidate set workloads excluded: 179
After filtering: 29 new workloads

Output written to: .../aggregated_workloads_final.txt

===============================================
Batch extraction complete!
Final output: .../aggregated_workloads_final.txt
===============================================
```

## Notes

- The script may show "free(): invalid pointer" errors at the end of each chunk due to PyTorch/TVM memory cleanup. This is expected and doesn't affect the results.
- Valid candidate models: ~826 (after filtering unsupported architectures)
- The script skips 174 architectures with unsupported relay frontend patterns
- Processing time: ~30-60 seconds per network depending on complexity

## Configuration

Default paths (can be modified in `extract_workloads_chunked.py`):
- OFA model: `/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth`
- Architecture config: `/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json`
- Architecture meta: `/home/srchand/Desktop/research/OFA_Obfs/transferability_matrix_try_final_remaining/transfer_meta.jsonl`


