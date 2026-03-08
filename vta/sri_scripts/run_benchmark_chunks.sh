#!/bin/bash
# Benchmark candidate set models on VTA in chunks
# Usage: ./run_benchmark_chunks.sh [start_chunk] [end_chunk] [chunk_size]
START_CHUNK=${1:-0}
END_CHUNK=${2:-50}
CHUNK_SIZE=${3:-10}
OUTPUT_DIR="benchmark_results"
mkdir -p "$OUTPUT_DIR/logs"
for chunk_id in $(seq $START_CHUNK $((END_CHUNK - 1))); do
    echo "Processing Chunk $chunk_id..."
    python benchmark_candidate_set_chunked.py --chunk_id $chunk_id --chunk_size $CHUNK_SIZE 2>&1 | tee "$OUTPUT_DIR/logs/chunk_${chunk_id}.log"
done
echo "All chunks complete. Running aggregation..."
python aggregate_benchmark_results.py --input_dir "$OUTPUT_DIR"
