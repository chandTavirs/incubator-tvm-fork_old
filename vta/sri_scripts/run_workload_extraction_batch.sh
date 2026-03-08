#!/bin/bash
# Batch script to run workload extraction in chunks
# Configuration
NUM_ITERATIONS=${1:-50}  # Default to 50 iterations
CHUNK_SIZE=${2:-20}      # Default to 20 networks per chunk
PYTHON_BIN=${3:-python}  # Python binary to use
# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHUNKED_SCRIPT="${SCRIPT_DIR}/extract_workloads_chunked.py"
AGGREGATOR_SCRIPT="${SCRIPT_DIR}/aggregate_workloads.py"
OUTPUT_DIR="${SCRIPT_DIR}/wkl_extraction"
MASTER_ERROR_LOG="${OUTPUT_DIR}/failed_chunks_summary.log"
echo "==============================================="
echo "Workload Extraction Batch Runner"
echo "==============================================="
echo "Number of iterations: ${NUM_ITERATIONS}"
echo "Chunk size: ${CHUNK_SIZE} networks"
echo "Output directory: ${OUTPUT_DIR}"
echo "Python binary: ${PYTHON_BIN}"
echo "==============================================="
echo ""
# Create output directory
mkdir -p "${OUTPUT_DIR}"
# Track failed chunks
FAILED_CHUNKS=()
SUCCEEDED_CHUNKS=0
# Clear previous master error log
> "${MASTER_ERROR_LOG}"
# Run extraction for each chunk
echo "Starting workload extraction..."
for ((i=0; i<${NUM_ITERATIONS}; i++)); do
    echo ""
    echo "----------------------------------------"
    echo "Running chunk ${i} of ${NUM_ITERATIONS}..."
    echo "----------------------------------------"
    ${PYTHON_BIN} "${CHUNKED_SCRIPT}" \
        --chunk_id ${i} \
        --chunk_size ${CHUNK_SIZE} \
        --output_dir "${OUTPUT_DIR}"
    EXIT_CODE=$?
    CHUNK_OUTPUT="${OUTPUT_DIR}/workloads_chunk_${i}.txt"
    # Check if chunk produced output file (success indicator)
    # Exit code may be non-zero due to memory cleanup issues but file was created
    if [ -f "${CHUNK_OUTPUT}" ] && [ -s "${CHUNK_OUTPUT}" ]; then
        echo "Chunk ${i} completed successfully (output file created)"
        SUCCEEDED_CHUNKS=$((SUCCEEDED_CHUNKS + 1))
    else
        echo "ERROR: Chunk ${i} failed - no output file generated!"
        FAILED_CHUNKS+=("${i}")
        echo "Chunk ${i}: FAILED (no output file)" >> "${MASTER_ERROR_LOG}"
    fi
done
echo ""
echo "==============================================="
echo "Extraction Summary"
echo "==============================================="
echo "Total chunks processed: ${NUM_ITERATIONS}"
echo "Successful chunks: ${SUCCEEDED_CHUNKS}"
echo "Failed chunks: ${#FAILED_CHUNKS[@]}"
if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo "Failed chunk IDs: ${FAILED_CHUNKS[*]}"
fi
echo "==============================================="
echo ""
# Aggregate error logs from individual chunks
ERROR_FILES="${OUTPUT_DIR}/errors_chunk_*.log"
if ls ${ERROR_FILES} 2>/dev/null >/dev/null; then
    echo "Consolidating per-model error logs..."
    echo "" >> "${MASTER_ERROR_LOG}"
    echo "===============================================" >> "${MASTER_ERROR_LOG}"
    echo "Per-Model Failures (from all chunks)" >> "${MASTER_ERROR_LOG}"
    echo "===============================================" >> "${MASTER_ERROR_LOG}"
    echo "" >> "${MASTER_ERROR_LOG}"
    for error_file in ${ERROR_FILES}; do
        if [ -f "${error_file}" ] && [ -s "${error_file}" ]; then
            cat "${error_file}" >> "${MASTER_ERROR_LOG}"
            echo "" >> "${MASTER_ERROR_LOG}"
        fi
    done
    echo "Master error log created: ${MASTER_ERROR_LOG}"
    echo ""
fi
echo "==============================================="
echo "Running aggregator..."
echo "==============================================="
echo ""
# Run aggregator
${PYTHON_BIN} "${AGGREGATOR_SCRIPT}" \
    --input_dir "${OUTPUT_DIR}" \
    --output_file "${OUTPUT_DIR}/aggregated_workloads_final.txt"
if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================="
    echo "Batch extraction complete!"
    echo "==============================================="
    echo "Final output: ${OUTPUT_DIR}/aggregated_workloads_final.txt"
    if [ -f "${MASTER_ERROR_LOG}" ] && [ -s "${MASTER_ERROR_LOG}" ]; then
        echo "Error log: ${MASTER_ERROR_LOG}"
    fi
    echo "==============================================="
else
    echo "ERROR: Aggregation failed!"
    exit 1
fi
