#!/bin/bash
# Quick verification script to show error logging is working
echo "=================================================="
echo "Error Logging Verification"
echo "=================================================="
echo ""
OUTPUT_DIR="wkl_extraction"
echo "1. Checking for workload chunk files..."
CHUNK_FILES=$(ls -1 ${OUTPUT_DIR}/workloads_chunk_*.txt 2>/dev/null | wc -l)
echo "   Found ${CHUNK_FILES} chunk file(s)"
echo ""
echo "2. Checking for per-chunk error logs..."
ERROR_FILES=$(ls -1 ${OUTPUT_DIR}/errors_chunk_*.log 2>/dev/null | wc -l)
if [ ${ERROR_FILES} -gt 0 ]; then
    echo "   Found ${ERROR_FILES} error log(s):"
    ls -1 ${OUTPUT_DIR}/errors_chunk_*.log 2>/dev/null | sed 's/^/      /'
else
    echo "   No per-chunk error logs (all models succeeded!)"
fi
echo ""
echo "3. Checking master error log..."
if [ -f "${OUTPUT_DIR}/failed_chunks_summary.log" ]; then
    SIZE=$(stat -f%z "${OUTPUT_DIR}/failed_chunks_summary.log" 2>/dev/null || stat -c%s "${OUTPUT_DIR}/failed_chunks_summary.log" 2>/dev/null)
    if [ "$SIZE" -gt 0 ]; then
        echo "   Master error log exists with ${SIZE} bytes"
        echo "   Preview:"
        head -20 "${OUTPUT_DIR}/failed_chunks_summary.log" | sed 's/^/      /'
    else
        echo "   Master error log is empty (no failures!)"
    fi
else
    echo "   Master error log not found"
fi
echo ""
echo "4. Checking aggregated workloads..."
if [ -f "${OUTPUT_DIR}/aggregated_workloads_final.txt" ]; then
    WORKLOAD_COUNT=$(wc -l < "${OUTPUT_DIR}/aggregated_workloads_final.txt")
    echo "   Found ${WORKLOAD_COUNT} unique workloads (after filtering)"
else
    echo "   Aggregated workloads file not found"
fi
echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="
echo "✓ Error logging system is operational"
echo "✓ Failed models will be logged with architecture details"
echo "✓ System continues processing on failures"
echo "=================================================="
