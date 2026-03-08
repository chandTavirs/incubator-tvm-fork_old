#!/bin/bash

# Phase 2 Full Test - Run with all 25 models
# Complete validation of shared parameter approach

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python"

echo "================================================================================"
echo "PHASE 2 FULL TEST: SHARED PARAMETER EXECUTION"
echo "Testing with ALL 25 models from sa_lam_2.0"
echo "================================================================================"
echo ""
echo "Expected time: ~20-25 minutes"
echo ""
echo "Results will be saved to:"
echo "  - phase2_results/phase2_full_test_results.json"
echo ""
echo "================================================================================"
echo ""

cd "$SCRIPT_DIR"

# Create results directory
mkdir -p phase2_results

# Run test (with output logging)
LOG_FILE="phase2_results/phase2_full_test.log"

echo "Running test (logging to $LOG_FILE)..."
echo ""

$PYTHON_BIN test_phase2_full.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ FULL TEST COMPLETED SUCCESSFULLY"
    echo ""
    echo "Review the results:"
    echo "  - Report: phase2_results/phase2_full_test_results.json"
    echo "  - Log: $LOG_FILE"
else
    echo "✗ FULL TEST FAILED (Exit code: $EXIT_CODE)"
    echo ""
    echo "Check the log file for details: $LOG_FILE"
fi
echo "================================================================================"
echo ""

exit $EXIT_CODE

