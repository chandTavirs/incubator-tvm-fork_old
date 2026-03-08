#!/bin/bash

# Phase 2 Quick Test - Run with 3 models
# Quick validation of shared parameter approach

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python"

echo "================================================================================"
echo "PHASE 2 QUICK TEST: SHARED PARAMETER EXECUTION"
echo "Testing with 3 models from sa_lam_2.0"
echo "================================================================================"
echo ""
echo "Expected time: ~3-5 minutes"
echo ""

cd "$SCRIPT_DIR"

# Run test
$PYTHON_BIN test_phase2_quick.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ QUICK TEST COMPLETED SUCCESSFULLY"
else
    echo "✗ QUICK TEST FAILED (Exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo ""

exit $EXIT_CODE

