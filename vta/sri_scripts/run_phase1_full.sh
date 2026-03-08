#!/bin/bash
#
# Run Phase 1 Analysis on Full OFA Candidate Set (25 models)
#
# This script runs the complete Phase 1 analysis on all 25 models
# from the sa_lam_2.0 experiment.
#

echo "================================================================================"
echo "PHASE 1: MULTI-GRAPH RUNTIME ANALYSIS - FULL RUN (25 MODELS)"
echo "================================================================================"
echo ""
echo "This will compile and analyze all 25 models from the sa_lam_2.0 experiment."
echo "Expected time: ~15-20 minutes"
echo ""
echo "Results will be saved to:"
echo "  - phase1_results/multi_graph_analysis_results.json"
echo "  - phase1_results/multi_graph_merged_structure.json"
echo "  - phase1_results/PHASE1_ANALYSIS_REPORT.md"
echo ""
echo "================================================================================"
echo ""

cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python run_phase1_analysis.py 2>&1 | tee phase1_results/phase1_full_run.log

echo ""
echo "================================================================================"
echo "PHASE 1 COMPLETE"
echo "================================================================================"
echo ""
echo "Review the results in:"
echo "  - phase1_results/PHASE1_ANALYSIS_REPORT.md (Human-readable report)"
echo "  - phase1_results/multi_graph_analysis_results.json (Detailed data)"
echo "  - phase1_results/phase1_full_run.log (Full execution log)"
echo ""

