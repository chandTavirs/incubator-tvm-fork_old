# Phase 1 Analysis - Real OFA Candidate Set

## Overview

This directory contains the Phase 1 analysis implementation for the Multi-Graph Runtime feasibility study. Phase 1 analyzes compiled TVM graphs from the OFA candidate set to determine if a unified multi-graph runtime is worth implementing.

## What Phase 1 Does

1. **Compiles Models**: Compiles models from the sa_lam_2.0 experiment
2. **Analyzes Graphs**: Examines graph structure, parameters, and storage requirements
3. **Identifies Sharing**: Determines which parameters are shared across models
4. **Calculates Savings**: Estimates memory reduction from parameter sharing
5. **Generates Reports**: Creates comprehensive feasibility report

## Files

### New Files (Created for Phase 1)

- **`run_phase1_analysis.py`**: Main script to run Phase 1 analysis
  - Compiles models from candidate set
  - Runs multi-graph analysis
  - Generates comprehensive reports
  
- **`extended_multi_graph_merger.py`**: Extended merger with save functionality
  - Extends base MultiGraphMerger
  - Adds JSON export capability
  
- **`PHASE1_RUN_GUIDE.md`**: This file

### Existing Files (Read-Only)

These files are used but NOT modified:

- **`execute_candidate_set_refactored.py`**: Model compilation utilities
- **`multi_graph_analyzer.py`**: Graph analysis engine
- **`multi_graph_merger.py`**: Base graph merger

## How to Run

### Quick Start

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
python run_phase1_analysis.py
```

### Configuration

Edit `Phase1Config` in `run_phase1_analysis.py`:

```python
class Phase1Config(Config):
    def __init__(self):
        super().__init__()
        
        # Experiment to analyze
        self.experiment_name = "sa_lam_2.0"
        
        # Limit models to compile (None = all)
        self.max_models_to_compile = 25  # Set to None for all models
        
        # Output directory
        self.analysis_output_dir = ".../phase1_results"
```

### Expected Runtime

- **With 5 models**: ~5-10 minutes
- **With 25 models**: ~30-60 minutes
- **All models (~800+)**: Several hours

### What to Expect

```
================================================================================
PHASE 1: MULTI-GRAPH RUNTIME ANALYSIS
Real OFA Candidate Set (sa_lam_2.0)
================================================================================

Setting up external imports...
Loading OFA model...
Setting up RPC connection...
Loaded XX schedule logs
Loading candidate set...
Found XX models in experiment 'sa_lam_2.0'

================================================================================
PHASE 1: COMPILING MODELS FOR ANALYSIS
================================================================================
Limiting to first 25 models

[1/25] Compiling arch_20250927_180844_0578...
  ✓ Success (1/1 compiled)
...

================================================================================
PHASE 1: MULTI-GRAPH ANALYSIS
================================================================================

Analyzing 25 graphs...
  [1/25] Analyzing arch_20250927_180844_0578...
  ...

Performing cross-graph analysis...

================================================================================
ANALYSIS REPORT
================================================================================

📊 OVERVIEW
────────────────────────────────────────────────────────────────────────────────
Total graphs analyzed: 25
Total unique parameters: XXX
Shared parameters: XXX
...

💾 STORAGE ANALYSIS
────────────────────────────────────────────────────────────────────────────────
Total storage (separate graphs): XX.XX MB
Estimated storage (merged): XX.XX MB
Storage reduction: XX.XX%

✅ FEASIBILITY ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
✓ All graphs have consistent input count: 1
✓ All graphs have consistent output count: 1
✓ High parameter sharing ratio: XX.X%
✓ Significant storage reduction achievable: XX.X%
```

## Output Files

All files are saved to `phase1_results/` directory:

### 1. `multi_graph_analysis_results.json`

Machine-readable analysis data:
- Per-graph statistics
- Parameter usage mappings
- Storage analysis
- Shared parameter list

### 2. `multi_graph_merged_structure.json`

Conceptual merged structure:
- Unified parameter pool
- Per-graph parameter mappings
- Usage statistics

### 3. `PHASE1_ANALYSIS_REPORT.md`

Comprehensive human-readable report:
- Executive summary
- Feasibility assessment
- Detailed statistics
- Next steps recommendations

## Interpreting Results

### High Parameter Sharing (≥70%)

**✅ HIGHLY RECOMMENDED to proceed to Phase 2**

- Excellent parameter reuse across models
- Significant memory savings achievable
- Strong ROI for implementing MultiGraphRuntime

**Action**: Proceed with Phase 2 implementation

### Medium Parameter Sharing (40-70%)

**⚠️ BENEFICIAL to proceed to Phase 2**

- Moderate parameter reuse
- Notable memory savings possible
- Consider Quick Win approach first

**Action**: Try SetInputZeroCopy approach or proceed to Phase 2

### Low Parameter Sharing (<40%)

**❌ NOT RECOMMENDED to proceed to Phase 2**

- Limited parameter reuse
- Minimal memory savings
- Implementation effort may not be justified

**Action**: Investigate why sharing is low or stay with current approach

## Customization

### Analyze Subset of Models

```python
# In Phase1Config
self.max_models_to_compile = 10  # Analyze only first 10 models
```

### Change Experiment

```python
# In Phase1Config
self.experiment_name = "different_experiment"  # Use different candidate set
```

### Different Output Directory

```python
# In Phase1Config
self.analysis_output_dir = "/path/to/custom/output"
```

## Troubleshooting

### Issue: Models fail to compile

**Symptom**: Many "FAILED" messages during compilation

**Solution**:
- Check OFA model path is correct
- Verify VTA environment is properly configured
- Check schedule logs are available

### Issue: Low parameter sharing

**Symptom**: Sharing ratio < 40%

**Possible Causes**:
- Models not from same OFA supernet
- Weight loading inconsistency
- Different quantization settings

**Solution**:
- Verify all models use same OFA checkpoint
- Check weight loading logic in `load_static_resnet_from_arch`

### Issue: Out of memory

**Symptom**: Script crashes during compilation

**Solution**:
- Reduce `max_models_to_compile`
- Run in smaller batches
- Use machine with more RAM

## Next Steps After Phase 1

Based on the analysis results:

### Option A: Proceed to Phase 2

If sharing ratio ≥ 70%:
1. Review `MULTI_GRAPH_RUNTIME_DESIGN.md`
2. Implement `MultiGraphRuntime` in C++
3. Create `relay.build_multi_graph()` function
4. Integrate with execution pipeline

### Option B: Quick Win Approach

If sharing ratio 40-70%:
1. Implement SetInputZeroCopy parameter caching
2. Pre-upload all OFA parameters once
3. Each runtime references shared device memory
4. See "Alternative: Quick Win Approach" in design doc

### Option C: Stay with Current

If sharing ratio < 40%:
1. Continue with current per-model runtime approach
2. Investigate why sharing is low
3. Focus on other optimizations

## Files Modified

**NONE** - This implementation creates new files only and does not modify any existing TVM/VTA code or scripts.

## Contact

For questions or issues:
- Review `MULTI_GRAPH_RUNTIME_DESIGN.md` for design details
- Check `PHASE1_IMPLEMENTATION_SUMMARY.md` for implementation info
- See `PHASE1_READY_TO_RUN.md` for additional context

