# Phase 1 Analysis - How to Run

## Two Scripts Available

### 1. Quick Test (3 models, ~2 minutes) ⚡
**Script:** `run_phase1_analysis_quick_test.py`
**Purpose:** Fast validation that Phase 1 components work
**Models:** 3 models only
**Time:** ~2 minutes

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python run_phase1_analysis_quick_test.py
```

**Output:**
- `phase1_results/quick_test_analysis_results.json`
- `phase1_results/quick_test_merged_structure.json`

---

### 2. Full Analysis (25 models, ~15-20 minutes) 🚀
**Script:** `run_phase1_analysis.py`
**Purpose:** Complete analysis of all models in sa_lam_2.0 experiment
**Models:** 25 models
**Time:** ~15-20 minutes

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python run_phase1_analysis.py
```

**Output:**
- `phase1_results/multi_graph_analysis_results.json`
- `phase1_results/multi_graph_merged_structure.json`
- `phase1_results/PHASE1_ANALYSIS_REPORT.md` ← Human-readable report

---

## Using the Convenience Script

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
./run_phase1_full.sh
```

This will:
1. Run the full 25-model analysis
2. Save output to log file
3. Display summary at the end

---

## What You Just Ran

Based on your output, you ran the **quick test** script:
```bash
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python run_phase1_analysis_quick_test.py
```

This is **correct** - it's supposed to only compile 3 models for quick validation.

---

## To Run Full 25-Model Analysis

Run this command instead:

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python run_phase1_analysis.py 2>&1 | tee phase1_results/phase1_full_run.log
```

This will:
- ✅ Compile all 25 models
- ✅ Analyze parameter sharing across all models
- ✅ Generate comprehensive report
- ✅ Save output to log file for review

---

## Key Differences

| Feature | Quick Test | Full Analysis |
|---------|-----------|---------------|
| Script | `run_phase1_analysis_quick_test.py` | `run_phase1_analysis.py` |
| Models | 3 | 25 |
| Time | ~2 min | ~15-20 min |
| Report | No markdown report | Full PHASE1_ANALYSIS_REPORT.md |
| Purpose | Validation | Decision making |

---

## Current Status

✅ Quick test completed successfully (3 models)
- 91.4% parameter sharing observed
- All components working correctly

⏭️ **Next:** Run full analysis with all 25 models

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python run_phase1_analysis.py
```

