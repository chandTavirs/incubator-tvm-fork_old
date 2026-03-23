# Phase B: OFA Weight Pool Runtime — Progress Log

**Date:** March 14, 2026  
**Status:** Step 1 + Step 2 COMPLETE ✅

---

## What Was Validated

### POC Run: 2 Subnets from `sa_lam_2.0`
- `arch_20250927_180844_0578` — 25 layer derivations (11 with kernel transforms, 22 decomposed)
- `arch_20250927_180844_0034` — 34 layer derivations (21 with kernel transforms, 30 decomposed)

### Results
| Metric | Value |
|---|---|
| Total layers checked | 59 |
| Matched (max_abs_diff < 1e-5) | **59 / 59** |
| Global max \|diff\| | **0.00e+00** |

**100% bitwise match with OFA model's `get_active_weights()` for ALL layers including kernel transforms (7→5, 7→5→3).**

---

## What Was Built

### `ofa_derivation_extractor.py`
Core extraction module:
- `LayerDerivation` — data class storing one conv's full derivation recipe:
  - base weight key, channel slice (out_start:out_end, in_start:in_end)
  - kernel transform sequence e.g. `[(7,5), (5,3)]` with pool keys
  - decompose_type (0-4)
- `OFADerivationExtractor` — walks OFA model to extract derivations for a subnet
  - Handles all decompose types (0-4)
  - Handles residual shortcuts
  - Handles mixed kernel sizes
- `derive_weight_numpy()` — mirrors OFA `get_active_weights()` in pure NumPy
  - Transform loop: crop from CURRENT start_filter → `F.linear` → reshape
  - Verified bitwise identical to PyTorch reference
- `validate_derivation_against_ofa()` — numerical cross-check

### `poc_derivation_validation.py`
End-to-end validation script (host-only, no VTA needed):
- Loads OFA model + weight pool
- Extracts derivation metadata for 2 subnets
- Validates every layer numerically
- Saves derivation JSON files

---

## Key Technical Insight: OFA Transform Loop

The OFA `get_active_weights()` transform sequence works as follows:
```
start_filter = base_conv.weight[out_s:out_e, in_s:in_e, :, :]   # full max_k spatial

for (src_ks, target_ks) in [(7,5), (5,3)]:   # if going 7→3
    # Center crop start_filter (which is now src_ks × src_ks after first step)
    crop_s, crop_e = sub_filter_start_end(src_ks, target_ks)
    _input = start_filter[:, :, crop_s:crop_e, crop_s:crop_e]   # [out, in, target_ks, target_ks]
    _flat = _input.reshape(-1, target_ks * target_ks)            # [out*in, target_ks^2]
    _out = F.linear(_flat, matrix)                               # [out*in, target_ks^2]  (matrix is square!)
    start_filter = _out.reshape(out_ch, in_ch, target_ks, target_ks)
```

The transform matrices are **square** (e.g. 9×9 for 3×3 kernels, 25×25 for 5×5).
They are initialized as identity matrices and learned during OFA training.

---

## Next Steps: Step 3 — Relay Graph Construction

Goal: Build a Relay IRModule where conv weights are expressed as:
```
base_var = relay.var("pool_...", shape=[max_out, max_in, max_k, max_k])
transform_var = relay.var("pool_...", shape=[tgt_ks^2, tgt_ks^2])

# Step 1: channel slice
w = strided_slice(base_var, [out_s, in_s, 0, 0], [out_e, in_e, max_k, max_k])

# Step 2: for each (src_ks, tgt_ks) in transform_sequence:
#   crop = strided_slice(w, [..., crop_s, crop_s], [..., crop_e, crop_e])
#   flat = reshape(crop, [-1, tgt_ks^2])
#   out  = nn.dense(flat, transform_var)       # [out*in, tgt_ks^2]
#   w    = reshape(out, [out_ch, in_ch, tgt_ks, tgt_ks])

conv_out = nn.conv2d(x, w, ...)
```

This Relay graph approach:
- Keeps OFA pool as `relay.var` (uploaded once, shared across all subnet runtimes)
- Adds ~3 ops per layer with transform (slice + dense + 2×reshape)
- No changes to TVM compiler needed — uses standard relay ops

**Estimated implementation time: 2–3 days**

---

## Files

| File | Purpose |
|---|---|
| `ofa_weight_pool_extractor.py` | Extract + save OFA base weights & transform matrices |
| `ofa_weight_pool/` | Saved pool (base_weights.npz, transform_matrices.npz, etc.) |
| `ofa_derivation_extractor.py` | **NEW** Extract weight derivation metadata per subnet |
| `ofa_relay_graph_builder.py` | **NEW** Build Relay graph with OFA pool variables (Step 3, in progress) |
| `poc_derivation_validation.py` | **NEW** End-to-end validation script |
| `poc_results/` | Validation results and saved derivation JSONs |

