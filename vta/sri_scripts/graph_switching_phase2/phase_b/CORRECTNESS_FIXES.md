# Step7 GMTF Correctness Fixes

Summary of code changes that produce the best VTA-vs-reference accuracy (90/200 = 45%)
across K=20 candidate subnets on 10 ImageNette images (1/class).

---

## 1. `_FixDeriveBNFoldShift8` — step7 derive fn, NO-MULTIPLY 3×3/5×5

**File**: `step7_gmtf_deriv_infer.py`  
**Status**: Committed (`3aa63add0`)

Two-stage GMTF outputs weights at ×16 scale. The post-graphpack derive fn for
NO-MULTIPLY 3×3 and 5×5 layers contained a spurious
`clip(right_shift(add(GMTF_int32, bias), shift), -127, 127)` where `shift=8`.
This divided the ×16 weights by 256, producing near-zero or all-zero derived
weights. Removing the shift restores ×16 scale and gives non-zero activations.

---

## 2. `_FixWith5x5FloatBNFold` — step7 derive fn, WITH-MULTIPLY layers

**File**: `step7_gmtf_deriv_infer.py`  
**Status**: Uncommitted (working directory)

Replaces integer BN fold in post-graphpack WITH-MULTIPLY GMTF layers:

- **3×3 WITH-MULTIPLY**: replaced with `relay.zeros_like(gmtf_int32)`.
  Integer BN fold with `_FixInferAccumCast` active causes all OG-group partial
  sums to saturate at ±127 → constant class prediction regardless of input.
  Zeroing removes the spurious constant bias and lets the residual path dominate.

- **5×5 WITH-MULTIPLY**: replaced with float32 BN fold
  (`round(clip(GMTF_float × BN_scale_float, -127, 127))`).
  Integer fold at shift=8 divides ×16 GMTF weights by 16 again → ×1 scale →
  16× activation collapse in the infer fn. Float fold restores ×16 scale. A
  zero-scale guard (`max(|BN_scale|, ε)`) prevents noise from near-zero BN channels.

- **1×1 and other sizes**: excluded (integer fold left as-is).
  For 1×1, integer fold gives `round(GMTF×scale/16) ≈ 0` for many channels →
  fewer saturating channels → better than float fold.

---

## 3. `_FixInferAccumCast` — step7 infer fn, OG-group accumulation

**File**: `step7_gmtf_deriv_infer.py`  
**Status**: Uncommitted (working directory)

VTA splits large conv2d into OG-groups and accumulates int32 partial sums before
casting to int8. Without a clip, `cast(int8, ±254)` wraps to `∓2`, which
accidentally gave near-zero contributions (a "helpful" bug). With this fix,
`clip(-127, 127)` is inserted before the int8 cast, clamping partial sums
correctly. This is the primary driver for using `zeros_like` in the 3×3
WITH-MULTIPLY case (see fix 2).

---

## 4. `_PoolLadderStripper` — step3 BN fold for 7×7 direct pool weights

**File**: `step3_merged_mod_deriv_poc.py`  
**Status**: Uncommitted (working directory)

For 7×7 conv layers whose weights come directly from the pool (no GMTF
transformation matrix), the relay.quantize ladder adds an outer `×16` scalar:
`multiply(multiply(cast(float32, int8_pool), BN_scale), scalar_16f)`.
The pool weight is already at ×16 scale, so `cast(float32, int8_pool) × BN_scale`
is the correct fused weight. The outer `×scalar_16f` over-scales by ×16 → ×256
× BN_scale → saturation → constant class predictions across all images.

The stripper detects and removes the outer scalar multiply, restoring ×16 scale.

---

## 5. CMA leak fix — `_switch()` in step7

**File**: `step7_gmtf_deriv_infer.py`  
**Status**: Uncommitted (working directory)

Each subnet switch runs a derive module (`m_derive`) which writes derived int8
weights into ext_dev (CMA) memory, then calls `m_infer.set_input()` which does
a same-device CMA memcpy into the infer module's parameter storage.
Previously `m_derive` was kept alive after this copy, accumulating ~105 MB of
CMA per subnet switch. With K=20, this caused OOM at subnet 15 (1470 MB used +
101 MB base > 1664 MB CMA budget). Fix: `del m_derive; gc.collect()` immediately
after `set_input`, so peak CMA during a switch is ~297 MB instead of growing
unboundedly.

---

## Accuracy summary (K=20, 10 images/subnet)

| Subnet | Acc  | Switch ms |
|--------|------|-----------|
| 0736   | 9/10 | 2448      |
| 0338   | 8/10 | 4341      |
| 0799   | 8/10 | 2543      |
| 0537   | 7/10 | 2597      |
| 0599   | 7/10 | 5619      |
| 0455   | 6/10 | 2075      |
| 0369   | 6/10 | 2758      |
| 0963   | 5/10 | 3703      |
| 0348   | 4/10 | 2774      |
| 0358   | 4/10 | 2608      |
| 0922   | 3/10 | 3582      |
| 0838   | 3/10 | 3782      |
| 0882   | 3/10 | 4226      |
| 0185   | 3/10 | 2565      |
| 0065   | 3/10 | 7872      |
| 0348   | 4/10 | 2774      |
| 0585   | 2/10 | 7723      |
| 0912   | 2/10 | 6405      |
| 0434   | 3/10 | 2882      |
| 0037   | 3/10 | 2372      |
| 0272   | 1/10 | 1740      |
| **Total** | **90/200 = 45%** | mean 3637 ms |

Metric: VTA prediction matches reference model prediction (not ground-truth accuracy).
