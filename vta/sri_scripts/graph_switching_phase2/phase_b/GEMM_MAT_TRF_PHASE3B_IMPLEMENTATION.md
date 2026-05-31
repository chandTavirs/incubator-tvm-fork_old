# GEMM_Mat_Trf Phase 3b — Relay Integration (runs during `m.run()`)

**Branch:** `gemm_mat_trf_phase2`
**Device:** ZCU104 @ `10.42.0.188:9091` — BATCH=1, BLOCK_IN=16, BLOCK_OUT=16, LOG_UOP_BUFF_SIZE=14
**Status:** GEMM_Mat_Trf executes on VTA hardware as part of the compiled inference graph, for every
OFA transform layer including the 1024×1024 (1,048,576-row) one. End-to-end `step3_merged_mod_deriv_poc.py`
runs to completion. Top-1 differs from the float PyTorch reference (int8-quantization + FPGA shift
rounding) — functional execution is the milestone; exact top-1 match is a separate quantization task.

---

## 1. Goal vs Phase 2

Phase 2 made `GEMM_Mat_Trf` work as a standalone VTA instruction (verified via the hardware benchmark).
Phase 3b wires it into the **Relay → graph_pack → relay.build** pipeline so the OFA kernel-derivation
transform `out[b,r] = sum_k inp[b,k]·T[r,k]` runs **on the accelerator during `m.run()`**, alongside the
conv2d ops — no CPU pre-materialization.

The OFA transform appears in Relay as:
```
strided_slice(base_conv_weight) → reshape([out_ch·in_ch, ks²]) → nn.dense(_, T) → reshape(...) → nn.conv2d
```
The `nn.dense(_, T)` is the target. `units=None`, square weight (9×9 or 25×25).

---

## 2. Why a new Relay op was needed

`nn.dense`'s `DenseRel` type relation (C++ `src/relay/op/nn/nn.h`) enforces, for packed 4-D tensors,
`data.shape[-3] == weight.shape[1]` (the `k_outer` dimension). The GMTF **large** weight is packed as
`(2·BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN) = (32,1,16,16)` — one full wgt entry per T row — while packed data
is `(n, 2, 1, 16)` (`k_o = 2`). `2 ≠ 1` → hard type error. Cannot be worked around with `nn.dense`.

So two custom ops were registered with a relaxed type relation.

### `src/relay/op/nn/nn.h` — `GmtfDenseRel`
Output shape = `(data[0], data[1], data[2], weight[2])`, no `k_o` matching constraint. Reused by both ops.

### `src/relay/op/nn/nn.cc` — two ops
- `vta.gmtf_dense_small`: data `(n,1,1,16)`, weight `(1,1,16,16)`, out `(n,1,1,16)`
- `vta.gmtf_dense_large`: data `(n,2,1,16)`, weight `(32,1,16,16)`, out `(n,2,1,16)`

Both use `DenseAttrs`, `GmtfDenseRel`, `TOpPattern = kOutEWiseFusable`, and a `relay.op.nn._make.*` maker.
**Requires a TVM rebuild** (`cmake --build build`).

---

## 3. Python wiring

### `vta/python/vta/top/vta_gmtf_op.py` (new)
- `gmtf_dense_small(data, weight, out_dtype)` / `gmtf_dense_large(...)` — thin wrappers over the C++ makers.
- Registers `FTVMStrategy` via `tvm.ir.register_op_attr("vta.gmtf_dense_*", "FTVMStrategy", fn)`.
  (Not `relay.op.op.register_strategy`, which asserts the fn is a `GenericFunc` — these VTA-only ops have
  no CPU fallback.) Strategy uses `_strat.wrap_compute_dense(topi_fn)` (out_dtype passed positionally).

### `vta/python/vta/top/vta_gmtf_dense.py` (new)
TOPI compute + schedule for both modes. The compute mirrors the Phase-2 intrinsic shapes; the schedule is
described in §5.

### `vta/python/vta/top/op.py`
Imports `vta_gmtf_op` for side-effect strategy registration (guarded so VTA still imports before the rebuild).

### `vta/python/vta/top/graphpack.py`
In the `nn.dense` handler, detect `units=None` with square 9×9 / 25×25 weight and emit the GMTF op instead:
- small: standard `_pack_weight_dense` → `(1,1,16,16)`; route to `gmtf_dense_small`.
- large: `_pack_weight_gmtf_large` → `(32,1,16,16)`; `units = 2·BLOCK_OUT`; route to `gmtf_dense_large`.
- `_chunk_packed_dense` gained `dense_fn` and `max_chunk` params; GMTF passes `max_chunk = 1<<30` so the
  transform is emitted as **ONE op** (no relay-level concatenate — see §4) and the schedule tiles it.
- The reshape-trimming logic recognizes the GMTF ops via an `_is_dense_like()` helper.

---

## 4. The scaling walls (in the order they were hit)

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `CopyIntrinInjector` lowering error | ReLU in ALU epilogue; double `pragma_alu`; store DMA on wrong axis | symmetric clip (no ReLU); one alu pragma per ewise op |
| 2 | `Allocation exceed bound local.inp_buffer` | schedule bulk-loaded the whole tensor | tile `n_batch` (TILE=512 small / 256 large) |
| 3 | `relay.build` 10-min timeout | relay-level chunking → 1024-way `concatenate` → `sch.normalize()` chokes | emit ONE GMTF op per transform; tile inside the schedule |
| 4 | runtime `runtime.cc:695 from!=Store ‖ to!=Load` | multi-tile schedule emits direct **STORE→LOAD** coproc dep (illegal) | conv2d-style outer-reduce load point (§5) |
| 5 | device crash, `code=1` / broken pipe at n_batch ≥ 262144 | load was **per-row** (`x_size=1`); ~8 insns/row × 262144 = exactly `VTA_MAX_XFER/16` → instruction-queue overflow | bulk load + UOP fold + bulk store (§5) |

---

## 5. The working schedule (`vta_gmtf_dense.py`)

The key is **conv2d, not dense_packed, as the reference**. conv2d never emits STORE→LOAD because it
accumulates over the `k_o` reduction — the acc buffer is persistent state, so the cross-tile buffer-reuse
dependency routes **STORE→COMPUTE→LOAD** (`coproc_dep_push(3,2)`, legal) instead of STORE→LOAD (`(3,1)`).

GMTF reduces entirely *inside* the intrinsic, so the schedule sees no reduction loop. We synthesize one:
split the intrinsic's reduce axis into an **outer load point** (extent 1) + inner (stays in the intrinsic),
and place it **outside the batch loop `d_b`**. That single placement achieves four things at once:
1. STORE→COMPUTE ring (no illegal STORE→LOAD).
2. **Bulk load** — one DMA for the whole tile (`x_size = TILE`), not per-row.
3. **UOP fold** — `d_b` folds into one `VTAUopLoopBegin(TILE, …)`, so instruction count is O(tiles) not O(rows).
4. Combined with a **bulk store** (`pragma` at the tile-batch axis `x_bn`), per-tile cost is ~8 instructions.

```python
# small (tensorize at d_bi):
x_bo, x_bn = s[out].split(x_b, factor=512); s[out].reorder(x_bo, x_co, x_bn, x_bi, x_ci)
s[dense].compute_at(s[out], x_co); ...                      # store_pt = x_co
d_ko, d_kii = s[dense].split(d_ki, factor=BLOCK_IN)
s[dense].reorder(d_ko, d_b, d_bi, d_ci, d_kii)              # d_ko OUTSIDE d_b  <- critical
s[cdata].compute_at(s[dense], d_ko); s[cweight].compute_at(s[dense], d_ko)
s[dense].tensorize(d_bi, gemm_mat_trf(env, large_mode=False))
s[out].pragma(x_bn, env.dma_copy)                          # bulk store

# large (tensorize at d_co):
x_bo, x_bn = s[out].split(x_b, factor=256); s[out].reorder(x_bo, x_bn, x_co, x_bi, x_ci)
s[dense].compute_at(s[out], x_bo); ...                      # store_pt = x_bo
d_kio, d_kii = s[dense].split(d_ki, factor=BLOCK_IN)
s[dense].reorder(d_kio, d_b, d_co, d_bi, d_ci, d_ko, d_kii) # d_kio OUTSIDE d_b <- critical
s[cdata].compute_at(s[dense], d_kio); s[cweight].compute_at(s[dense], d_kio)
s[dense].tensorize(d_co, gemm_mat_trf(env, large_mode=True))
s[out].pragma(x_bn, env.dma_copy)
```

No virtual threading, no `n_batch` branching — single-tile and multi-tile use the same structure.

**TIR verification (local, no device):** dump the lowered module and check:
- `coproc_dep_push(3, 1)` is **absent** (no STORE→LOAD)
- load shows `VTALoadBuffer2D(..., x_size=TILE, ...)` (bulk)
- gemm shows `VTAUopLoopBegin(TILE, …)` (folded)
- store shows `VTAStoreBuffer2D(..., x_size=TILE, ...)` (bulk)

---

## 6. Verification & limits

`test_gmtf_tiled_hw.py` — standalone HW correctness sweep of the exact schedule.

| Finding | Result |
|---------|--------|
| Runs on HW, all sizes 256 … 1,048,576 (small + large) | no crash, no STORE→LOAD, no acc/inp overflow |
| Device limit found (pre-fix) | per-row instructions overflowed `VTA_MAX_XFER` (1<<25 B) at n_batch≥262144; fixed by §5 |
| CMA allocation (`probe_cma.py`) | 255 MB cumulative / 16 MB single — not a constraint |
| tile-count vs total-rows (`probe_limit.py`) | 2048 tiles fine; limit was total instructions, now O(tiles) |
| Numerical vs numpy floor-shift reference | ±1-3 LSB on **negative** acc values |

**±1-3 LSB note:** the VTA *sim* does signed `int32 x >> y` (arithmetic floor, == `np.right_shift`); the
*FPGA bitstream* rounds negatives slightly differently. The Phase-2 benchmark used ReLU clip `[0,127]` so it
never exercised negatives. Within int8 quantization noise for weight derivation. The GMTF ALU epilogue uses
symmetric clip `[-127,127]` (no ReLU) because transform outputs are signed.

---

## 7. Files

**C++ (needs rebuild):** `src/relay/op/nn/nn.h`, `src/relay/op/nn/nn.cc`
**Python:** `vta/python/vta/top/vta_gmtf_dense.py` (new), `vta/python/vta/top/vta_gmtf_op.py` (new),
`vta/python/vta/top/graphpack.py`, `vta/python/vta/top/op.py`
**Scripts (phase_b/):** `step3_merged_mod_deriv_poc.py --enable-dynamic-dense-quant` (pipeline; the flag is
required — without it `_materialize_int8_pool_constants` hits a BroadcastRel type error unrelated to GMTF),
`test_gmtf_tiled_hw.py`, `probe_cma.py`, `probe_limit.py`, `step3_merged_hw_gemm_mat_trf.py` (alt CPU-materialize path).

---

## 8. Open items

1. **Top-1 match.** VTA top-1 differs from the float reference. Dominant cause is int8 quantization of the
   derived weights, not GMTF. Quick diagnostic: compare GMTF-path vs CPU-numpy-path top-1 on the same subnet —
   if equal, GMTF is provably correct and the gap is purely quantization. Then tune the transform quant scale.
2. **FPGA SHR rounding** on negatives (±1-3 LSB) — characterize and match the reference if exact bits are needed.
