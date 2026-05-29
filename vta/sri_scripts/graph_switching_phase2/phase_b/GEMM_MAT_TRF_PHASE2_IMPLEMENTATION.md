# GEMM_Mat_Trf Phase 2 — End-to-End Implementation Notes

**Branch:** `gemm_mat_trf_phase2` / `gemm_mat_trf`  
**Device:** ZCU104 @ `10.42.0.188:9091` — BATCH=1, BLOCK_IN=16, BLOCK_OUT=16  
**Final status:** Both small (9×9) and large (25×25) modes verified on hardware, all correctness checks passing.

---

## 1. Background

OFA (Once-For-All) networks derive smaller conv kernels at runtime from a stored maximum-size
kernel and a spatial-crop transform matrix:

```
out[b, r] = sum_k( inp[b, k] * T[r, k] )
```

Phase 1 (previous session) offloaded this to CPU materialization and demonstrated end-to-end
VTA inference on the ZCU104. Phase 2 implements this operation in the **VTA GEMM datapath**
itself as a new instruction — `GEMM_Mat_Trf` — that handles it on-chip.

---

## 2. Instruction Encoding

The VTA GEMM instruction word has `FILL_FIELD_GEMM` — spare bits at the top of the 128-bit
instruction. With `LOG_UOP_BUFF_SIZE = 14` (UOP buffer = 16 K entries), `FILL_WIDTH_GEMM = 3`,
freeing bits 61–63.

### Critical config constraint

Standard ZCU104 uses `LOG_UOP_BUFF_SIZE = 15`. This gives `FILL_WIDTH_GEMM = 1`, only one
spare bit, which is enough for MODE but not DIM. **Must use `LOG_UOP_BUFF_SIZE = 14`.**
The custom config lives at `3rdparty/vta-hw/config/vta_config_gemm_mat.json`.

### Bit fields (in `FILL_FIELD_GEMM`)

| Bit | Macro | Meaning |
|-----|-------|---------|
| 61 | `VTA_GEMM_MAT_TRF_MODE_BIT` | `0` = standard GEMM, `1` = GEMM_Mat_Trf |
| 62 | `VTA_GEMM_MAT_TRF_DIM_BIT`  | `0` = 9×9 (3×3 kernel), `1` = 25×25 (5×5 kernel) |

The `empty_0` field encoding passed to the runtime:

| `empty_0` | Mode | Constant |
|-----------|------|----------|
| `0x1` | Small (9×9) | `VTA_GEMM_MAT_TRF_EMPTY0_SMALL` |
| `0x3` | Large (25×25) | `VTA_GEMM_MAT_TRF_EMPTY0_LARGE` |

**Note:** Bit 7 of VTAGemInsn is `reset_reg`, NOT a free bit. An earlier attempt to use bit 7
produced random results because it toggled the accumulator reset. FILL_FIELD_GEMM (bits 61–62)
is the correct location.

---

## 3. HLS Implementation (`hardware/xilinx/src/vta.cc`)

The entire compute-stage implementation is in `vta.cc`. Phase 2 added the following new
functions and patched the `compute()` dispatcher.

### 3.1 Buffer read/write helpers

#### `read_inp_vec_small(src_idx, inp_mem, vec[9])`
Reads 9 int8 elements from `inp_mem[src_idx][bus_word=0][bytes 0..8]`.  
Zero-padded bytes 9–15 are ignored.

#### `read_inp_vec_large(src_idx, inp_mem, vec[25])`
With `BLOCK_IN = 16` (i.e., `INP_MAT_AXI_RATIO = 1`):
- `inp_mem[src_idx][word 0]` → `vec[0..15]`
- `inp_mem[src_idx + 1][word 0]` → `vec[16..24]`  

With `BLOCK_IN ≥ 32` (i.e., `INP_MAT_AXI_RATIO ≥ 2`): all 25 elements fit in one entry
(word 0 holds 0..15, word 1 holds 16..24). The function uses an `#if INP_MAT_AXI_RATIO`
guard to select the right branch.

#### `read_trf_row_small(trf_base, row_within, wgt_mem, row[9])`
Reads bus word `row_within` from `wgt_mem[trf_base]`. All 9 T rows are packed into
bus words 0–8 of a **single** wgt_mem entry.

#### `read_trf_row_large(row_idx, wgt_mem, row[25])`
Reads T row from a **separate** wgt_mem entry at index `row_idx`:
- `wgt_mem[row_idx][word 0]` → `row[0..15]`
- `wgt_mem[row_idx][word 1]` → `row[16..24]`  

25 entries total, one per T row.

#### `write_trf_result_small(dst_idx, results[9], acc_mem, out_mem)`
Writes 9 int32 values into `acc_mem[dst_idx]` and int8-truncated into `out_mem[dst_idx]`.
Elements 9–15 of each entry are left as zero.

#### `write_trf_result_large(dst_idx, results[25], acc_mem, out_mem)`
With `VTA_MAT_TRF_ACC_STRIDE_LARGE = 2`:
- `results[0..15]` → `acc_mem[dst_idx]` and `out_mem[dst_idx]`
- `results[16..24]` → `acc_mem[dst_idx + 1]` and `out_mem[dst_idx + 1]`

### 3.2 Compute cores

#### `gemm_mat_trf_small(insn, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem)`
9×9 matrix transform for 3×3 kernels. Outer loop structure mirrors standard GEMM
(iter_out / iter_in / UOP triple loop). Per UOP:
1. Read input vector (9 elements) via `read_inp_vec_small`
2. Read T[9][9] from a single packed wgt_mem entry via `read_trf_row_small` (9 calls)
3. Compute `out[i] = sum_j(inp[j] * T[i][j])` — **direct assign, no accumulation**
4. Write via `write_trf_result_small`

`dst_factor` and `src_factor` step by `VTA_MAT_TRF_INP_STRIDE_SMALL = 1`.

#### `gemm_mat_trf_large(insn, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem)`
25×25 matrix transform for 5×5 kernels. Same outer loop structure. Per UOP:
1. Read 25-element input vector via `read_inp_vec_large` (2 inp_mem reads)
2. Read T[25][25] from 25 separate wgt_mem entries via `read_trf_row_large`
3. Compute `out[i] = sum_j(inp[j] * T[i][j])` — direct assign
4. Write via `write_trf_result_large` (2 acc_mem / out_mem writes)

`dst_factor` and `src_factor` **must** step by `VTA_MAT_TRF_ACC/INP_STRIDE_LARGE = 2`.

#### `gemm_mat_trf(insn, ...)` — dispatcher
```c
if (insn_raw.range(VTA_GEMM_MAT_TRF_DIM_BIT, VTA_GEMM_MAT_TRF_DIM_BIT) == 1)
    gemm_mat_trf_large(...)
else
    gemm_mat_trf_small(...)
```

#### Patch to `compute()` dispatcher
```c
} else if (insn.generic.opcode == VTA_OPCODE_GEMM) {
    if (raw_copy.range(VTA_GEMM_MAT_TRF_MODE_BIT, VTA_GEMM_MAT_TRF_MODE_BIT) == 1)
        gemm_mat_trf(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
    else
        gemm(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
```

### 3.3 Memory layout summary

| Mode | T matrix (wgt_mem) | Input (inp_mem) | Output (acc/out_mem) |
|------|--------------------|-----------------|----------------------|
| Small (9×9) | 1 entry: T[i][j] = bus_word[i][byte j], i,j < 9 | 1 entry/batch: bytes 0–8 | 1 entry/batch: elements 0–8 |
| Large (25×25) | 25 entries: entry i = T row i; word 0 = T[i][0..15], word 1 = T[i][16..24] | 2 entries/batch: entry 0 = inp[0..15], entry 1 = inp[16..24] | 2 entries/batch: entry 0 = out[0..15], entry 1 = out[16..24] |

### 3.4 HLS pragmas
- `#pragma HLS ARRAY_PARTITION variable=in_vec complete dim=1` on the input vector
- `#pragma HLS ARRAY_PARTITION variable=t_mat complete dim=2` on the T matrix
- `#pragma HLS PIPELINE` on the T matrix read loop
- `#pragma HLS UNROLL` on the compute inner loop

---

## 4. C Simulation (CSIM) Gate

### Testbench: `tests/hardware/common/test_lib.cc` — `gemm_mat_trf_test(batch, large_mode)`

Tests both modes with batch sizes 1, 4, 16:

```c
status |= gemm_mat_trf_test(1,  false);   // 9x9, batch=1
status |= gemm_mat_trf_test(4,  false);   // 9x9, batch=4
status |= gemm_mat_trf_test(16, false);   // 9x9, batch=16
status |= gemm_mat_trf_test(1,  true);    // 25x25, batch=1
status |= gemm_mat_trf_test(4,  true);    // 25x25, batch=4
status |= gemm_mat_trf_test(16, true);    // 25x25, batch=16
```

The test function:
1. Allocates random int8 inputs and a random int8 T matrix
2. Constructs the full VTA instruction sequence:
   - `LOAD_UOP` (no dep)
   - `LOAD_INP` (no dep)
   - `LOAD_WGT` (push next → l2g)
   - `GEMM_MAT_TRF` (pop prev ← l2g, push next → g2s) with `empty_0 = 0x1 or 0x3`
   - `STORE` (pop prev ← g2s, push prev → s2g)
   - `FINISH` (pop next ← s2g)
3. UOP buffer: `uop_buf[b] = {dst_idx=b*STRIDE, src_idx=b*STRIDE, wgt_idx=0}`
4. Packs inputs into inp_mem entries (respecting STRIDE)
5. Packs T matrix into wgt_mem entries (packed or row-per-entry format)
6. Runs csim, reads output, verifies against reference:
   `ref[b][r] = clip(sum_k(inp[b][k] * T[r][k]) >> 8, 0, 127)`

### CSIM results (`build/gemm_mat_trf_csim/vivado_hls.log`)
```
INFO - GEMM_Mat_Trf test (dim=9)  PASSED   [batch=1]
INFO - GEMM_Mat_Trf test (dim=9)  PASSED   [batch=4]
INFO - GEMM_Mat_Trf test (dim=9)  PASSED   [batch=16]
INFO - GEMM_Mat_Trf test (dim=25) PASSED   [batch=1]
INFO - GEMM_Mat_Trf test (dim=25) PASSED   [batch=4]
INFO - GEMM_Mat_Trf test (dim=25) PASSED   [batch=16]
```

All 6 tests pass. The CSIM verified behavioral correctness before bitstream generation.

---

## 5. HLS Errors Encountered and Fixed

### 5.1 `ERROR: [IMPL 213-28] Failed to generate IP` (vta_fetch)

The CSIM log ends with an IP generation error for `vta_fetch`. This is a **red herring** —
the `vta_fetch` IP synthesis failed because the tcl build script attempted to generate the
full IP (synthesis + export), which requires timing closure and is not part of CSIM.
The CSIM (C simulation) itself passed all tests; the IP error occurs in the subsequent
synthesis step that is irrelevant to functional verification.

**Resolution:** CSIM correctness confirmed. Bitstream was generated via a separate Vivado
implementation flow (not via the HLS IP export path).

### 5.2 `empty_0` bit field collision with `reset_reg`

**Problem:** An early implementation placed the mode flag at bit 7 of the GEMM instruction
word. In VTAGemInsn, bit 7 is `reset_reg`. Setting mode=1 at bit 7 accidentally caused the
accumulator to reset on every GEMM_Mat_Trf call, producing zeros.

**Fix:** Used `FILL_FIELD_GEMM` (bits 61–62) instead of bit 7. Required changing
`LOG_UOP_BUFF_SIZE` from 15 to 14 to ensure `FILL_WIDTH_GEMM ≥ 3`.

### 5.3 `INP_MAT_AXI_RATIO` portability issue

**Problem:** The initial `read_inp_vec_large` always read two separate inp_mem entries
(`inp_mem[src_idx]` and `inp_mem[src_idx + 1]`). On configurations with `BLOCK_IN ≥ 32`
(AXI ratio ≥ 2), both halves of the 25-element vector fit in a single entry's two bus words.
Using `src_idx + 1` reads the wrong entry.

**Fix:** Added an `#if INP_MAT_AXI_RATIO >= 2` guard:
```c
#if INP_MAT_AXI_RATIO >= 2
  bus_T word1 = inp_mem[src_idx][1];   // second bus word, same entry
#else
  bus_T word1 = inp_mem[src_idx + 1][0];  // next entry, first bus word
#endif
```

The ZCU104 config has `BLOCK_IN = 16`, so `INP_MAT_AXI_RATIO = 1` and the `else` branch is
active on the deployed bitstream. The guard ensures portability to wider-bus configurations.

---

## 6. VTA Runtime Wiring (`vta/runtime/runtime.cc`)

### New C functions exposed to TVM

```c
// Generic: caller specifies mat_trf_mode directly
int VTAPushGEMMMatTrfOp(void** uop_handle, int (*finit)(void*),
                         void* signature, int nbytes, int mat_trf_mode);

// Convenience wrappers
int VTAPushGEMMMatTrfOpSmall(void** uop_handle, int (*finit)(void*),
                              void* signature, int nbytes);
int VTAPushGEMMMatTrfOpLarge(void** uop_handle, int (*finit)(void*),
                              void* signature, int nbytes);
```

### `PushGEMMMatTrfOp` (private method on `CommandQueue`)

Identical to `PushGEMMOp` except it sets `insn->empty_0 = mat_trf_mode` in the
instruction word. All other fields (iter_out, iter_in, dst/src/wgt factors, UOP loop)
are populated identically.

```c
#if FILL_WIDTH_GEMM >= 2
    insn->empty_0 = static_cast<uint64_t>(mat_trf_mode);
#else
    CHECK(false) << "VTAPushGEMMMatTrfOp requires FILL_WIDTH_GEMM >= 2";
#endif
```

### Python bindings (`vta/python/vta/environment.py`)

```python
self.vta_push_uop_mat_trf_small = tvm.tir.StringImm("VTAPushGEMMMatTrfOpSmall")
self.vta_push_uop_mat_trf_large = tvm.tir.StringImm("VTAPushGEMMMatTrfOpLarge")
```

Both are added to `gemm_uop_scopes` in `transform.py` (FoldUopLoop) so that TVM's
outer loop folding applies correctly to gemm_mat_trf kernels.

---

## 7. TVM Tensor Intrinsic (`vta/python/vta/intrin.py`)

### `gemm_mat_trf(env, mock=False, large_mode=False)`

Returns a `TensorIntrin` for use with `schedule.tensorize()`.

#### Small mode (`large_mode=False`)

- **Tensorize axis:** `xbi` (innermost batch axis)
- **Outer loops after tensorize:** `(xbo=n_batch, xco=1)` → `n_batch` UOPs
- **UOP strides:** `dst_factor = 1`, `src_factor = 1`, `wgt_factor = 0`
- **Placeholder shapes:**
  - `inp`: `(BATCH, BLOCK_IN)` — 1 inp entry
  - `trf_mat`: `(BLOCK_OUT, BLOCK_IN)` — 1 wgt entry, T[i][j] = `trf_mat[i, j]`
  - `out`: `(BATCH, BLOCK_OUT)` — 1 acc entry
- **Reduce axis:** `k ∈ [0, BLOCK_IN)`

#### Large mode (`large_mode=True`)

- **Tensorize axis:** `xco` (output tile axis, range 2)
- **Outer loops after tensorize:** `(xbo=n_batch)` → `n_batch` UOPs
- **UOP strides:** `dst_factor = 2`, `src_factor = 2`, `wgt_factor = 0`
  (derived by FoldUopLoop from the output/input tensors having 2 tiles per batch)
- **Placeholder shapes:**
  - `inp`: `(2, BATCH, BLOCK_IN)` — 2 inp entries per batch
  - `trf_mat`: `(2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)` — 32 wgt entries
  - `out`: `(2, BATCH, BLOCK_OUT)` — 2 acc entries per batch
- **Reduce axes:** `ko ∈ [0, BLOCK_OUT)`, `ki ∈ [0, BLOCK_IN)`

#### Why `ko ∈ [0, BLOCK_OUT)` instead of `[0, 2)`

TVM's `CopyIntrinInjector` validates that any wgt-scope buffer used in a tensorize region
satisfies `_fold_buffer_dim` — the innermost contiguous dimensions of the buffer must
multiply to `BLOCK_OUT × BLOCK_IN = 256` (one VTA wgt entry). If `ko ∈ [0, 2)`, the
footprint for dimension 2 is size 2, and `2 × BLOCK_IN = 32 ≠ 256`. TVM fails.

Using `ko ∈ [0, BLOCK_OUT = 16)` makes the footprint cover ALL bus words of each entry,
so dimension 2 has size BLOCK_OUT and `BLOCK_OUT × BLOCK_IN = 256`. ✓

The T matrix is zero-padded in bus words 2–15, so those terms contribute nothing to the
sum. `inp` is accessed as `inp[tvm.tir.min(ko, 1), bi, ki]` — for `ko ≥ 2`, inp tile 1
is broadcast, but multiplied by zero T rows, so the sum is unaffected.

```python
out = te.compute(
    (2, env.BATCH, env.BLOCK_OUT),
    lambda co, bi, ci: te.sum(
        inp[tvm.tir.min(ko, 1), bi, ki].astype(out_dtype)
        * trf_mat[co * env.BLOCK_OUT + ci, 0, ko, ki].astype(out_dtype),
        axis=[ko, ki],
    ),
)
```

#### `intrin_func` (both modes)

```python
irb.scope_attr(dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE))
irb.scope_attr(dev.vta_axis, "coproc_uop_scope", push_uop_scope)
# push_uop_scope = vta_push_uop_mat_trf_small or vta_push_uop_mat_trf_large

irb.emit(tvm.tir.call_intrin("int32", "tir.vta.uop_push",
    0, 0,
    dout.access_ptr("rw", "int32"),   # dst_idx
    dinp.access_ptr("r", "int32"),    # src_idx
    dtrf_mat.access_ptr("r", "int32"), # wgt_idx
    0, 0, 0))
```

---

## 8. Schedule Builder (`test_benchmark_gemm_mat_trf_hw.py`)

### Small mode: `build_gemm_mat_trf_schedule(env, n_batch)`

```python
data_shape = (n_batch, 1, BATCH, BLOCK_IN)    # 1 inp entry/batch
trf_shape  = (1, 1, BLOCK_OUT, BLOCK_IN)      # 1 wgt entry (9 rows packed)
out_shape  = (n_batch, 1, BATCH, BLOCK_OUT)   # 1 acc entry/batch
# ...
s[res_gemm].tensorize(xbi, gemm_mat_trf(env, large_mode=False))
```

### Large mode: `build_gemm_mat_trf_large_schedule(env, n_batch)`

```python
data_shape = (n_batch, 2, BATCH, BLOCK_IN)        # 2 inp entries/batch
trf_shape  = (2 * BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)  # 32 wgt entries
out_shape  = (n_batch, 2, BATCH, BLOCK_OUT)        # 2 acc entries/batch

ko = te.reduce_axis((0, BLOCK_OUT), name="ko")    # full BLOCK_OUT range
ki = te.reduce_axis((0, BLOCK_IN),  name="ki")

res_gemm = te.compute(out_shape,
    lambda bo, co, bi, ci: te.sum(
        data_buf[bo, tvm.tir.min(ko, 1), bi, ki].astype(env.acc_dtype)
        * trf_buf[co * BLOCK_OUT + ci, 0, ko, ki].astype(env.acc_dtype),
        axis=[ko, ki]),
    name="res_gemm")
# ...
xbo, xco, xbi, xci = s[res_gemm].op.axis
s[res_gemm].tensorize(xco, gemm_mat_trf(env, large_mode=True))
```

**Buffer capacity constraint:** VTA inp_mem holds 2048 entries. Large mode uses 2 entries/
batch, so `n_batch ≤ 1024`. Small mode uses 1 entry/batch, so `n_batch ≤ 2048`.

### Data packing for large mode

```python
# inp[b][0..15]  → data_np[b, 0, 0, :]
# inp[b][16..24] → data_np[b, 1, 0, 0:9]   (bytes 9..15 zero-padded)
data_np = np.zeros(data_shape, dtype=...)
data_np[:, 0, 0, :]  = random_int8(n_batch, BLOCK_IN)
data_np[:, 1, 0, :9] = random_int8(n_batch, 9)

# T[i][0..15]  → trf_np[i, 0, 0, :]       (bus word 0)
# T[i][16..24] → trf_np[i, 0, 1, 0:9]    (bus word 1, bytes 9..15 zero-padded)
# rows 25..31 stay zero
trf_np = np.zeros(trf_shape, dtype=...)
trf_np[:DIM, 0, 0, :]  = random_int8(DIM, BLOCK_IN)
trf_np[:DIM, 0, 1, :9] = random_int8(DIM, 9)
```

### Reference computation (`numpy_ref_large`)

```python
inp = data_np[:, j//BLOCK_IN, 0, j%BLOCK_IN]  # reconstruct flat inp[b][j]
T   = trf_np[i, 0, j//BLOCK_IN, j%BLOCK_IN]   # reconstruct T[i][j]
acc = inp @ T.T                                # (n_batch, DIM)
acc = np.right_shift(acc, 8)
acc = np.clip(acc, 0, 127)
ref[:, i//BLOCK_OUT, 0, i%BLOCK_OUT] = acc[:, i]   # pack into (n_batch, 2, 1, 16)
```

---

## 9. Checkpoint: Small Mode Hardware Benchmark

Small mode was the first hardware validation milestone. Run after bitstream programming:

```
[Small mode — 9x9 transform (3x3 kernel), empty_0=1]

  n_batch=1    correctness: PASS    time: 0.064 ms   (0.003 GOPS)
  n_batch=4    correctness: PASS    time: 0.064 ms   (0.010 GOPS)
  n_batch=2048 correctness: PASS    time: 0.639 ms   (0.519 GOPS)
```

Small mode is compute-bound at n_batch=2048. 0.52 GOPS vs peak theoretical ~1.5 GOPS
(9×9 = 81 MACs per batch at 300 MHz) — overhead comes from instruction dispatch and
wgt DMA latency, which dominates at small n_batch.

---

## 10. Large Mode Hardware Benchmark

After implementing the TVM schedule with the `min(ko,1)` trick and tensorize-at-xco:

```
[Large mode — 25x25 transform (5x5 kernel), empty_0=3]

  n_batch=1    correctness: PASS    time: 0.062 ms   (0.020 GOPS)
  n_batch=4    correctness: PASS    time: 0.064 ms   (0.078 GOPS)
  n_batch=1024 correctness: PASS    time: 0.720 ms   (1.777 GOPS)
```

Large mode peaks at ~1.78 GOPS (n_batch=1024) — higher GOPS than small mode because the
25×25 = 625 MACs per batch element gives better compute-to-overhead ratio vs 9×9 = 81 MACs.

Maximum n_batch for large mode = 1024 (2 inp entries/batch × 1024 = 2048 = inp_mem capacity).

---

## 11. TVM Build Errors Encountered During Schedule Development

These errors appeared when building the large mode TVM schedule and required the
`min(ko,1)` solution described in section 7.

### 11.1 `BindBuffer: unmet assertion on local.wgt_buffer.shape[2]`

**Source:** `StorageFlattener::HandleBufferBindScope` in TVM's storage flattening pass.

**Cause:** When tensorizing at `xco` (range 2), TVM computes the trf_buf footprint for
the inner region as `(32, 1, 2, 16)` — only bus words 0 and 1 are accessed (`ko ∈ {0,1}`).
TVM then checks that the intrinsic's declared buffer shape matches this footprint.
With `trf_mat.shape = (32, 1, 16, 16)` (full entry), `shape[2] = 16 ≠ 2`. Assertion fails.

**Attempted fix:** Changed `trf_mat.shape[2]` to `2` → passed BindBuffer but triggered the
next error below.

### 11.2 `RuntimeError: scope local.wgt_buffer needs to have block=256`

**Source:** `CopyIntrinInjector::MatchCopyPattern` → Python `_inject_copy` →
`_get_2d_pattern` → `_fold_buffer_dim` in `vta/python/vta/transform.py`.

**Cause:** VTA's copy injector validates every DRAM→wgt DMA by checking that the innermost
contiguous block of the destination buffer equals `BLOCK_OUT × BLOCK_IN = 256`. With
`trf_mat.shape = (32, 1, 2, 16)`, the inner product is `2 × 16 = 32 ≠ 256`. Also: after
tensorize, TVM rewrites the DRAM→wgt DMA to only transfer the footprint `(32, 1, 2, 16)`
= 1024 bytes (4 entries), whereas the hardware reads 32 entries (8192 bytes).

**Root cause:** Using `ko ∈ [0, 2)` causes TVM to "fold" the DMA down to only the accessed
footprint, breaking the VTA block-size requirement and producing an incorrect transfer size.

**Final fix:** Extend `ko ∈ [0, BLOCK_OUT = 16)` so the trf footprint spans the full entry.
Access `inp[tvm.tir.min(ko, 1), ...]` to broadcast the two valid input tiles to all ko values;
T is zero in bus words 2–15, so those ko terms contribute zero to the sum. The full
`(32, 1, 16, 16)` shape passes `_fold_buffer_dim` (16×16=256) and the DMA correctly
transfers 8192 bytes (32 full entries).

### 11.3 Inp buffer overflow for `n_batch=2048`

**Error:** `Allocation exceed bound of memory tag local.inp_buffer (524288 vs. 262144 bits)`

**Cause:** Large mode needs 2 inp entries per batch. At n_batch=2048: 4096 entries × 16 bytes
= 65536 bytes = 524288 bits. VTA inp_mem max = 2048 entries = 32768 bytes = 262144 bits.

**Fix:** Cap `n_batch` at 1024 for large mode.

---

## 12. File Summary

### HLS / Hardware (`3rdparty/vta-hw/`)

| File | What changed |
|------|-------------|
| `include/vta/hw_spec.h` | `VTA_GEMM_MAT_TRF_{MODE,DIM}_BIT`, `VTA_GEMM_MAT_TRF_EMPTY0_{SMALL,LARGE}`, `VTA_MAT_TRF_DIM_{SMALL,LARGE}`, stride macros, wgt-row count macros |
| `hardware/xilinx/src/vta.cc` | 7 new functions (helpers + cores + dispatcher); `compute()` dispatch patch; `#if INP_MAT_AXI_RATIO` portability guard |
| `hardware/xilinx/src/vta.h` | Prototypes for 7 new functions |
| `hardware/xilinx/sim/vta_test.cc` | 6 `gemm_mat_trf_test()` calls |
| `tests/hardware/common/test_lib.cc` | `gemm_mat_trf_test()` implementation |
| `tests/hardware/common/test_lib.h` | `gemm_mat_trf_test()` declaration |
| `config/vta_config_gemm_mat.json` | Custom config: LOG_UOP_BUFF_SIZE=14 |

### TVM Runtime (`vta/runtime/`)

| File | What changed |
|------|-------------|
| `runtime.cc` | `PushGEMMMatTrfOp()`, `VTAPushGEMMMatTrfOp()`, `VTAPushGEMMMatTrfOpSmall()`, `VTAPushGEMMMatTrfOpLarge()` |
| `runtime.h` | Declarations + doc comments for the 3 exported functions |

### TVM Python (`vta/python/vta/`)

| File | What changed |
|------|-------------|
| `environment.py` | `vta_push_uop_mat_trf_small`, `vta_push_uop_mat_trf_large` string imms |
| `transform.py` | `gemm_uop_scopes` set extended with both mat_trf scope strings |
| `intrin.py` | `gemm_mat_trf()` rewritten: `large_mode=False` path unchanged, `large_mode=True` path added with `min(ko,1)` trick and `(2,BATCH,BLOCK_IN)` / `(2*BLOCK_OUT,1,BLOCK_OUT,BLOCK_IN)` / `(2,BATCH,BLOCK_OUT)` shapes |

### Tests

| File | What changed |
|------|-------------|
| `vta/tests/python/integration/test_benchmark_gemm_mat_trf_hw.py` | Added `build_gemm_mat_trf_large_schedule()`, `numpy_ref_large()`, `run_test_large()`; updated `_run()` with large mode tests (n_batch=1,4,1024) |

---

## 13. Quick-Start: Running the Benchmark

```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork/tvm
VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 \
    python vta/tests/python/integration/test_benchmark_gemm_mat_trf_hw.py
```

Expected output:
```
GEMM_Mat_Trf Hardware Benchmark
  Config: BATCH=1  BLOCK_IN=16  BLOCK_OUT=16

[Small mode — 9x9 transform (3x3 kernel), empty_0=1]
  n_batch=1    correctness: PASS    time: ~0.06 ms
  n_batch=4    correctness: PASS    time: ~0.06 ms
  n_batch=2048 correctness: PASS    time: ~0.64 ms  (~0.52 GOPS)

[Large mode — 25x25 transform (5x5 kernel), empty_0=3]
  n_batch=1    correctness: PASS    time: ~0.06 ms
  n_batch=4    correctness: PASS    time: ~0.06 ms
  n_batch=1024 correctness: PASS    time: ~0.72 ms  (~1.78 GOPS)

All tests passed.
```
