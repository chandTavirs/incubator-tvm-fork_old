# GEMM_Mat_Trf: VTA Hardware Instruction for OFA Kernel Transforms

## Overview

GEMM_Mat_Trf is a new VTA instruction that computes dense matrix transforms in hardware,
targeting Once-For-All (OFA) dynamic weight derivation:

```
out[b, r] = sum_k( inp[b, k] * T[r, k] )    for each batch row b
```

where `T` is the transform matrix and `inp` contains packed input blocks. This is the
per-batch-row dot product used when deriving a smaller OFA kernel from a stored
transform matrix.

## Instruction Encoding

The instruction reuses the VTA GEMM instruction word, using the `FILL_FIELD_GEMM` bits
(bits 61–63 of word 0, available when `LOG_UOP_BUFF_SIZE ≤ 14`):

| Bit | Field                        | Value |
|-----|------------------------------|-------|
| 61  | `VTA_GEMM_MAT_TRF_MODE_BIT` | 1 = GEMM_Mat_Trf, 0 = standard GEMM |
| 62  | `VTA_GEMM_MAT_TRF_DIM_BIT`  | 0 = 9×9 (3×3 kernel), 1 = 25×25 (5×5 kernel) |

`empty_0` field mapping: `0x1` = small mode (9×9), `0x3` = large mode (25×25).

**Note:** Bit 7 of VTAGemInsn is `reset_reg` — not a free bit. Must use FILL_FIELD_GEMM.
With `LOG_UOP_BUFF_SIZE=15`, `FILL_WIDTH_GEMM=1` and `DIM_BIT=64` (out of range).
**Use `LOG_UOP_BUFF_SIZE=14`** to get `FILL_WIDTH_GEMM=3`.

## Config: `vta_config_gemm_mat.json`

Base: ZCU104 standard config with one change: `LOG_UOP_BUFF_SIZE: 15 → 14`.

- `BLOCK_IN = 16`, `BLOCK_OUT = 16`
- `FILL_WIDTH_GEMM = 3` (bits 61–63 free)
- UOP buffer: 16384 entries (vs 32768 in standard config)
- IP dir: `zcu104_1x16x16_i8w8a32_14_15_18_17`

## Buffer Layout

### Small mode (dim=9, 3×3 kernel)

| Buffer   | Entries per batch row | Layout |
|----------|-----------------------|--------|
| `inp_mem`  | 1 (`STRIDE=1`)      | bytes 0–8: input vector; bytes 9–15: zeros |
| `wgt_mem`  | 9 (one per T row)   | each entry: bytes 0–8 = T[row][0..8]; rest zeros |
| `acc_mem`/`out_mem` | 1 (`STRIDE=1`) | 9-element output |

### Large mode (dim=25, 5×5 kernel)

| Buffer   | Entries per batch row | Layout |
|----------|-----------------------|--------|
| `inp_mem`  | 2 (`STRIDE=2`)      | entry 0: bytes 0–15; entry 1: bytes 0–8 (elems 16–24) |
| `wgt_mem`  | 25 (one per T row)  | each entry: bytes 0–15 = T[row][0..15]; bytes 16–24 = T[row][16..24] (large only) |
| `acc_mem`/`out_mem` | 2 (`STRIDE=2`) | same layout as inp |

`wgt_factor_in = wgt_factor_out = 0` — transform matrix is fixed across all batch rows.
UOPs encode `dst_idx = b * STRIDE`, `src_idx = b * STRIDE`, `wgt_idx = 0`.

## Instruction Dependency Chain

```
LOAD_UOP  (no dep)
LOAD_INP  (no dep)
LOAD_WGT  (push_next → l2g)
GEMM_MAT_TRF (pop_prev ← l2g, push_next → g2s)
STORE     (pop_prev ← g2s, push_prev → s2g)
FINISH    (pop_next ← s2g)
```

## Files Modified

### vta-hw (HLS + testbench)

| File | Change |
|------|--------|
| `include/vta/hw_spec.h` | `VTA_GEMM_MAT_TRF_MODE_BIT`, `VTA_GEMM_MAT_TRF_DIM_BIT`, `VTA_MAT_TRF_DIM_{LARGE,SMALL}`, parameterized stride macros |
| `hardware/xilinx/src/vta.cc` | `read_inp_vec_{large,small}`, `read_trf_row_{large,small}`, `write_trf_result_{large,small}`, `gemm_mat_trf_{large,small}()`, `gemm_mat_trf()` dispatcher, patched `compute()` dispatch; `#if INP_MAT_AXI_RATIO` guards for BLOCK_IN portability |
| `hardware/xilinx/src/vta.h` | Prototypes for all new functions |
| `hardware/xilinx/sim/vta_test.cc` | 6 test calls: small + large × batch=1,4,16 |
| `tests/hardware/common/test_lib.cc` | `gemm_mat_trf_test()` implementation (~194 lines) |
| `tests/hardware/common/test_lib.h` | `gemm_mat_trf_test()` declaration |
| `config/vta_config_gemm_mat.json` | Custom config (LOG_UOP_BUFF_SIZE=14) |
| `config/vta_config_gemm_mat.tcl` | TCL export of custom config |
| `hardware/xilinx/scripts/vivado.tcl` | `ip_path` updated to `zcu104_1x16x16_i8w8a32_14_15_18_17` |

### TVM (Python runtime + Relay)

| File | Change |
|------|--------|
| `cmake/modules/VTA.cmake` | Added `zcu104` to pynq FPGA driver branch (fixes `VTAMemFree` undefined symbol) |
| `vta/python/vta/beh/instructions/gemm.py` | `gemm_trf_core()` behavioral simulator, mode dispatch |
| `vta/python/vta/intrin.py` | `gemm_mat_trf()` TVM tensor intrinsic |
| `vta/python/vta/runtime_gemm_mat_trf.py` | CPU numpy reference implementation |
| `vta/python/vta/relay_passes/lower_gemm_mat_trf.py` | Relay lowering pass (CPU + VTA composite modes) |
| `vta/python/vta/phase_gemm_mat_trf.py` | End-to-end pipeline utilities |
| `vta/tests/python/test_gemm_mat_trf_basic.py` | Unit tests |
| `vta/tests/python/bench_gemm_mat_trf.py` | Latency benchmark |

## Build and Run

### HLS csim

```bash
vivado_hls -f <vta-hw>/hardware/xilinx/scripts/hls.tcl \
           <vta-hw> \
           <vta-hw>/config/vta_config_gemm_mat.tcl
```

### IP generation + bitstream

```bash
cd <vta-hw>
make ip    # uses vta_config_gemm_mat.json automatically
# Bitstream: open Vivado GUI, source hardware/xilinx/scripts/vivado.tcl
```

### Device rebuild (ZCU104)

After syncing `hw_spec.h` and the updated `cmake/modules/VTA.cmake` to the device:
```bash
# On device: ensure config/vta_target.json has "TARGET": "zcu104"
# and cmake/config.cmake has USE_VTA_FPGA ON
cd ~/tvm_il/build && cmake .. && make vta -j4
```

## Implementation Status

| Phase | Item | Status |
|-------|------|--------|
| Phase 1 | CPU-side materialization | ✅ Done |
| Phase 1 | Behavioral simulator | ✅ Done |
| Phase 1 | Relay pass (CPU mode) | ✅ Done |
| Phase 1 | End-to-end on ZCU104 (CPU path) | ✅ Done (~248ms, 12 transforms) |
| Phase 2 | HLS implementation (vta.cc) | ✅ Done |
| Phase 2 | C++ testbench (all 6 cases pass) | ✅ Done |
| Phase 2 | IP generation (csim validated) | ✅ Done |
| Phase 2 | Bitstream + device deployment | ✅ Done |
| Phase 2 | cmake zcu104 fix (VTAMemFree) | ✅ Done |
| Phase 3 | `VTAPushGEMMMatTrfOp` in VTA C++ runtime | 🔲 TODO |
| Phase 3 | Fix `intrin.py` emission (empty_0 bits) | 🔲 TODO |
| Phase 3 | Standalone Python benchmark | 🔲 TODO |
| Phase 3 | Relay composite → hardware path | 🔲 TODO |
| Phase 3 | ImageNet accuracy validation | 🔲 TODO |
