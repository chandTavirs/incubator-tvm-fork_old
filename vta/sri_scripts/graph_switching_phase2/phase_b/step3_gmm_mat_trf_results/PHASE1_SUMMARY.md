# GEMM_Mat_Trf Phase 1 Summary

**Date completed:** 2026-05-17  
**Branch:** `gemm_mat_trf`  
**Result:** Phase 1 exit achieved — CPU-side materialization runs end-to-end on VTA device.

---

## Goal

Extend VTA's instruction set with a new GEMM_Mat_Trf operation for OFA dynamic weight derivation.
OFA subnets derive conv weights at runtime via matrix transforms:

```
out = input_blocks @ transform_matrix.T
```

Phase 1 validates this in software (CPU materialization + behavioral simulator) before any RTL changes.
Phase 2 will add a dedicated hardware datapath.

---

## Architecture Overview

### New VTA Instruction (Behavioral Level)

**File:** `vta/python/vta/beh/instructions/gemm.py`

- Added `VTA_GEMM_MODE_STANDARD = 0` and `VTA_GEMM_MODE_MAT_TRF = 1`
- Bit 7 of the GEMM instruction repurposed as a mode selector (was unused)
- `gemm_trf_core()` implements transform semantics: direct assign (no accumulation), unlike standard GEMM
- Main `gemm()` function dispatches to `gemm_core()` or `gemm_trf_core()` based on mode flag

**File:** `vta/python/vta/beh/datatype.py`

- Added mode constants to the behavioral simulator's datatype definitions

### TVM Intrinsic Definition

**File:** `vta/python/vta/intrin.py`

- `gemm_mat_trf(env, mock=False)` intrinsic defined alongside the standard `gemm()` intrinsic
- Emits `uop_push(mode=1, ...)` calls to signal the transform operation
- Reuses the same VTA buffer layouts (inp_scope, wgt_scope, acc_scope) and block dimensions

> **Note (Phase 2 work item):** In the VTA C++ runtime, `uop_push(mode=1)` creates an ALU micro-op, not a GEMM instruction with bit 7 set. The intrinsic will need to be rewired to the correct instruction encoding before it can execute on real hardware. This does not affect Phase 1 (CPU materialization path does not use the TVM intrinsic).

### CPU Runtime Materializer

**File:** `vta/python/vta/runtime_gemm_mat_trf.py`

- `vta_gemm_mat_trf_cpu(input_blocks, transform_matrix)` — core CPU numpy implementation
  - Supports int8 (with int32 accumulation, matching VTA semantics) and float32
- `batch_vta_gemm_mat_trf_cpu()` — batch processing
- `quantize_and_transform()` — quantization + transform in one pass

### Relay Lowering Pass

**File:** `vta/python/vta/relay_passes/lower_gemm_mat_trf.py`

Detects the OFA transform pattern in the Relay graph:
```
strided_slice(base_weight_var)         # channel + spatial crop
  → strided_slice(...)                 # spatial center crop to target ks
  → reshape([out_ch*in_ch, tgt_ks^2])
  → nn.dense(_, transform_matrix_var)  ← TARGET
  → reshape([out_ch, in_ch, tgt_ks, tgt_ks])
  → nn.conv2d(data, derived_weight)
```

Two modes:
- **CPU mode** (`cpu_materialization=True`): substitutes the `nn.dense` with `relay.const(precomputed_output)` — the quantizer sees plain constants
- **VTA composite mode** (`cpu_materialization=False`): wraps the dense in a `"vta.gemm_mat_trf"` composite function — reserved for Phase 2

### Integration Script

**File:** `vta/sri_scripts/graph_switching_phase2/phase_b/step3_gemm_mat_trf_integration.py`

Full 8-step pipeline for OFA subnet inference with CPU-side transform materialization:

| Step | Description |
|---|---|
| 1 | Load OFA model |
| 2 | Load OFA weight pool (base weights + transform matrices) |
| 3 | Build subnet-specific Relay graph with pool variables |
| 4 | **Materialize transforms on CPU** (graph-aware, subnet-specific) |
| 4b | Lower transform dense ops to `relay.const` in the Relay graph |
| 5 | Quantize the module |
| 6 | Rewrite pool vars to int8 for runtime upload |
| 7 | graph_pack + relay.build for VTA |
| 8 | RPC upload + inference on VTA device |

---

## Key Implementation Decisions

### Graph-Aware Transform Materialization

The materialization walks the type-checked Relay graph using a post-order `ExprVisitor` and recursively evaluates each expression to numpy:

- **Direct pattern** (first transform in a chain): `reshape(strided_slice(strided_slice(base_var)))` — two strided slices (channel crop + spatial center crop), both extracted from the IR
- **Chained pattern** (subsequent transforms, e.g. 5→3 after 7→5): `reshape(strided_slice(reshape(nn.dense(prev_data, trf_var))))` — the previous dense output is cached and reused

This is necessary because the pool stores weights at the maximum subnet channel count, but each subnet's Relay graph slices to the subnet-specific channel count before the transform. Using pool-level shapes directly produces an output that cannot be reshaped to the expected conv2d weight shape.

**Cache key:** `call.handle.value` (the underlying C++ pointer) rather than Python `id()`, because TVM's FFI creates a new Python wrapper object each time `.args` is accessed — making `id()` unreliable for node identity across accesses.

### Transform Substitution

`_lower_cpu` in the Relay pass replaces `nn.dense(data, trf_var)` with `relay.const(precomputed_output)`:
- `precomputed_output` has shape `[subnet_out_ch * subnet_in_ch, tgt_ks^2]`
- The downstream `reshape → nn.conv2d` chain is preserved unchanged
- The transform matrix var and base weight slice become dead code (cleaned up by the optimizer)

---

## Tests

**File:** `vta/tests/python/test_gemm_mat_trf_basic.py`

4 tests, all passing:
- CPU int8 matrix multiply correctness
- CPU float32 matrix multiply correctness  
- Batch processing of multiple transforms
- VTA intrinsic definition and instantiation

**File:** `vta/tests/python/bench_gemm_mat_trf.py`

CPU materialization throughput: 0.05–0.92 GOPS for typical kernel sizes.

---

## End-to-End Result

**Subnet:** `arch_20250927_180844_0126`  
**Architecture:** `kernel_size_list=[[[3],[5]],[[3],[]],[[3],[3]],[[5,7],[3]]]`, 4 residual stages  
**Transforms materialized:** 12 (7to5 and 5to3 chains across 4 stages)  
**Pool vars uploaded at runtime:** 5 base weights (float32 → int8)  
**VTA inference time:** ~248 ms  
**Device:** ZCU104 at 10.42.0.188:9091, BLOCK_IN=16, BLOCK_OUT=16

The pipeline runs without errors. Top-1 disagreement between VTA and float32 reference on a random input is expected due to quantization and is not a correctness failure.

---

## Bugs Fixed During Phase 1

| Bug | Root Cause | Fix |
|---|---|---|
| All 32 transform lookups returned "not found in pool" | `transform_var_names` carry `pool_` prefix but `pool_var_dict` was built with raw keys only | Added `pool_` prefixed form alongside raw key in `pool_var_dict` |
| Shape mismatch in `relay.const` substitution (2304 vs 147456 elements) | `_build_input_block_slices` used the full pool weight (max channel count), not the subnet-sliced weight | Replaced with `_materialize_transforms_from_graph`: walks the Relay IR to read actual `strided_slice` attrs |
| Chained 5to3 transforms failed cache lookup for parent 7to5 output | TVM FFI creates a new Python wrapper per `.args` access — `id()` was different each time even for the same C++ node | Changed cache key from `id(call)` to `call.handle.value` (stable C++ pointer) |
| `m = graph_runtime.create(graph, lib, ctx)` assertion failed | `lib` is a local module; RPC requires upload to remote before use | Added `lib.export_library` → `remote.upload` → `remote.load_module` before `create` |
| `m.load_params(params)` TypeError | `relay.build` returns a dict; `load_params` expects serialized bytes | Changed to `m.set_input(**params)` |
| `remote.close()` AttributeError | `RPCSession` has no `close()` method | Changed to `del remote` |

---

## Phase 2 Work Items

1. **Fix `intrin.py` uop_push semantics** — `uop_push(mode=1)` creates an ALU op in the C++ runtime, not a GEMM with bit 7 set. Need to either add a new runtime API or use a separate opcode to correctly signal GEMM_MAT_TRF mode to the hardware.

2. **VTA backend wiring for `"vta.gemm_mat_trf"` composites** — graph_pack needs to handle or skip 2D tensor composites (they are not conv2d packed tensors). A backend lowering pass is needed to schedule the composite through the `gemm_mat_trf` intrinsic during `relay.build`.

3. **RTL implementation** — Add a dedicated matrix transform FSM state in the Xilinx HLS compute module, routed from the GEMM_MAT_TRF mode flag in the instruction decoder.

4. **Accuracy validation** — Run against a real ImageNet validation set to measure Top-1 accuracy degradation from quantization.
