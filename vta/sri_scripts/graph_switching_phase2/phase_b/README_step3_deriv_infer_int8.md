# Step3 Split Deriv/Infer Int8 Boundary

## What this adds

`step3_relay_build_poc_deriv_infer_int8.py` runs the split-module flow with a configurable
boundary contract between `mod_deriv` and `mod_infer`.

- `int8` mode (default):
  - `mod_deriv` outputs quantized `int8` derived weights using fixed `GLOBAL_SCALE`.
  - `mod_infer` consumes `int8` tensors and dequantizes internally.
- `float32` mode (fallback):
  - behaves like the original split boundary for A/B debugging.

The boundary transform is implemented in:
- `ofa_relay_graph_builder.py` via `split_derivation_and_inference_modules_quant_boundary`.

## Quick run

```bash
python tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_relay_build_poc_deriv_infer_int8.py \
  --num-subnets=1 --skip-cpu --enable-dynamic-dense-quant
```

## Fallback A/B run (float32 boundary)

```bash
python tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_relay_build_poc_deriv_infer_int8.py \
  --num-subnets=1 --skip-cpu --enable-dynamic-dense-quant \
  --derived-weight-dtype=float32
```

## Notes

- No new Python dependencies were introduced.
- Summary JSON is written to:
  - `tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_results/step3_deriv_infer_int8_summary.json`

