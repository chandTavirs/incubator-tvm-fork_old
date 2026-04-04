# Step3 Merged Mod Deriv POC

This POC uses a single Relay module that includes both:
- runtime derivation from `pool_*` tensors, and
- inference conv graph.

## Why

The split-mod path uploads large `derived_w_*` tensors to VTA runtime inputs and can hit memory pressure.
This merged path uploads only pool tensors and computes transformed weights inside the compiled graph.

## Script

- `tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_merged_mod_deriv_poc.py`

## Quick Run

```bash
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python \
/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_merged_mod_deriv_poc.py \
--num-subnets=1 --enable-dynamic-dense-quant
```

## Typical Dynamic Run (no CPU validation stage)

```bash
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python \
/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_merged_mod_deriv_poc.py \
--num-subnets=1 --skip-cpu --enable-dynamic-dense-quant
```

## Static Debug Mode

```bash
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python \
/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_merged_mod_deriv_poc.py \
--num-subnets=1 --enable-dynamic-dense-quant --static-debug-mode
```

## Outputs

- Relay IR dump: `step3_results/merged_relay_ir_<subnet>.txt`
- Built graph JSON: `step3_results/merged_graph_<subnet>.json`
- Run summary: `step3_results/step3_merged_summary.json`

