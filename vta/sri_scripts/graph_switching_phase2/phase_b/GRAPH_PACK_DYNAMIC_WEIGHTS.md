# graph_pack_dynamic_weights
`graph_pack_dynamic_weights` is a dynamic-weight-safe wrapper around `vta.top.graph_pack`.
## Why
Quantized OFA graphs with runtime-derived weights include branches like:
- `strided_slice`
- `reshape`
- `nn.dense` (kernel transform)
- `reshape`
- `nn.conv2d`
Classic `graph_pack` start/stop boundaries can miss these paths or trigger shape/rank failures.
## API
Location: `tvm/vta/python/vta/top/graphpack.py`
```python
graph_pack_dynamic_weights(
    expr,
    bfactor,
    blockin,
    blockout,
    weight_bits,
    start_name="nn.max_pool2d",
    stop_name="nn.global_avg_pool2d",
    start_name_idx=None,
    stop_name_idx=None,
    count_meta=False,
    device_annot=False,
    annot_start_name="nn.conv2d",
    annot_end_name="annotation.stop_fusion",
    pack_all=True,
    allow_fallback=True,
    return_status=False,
)
```
### Dynamic-specific behavior
- `pack_all=True` auto-discovers first/last op indices from ANF and packs full graph region.
- `allow_fallback=True` falls back to unpacked expression for known dynamic graph transpose rank mismatch.
- `return_status=True` returns `(packed_expr, used_graph_pack, fallback_reason)`.
## Step3 integration
`step3_relay_build_poc.py` now uses this API in Step 3B.
## Smoke test
Run from repo root:
```bash
cd /home/srchand/Desktop/research/TVM_Intel_Fork
PYTHONPATH=/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/python:/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/python \
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python \
/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/graph_switching_phase2/phase_b/test_graph_pack_dynamic_weights_smoke.py
```
## Notes
- This is an additive API; existing `graph_pack` is unchanged.
- The fallback path is intentionally narrow to known dynamic graph failure signatures.
