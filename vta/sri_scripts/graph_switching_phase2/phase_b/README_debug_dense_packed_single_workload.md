# Debug Dense Packed Single Workload

This helper isolates a single workload:

- `('dense_packed.vta', ('TENSOR', (256, 1, 1, 16), 'int8'), ('TENSOR', (1, 1, 16, 16), 'int8'), None, 'int32')`

It splits debug into:

1. local schedule/lower/build
2. optional remote RPC execution (without programming bitstream)

## Script

- `tvm/vta/sri_scripts/graph_switching_phase2/phase_b/debug_dense_packed_single_workload.py`

## Quick Runs

```bash
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python -u /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/graph_switching_phase2/phase_b/debug_dense_packed_single_workload.py --mode local
```

```bash
/home/srchand/anaconda3/envs/tvm-build-il-2/bin/python -u /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/graph_switching_phase2/phase_b/debug_dense_packed_single_workload.py --mode both --use-tracker --tracker-host 127.0.0.1 --tracker-port 9190 --rpc-key zcu104
```

Add `--reconfig-runtime` if you want to explicitly call `vta.reconfig_runtime(remote)` before the run.

