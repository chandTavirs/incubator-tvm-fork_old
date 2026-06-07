"""Probe the VTA device CMA allocation limit and isolate alloc vs execution.

Usage: VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 python probe_cma.py
"""
import os, sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TVM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
for p in [os.path.join(TVM_ROOT, "python"), os.path.join(TVM_ROOT, "vta", "python")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import tvm
import vta
import vta.testing


def _run(env, remote):
    vta.reconfig_runtime(remote)
    ctx = remote.ext_dev(0)
    print("=" * 56)
    print("CMA single-allocation probe (int8 buffers on ext_dev)")
    print("=" * 56)
    held = []  # keep references so they aren't freed between steps
    for mb in [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
               64, 64, 64, 64, 64, 64, 64, 64]:  # push cumulatively until CMA fails
        n = mb * 1024 * 1024
        try:
            a = tvm.nd.array(np.zeros((n,), dtype="int8"), ctx)
            a.copyfrom(np.ones((n,), dtype="int8"))  # force a real DMA write
            held.append(a)
            print("  alloc+write %4d MB  OK  (cumulative %d MB)" % (mb, sum(
                int(np.prod(x.shape)) for x in held) // (1024 * 1024)))
        except Exception as e:
            print("  alloc+write %4d MB  FAIL: %s" % (mb, str(e).splitlines()[-1][:80]))
            break
    del held
    print("-" * 56)
    # Now test ONE fresh large single allocation (no cumulative) at the crash size
    for mb in [4, 8, 16]:
        n = mb * 1024 * 1024
        try:
            a = tvm.nd.array(np.zeros((n,), dtype="int8"), ctx)
            a.copyfrom(np.ones((n,), dtype="int8"))
            print("  single %4d MB  OK" % mb)
            del a
        except Exception as e:
            print("  single %4d MB  FAIL: %s" % (mb, str(e).splitlines()[-1][:80]))


if __name__ == "__main__":
    vta.testing.run(_run)
