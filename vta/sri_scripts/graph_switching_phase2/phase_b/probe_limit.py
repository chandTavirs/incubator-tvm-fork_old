"""Discriminate the GMTF execution limit: tile-count vs total-rows.

Usage: VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 python probe_limit.py
"""
import os, sys, time
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TVM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
for p in [os.path.join(TVM_ROOT, "python"), os.path.join(TVM_ROOT, "vta", "python")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import tvm
from tvm import te, topi
from tvm.contrib import utils as tvm_utils
import vta
import vta.testing
from vta import intrin as vta_intrin

env = vta.get_env(); B, BI, BO = env.BATCH, env.BLOCK_IN, env.BLOCK_OUT


def _collect(output):
    c, ei, ew, dr = [], [], [], []
    def _t(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                (c if not op.axis else ew).append(op)
            for tt in op.input_tensors:
                if isinstance(tt.op, tvm.te.PlaceholderOp): ei.append((op, tt))
                else: _t(tt.op)
        else: dr.append(op)
    _t(output.op); return c, ei, ew, dr[0].output(0)


def build_small(n_batch, TILE):
    data = te.placeholder((n_batch, 1, B, BI), env.inp_dtype, "data")
    wgt = te.placeholder((1, 1, BO, BI), env.wgt_dtype, "wgt")
    ki = te.reduce_axis((0, BI), "ki")
    res = te.compute((n_batch, 1, B, BO),
        lambda b, co, bi, ci: te.sum(
            data[b, 0, bi, ki].astype(env.acc_dtype) * wgt[0, 0, ci, ki].astype(env.acc_dtype),
            axis=[ki]), name="res", tag="dense_gmtf_small")
    out = topi.cast(topi.clip(topi.right_shift(res, 8), -127, 127), env.inp_dtype)
    s = te.create_schedule(out.op)
    c, ei, ew, ds = _collect(out)
    cd = s.cache_read(data, env.inp_scope, [ds]); cw = s.cache_read(wgt, env.wgt_scope, [ds])
    s[ds].set_scope(env.acc_scope)
    cre = [s.cache_read(t, env.acc_scope, [cc]) for cc, t in ei]
    for op in ew: s[op].set_scope(env.acc_scope); s[op].pragma(s[op].op.axis[0], env.alu)
    for op in c: s[op].compute_inline()
    xb, xco, xbi, xci = s[out].op.axis
    xbo, xbn = s[out].split(xb, factor=TILE)
    s[out].reorder(xbo, xco, xbn, xbi, xci)
    s[ds].compute_at(s[out], xco)
    for op in ew: s[op].compute_at(s[out], xco)
    for t in cre: s[t].compute_at(s[out], xco); s[t].pragma(s[t].op.axis[0], env.dma_copy)
    db, dco, dbi, dci = s[ds].op.axis; (dki,) = s[ds].op.reduce_axis
    dko, dkii = s[ds].split(dki, factor=BI)
    s[ds].reorder(db, dko, dbi, dci, dkii)
    s[cd].compute_at(s[ds], dko); s[cw].compute_at(s[ds], dko)
    s[cd].pragma(s[cd].op.axis[0], env.dma_copy); s[cw].pragma(s[cw].op.axis[0], env.dma_copy)
    s[ds].tensorize(dbi, vta_intrin.gemm_mat_trf(env, mock=False, large_mode=False))
    s[out].pragma(xci, env.dma_copy)
    return s, [data, wgt, out]


def run(remote, n_batch, TILE):
    n_tiles = (n_batch + TILE - 1) // TILE
    s, args = build_small(n_batch, TILE)
    with vta.build_config():
        m = vta.build(s, args, "ext_dev", env.target_host, name="g")
    tmp = tvm_utils.tempdir(); fp = tmp.relpath("g.o"); m.save(fp)
    remote.upload(fp); f = remote.load_module("g.o")
    ctx = remote.ext_dev(0)
    da = tvm.nd.array(np.zeros((n_batch, 1, B, BI), np.int8), ctx)
    wa = tvm.nd.array(np.zeros((1, 1, BO, BI), np.int8), ctx)
    oa = tvm.nd.array(np.zeros((n_batch, 1, B, BO), np.int8), ctx)
    t0 = time.time()
    try:
        f(da, wa, oa)
        ctx.sync()
        dt = (time.time() - t0) * 1000
        print("  n_batch=%-8d TILE=%-4d tiles=%-5d  OK   %.0f ms" % (n_batch, TILE, n_tiles, dt))
        return True
    except Exception as e:
        print("  n_batch=%-8d TILE=%-4d tiles=%-5d  CRASH: %s"
              % (n_batch, TILE, n_tiles, str(e).splitlines()[-1][:70]))
        return False


def _run(env_, remote):
    vta.reconfig_runtime(remote)
    print("=" * 64)
    print("GMTF small execution-limit probe (tile-count vs total-rows)")
    print("=" * 64)
    # Fixed total rows = 131072 (known OK at TILE=512=256 tiles), vary TILE → vary tiles
    print("[A] same total rows (131072), more tiles via smaller TILE:")
    for TILE in [512, 256, 128, 64]:
        if not run(remote, 131072, TILE):
            break
    # Fixed TILE=512, increase rows until crash
    print("[B] TILE=512, increasing rows:")
    for nb in [131072, 163840, 196608, 229376, 262144]:
        if not run(remote, nb, 512):
            break


if __name__ == "__main__":
    vta.testing.run(_run)
