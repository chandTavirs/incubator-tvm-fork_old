"""Standalone hardware test for the NEW tiled GMTF schedules (vta_gmtf_dense.py).

Builds dense_pack_gmtf_small/large with the multi-tile + outer-reduce-load-point
schedule, runs on the ZCU104, and verifies correctness against a numpy reference.
Sweeps n_batch from single-tile up to multi-tile sizes to catch acc/inp overflow
or out-of-bounds at runtime.

Usage:
    VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 python test_gmtf_tiled_hw.py
"""
from __future__ import absolute_import, print_function

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TVM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
for p in [os.path.join(TVM_ROOT, "python"), os.path.join(TVM_ROOT, "vta", "python")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tvm
from tvm import te, topi, rpc
from tvm.contrib import utils as tvm_utils
import vta
import vta.testing
from vta import intrin as vta_intrin

env = vta.get_env()
B, BI, BO = env.BATCH, env.BLOCK_IN, env.BLOCK_OUT
DIM_SMALL, DIM_LARGE = 9, 25


# ---- schedule logic copied verbatim from vta_gmtf_dense.py ----
def _collect(output):
    const_ops, ewise_inputs, ewise_ops, dense_res = [], [], [], []
    def _t(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                (const_ops if not op.axis else ewise_ops).append(op)
            for tt in op.input_tensors:
                if isinstance(tt.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tt))
                else:
                    _t(tt.op)
        else:
            dense_res.append(op)
    _t(output.op)
    return const_ops, ewise_inputs, ewise_ops, dense_res[0].output(0)


def build_small(n_batch, TILE=512):
    data = te.placeholder((n_batch, 1, B, BI), env.inp_dtype, "data")
    wgt = te.placeholder((1, 1, BO, BI), env.wgt_dtype, "wgt")
    ki = te.reduce_axis((0, BI), "ki")
    res = te.compute((n_batch, 1, B, BO),
        lambda b, co, bi, ci: te.sum(
            data[b, 0, bi, ki].astype(env.acc_dtype) * wgt[0, 0, ci, ki].astype(env.acc_dtype),
            axis=[ki]), name="res", tag="dense_gmtf_small")
    # epilogue via topi (broadcast-tagged, like the relay-lowered right_shift/clip/cast)
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
    s[out].reorder(xbo, xco, xbn, xbi, xci); spt = xco
    s[ds].compute_at(s[out], spt)
    for op in ew: s[op].compute_at(s[out], spt)
    for t in cre: s[t].compute_at(s[out], spt); s[t].pragma(s[t].op.axis[0], env.dma_copy)
    db, dco, dbi, dci = s[ds].op.axis; (dki,) = s[ds].op.reduce_axis
    dko, dkii = s[ds].split(dki, factor=BI)
    s[ds].reorder(dko, db, dbi, dci, dkii)
    s[cd].compute_at(s[ds], dko); s[cw].compute_at(s[ds], dko)
    s[cd].pragma(s[cd].op.axis[0], env.dma_copy); s[cw].pragma(s[cw].op.axis[0], env.dma_copy)
    s[ds].tensorize(dbi, vta_intrin.gemm_mat_trf(env, mock=False, large_mode=False))
    s[out].pragma(xbn, env.dma_copy)
    return s, [data, wgt, out]


def build_large(n_batch, TILE=256):
    data = te.placeholder((n_batch, 2, B, BI), env.inp_dtype, "data")
    wgt = te.placeholder((2 * BO, 1, BO, BI), env.wgt_dtype, "wgt")
    ko = te.reduce_axis((0, BO), "ko"); ki = te.reduce_axis((0, BI), "ki")
    res = te.compute((n_batch, 2, B, BO),
        lambda b, co, bi, ci: te.sum(
            data[b, tvm.tir.min(ko, 1), bi, ki].astype(env.acc_dtype)
            * wgt[co * BO + ci, 0, ko, ki].astype(env.acc_dtype), axis=[ko, ki]),
        name="res", tag="dense_gmtf_large")
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
    s[out].reorder(xbo, xbn, xco, xbi, xci); spt = xbo
    s[ds].compute_at(s[out], spt)
    for op in ew: s[op].compute_at(s[out], spt)
    for t in cre: s[t].compute_at(s[out], spt); s[t].pragma(s[t].op.axis[0], env.dma_copy)
    db, dco, dbi, dci = s[ds].op.axis; dko, dki = s[ds].op.reduce_axis
    dkio, dkii = s[ds].split(dki, factor=BI)
    s[ds].reorder(dkio, db, dco, dbi, dci, dko, dkii)
    s[cd].compute_at(s[ds], dkio); s[cw].compute_at(s[ds], dkio)
    s[cd].pragma(s[cd].op.axis[0], env.dma_copy); s[cw].pragma(s[cw].op.axis[0], env.dma_copy)
    s[ds].tensorize(dco, vta_intrin.gemm_mat_trf(env, mock=False, large_mode=True))
    s[out].pragma(xbn, env.dma_copy)
    return s, [data, wgt, out]


def ref_small(data_np, trf_np):
    n = data_np.shape[0]
    inp = data_np[:, 0, 0, :].astype(np.int32)        # (n,16)
    T = trf_np[0, 0, :, :].astype(np.int32)           # (16,16)
    acc = inp @ T.T
    acc = np.clip(np.right_shift(acc, 8), -127, 127)
    out = np.zeros((n, 1, B, BO), dtype=np.int8)
    out[:, 0, 0, :] = acc.astype(np.int8)
    return out


def ref_large(data_np, trf_np):
    n = data_np.shape[0]
    inp = np.zeros((n, 32), dtype=np.int32)
    inp[:, :16] = data_np[:, 0, 0, :].astype(np.int32)
    inp[:, 16:] = data_np[:, 1, 0, :].astype(np.int32)
    T = np.zeros((32, 32), dtype=np.int32)
    for i in range(32):
        T[i, :16] = trf_np[i, 0, 0, :].astype(np.int32)
        T[i, 16:] = trf_np[i, 0, 1, :].astype(np.int32)
    acc = inp @ T.T                                    # (n,32)
    acc = np.clip(np.right_shift(acc, 8), -127, 127)
    out = np.zeros((n, 2, B, BO), dtype=np.int8)
    out[:, 0, 0, :] = acc[:, :16].astype(np.int8)
    out[:, 1, 0, :] = acc[:, 16:].astype(np.int8)
    return out


def run_case(remote, mode, n_batch):
    rng = np.random.default_rng(0)
    if mode == "small":
        s, (data, wgt, out) = build_small(n_batch)
        data_np = np.zeros((n_batch, 1, B, BI), np.int8)
        data_np[:, 0, 0, :] = rng.integers(-8, 8, size=(n_batch, BI), dtype=np.int8)
        trf_np = np.zeros((1, 1, BO, BI), np.int8)
        trf_np[0, 0, :, :] = rng.integers(-8, 8, size=(BO, BI), dtype=np.int8)
        ref = ref_small(data_np, trf_np)
    else:
        s, (data, wgt, out) = build_large(n_batch)
        data_np = np.zeros((n_batch, 2, B, BI), np.int8)
        data_np[:, 0, 0, :] = rng.integers(-8, 8, size=(n_batch, BI), dtype=np.int8)
        data_np[:, 1, 0, :] = rng.integers(-8, 8, size=(n_batch, BI), dtype=np.int8)
        trf_np = np.zeros((2 * BO, 1, BO, BI), np.int8)
        trf_np[:, 0, 0, :] = rng.integers(-8, 8, size=(2 * BO, BI), dtype=np.int8)
        trf_np[:, 0, 1, :] = rng.integers(-8, 8, size=(2 * BO, BI), dtype=np.int8)
        ref = ref_large(data_np, trf_np)

    with vta.build_config():
        m = vta.build(s, [data, wgt, out], "ext_dev", env.target_host, name="gmtf")
    tmp = tvm_utils.tempdir(); fp = tmp.relpath("gmtf.o"); m.save(fp)
    remote.upload(fp); f = remote.load_module("gmtf.o")
    ctx = remote.ext_dev(0)
    da = tvm.nd.array(data_np, ctx); wa = tvm.nd.array(trf_np, ctx)
    oa = tvm.nd.array(np.zeros(ref.shape, np.int8), ctx)
    f(da, wa, oa)
    res = oa.asnumpy()
    ok = np.array_equal(res, ref)
    maxdiff = int(np.abs(res.astype(np.int32) - ref.astype(np.int32)).max())
    print("  %-5s n_batch=%-8d %s  maxdiff=%d" % (mode, n_batch, "PASS" if ok else "FAIL", maxdiff))
    if not ok:
        diff = np.argwhere(res != ref)
        print("    mismatches: %d / %d  first at %s  got=%d ref=%d"
              % (len(diff), res.size, tuple(diff[0]), res[tuple(diff[0])], ref[tuple(diff[0])]))
    return ok


def _run(env_, remote):
    vta.reconfig_runtime(remote)
    print("=" * 60)
    print("GMTF tiled-schedule HW test  BATCH=%d BLOCK_IN=%d BLOCK_OUT=%d" % (B, BI, BO))
    print("=" * 60)
    allok = True
    for mode in ["small", "large"]:
        for nb in [256, 1024, 262144, 1048576]:
            try:
                allok &= run_case(remote, mode, nb)
            except Exception as e:
                print("  %-5s n_batch=%-8d CRASH: %s" % (mode, nb, str(e).splitlines()[-1][:80]))
                break
    print("=" * 60)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    vta.testing.run(_run)
