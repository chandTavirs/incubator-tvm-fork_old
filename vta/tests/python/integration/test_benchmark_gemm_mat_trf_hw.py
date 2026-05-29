# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Modified by contributors from Intel Labs

"""Hardware benchmark for the GEMM_Mat_Trf VTA instruction on ZCU104.

Computes out[b][i] = sum_j( inp[b][j] * T[i][j] ) for i,j in [0, DIM_SMALL=9)
using the small-mode (3×3 kernel) GEMM_Mat_Trf instruction.

Memory layout used by the hardware (packed wgt layout):
  inp_mem[b]  : input vector for batch b (9 elements at bytes 0..8, rest zero-padded)
  wgt_mem[0]  : single packed entry; bus word i = T row i; T[i][j] = byte j of bus word i
  acc_mem[b]  : result vector for batch b (elements 0..8 valid, 9..15 are 0)

UOP encoding:
  n_batch UOPs, each with wgt_idx=0 (T is fixed), src_idx=b, dst_idx=b.
  FoldUopLoop produces VTAUopLoopBegin(n_batch, dst_stride=1, src_stride=1, wgt_stride=0).

Large mode (DIM_LARGE=25): uses build_gemm_mat_trf_large_schedule() with
2 acc/inp entries per batch, tensorized at xco so FoldUopLoop gives dst/src stride=2.

Usage (bitstream already programmed on device):
    VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 \\
        python test_benchmark_gemm_mat_trf_hw.py
"""

from __future__ import absolute_import, print_function

from datetime import time
from time import sleep

import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm.contrib import utils
import vta
from vta import intrin as vta_intrin
from vta import reconfig_runtime
import vta.testing

# Transform dimensions (must match HW constants VTA_MAT_TRF_DIM_{SMALL,LARGE})
_DIM_SMALL = 9
_DIM_LARGE = 25


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def build_gemm_mat_trf_schedule(env, n_batch, large_mode=False):
    """Return (schedule, [data_ph, trf_ph, res_ph], (data_shape, trf_shape, out_shape)).

    Small mode (DIM_SMALL=9, fits in one BLOCK_OUT=16 tile):
      data : (n_batch, 1, BATCH, BLOCK_IN)       -> n_batch inp entries; inp[b][j]=data[b,0,0,j]
      trf  : (1, 1, BLOCK_OUT, BLOCK_IN)         -> 1 packed wgt entry; T[i][j]=trf[0,0,i,j]
      out  : (n_batch, 1, BATCH, BLOCK_OUT)       -> n_batch acc entries; out[b,0,0,i] valid i<DIM
    """
    assert not large_mode, "Use build_gemm_mat_trf_large_schedule() for large mode"
    DIM       = _DIM_SMALL
    BATCH     = env.BATCH
    BLOCK_IN  = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT

    data_shape = (n_batch, 1, BATCH,     BLOCK_IN)
    trf_shape  = (1,       1, BLOCK_OUT, BLOCK_IN)  # 1 packed wgt entry; T[i][j]=trf[0,0,i,j]
    out_shape  = (n_batch, 1, BATCH,     BLOCK_OUT)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    trf  = te.placeholder(trf_shape,  name="trf",  dtype=env.wgt_dtype)

    # Buffers in VTA scopes (DMA will copy these from DRAM)
    data_buf = te.compute(data_shape, lambda *i: data(*i), name="data_buf")
    trf_buf  = te.compute(trf_shape,  lambda *i: trf(*i),  name="trf_buf")

    # trf_buf[0, 0, ci, ki]: accesses the single packed wgt entry at wgt_idx=0.
    # Hardware reads T[ci][ki] from wgt_mem[trf_base][bus_word=ci][byte=ki].
    # TVM required region = [1,1,16,16] = exactly 1 wgt entry → DMA loads all 256 bytes
    # of the packed entry, which contains all 9 T rows.
    ki = te.reduce_axis((0, BLOCK_IN), name="ki")
    res_gemm = te.compute(
        out_shape,
        lambda bo, co, bi, ci: te.sum(
            data_buf[bo, 0, bi, ki].astype(env.acc_dtype)
            * trf_buf[0, 0, ci, ki].astype(env.acc_dtype),
            axis=[ki],
        ),
        name="res_gemm",
    )

    res_shf = te.compute(out_shape, lambda *i: res_gemm(*i) >> 8, name="res_shf")
    res_max = te.compute(out_shape, lambda *i: tvm.te.max(res_shf(*i), 0), name="res_max")
    res_min = te.compute(
        out_shape,
        lambda *i: tvm.te.min(res_max(*i), (1 << (env.INP_WIDTH - 1)) - 1),
        name="res_min",
    )
    res = te.compute(out_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")

    s = te.create_schedule(res.op)
    s[data_buf].set_scope(env.inp_scope)
    s[trf_buf].set_scope(env.wgt_scope)
    s[res_gemm].set_scope(env.acc_scope)
    s[res_shf].set_scope(env.acc_scope)
    s[res_max].set_scope(env.acc_scope)
    s[res_min].set_scope(env.acc_scope)

    # DMA: load all DIM T-matrix rows into wgt_mem[0..DIM-1] (once, outside compute loop).
    s[trf_buf].pragma(s[trf_buf].op.axis[0], env.dma_copy)

    # DMA: load all n_batch input vectors into inp_mem (once, outside compute loop).
    s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)

    # Tensorize at xbi (batch-inner axis).  Outer loops after tensorize: (xbo, xco=1).
    # xco has range 1 → wgt_idx stays at tile (0,0) = 0 for every UOP.
    # FoldUopLoop folds xbo → VTAUopLoopBegin(n_batch, dst_stride=1, src_stride=1, wgt_stride=0).
    xbo, xco, xbi, xci = s[res_gemm].op.axis
    gemm_mat_trf_intrin = vta_intrin.gemm_mat_trf(env, mock=False, large_mode=large_mode)
    s[res_gemm].tensorize(xbi, gemm_mat_trf_intrin)

    s[res_shf].pragma(s[res_shf].op.axis[0], env.alu)
    s[res_max].pragma(s[res_max].op.axis[0], env.alu)
    s[res_min].pragma(s[res_min].op.axis[0], env.alu)
    s[res].pragma(s[res].op.axis[0], env.dma_copy)

    return s, [data, trf, res], (data_shape, trf_shape, out_shape)


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def numpy_ref(data_np, trf_np, out_shape, env):
    """Ground-truth for small-mode GEMM_Mat_Trf (shift-right + clip).

    data_np : (n_batch, 1, BATCH, BLOCK_IN)   — inp[b][j] = data_np[b, 0, 0, j] for j < DIM_SMALL
    trf_np  : (DIM_SMALL, 1, BLOCK_OUT, BLOCK_IN) — T[i][j] = trf_np[i, 0, 0, j]

    out[b][i] = clip( sum_j(inp[b][j] * T[i][j]) >> 8, 0, 127 ) for i,j in [0, DIM_SMALL).
    Positions out_shape[..., DIM_SMALL:] are 0 (hardware zeroes those).
    """
    DIM = _DIM_SMALL
    inp = data_np[:, 0, 0, :DIM].astype(np.int32)     # (n_batch, DIM)
    T   = trf_np[0, 0, :DIM, :DIM].astype(np.int32)   # (DIM, DIM) packed in bus words 0..DIM-1
    acc = inp @ T.T  # (n_batch, DIM): acc[b][i] = sum_j( inp[b][j] * T[i][j] )
    acc = np.right_shift(acc, 8)
    acc = np.clip(acc, 0, (1 << (env.INP_WIDTH - 1)) - 1)
    ref = np.zeros(out_shape, dtype=env.inp_dtype)
    ref[:, 0, 0, :DIM] = acc.astype(env.inp_dtype)
    return ref


# ---------------------------------------------------------------------------
# Large mode schedule (DIM_LARGE=25, 5x5 kernel)
# ---------------------------------------------------------------------------

def build_gemm_mat_trf_large_schedule(env, n_batch):
    """Return (schedule, [data_ph, trf_ph, res_ph], (data_shape, trf_shape, out_shape)).

    Large mode (DIM_LARGE=25, spans 2 BLOCK_IN/BLOCK_OUT tiles):
      data : (n_batch, 2, BATCH, BLOCK_IN)          -> 2 inp entries per batch
      trf  : (2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)  -> 32 wgt entries (rows 0..24 valid)
               T[i][j] = trf[i, 0, j//BLOCK_IN, j%BLOCK_IN]
      out  : (n_batch, 2, BATCH, BLOCK_OUT)          -> 2 acc entries per batch

    Tensorize at xco (range 2); outer loop xbo (range n_batch) -> n_batch UOPs.
    FoldUopLoop produces dst_factor=2, src_factor=2, wgt_factor=0.
    """
    DIM      = _DIM_LARGE
    BATCH    = env.BATCH
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT

    data_shape = (n_batch, 2, BATCH, BLOCK_IN)
    trf_shape  = (2 * BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)
    out_shape  = (n_batch, 2, BATCH, BLOCK_OUT)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    trf  = te.placeholder(trf_shape,  name="trf",  dtype=env.wgt_dtype)

    data_buf = te.compute(data_shape, lambda *i: data(*i), name="data_buf")
    trf_buf  = te.compute(trf_shape,  lambda *i: trf(*i),  name="trf_buf")

    # ko spans [0, BLOCK_OUT) so trf's footprint covers all bus words (passes _fold_buffer_dim).
    # T is zero in bus_words 2..BLOCK_OUT-1; inp is accessed as inp[min(ko,1)] so tile 1
    # is broadcast to ko=1..BLOCK_OUT-1, multiplied by zero T entries — sum is still correct.
    ko = te.reduce_axis((0, BLOCK_OUT), name="ko")
    ki = te.reduce_axis((0, BLOCK_IN), name="ki")
    res_gemm = te.compute(
        out_shape,
        lambda bo, co, bi, ci: te.sum(
            data_buf[bo, tvm.tir.min(ko, 1), bi, ki].astype(env.acc_dtype)
            * trf_buf[co * BLOCK_OUT + ci, 0, ko, ki].astype(env.acc_dtype),
            axis=[ko, ki],
        ),
        name="res_gemm",
    )

    res_shf = te.compute(out_shape, lambda *i: res_gemm(*i) >> 8, name="res_shf")
    res_max = te.compute(out_shape, lambda *i: tvm.te.max(res_shf(*i), 0), name="res_max")
    res_min = te.compute(
        out_shape,
        lambda *i: tvm.te.min(res_max(*i), (1 << (env.INP_WIDTH - 1)) - 1),
        name="res_min",
    )
    res = te.compute(out_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")

    s = te.create_schedule(res.op)
    s[data_buf].set_scope(env.inp_scope)
    s[trf_buf].set_scope(env.wgt_scope)
    s[res_gemm].set_scope(env.acc_scope)
    s[res_shf].set_scope(env.acc_scope)
    s[res_max].set_scope(env.acc_scope)
    s[res_min].set_scope(env.acc_scope)

    s[trf_buf].pragma(s[trf_buf].op.axis[0], env.dma_copy)
    s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)

    # Tensorize at xco (axis 1, range 2): outer loop is xbo (range n_batch).
    # FoldUopLoop sees dst/src advance by 2 per xbo step -> dst_factor=2, src_factor=2.
    xbo, xco, xbi, xci = s[res_gemm].op.axis
    large_intrin = vta_intrin.gemm_mat_trf(env, mock=False, large_mode=True)
    s[res_gemm].tensorize(xco, large_intrin)

    s[res_shf].pragma(s[res_shf].op.axis[0], env.alu)
    s[res_max].pragma(s[res_max].op.axis[0], env.alu)
    s[res_min].pragma(s[res_min].op.axis[0], env.alu)
    s[res].pragma(s[res].op.axis[0], env.dma_copy)

    return s, [data, trf, res], (data_shape, trf_shape, out_shape)


def numpy_ref_large(data_np, trf_np, out_shape, env):
    """Ground-truth for large-mode GEMM_Mat_Trf (shift-right + clip).

    data_np : (n_batch, 2, BATCH, BLOCK_IN)            -- inp[b][j] = data_np[b, j//16, 0, j%16]
    trf_np  : (2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)    -- T[i][j]   = trf_np[i, 0, j//16, j%16]
    """
    DIM      = _DIM_LARGE
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT
    n_batch  = data_np.shape[0]

    inp = np.zeros((n_batch, DIM), dtype=np.int32)
    for j in range(DIM):
        inp[:, j] = data_np[:, j // BLOCK_IN, 0, j % BLOCK_IN].astype(np.int32)

    T = np.zeros((DIM, DIM), dtype=np.int32)
    for i in range(DIM):
        for j in range(DIM):
            T[i, j] = int(trf_np[i, 0, j // BLOCK_IN, j % BLOCK_IN])

    acc = inp @ T.T
    acc = np.right_shift(acc, 8)
    acc = np.clip(acc, 0, (1 << (env.INP_WIDTH - 1)) - 1)

    ref = np.zeros(out_shape, dtype=env.inp_dtype)
    for i in range(DIM):
        ref[:, i // BLOCK_OUT, 0, i % BLOCK_OUT] = acc[:, i].astype(env.inp_dtype)
    return ref


def run_test_large(env, remote, n_batch, check=True, n_repeat=20):
    """Build, upload, execute and optionally verify large-mode GEMM_Mat_Trf."""
    DIM      = _DIM_LARGE
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT
    tag = f"n_batch={n_batch} mode=large(DIM={DIM})"
    print(f"\n  {tag}")

    s, (data_ph, trf_ph, res_ph), (data_shape, trf_shape, out_shape) = \
        build_gemm_mat_trf_large_schedule(env, n_batch)

    with vta.build_config():
        mod = vta.build(
            s, [data_ph, trf_ph, res_ph],
            "ext_dev", env.target_host,
            name="gemm_mat_trf_large",
        )

    tmp = utils.tempdir()
    lib_path = tmp.relpath("gemm_mat_trf_large.o")
    mod.save(lib_path)
    remote.upload(lib_path)
    f = remote.load_module("gemm_mat_trf_large.o")

    ctx = remote.ext_dev(0)

    # Pack inp[b][j] for j=0..24 into data_np[b, j//16, 0, j%16]
    data_np = np.zeros(data_shape, dtype=data_ph.dtype)
    data_np[:, 0, 0, :]  = np.random.randint(-64, 64, size=(n_batch, BLOCK_IN),  dtype=data_ph.dtype)
    data_np[:, 1, 0, :9] = np.random.randint(-64, 64, size=(n_batch, 9),         dtype=data_ph.dtype)

    # Pack T[i][j] for i,j=0..24 into trf_np[i, 0, j//16, j%16]; rows 25..31 stay zero
    trf_np = np.zeros(trf_shape, dtype=trf_ph.dtype)
    trf_np[:DIM, 0, 0, :]  = np.random.randint(-64, 64, size=(DIM, BLOCK_IN), dtype=trf_ph.dtype)
    trf_np[:DIM, 0, 1, :9] = np.random.randint(-64, 64, size=(DIM, 9),        dtype=trf_ph.dtype)

    res_np = np.zeros(out_shape, dtype=res_ph.dtype)

    data_arr = tvm.nd.array(data_np, ctx)
    trf_arr  = tvm.nd.array(trf_np,  ctx)
    res_arr  = tvm.nd.array(res_np,  ctx)

    f(data_arr, trf_arr, res_arr)

    if check:
        ref = numpy_ref_large(data_np, trf_np, out_shape, env)
        tvm.testing.assert_allclose(res_arr.asnumpy(), ref)
        print("    correctness: PASS")

    time_f = f.time_evaluator("gemm_mat_trf_large", ctx, number=n_repeat)
    cost    = time_f(data_arr, trf_arr, res_arr)
    total_muls = n_batch * DIM * DIM
    gops = (2 * total_muls / cost.mean) / 1e9
    print(f"    time: {cost.mean * 1e3:.3f} ms  ({gops:.3f} GOPS)")
    return cost


# ---------------------------------------------------------------------------
# Single test run (small mode)
# ---------------------------------------------------------------------------

def run_test(env, remote, n_batch, large_mode=False, check=True, n_repeat=20):
    """Build, upload, execute and optionally verify one small-mode GEMM_Mat_Trf configuration."""
    assert not large_mode, "Use run_test_large() for large mode"
    DIM = _DIM_SMALL
    tag = f"n_batch={n_batch} mode=small(DIM={DIM},empty_0=1)"
    print(f"\n  {tag}")

    s, (data_ph, trf_ph, res_ph), (data_shape, trf_shape, out_shape) = \
        build_gemm_mat_trf_schedule(env, n_batch, large_mode)

    with vta.build_config():
        mod = vta.build(
            s, [data_ph, trf_ph, res_ph],
            "ext_dev", env.target_host,
            name="gemm_mat_trf",
        )

    tmp = utils.tempdir()
    lib_path = tmp.relpath("gemm_mat_trf.o")
    mod.save(lib_path)
    remote.upload(lib_path)
    f = remote.load_module("gemm_mat_trf.o")

    ctx = remote.ext_dev(0)

    # Input vectors: n_batch rows, each with DIM meaningful elements (rest zero-padded)
    data_np = np.zeros(data_shape, dtype=data_ph.dtype)
    data_np[:, 0, 0, :DIM] = np.random.randint(-64, 64, size=(n_batch, DIM), dtype=data_ph.dtype)

    # T matrix (packed): T[i][j] stored in trf_np[0, 0, i, j]; rows DIM..BLOCK_OUT-1 zero-padded.
    trf_np = np.zeros(trf_shape, dtype=trf_ph.dtype)
    trf_np[0, 0, :DIM, :DIM] = np.random.randint(-64, 64, size=(DIM, DIM), dtype=trf_ph.dtype)

    res_np = np.zeros(out_shape, dtype=res_ph.dtype)

    data_arr = tvm.nd.array(data_np, ctx)
    trf_arr  = tvm.nd.array(trf_np,  ctx)
    res_arr  = tvm.nd.array(res_np,  ctx)

    f(data_arr, trf_arr, res_arr)

    if check:
        ref = numpy_ref(data_np, trf_np, out_shape, env)
        tvm.testing.assert_allclose(res_arr.asnumpy(), ref)
        print("    correctness: PASS")

    time_f = f.time_evaluator("gemm_mat_trf", ctx, number=n_repeat)
    cost    = time_f(data_arr, trf_arr, res_arr)
    total_muls = n_batch * DIM * DIM          # DIM MACs per output element, DIM outputs per batch
    gops = (2 * total_muls / cost.mean) / 1e9
    print(f"    time: {cost.mean * 1e3:.3f} ms  ({gops:.3f} GOPS)")
    return cost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run(env, remote):
    reconfig_runtime(remote)

    BATCH     = env.BATCH
    BLOCK_IN  = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT

    print("=" * 64)
    print("GEMM_Mat_Trf Hardware Benchmark")
    print(f"  Config: BATCH={BATCH}  BLOCK_IN={BLOCK_IN}  BLOCK_OUT={BLOCK_OUT}")
    print("=" * 64)

    print("\n[Small mode — 9x9 transform (3x3 kernel), empty_0=1]")
    run_test(env, remote, n_batch=1,    check=True)
    run_test(env, remote, n_batch=4,    check=True)
    run_test(env, remote, n_batch=2048, check=True)

    # Large mode uses 2 inp/acc entries per batch; VTA inp_buffer holds 2048 entries total,
    # so max n_batch for large mode = 1024.
    print("\n[Large mode — 25x25 transform (5x5 kernel), empty_0=3]")
    run_test_large(env, remote, n_batch=1,    check=True)
    run_test_large(env, remote, n_batch=4,    check=True)
    run_test_large(env, remote, n_batch=1024, check=True)

    print("\n" + "=" * 64)
    print("All tests passed.")


if __name__ == "__main__":
    vta.testing.run(_run)
