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
"""GEMM_Mat_Trf dense compute and schedule for VTA (small and large mode).

Small mode (9×9 / 3×3 kernel):
  weight: (1, 1, BLOCK_OUT, BLOCK_IN) — all T rows in bus words 0..8
  Uses gemm_mat_trf intrinsic with large_mode=False, tensorizes at xbi.

Large mode (25×25 / 5×5 kernel):
  weight: (2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN) — one wgt entry per T row
  Uses gemm_mat_trf intrinsic with large_mode=True, tensorizes at xco.
  min(ko,1) trick keeps trf footprint at 256 elements for _fold_buffer_dim.
"""

import numpy as np
import tvm
from tvm import te, topi
from tvm import autotvm

from ..environment import get_env
from .. import intrin


# ---------------------------------------------------------------------------
# Small mode  (9×9 / 3×3 kernel)
# ---------------------------------------------------------------------------

@autotvm.register_topi_compute("dense_pack_gmtf_small.vta")
def dense_pack_gmtf_small(cfg, data, weight, bias=None, out_dtype=None):
    """GEMM_Mat_Trf small-mode compute.

    data   shape: (n_batch, 1, BATCH, BLOCK_IN)
    weight shape: (1, 1, BLOCK_OUT, BLOCK_IN)
    out    shape: (n_batch, 1, BATCH, BLOCK_OUT)

    Identical to dense_packed in structure; different intrinsic at tensorize.
    """
    if len(data.shape) != 4 or len(weight.shape) != 4:
        raise topi.InvalidShapeError()

    env = get_env()
    ishape = topi.utils.get_const_tuple(data.shape)
    wshape = topi.utils.get_const_tuple(weight.shape)

    n_batch   = ishape[0]
    BATCH     = ishape[2]
    BLOCK_IN  = ishape[3]
    BLOCK_OUT = wshape[2]

    oshape = (n_batch, 1, BATCH, BLOCK_OUT)

    ki = te.reduce_axis((0, BLOCK_IN), name="ki")

    res = te.compute(
        oshape,
        lambda b_o, co, b_i, ci: te.sum(
            data[b_o, 0, b_i, ki].astype(out_dtype)
            * weight[0, 0, ci, ki].astype(out_dtype),
            axis=[ki],
        ),
        name="res",
        tag="dense_gmtf_small",
    )

    cfg.add_flop(2 * np.prod(topi.utils.get_const_tuple(oshape)) * BLOCK_IN)
    return res


@autotvm.register_topi_schedule("dense_pack_gmtf_small.vta")
def schedule_dense_pack_gmtf_small(cfg, outs):
    """GEMM_Mat_Trf small-mode schedule.

    Mirrors schedule_dense_packed but tensorizes at xbi using
    gemm_mat_trf(large_mode=False) instead of env.gemm.
    """
    assert len(outs) == 1
    output = outs[0]

    const_ops    = []
    ewise_inputs = []
    ewise_ops    = []
    dense_res    = []

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "dense_gmtf_small", "unexpected tag: %s" % op.tag
            dense_res.append(op)

    _traverse(output.op)
    assert len(dense_res) == 1
    dense_stage = dense_res[0].output(0)

    s   = te.create_schedule(output.op)
    env = get_env()

    data, weight = dense_stage.op.input_tensors

    cdata   = s.cache_read(data,   env.inp_scope, [dense_stage])
    cweight = s.cache_read(weight, env.wgt_scope, [dense_stage])
    s[dense_stage].set_scope(env.acc_scope)

    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(s.cache_read(tensor, env.acc_scope, [consumer]))

    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)
    for op in const_ops:
        s[op].compute_inline()

    # Tile n_batch into TILE-sized chunks.  The cross-tile STORE→LOAD dependency
    # (illegal in VTA, runtime.cc:695) is avoided by exposing an OUTER reduce axis as
    # the cache-read load point — exactly how conv2d places loads at k_o.  The reduction
    # makes the acc buffer persistent state, so the buffer-reuse dependency routes
    # STORE→COMPUTE→LOAD through the compute stage instead of a direct STORE→LOAD.
    # TILE=512: on-chip tile = 512 acc entries < 2048.
    TILE = 512
    x_b, x_co, x_bi, x_ci = s[output].op.axis  # (n_batch, 1, BATCH=1, BLOCK_OUT=16)
    x_bo, x_bn = s[output].split(x_b, factor=TILE)
    s[output].reorder(x_bo, x_co, x_bn, x_bi, x_ci)
    store_pt = x_co

    s[dense_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)
    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    # Split the reduce axis ki → (d_ko outer, d_kii inner).  d_ko (extent 1) is the
    # conv2d-style load point and is placed OUTSIDE the batch loop d_b, so:
    #   - the cache-read load is BULK (one DMA for the whole tile, not per-row), and
    #   - FoldUopLoop folds d_b into a single UOP GEMM (one instruction, not per-row).
    # This keeps instruction count O(tiles), not O(rows) — critical: per-row instructions
    # overflow VTA_MAX_XFER (2^21 insns) at ~262144 rows.  Loads at d_ko (inside the
    # compute scope) still give the STORE→COMPUTE ring (no illegal STORE→LOAD).
    d_b, d_co, d_bi, d_ci = s[dense_stage].op.axis
    (d_ki,) = s[dense_stage].op.reduce_axis
    d_ko, d_kii = s[dense_stage].split(d_ki, factor=env.BLOCK_IN)
    s[dense_stage].reorder(d_ko, d_b, d_bi, d_ci, d_kii)
    s[cdata].compute_at(s[dense_stage],   d_ko)
    s[cweight].compute_at(s[dense_stage], d_ko)
    s[cdata].pragma(s[cdata].op.axis[0],     env.dma_copy)
    s[cweight].pragma(s[cweight].op.axis[0], env.dma_copy)

    s[dense_stage].tensorize(
        d_bi,
        intrin.gemm_mat_trf(env, mock=False, large_mode=False),
    )
    # Bulk store: pragma at x_bn (tile batch axis) stores the whole tile in ONE DMA.
    s[output].pragma(x_bn, env.dma_copy)

    return s


# ---------------------------------------------------------------------------
# Large mode  (25×25 / 5×5 kernel)
# ---------------------------------------------------------------------------

@autotvm.register_topi_compute("dense_pack_gmtf_large.vta")
def dense_pack_gmtf_large(cfg, data, weight, bias=None, out_dtype=None):
    """GEMM_Mat_Trf large-mode compute.

    data   shape: (n_batch, 2, BATCH, BLOCK_IN)
    weight shape: (2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)
    out    shape: (n_batch, 2, BATCH, BLOCK_OUT)
    """
    if len(data.shape) != 4 or len(weight.shape) != 4:
        raise topi.InvalidShapeError()

    env = get_env()
    ishape = topi.utils.get_const_tuple(data.shape)
    wshape = topi.utils.get_const_tuple(weight.shape)

    n_batch  = ishape[0]
    BATCH    = ishape[2]
    BLOCK_IN = ishape[3]
    BLOCK_OUT = wshape[2]

    oshape = (n_batch, 2, BATCH, BLOCK_OUT)

    ko = te.reduce_axis((0, BLOCK_OUT), name="ko")
    ki = te.reduce_axis((0, BLOCK_IN),  name="ki")

    res = te.compute(
        oshape,
        lambda b_o, co, b_i, ci: te.sum(
            data[b_o, tvm.tir.min(ko, 1), b_i, ki].astype(out_dtype)
            * weight[co * BLOCK_OUT + ci, 0, ko, ki].astype(out_dtype),
            axis=[ko, ki],
        ),
        name="res",
        tag="dense_gmtf_large",
    )

    cfg.add_flop(2 * np.prod(topi.utils.get_const_tuple(oshape)) * BLOCK_OUT * BLOCK_IN)
    return res


@autotvm.register_topi_schedule("dense_pack_gmtf_large.vta")
def schedule_dense_pack_gmtf_large(cfg, outs):
    """GEMM_Mat_Trf large-mode schedule.

    Mirrors build_gemm_mat_trf_large_schedule from the benchmark but structured
    for relay.build: uses cache_read to set VTA scope, then tensorizes at xco (range 2).
    Epilogue ops (right_shift, clip, cast, copy, stop_fusion) from _PoolLadderStripper
    are scheduled as VTA ALU instructions.
    """
    assert len(outs) == 1
    output = outs[0]

    const_ops   = []
    ewise_inputs = []
    ewise_ops   = []
    dense_res   = []

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "dense_gmtf_large", "unexpected tag: %s" % op.tag
            dense_res.append(op)

    _traverse(output.op)
    assert len(dense_res) == 1
    dense_stage = dense_res[0].output(0)

    s   = te.create_schedule(output.op)
    env = get_env()

    data, weight = dense_stage.op.input_tensors

    # DMA: DRAM → inp_scope / wgt_scope
    cdata   = s.cache_read(data,   env.inp_scope, [dense_stage])
    cweight = s.cache_read(weight, env.wgt_scope, [dense_stage])
    s[dense_stage].set_scope(env.acc_scope)

    # Cache-read any ewise (epilogue) inputs into acc_scope
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(s.cache_read(tensor, env.acc_scope, [consumer]))

    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)
    for op in const_ops:
        s[op].compute_inline()

    # Tile n_batch.  Cross-tile STORE→LOAD (illegal, runtime.cc:695) is avoided by
    # exposing an OUTER reduce axis as the cache-read load point (conv2d's k_o pattern),
    # which routes the buffer-reuse dependency STORE→COMPUTE→LOAD through compute.
    # TILE=256: on-chip tile = 512 acc entries (2/batch elem) < 2048.
    TILE = 256
    x_b, x_co, x_bi, x_ci = s[output].op.axis  # (n_batch, 2, BATCH=1, BLOCK_OUT=16)
    x_bo, x_bn = s[output].split(x_b, factor=TILE)
    s[output].reorder(x_bo, x_bn, x_co, x_bi, x_ci)
    store_pt = x_bo

    s[dense_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)
    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    # Split inner reduce ki → (d_kio outer load point, d_kii inner).  d_kio (extent 1)
    # is the conv2d-style load point, placed OUTSIDE the batch loop d_b so the load is
    # BULK and d_b folds into one UOP GEMM (instruction count O(tiles), not O(rows) —
    # per-row instructions overflow VTA_MAX_XFER at ~262144 rows).  Load at d_kio (in
    # the compute scope) still gives the STORE→COMPUTE ring.
    d_b, d_co, d_bi, d_ci = s[dense_stage].op.axis
    d_ko, d_ki = s[dense_stage].op.reduce_axis
    d_kio, d_kii = s[dense_stage].split(d_ki, factor=env.BLOCK_IN)
    s[dense_stage].reorder(d_kio, d_b, d_co, d_bi, d_ci, d_ko, d_kii)
    s[cdata].compute_at(s[dense_stage],   d_kio)
    s[cweight].compute_at(s[dense_stage], d_kio)
    s[cdata].pragma(s[cdata].op.axis[0],     env.dma_copy)
    s[cweight].pragma(s[cweight].op.axis[0], env.dma_copy)

    # Tensorize at d_co (range 2); FoldUopLoop folds d_b (TILE) with stride-2
    # → VTAUopLoopBegin(TILE, dst_factor=2, src_factor=2, wgt_factor=0).
    s[dense_stage].tensorize(
        d_co,
        intrin.gemm_mat_trf(env, mock=False, large_mode=True),
    )

    # DMA STORE of the tile (x_bn spans TILE elements).
    s[output].pragma(x_bn, env.dma_copy)

    return s
