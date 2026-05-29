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
"""VTA related intrinsics"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te


def gemm(env, mock=False):
    """Matrix-matrix multiply intrinsic

    Parameters
    ----------
    env : Environment
        The Environment

    mock : bool
        Whether create a mock version.
    """
    wgt_lanes = env.WGT_ELEM_BITS // env.WGT_WIDTH
    assert wgt_lanes == env.BLOCK_OUT * env.BLOCK_IN
    wgt_shape = (env.BLOCK_OUT, env.BLOCK_IN)
    assert wgt_shape[0] * wgt_shape[1] == wgt_lanes

    inp_lanes = env.INP_ELEM_BITS // env.INP_WIDTH
    assert inp_lanes == env.BATCH * env.BLOCK_IN
    inp_shape = (env.BATCH, env.BLOCK_IN)
    assert inp_shape[0] * inp_shape[1] == inp_lanes

    out_lanes = env.ACC_ELEM_BITS // env.ACC_WIDTH
    assert out_lanes == env.BATCH * env.BLOCK_OUT
    out_shape = (env.BATCH, env.BLOCK_OUT)
    assert out_shape[0] * out_shape[1] == out_lanes

    wgt = te.placeholder(
        (wgt_shape[0], wgt_shape[1]), dtype="int%d" % env.WGT_WIDTH, name=env.wgt_scope
    )
    inp = te.placeholder(
        (inp_shape[0], inp_shape[1]), dtype="int%d" % env.INP_WIDTH, name=env.inp_scope
    )
    k = te.reduce_axis((0, wgt_shape[1]), name="k")
    out_dtype = "int%d" % env.ACC_WIDTH
    out = te.compute(
        (out_shape[0], out_shape[1]),
        lambda i, j: te.sum(inp[i, k].astype(out_dtype) * wgt[j, k].astype(out_dtype), axis=[k]),
        name="out",
    )
    wgt_layout = tvm.tir.decl_buffer(
        wgt.shape,
        wgt.dtype,
        env.wgt_scope,
        scope=env.wgt_scope,
        offset_factor=wgt_lanes,
        data_alignment=wgt_lanes,
    )
    inp_layout = tvm.tir.decl_buffer(
        inp.shape,
        inp.dtype,
        env.inp_scope,
        scope=env.inp_scope,
        offset_factor=inp_lanes,
        data_alignment=inp_lanes,
    )
    out_layout = tvm.tir.decl_buffer(
        out.shape,
        out.dtype,
        env.acc_scope,
        scope=env.acc_scope,
        offset_factor=out_lanes,
        data_alignment=out_lanes,
    )

    def intrin_func(ins, outs):
        """Matrix-matrix multiply intrinsic function"""
        dinp, dwgt = ins
        dout = outs[0]

        def instr(index):
            """Generate matrix-matrix multiply VTA instruction"""
            irb = tvm.tir.ir_builder.create()
            dev = env.dev
            irb.scope_attr(dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE))
            irb.scope_attr(dev.vta_axis, "coproc_uop_scope", dev.vta_push_uop)
            if index in (0, 2):
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        0,
                        0,
                        dout.access_ptr("rw", "int32"),
                        dinp.access_ptr("r", "int32"),
                        dwgt.access_ptr("r", "int32"),
                        0,
                        0,
                        0,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        0,
                        1,
                        dout.access_ptr("rw", "int32"),
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                )
            return irb.get()

        # return a triple of normal-set, reset, update
        nop = tvm.tir.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(0), instr(1), instr(2))

    return te.decl_tensor_intrin(
        out.op, intrin_func, name="GEMM", binds={inp: inp_layout, wgt: wgt_layout, out: out_layout}
    )


def gemm_mat_trf(env, mock=False, large_mode=False):
    """Matrix transform intrinsic for dense-transform subgraphs.

    Small mode (large_mode=False): 9x9 (3x3 kernel). Tensorize at xbi.
    Large mode (large_mode=True): 25x25 (5x5 kernel). Tensorize at xco.
      Uses 2 inp/acc entries and 2*BLOCK_OUT wgt entries per batch element.

    Parameters
    ----------
    env : Environment
    mock : bool
        Return no-ops instead of real instructions (for scheduling without execution).
    large_mode : bool
        False -> 9x9 (3x3 kernel), VTAPushGEMMMatTrfOpSmall, tensorize at xbi.
        True  -> 25x25 (5x5 kernel), VTAPushGEMMMatTrfOpLarge, tensorize at xco.
    """
    wgt_lanes = env.WGT_ELEM_BITS // env.WGT_WIDTH
    assert wgt_lanes == env.BLOCK_OUT * env.BLOCK_IN

    inp_lanes = env.INP_ELEM_BITS // env.INP_WIDTH
    assert inp_lanes == env.BATCH * env.BLOCK_IN

    out_lanes = env.ACC_ELEM_BITS // env.ACC_WIDTH
    assert out_lanes == env.BATCH * env.BLOCK_OUT

    out_dtype = "int%d" % env.ACC_WIDTH

    push_uop_scope = (
        env.dev.vta_push_uop_mat_trf_large if large_mode
        else env.dev.vta_push_uop_mat_trf_small
    )

    if not large_mode:
        # 1 inp entry, 1 wgt entry (all 9 T rows packed in bus words 0..8),
        # 1 acc entry per batch element.  Tensorize at xbi; outer=(xbo, xco=1).
        trf_mat = te.placeholder(
            (env.BLOCK_OUT, env.BLOCK_IN), dtype="int%d" % env.WGT_WIDTH, name=env.wgt_scope
        )
        inp = te.placeholder(
            (env.BATCH, env.BLOCK_IN), dtype="int%d" % env.INP_WIDTH, name=env.inp_scope
        )
        k = te.reduce_axis((0, env.BLOCK_IN), name="k")
        out = te.compute(
            (env.BATCH, env.BLOCK_OUT),
            lambda i, j: te.sum(
                inp[i, k].astype(out_dtype) * trf_mat[j, k].astype(out_dtype), axis=[k]
            ),
            name="out",
        )
    else:
        # 2 inp entries, 2*BLOCK_OUT wgt entries (one T row each, rows 25..31 zero-padded),
        # 2 acc entries per batch element.  Tensorize at xco; outer=(xbo).
        # trf_mat[row, 0, bus_word, byte] = T[row][bus_word*BLOCK_IN + byte].
        #
        # ko spans [0, BLOCK_OUT) so the trf footprint covers all bus words (BLOCK_OUT*BLOCK_IN=256
        # elements), satisfying VTA's _fold_buffer_dim block-size check.  T is zero in bus words
        # 2..BLOCK_OUT-1, so terms for ko >= 2 contribute zero.  inp is accessed as
        # inp[min(ko, 1)] so inp_tile[1] is broadcast to ko=1..BLOCK_OUT-1; multiplied by
        # T's zero bus words it still contributes nothing to the sum.
        trf_mat = te.placeholder(
            (2 * env.BLOCK_OUT, 1, env.BLOCK_OUT, env.BLOCK_IN),
            dtype="int%d" % env.WGT_WIDTH, name=env.wgt_scope,
        )
        inp = te.placeholder(
            (2, env.BATCH, env.BLOCK_IN), dtype="int%d" % env.INP_WIDTH, name=env.inp_scope
        )
        ko = te.reduce_axis((0, env.BLOCK_OUT), name="ko")
        ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
        out = te.compute(
            (2, env.BATCH, env.BLOCK_OUT),
            lambda co, bi, ci: te.sum(
                inp[tvm.tir.min(ko, 1), bi, ki].astype(out_dtype)
                * trf_mat[co * env.BLOCK_OUT + ci, 0, ko, ki].astype(out_dtype),
                axis=[ko, ki],
            ),
            name="out",
        )

    trf_mat_layout = tvm.tir.decl_buffer(
        trf_mat.shape, trf_mat.dtype, env.wgt_scope,
        scope=env.wgt_scope, offset_factor=wgt_lanes, data_alignment=wgt_lanes,
    )
    inp_layout = tvm.tir.decl_buffer(
        inp.shape, inp.dtype, env.inp_scope,
        scope=env.inp_scope, offset_factor=inp_lanes, data_alignment=inp_lanes,
    )
    out_layout = tvm.tir.decl_buffer(
        out.shape, out.dtype, env.acc_scope,
        scope=env.acc_scope, offset_factor=out_lanes, data_alignment=out_lanes,
    )

    def intrin_func(ins, outs):
        dinp, dtrf_mat = ins
        dout = outs[0]

        def instr(reset_out):
            irb = tvm.tir.ir_builder.create()
            dev = env.dev
            irb.scope_attr(dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE))
            irb.scope_attr(dev.vta_axis, "coproc_uop_scope", push_uop_scope)
            if reset_out:
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32", "tir.vta.uop_push",
                        0, 1,
                        dout.access_ptr("rw", "int32"),
                        0, 0, 0, 0, 0,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32", "tir.vta.uop_push",
                        0, 0,
                        dout.access_ptr("rw", "int32"),
                        dinp.access_ptr("r", "int32"),
                        dtrf_mat.access_ptr("r", "int32"),
                        0, 0, 0,
                    )
                )
            return irb.get()

        nop = tvm.tir.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(reset_out=False), instr(reset_out=True), instr(reset_out=False))

    return te.decl_tensor_intrin(
        out.op, intrin_func, name="GEMM_MAT_TRF",
        binds={inp: inp_layout, trf_mat: trf_mat_layout, out: out_layout},
    )
