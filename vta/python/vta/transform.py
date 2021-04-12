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

"""Additional Transformation Passes. for VTA"""
# pylint: disable=len-as-condition, no-else-return, unused-argument, invalid-name
import math
import numpy as np
import tvm
from tvm import te
from tvm.topi import utils

from .environment import get_env


def _match_pragma(stmt, key):
    """Internal helper to match stmt to pragma stmt.

    Parameters
    ----------
    stmt : Stmt
        The AttrStmt

    key : str
        The pragma key
    """
    return ((stmt.attr_key == "pragma_" + key) or
            (stmt.attr_key == "pragma_scope" and stmt.value.value == key))


def FoldUopLoop():
    """Detect and fold uop loop.

    VTA support uop programming model
    that recognizes loop structure.
    This pass detect the loop structure
    and extract that into uop loop AST.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _fold_outermost_loop(body):
        stmt = body
        if not isinstance(stmt, tvm.tir.For):
            return None, body, None

        loop_var = stmt.loop_var
        gemm_offsets = [None, None, None]
        fail = [False]
        builtin_uop_push = tvm.ir.Op.get("tir.vta.uop_push")

        def _post_order(op):
            assert isinstance(op, tvm.tir.Call)
            base_args = 2
            if op.op.same_as(builtin_uop_push):
                args = []
                args += op.args[:base_args]
                for i in range(3):
                    m = tvm.arith.detect_linear_equation(op.args[i + base_args], [loop_var])
                    if not m:
                        fail[0] = True
                        return op
                    if gemm_offsets[i] is not None:
                        if not tvm.ir.structural_equal(m[0], gemm_offsets[i]):
                            fail[0] = True
                            return op
                        args.append(m[1])
                    else:
                        gemm_offsets[i] = m[0]
                        args.append(m[1])
                args += op.args[base_args + 3 :]
                return tvm.tir.call_intrin("int32", builtin_uop_push, *args)
            if op.op.name not in ("tir.vta.command_handle", "tir.tvm_thread_context"):
                raise RuntimeError("unexpected op %s" % op)
            return op

        ret = tvm.tir.stmt_functor.ir_transform(stmt.body, None, _post_order, ["tir.Call"])

        if not fail[0] and all(x is not None for x in gemm_offsets):

            def _visit(op):
                if op.same_as(loop_var):
                    fail[0] = True

            tvm.tir.stmt_functor.post_order_visit(ret, _visit)
            if not fail[0]:
                begin = tvm.tir.call_extern("int32", "VTAUopLoopBegin", stmt.extent, *gemm_offsets)
                end = tvm.tir.call_extern("int32", "VTAUopLoopEnd")
                return [begin, ret, end]
        raise ValueError("Failed to fold the GEMM instructions..")

    def _do_fold(stmt):
        env = get_env()
        if (
            stmt.attr_key == "coproc_uop_scope"
            and isinstance(stmt.value, tvm.tir.StringImm)
            and stmt.value.value == env.dev.vta_push_uop.value
        ):
            body = stmt.body
            begins = []
            ends = []
            try:
                begin, body, end = _fold_outermost_loop(body)
                if begin is not None:
                    begins.append(begin)
                if end is not None:
                    ends.append(end)
                begin, body, end = _fold_outermost_loop(body)
                if begin is not None:
                    begins.append(begin)
                if end is not None:
                    ends.append(end)
            except ValueError:
                pass
            if body == stmt.body:
                return stmt
            ends = list(reversed(ends))
            body = tvm.tir.stmt_seq(*(begins + [body] + ends))
            return tvm.tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, body)
        return None

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, _do_fold, None, ["tir.AttrStmt"])
        )

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0, name="tir.vta.FoldUopLoop")


def CPUAccessRewrite():
    """Detect CPU access to VTA buffer and get address correctly.

    VTA's buffer is an opaque handle that do not
    correspond to address in CPU.
    This pass detect CPU access and rewrite to use pointer
    returned VTABufferCPUPtr for CPU access.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _ftransform(f, mod, ctx):
        rw_info = {}
        env = get_env()

        def _post_order(op):
            if isinstance(op, tvm.tir.Allocate):
                buffer_var = op.buffer_var
                if not buffer_var in rw_info:
                    return None
                new_var = rw_info[buffer_var]
                let_stmt = tvm.tir.LetStmt(
                    new_var,
                    tvm.tir.call_extern(
                        "handle", "VTABufferCPUPtr", env.dev.command_handle, buffer_var
                    ),
                    op.body,
                )
                alloc = tvm.tir.Allocate(buffer_var, op.dtype, op.extents, op.condition, let_stmt)
                del rw_info[buffer_var]
                return alloc
            if isinstance(op, tvm.tir.Load):
                buffer_var = op.buffer_var
                if not buffer_var in rw_info:
                    rw_info[buffer_var] = te.var(buffer_var.name + "_ptr", "handle")
                new_var = rw_info[buffer_var]
                return tvm.tir.Load(op.dtype, new_var, op.index)
            if isinstance(op, tvm.tir.Store):
                buffer_var = op.buffer_var
                if not buffer_var in rw_info:
                    rw_info[buffer_var] = te.var(buffer_var.name + "_ptr", "handle")
                new_var = rw_info[buffer_var]
                return tvm.tir.Store(new_var, op.value, op.index)
            raise RuntimeError("not reached")

        stmt_in = f.body
        stmt = tvm.tir.stmt_functor.ir_transform(
            stmt_in, None, _post_order, ["tir.Allocate", "tir.Load", "tir.Store"]
        )

        for buffer_var, new_var in rw_info.items():
            stmt = tvm.tir.LetStmt(
                new_var,
                tvm.tir.call_extern(
                    "handle", "VTABufferCPUPtr", env.dev.command_handle, buffer_var
                ),
                stmt,
            )
        return f.with_body(stmt)

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.vta.CPUAccessRewrite"
    )


def LiftAllocToScopeBegin():
    """Lift allocate to beginning of the current scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _ftransform(f, mod, ctx):
        lift_stmt = [[]]

        def _merge_block(slist, body):
            for op in slist:
                if op.body == body:
                    body = op
                elif isinstance(op, tvm.tir.Allocate):
                    body = tvm.tir.Allocate(op.buffer_var, op.dtype, op.extents, op.condition, body)
                elif isinstance(op, tvm.tir.AttrStmt):
                    body = tvm.tir.AttrStmt(op.node, op.attr_key, op.value, body)
                elif isinstance(op, tvm.tir.For):
                    body = tvm.tir.For(
                        op.loop_var,
                        op.min,
                        op.extent,
                        op.kind,
                        body,
                        op.thread_binding,
                        op.annotations,
                    )
                else:
                    raise RuntimeError("unexpected op")
            del slist[:]
            return body

        def _pre_order(op):
            if isinstance(op, tvm.tir.For):
                lift_stmt.append([])
            elif isinstance(op, tvm.tir.AttrStmt):
                if op.attr_key == "virtual_thread":
                    lift_stmt.append([])

        def _post_order(op):
            if isinstance(op, tvm.tir.Allocate):
                lift_stmt[-1].append(op)
                return op.body
            if isinstance(op, tvm.tir.AttrStmt):
                if op.attr_key == "storage_scope":
                    lift_stmt[-1].append(op)
                    return op.body
                if op.attr_key == "virtual_thread":
                    return _merge_block(lift_stmt.pop() + [op], op.body)
                return op
            if isinstance(op, tvm.tir.For):
                return _merge_block(lift_stmt.pop() + [op], op.body)
            raise RuntimeError("not reached")

        stmt_in = f.body
        stmt = tvm.tir.stmt_functor.ir_transform(
            stmt_in, _pre_order, _post_order, ["tir.Allocate", "tir.AttrStmt", "tir.For"]
        )
        assert len(lift_stmt) == 1
        return f.with_body(_merge_block(lift_stmt[0], stmt))

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.vta.LiftAllocToScopeBegin"
    )


def InjectSkipCopy():
    """Pass to inject skip copy stmt, used for debug purpose.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _do_fold(stmt):
        if _match_pragma(stmt, "skip_dma_copy"):
            return tvm.tir.Evaluate(0)
        return None

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, _do_fold, None, ["tir.AttrStmt"])
        )

    return tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0, name="tir.vta.InjectSkipCopy")


def InjectCoProcSync():
    """Pass inject coproc sync

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _ftransform(f, *_):
        success = [False]

        def _do_fold(stmt):
            if _match_pragma(stmt, "coproc_sync"):
                success[0] = True
                sync = tvm.tir.Call("int32", "vta.coproc_sync", [])
                return tvm.tir.SeqStmt([stmt.body, tvm.tir.Evaluate(sync)])
            if _match_pragma(stmt, "trim_loop"):
                op = stmt.body
                assert isinstance(op, tvm.tir.For)
                return tvm.tir.For(
                    op.loop_var, op.min, 2, op.kind, op.body, op.thread_binding, op.annotations
                )
            return None

        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, None, _do_fold, ["tir.AttrStmt"])
        )

    return tvm.transform.Sequential(
        [
            tvm.tir.transform.prim_func_pass(_ftransform, 0, "tir.vta.InjectCoProcSync"),
            tvm.tir.transform.CoProcSync(),
        ],
        opt_level=0,
        name="tir.vta.InjectCoProcSync",
    )


def InjectDMAIntrin():
    """Pass to inject DMA copy intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _check_compact(buf):
        ndim = len(buf.shape)
        size = tvm.tir.const(1, buf.shape[0].dtype)
        for i in reversed(range(ndim)):
            if not utils.equal_const_int(size - buf.strides[i], 0):
                raise RuntimeError(
                    "Cannot prove compact: shape=%s, strides=%s" % (buf.shape, buf.strides)
                )
            size = size * buf.shape[i]

    def _fold_buffer_dim(buf, scope, elem_block):
        ndim = len(buf.shape)
        x_size = 1
        base = 0
        for i in range(1, ndim + 1):
            if not utils.equal_const_int(buf.strides[ndim - i] - x_size, 0):
                raise RuntimeError("scope %s needs to have block=%d" % (scope, elem_block))
            x_size = x_size * buf.shape[ndim - i]
            if utils.equal_const_int(x_size - elem_block, 0):
                base = i + 1
                break
        if base == 0:
            raise RuntimeError(
                "scope %s need to have block=%d, shape=%s" % (scope, elem_block, buf.shape)
            )
        shape = [elem_block]
        strides = [1]

        if base < ndim + 1 and not utils.equal_const_int(buf.strides[ndim - base], elem_block):
            shape.append(1)
            strides.append(elem_block)

        analyzer = tvm.arith.Analyzer()
        while base < ndim + 1:
            x_size = 1
            x_stride = buf.strides[ndim - base]
            next_base = base
            if not utils.equal_const_int(idxm(x_stride, elem_block), 0):
                raise RuntimeError(
                    "scope %s need to have block=%d, shape=%s, strides=%s"
                    % (scope, elem_block, buf.shape, buf.strides)
                )
            for i in range(base, ndim + 1):
                k = ndim - i
                if not utils.equal_const_int(x_size * x_stride - buf.strides[k], 0):
                    break
                x_size = x_size * buf.shape[k]
                next_base = i + 1
            shape.append(analyzer.simplify(x_size))
            strides.append(x_stride)
            assert next_base != base
            base = next_base

        strides = list(reversed(strides))
        shape = list(reversed(shape))
        return shape, strides

    def _get_2d_pattern(buf, elem_width, elem_bytes, dtype, scope, allow_fold):
        elem_block = elem_bytes * 8 // elem_width
        if buf.dtype != dtype:
            raise RuntimeError("Expect buffer type to be %s instead of %s" % (dtype, buf.dtype))
        shape, strides = buf.shape, buf.strides
        if not utils.equal_const_int(idxm(buf.elem_offset, elem_block), 0):
            raise RuntimeError("scope %s need to have block=%d" % (scope, elem_block))
        if allow_fold:
            shape, strides = _fold_buffer_dim(buf, scope, elem_block)
        else:
            shape = list(x for x in shape)
            strides = list(x for x in strides)

        def raise_error():
            """Internal function to raise error """
            raise RuntimeError(
                (
                    "Scope[%s]: cannot detect 2d pattern with elem_block=%d:"
                    + " shape=%s, strides=%s"
                )
                % (scope, elem_block, buf.shape, buf.strides)
            )

        ndim = len(shape)

        # Check if the inner-tensor is already flat
        flat = utils.equal_const_int(shape[-1], elem_block)

        if flat:
            if not utils.equal_const_int(strides[-1], 1):
                raise_error()

            if ndim == 1:
                x_size = 1
                x_stride = 1
                y_size = 1
                return x_size, y_size, x_stride, idxd(buf.elem_offset, elem_block)
            if not utils.equal_const_int(strides[-2] - elem_block, 0):
                raise_error()

            if ndim == 2:
                x_size = shape[-2]
                x_stride = shape[-2]
                y_size = 1
                return x_size, y_size, x_stride, idxd(buf.elem_offset, elem_block)
            if not utils.equal_const_int(idxm(strides[-3], elem_block), 0):
                raise_error()

            if ndim == 3:
                x_size = shape[-2]
                x_stride = idxd(strides[-3], elem_block)
                y_size = shape[-3]
                return x_size, y_size, x_stride, idxd(buf.elem_offset, elem_block)

        else:
            if not utils.equal_const_int(strides[-1], 1):
                raise_error()
            if not utils.equal_const_int(strides[-2] - shape[-1], 0):
                raise_error()
            if not utils.equal_const_int(shape[-1] * shape[-2], elem_block):
                raise_error()

            if ndim == 2:
                x_size = 1
                x_stride = 1
                y_size = 1
                return x_size, y_size, x_stride, idxd(buf.elem_offset, elem_block)
            if not utils.equal_const_int(strides[-3], elem_block):
                raise_error()

            if ndim == 3:
                x_size = shape[-3]
                x_stride = shape[-3]
                y_size = 1
                return x_size, y_size, x_stride, idxd(buf.elem_offset, elem_block)
            if not utils.equal_const_int(idxm(strides[-4], elem_block), 0):
                raise_error()

            if ndim == 4:
                x_size = shape[-3]
                x_stride = idxd(strides[-4], elem_block)
                y_size = shape[-4]
                return x_size, y_size, x_stride, idxd(buf.elem_offset, elem_block)

        raise_error()


    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        env = get_env()
        is_pad_min_value = 0
        if isinstance(pad_value, tvm.tir.expr.IntImm) and pad_value.value != 0:
            if dst.scope == env.acc_scope \
               and pad_value.value == np.int32(1 << (env.ACC_WIDTH - 1)):
                is_pad_min_value = 1
                assert env.ACC_WIDTH == 32, "Only 32-bit acc supported for neg pad value"
            else:
                print("WARNING: Unsupported pad value, assuming zero")
        else:
            _ = pad_value

        if dst.scope == "global":
            # Store
            if pad_before or pad_after:
                raise RuntimeError("Do not support copy into DRAM with pad")
            if src.scope == env.acc_scope:
                elem_width = env.OUT_WIDTH
                elem_bytes = env.OUT_ELEM_BYTES
                mem_type = env.dev.MEM_ID_OUT
                data_type = "int%d" % env.OUT_WIDTH
                task_qid = env.dev.QID_STORE_OUT
            else:
                raise RuntimeError("Do not support copy %s->dram" % (src.scope))
            _check_compact(src)
            x_size, y_size, x_stride, offset = _get_2d_pattern(
                dst, elem_width, elem_bytes, data_type, src.scope, allow_fold=True
            )
            irb = tvm.tir.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope", env.dev.get_task_qid(task_qid))
            irb.emit(
                tvm.tir.call_extern(
                    "int32",
                    "VTAStoreBuffer2D",
                    env.dev.command_handle,
                    src.access_ptr("r", "int32"),
                    mem_type,
                    dst.data,
                    offset,
                    x_size,
                    y_size,
                    x_stride,
                )
            )
            return irb.get()
        elif src.scope == "global":
            if dst.scope == env.acc_scope:
                elem_width = env.ACC_WIDTH
                elem_bytes = env.ACC_ELEM_BYTES
                mem_type = env.dev.MEM_ID_ACC
                data_type = "int%d" % env.ACC_WIDTH
                task_qid = env.dev.QID_LOAD_OUT
            elif dst.scope == env.inp_scope:
                elem_width = env.INP_WIDTH
                elem_bytes = env.INP_ELEM_BYTES
                mem_type = env.dev.MEM_ID_INP
                data_type = "int%d" % env.INP_WIDTH
                task_qid = env.dev.QID_LOAD_INP
            elif dst.scope == env.wgt_scope:
                elem_width = env.WGT_WIDTH
                elem_bytes = env.WGT_ELEM_BYTES
                mem_type = env.dev.MEM_ID_WGT
                data_type = "int%d" % env.WGT_WIDTH
                task_qid = env.dev.QID_LOAD_WGT
            else:
                raise RuntimeError("Do not support copy dram->%s" % (dst.scope))
            # collect pad statistics
            if pad_before:
                assert pad_after
                ndim = len(pad_before)
                if ndim <= 2 or ndim > 5:
                    raise ValueError("Limitation of 2D pad load forbid ndim=%d" % ndim)
                if ndim == 5:
                    # This case occurs when batch size N > 1
                    y_pad_before = pad_before[1]
                    x_pad_before = pad_before[2]
                    y_pad_after = pad_after[1]
                    x_pad_after = pad_after[2]
                    for dim in range(3, ndim):
                        if not utils.equal_const_int(pad_before[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                        if not utils.equal_const_int(pad_after[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                else:
                    y_pad_before = pad_before[0]
                    x_pad_before = pad_before[1]
                    y_pad_after = pad_after[0]
                    x_pad_after = pad_after[1]
                    for dim in range(2, ndim):
                        if not utils.equal_const_int(pad_before[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                        if not utils.equal_const_int(pad_after[dim], 0):
                            raise ValueError("Do not support pad on the innermost block")
                allow_fold = False
            else:
                x_pad_before = 0
                y_pad_before = 0
                x_pad_after = 0
                y_pad_after = 0
                allow_fold = True

            _check_compact(dst)
            x_size, y_size, x_stride, offset = _get_2d_pattern(
                src, elem_width, elem_bytes, data_type, dst.scope, allow_fold=allow_fold
            )

            irb = tvm.tir.ir_builder.create()
            irb.scope_attr(env.dev.vta_axis, "coproc_scope", env.dev.get_task_qid(task_qid))

            irb.emit(
                tvm.tir.call_extern(
                    "int32",
                    "VTALoadBuffer2D",
                    env.dev.command_handle,
                    src.data,
                    offset,
                    x_size,
                    y_size,
                    x_stride,
                    x_pad_before,
                    y_pad_before,
                    x_pad_after,
                    y_pad_after,
                    is_pad_min_value,
                    dst.access_ptr("r", "int32"),
                    mem_type,
                )
            )
            return irb.get()

        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope, dst.scope))

    return tvm.tir.transform.InjectCopyIntrin("dma_copy", _inject_copy)


def _get_gemm_intrin_buffer():
    env = get_env()
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

    return wgt_layout, inp_layout, out_layout


def InjectConv2DTransposeSkip():
    """Pass to skip 0-weights in conv2d transpose with stride > 1.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _ftransform(func, mod, ctx):
        env = get_env()
        dwgt, dinp, dout = _get_gemm_intrin_buffer()

        calls = []
        selects = []

        def _find_basics(op):
            if isinstance(op, tvm.tir.BufferLoad):
                calls.append(op)
            elif isinstance(op, tvm.tir.Select):
                selects.append(op)

        def _do_fold(op):
            if _match_pragma(op, "conv2d_transpose_gemm"):
                is_init = ".init" in str(op)
                tvm.tir.stmt_functor.post_order_visit(op, _find_basics)

                if is_init:
                    # create inner most block
                    irb = tvm.tir.ir_builder.create()
                    dev = env.dev
                    irb.scope_attr(dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE))
                    irb.scope_attr(dev.vta_axis, "coproc_uop_scope", dev.vta_push_uop)
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
                    inner = irb.get()
                    # TODO(@tmoreau89): This is only a temporary fix, please take a look.
                    body = op.body.body
                    while isinstance(body, tvm.tir.IfThenElse):
                        body = body.then_case
                    args = body.indices
                    res_buffer = body.buffer
                    tpl = (args[0], 1, args[1], 1, args[2], 1, args[3], 1, 0, 1, 0, env.BLOCK_OUT)
                    inner = tvm.tir.AttrStmt(
                        [dout, res_buffer],
                        "buffer_bind_scope",
                        tvm.tir.call_intrin("handle", "tir.tvm_tuple", *tpl),
                        inner,
                    )
                    return inner
                else:
                    conv_call, data_call, kernel_call = calls[-3:]
                    pad_data_tensor = data_call.buffer
                    kernel_tensor = kernel_call.buffer
                    res_tensor = conv_call.buffer

                    if selects:
                        condition = selects[0].condition
                    else:
                        condition = tvm.tir.const(1, "int")

                    # create inner most block
                    irb = tvm.tir.ir_builder.create()
                    with irb.if_scope(condition):
                        dev = env.dev
                        irb.scope_attr(
                            dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE)
                        )
                        irb.scope_attr(dev.vta_axis, "coproc_uop_scope", dev.vta_push_uop)
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
                    inner = irb.get()

                    args = conv_call.indices
                    tpl = (args[0], 1, args[1], 1, args[2], 1, args[3], 1, 0, 1, 0, env.BLOCK_OUT)
                    inner = tvm.tir.AttrStmt(
                        [dout, res_tensor],
                        "buffer_bind_scope",
                        tvm.tir.call_intrin("handle", "tir.tvm_tuple", *tpl),
                        inner,
                    )
                    args = kernel_call.indices
                    tpl = (
                        args[0],
                        1,
                        args[1],
                        1,
                        args[2],
                        1,
                        args[3],
                        1,
                        0,
                        env.BLOCK_OUT,
                        0,
                        env.BLOCK_IN,
                    )
                    inner = tvm.tir.AttrStmt(
                        [dwgt, kernel_tensor],
                        "buffer_bind_scope",
                        tvm.tir.call_intrin("handle", "tir.tvm_tuple", *tpl),
                        inner,
                    )
                    args = data_call.indices
                    tpl = (args[0], 1, args[1], 1, args[2], 1, args[3], 1, 0, 1, 0, env.BLOCK_IN)
                    inner = tvm.tir.AttrStmt(
                        [dinp, pad_data_tensor],
                        "buffer_bind_scope",
                        tvm.tir.call_intrin("handle", "tir.tvm_tuple", *tpl),
                        inner,
                    )
                    return inner
            return None

        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(func.body, _do_fold, None, ["tir.AttrStmt"])
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.vta.InjectConv2DTrasnposeSkip"
    )


def AnnotateALUCoProcScope():
    """Pass to insert ALU instruction.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _ftransform(func, mod, ctx):
        env = get_env()

        def _do_fold(stmt):
            if _match_pragma(stmt, "alu"):
                irb = tvm.tir.ir_builder.create()
                irb.scope_attr(
                    env.dev.vta_axis, "coproc_scope", env.dev.get_task_qid(env.dev.QID_COMPUTE)
                )
                irb.scope_attr(
                    env.dev.vta_axis, "coproc_uop_scope", tvm.tir.StringImm("VTAPushALUOp")
                )
                irb.emit(stmt)
                return irb.get()
            if _match_pragma(stmt, "skip_alu"):
                return tvm.tir.Evaluate(0)
            return stmt

        return func.with_body(
            tvm.tir.stmt_functor.ir_transform(func.body, None, _do_fold, ["tir.AttrStmt"])
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.vta.AnnotateALUCoProcScope"
    )

# lowering optimization for comm reducer lowering
comm_suppress_init = []
comm_rewrite = []

def InjectALUIntrin():
    """Pass to inject ALU micro-ops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """

    def _ftransform(func, mod, ctx):
        env = get_env()
        idxm = tvm.tir.indexmod
        idxd = tvm.tir.indexdiv
        analyzer = tvm.arith.Analyzer()

        def _do_fold(stmt):
            def _equal(x, y):
                return tvm.ir.structural_equal(analyzer.simplify(x - y), 0)

            def _flatten_loop(src_coeff, dst_coeff, extents):
                src_coeff = list(src_coeff)
                dst_coeff = list(dst_coeff)
                extents = list(extents)
                rev_src_coeff = [src_coeff.pop()]
                rev_dst_coeff = [dst_coeff.pop()]
                rev_extents = []
                assert src_coeff
                vsrc = src_coeff.pop()
                vdst = dst_coeff.pop()
                vext = extents.pop()
                while src_coeff:
                    next_src = src_coeff.pop()
                    next_dst = dst_coeff.pop()
                    next_ext = extents.pop()

                    if _equal(next_src, vsrc * vext) and _equal(next_dst, vdst * vext):
                        vext = analyzer.simplify(vext * next_ext)
                    else:
                        rev_src_coeff.append(vsrc)
                        rev_dst_coeff.append(vdst)
                        rev_extents.append(vext)
                        vsrc = next_src
                        vdst = next_dst
                        vext = next_ext
                rev_src_coeff.append(vsrc)
                rev_dst_coeff.append(vdst)
                rev_extents.append(vext)
                rev_src_coeff.reverse()
                rev_dst_coeff.reverse()
                rev_extents.reverse()

                return rev_src_coeff, rev_dst_coeff, rev_extents

            if stmt in comm_suppress_init: # will be suppressed
                irb = tvm.tir.ir_builder.create()
                return irb.get()
            elif _match_pragma(stmt, "alu"):
                # Get to the innermost loop body
                loop_body = stmt.body
                nest_size = 0
                while isinstance(loop_body, tvm.tir.For):
                    loop_body = loop_body.body
                    nest_size += 1
                # Get the src/dst arguments
                dst_var = loop_body.buffer_var
                dst_idx = loop_body.index
                # Derive loop variables and extents
                tmp_body = stmt.body
                indices = []
                extents = []
                for _ in range(nest_size):
                    indices.append(tmp_body.loop_var)
                    extents.append(tmp_body.extent)
                    tmp_body = tmp_body.body
                # Derive opcode
                if isinstance(loop_body.value, tvm.tir.Add):
                    if loop_body in comm_rewrite: # replace add with assignment
                        alu_opcode = env.dev.ALU_OPCODE_MOV
                        lhs = loop_body.value.a
                        rhs = loop_body.value.b
                    else:
                        alu_opcode = env.dev.ALU_OPCODE_ADD
                        lhs = loop_body.value.a
                        rhs = loop_body.value.b
                elif isinstance(loop_body.value, tvm.tir.Sub):
                    lhs = loop_body.value.a
                    rhs = loop_body.value.b
                    # cannot subtract, must add with negate
                    if isinstance(lhs, tvm.tir.IntImm):
                        lhs = -1 * lhs
                    elif isinstance(rhs, tvm.tir.IntImm):
                        rhs = -1 * rhs
                    else:
                        assert False # must invert 1 operand
                    alu_opcode = env.dev.ALU_OPCODE_ADD
                elif isinstance(loop_body.value, tvm.tir.Mul):
                    alu_opcode = env.dev.ALU_OPCODE_MUL
                    lhs = loop_body.value.a
                    rhs = loop_body.value.b
                elif isinstance(loop_body.value, tvm.tir.Min):
                    alu_opcode = env.dev.ALU_OPCODE_MIN
                    lhs = loop_body.value.a
                    rhs = loop_body.value.b
                elif isinstance(loop_body.value, tvm.tir.Div):
                    alu_opcode = env.dev.ALU_OPCODE_SHR
                    lhs = loop_body.value.a
                    imm = loop_body.value.b.value
                    assert (1 << int(math.log2(imm))) == imm, "Only power of two Div supported"
                    rhs = tvm.ir.make_node("IntImm", dtype="int32", value=int(math.log2(imm)))
                elif isinstance(loop_body.value, tvm.tir.Max):
                    if loop_body in comm_rewrite: # replace max with mov
                        alu_opcode = env.dev.ALU_OPCODE_MOV
                        lhs = loop_body.value.a
                        rhs = loop_body.value.b
                    elif isinstance(loop_body.value.a, tvm.tir.Min): # nested max/min = clip
                        assert loop_body.value.a.b * -1 == loop_body.value.b
                        alu_opcode = env.dev.ALU_OPCODE_CLP
                        lhs = loop_body.value.a.a
                        rhs = loop_body.value.a.b
                    else:
                        alu_opcode = env.dev.ALU_OPCODE_MAX
                        lhs = loop_body.value.a
                        rhs = loop_body.value.b
                elif isinstance(loop_body.value, tvm.tir.Call):
                    if loop_body.value.op.name == "tir.shift_left":
                        alu_opcode = env.dev.ALU_OPCODE_SHR
                        lhs = loop_body.value.args[0]
                        rhs = analyzer.simplify(-loop_body.value.args[1])
                    elif loop_body.value.op.name == "tir.shift_right":
                        alu_opcode = env.dev.ALU_OPCODE_SHR
                        lhs = loop_body.value.args[0]
                        rhs = loop_body.value.args[1]
                    else:
                        raise RuntimeError(
                            "Function call not recognized %s" % (loop_body.value.name)
                        )
                elif isinstance(loop_body.value, tvm.tir.Load):
                    alu_opcode = env.dev.ALU_OPCODE_SHR
                    lhs = loop_body.value
                    rhs = tvm.tir.const(0, "int32")
                elif isinstance(loop_body.value, tvm.tir.expr.IntImm):
                    if loop_body.value >= -1 * (1 << 15) and loop_body.value < (1 << 15):
                        alu_opcode = env.dev.ALU_OPCODE_MOV
                        lhs = loop_body
                        rhs = analyzer.simplify(loop_body.value)
                    else:
                        assert False,\
                               "ALU MOV imm must fit in int16, not: %d" % (loop_body.value)
                else:
                    raise RuntimeError(
                        "Expression not recognized %s, %s, %s"
                        % (type(loop_body.value), str(loop_body.value), str(stmt))
                    )

                # Derive array index coefficients
                dst_coeff = tvm.arith.detect_linear_equation(dst_idx, indices)
                # Check if lhs/rhs is immediate
                use_imm = False
                imm_val = None
                if isinstance(rhs, tvm.tir.IntImm):
                    assert lhs.buffer_var.same_as(dst_var)
                    src_coeff = tvm.arith.detect_linear_equation(lhs.index, indices)
                    use_imm = True
                    imm_val = rhs
                if isinstance(lhs, tvm.tir.IntImm):
                    assert rhs.buffer_var.same_as(dst_var)
                    src_coeff = tvm.arith.detect_linear_equation(rhs.index, indices)
                    use_imm = True
                    imm_val = lhs
                if imm_val is None:
                    imm_val = 0
                    assert lhs.buffer_var.same_as(dst_var) and rhs.buffer_var.same_as(dst_var)
                    src_lhs_coeff = tvm.arith.detect_linear_equation(lhs.index, indices)
                    src_rhs_coeff = tvm.arith.detect_linear_equation(rhs.index, indices)
                    # Determine which side has the same coefficients
                    lhs_equal = True
                    rhs_equal = True
                    for i, coef in enumerate(dst_coeff):
                        if not tvm.ir.structural_equal(coef, src_lhs_coeff[i]):
                            lhs_equal = False
                        if not tvm.ir.structural_equal(coef, src_rhs_coeff[i]):
                            rhs_equal = False
                    # Make sure at least one of the source is identical to the
                    # destination (in-place computation)
                    assert lhs_equal or rhs_equal
                    # Assign the source coefficients
                    if lhs_equal:
                        src_coeff = src_rhs_coeff
                    else:
                        src_coeff = src_lhs_coeff

                # Ensure that we have the proper tensor dimensions in the
                # innermost loop (pattern match)
                src_coeff = list(src_coeff)
                dst_coeff = list(dst_coeff)
                extents = list(extents)
                assert len(src_coeff) > 1
                assert len(dst_coeff) > 1
                assert len(extents) != 0
                assert tvm.ir.structural_equal(
                    analyzer.simplify(idxm(src_coeff[-1], env.BATCH * env.BLOCK_OUT)), 0
                )
                assert tvm.ir.structural_equal(
                    analyzer.simplify(idxm(dst_coeff[-1], env.BATCH * env.BLOCK_OUT)), 0
                )
                assert tvm.ir.structural_equal(src_coeff[-2], 1)
                assert tvm.ir.structural_equal(dst_coeff[-2], 1)
                if env.BATCH > 1:
                    assert len(src_coeff) > 2
                    assert len(dst_coeff) > 2
                    assert len(extents) > 1
                    assert tvm.ir.structural_equal(src_coeff[-3], env.BLOCK_OUT)
                    assert tvm.ir.structural_equal(dst_coeff[-3], env.BLOCK_OUT)

                # Apply tensorization of the loop coefficients
                src_offset = src_coeff[-1]
                dst_offset = dst_coeff[-1]
                if env.BATCH == 1:
                    src_coeff = src_coeff[:-2]
                    dst_coeff = dst_coeff[:-2]
                    extents = extents[:-1]
                else:
                    src_coeff = src_coeff[:-3]
                    dst_coeff = dst_coeff[:-3]
                    extents = extents[:-2]
                src_coeff.append(src_offset)
                dst_coeff.append(dst_offset)
                src_coeff = [analyzer.simplify(c // (env.BATCH * env.BLOCK_OUT)) for c in src_coeff]
                dst_coeff = [analyzer.simplify(c // (env.BATCH * env.BLOCK_OUT)) for c in dst_coeff]

                # Flatten the outer loops
                if extents:
                    src_coeff, dst_coeff, extents = _flatten_loop(src_coeff, dst_coeff, extents)

                # Batch Tiling
                # Replace 3 UopLoopBegin with 2 UopLoopBegin
                # followed by sequence of for loops
                if len(extents) > 2:
                    loop_body = stmt.body
                    loop_vars = []
                    loop_mins = []
                    loop_extents = []
                    for_type = loop_body.kind

                    # Collect for loop info
                    while isinstance(loop_body, tvm.tir.For):
                        loop_vars.append(loop_body.loop_var)
                        loop_mins.append(loop_body.min)
                        loop_extents.append(loop_body.extent)
                        loop_body = loop_body.body
                    begins = []
                    ends = []

                    # Create 2 topmost VTAUopLoop s
                    for idx in range(2):
                        begins.append(tvm.tir.call_extern(
                            "int32", "VTAUopLoopBegin",
                            extents[idx], dst_coeff[idx], src_coeff[idx], 0))
                        ends.append(tvm.tir.call_extern(
                            "int32", "VTAUopLoopEnd"))

                    start_idx = 2 # VTA ISA supports two VTAUopLoops
                    if env.BATCH == 1:
                        end_idx = -1
                    else:
                        end_idx = -2
                    loop_vars = loop_vars[start_idx:end_idx]
                    loop_mins = loop_mins[start_idx:end_idx]
                    loop_extents = loop_extents[start_idx:end_idx]
                    vta_factor = env.BATCH * env.BLOCK_OUT
                    if use_imm is True:
                        src_rhs_coeff = tvm.arith.detect_linear_equation(lhs.index, indices)
                        src_lhs_coeff = tvm.arith.detect_linear_equation(dst_idx, indices)
                    src_ptr = idxd(src_rhs_coeff[-1], vta_factor)
                    dst_ptr = idxd(src_lhs_coeff[-1], vta_factor)
                    src_coeff_set = src_rhs_coeff[2:end_idx-1]
                    dst_coeff_set = src_lhs_coeff[2:end_idx-1]

                    # Adjust VTAUopPush src and dst indices
                    for idx, var in enumerate(loop_vars):
                        src_ptr = tvm.tir.expr.Add(
                            src_ptr, tvm.tir.expr.Mul(
                                var,
                                idxd(src_coeff_set[idx], vta_factor)
                            ))
                        dst_ptr = tvm.tir.expr.Add(
                            dst_ptr, tvm.tir.expr.Mul(
                                var,
                                idxd(dst_coeff_set[idx], vta_factor)
                            ))

                    body = tvm.tir.call_extern(
                        "int32", "VTAUopPush",
                        1, 0,
                        dst_ptr, src_ptr, 0,
                        alu_opcode, int(use_imm), imm_val)

                    body = tvm.tir.stmt.Evaluate(body)
                    for idx in range(len(loop_vars)-1, -1, -1):
                        body = tvm.tir.For(
                            loop_vars[idx], loop_mins[idx],
                            loop_extents[idx],
                            for_type,
                            body)

                    final = tvm.tir.stmt_seq(*(begins + [body] + ends))
                    return final

                # Insert ALU micro-ops
                irb = tvm.tir.ir_builder.create()
                for idx, extent in enumerate(extents):
                    irb.emit(
                        tvm.tir.call_extern(
                            "int32", "VTAUopLoopBegin", extent, dst_coeff[idx], src_coeff[idx], 0
                        )
                    )
                use_imm = int(use_imm)
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        1,
                        0,
                        dst_coeff[len(dst_coeff) - 1],
                        src_coeff[len(src_coeff) - 1],
                        0,
                        alu_opcode,
                        use_imm,
                        imm_val,
                    )
                )
                for extent in extents:
                    irb.emit(tvm.tir.call_extern("int32", "VTAUopLoopEnd"))
                return irb.get()
            return stmt

        def _alu_Mutator_(stmt):
            if isinstance(stmt, tvm.tir.stmt.AttrStmt) and \
                isinstance(stmt.body, tvm.tir.stmt.AttrStmt):
                alu_stmt = stmt.body.body
                loop_vars = []
                min_vals = []
                extents = []
                for_types = []

                if isinstance(alu_stmt, tvm.tir.stmt.For):
                    while isinstance(alu_stmt, tvm.tir.stmt.For):
                        # Collect for-loop structure
                        loop_vars.append(alu_stmt.loop_var)
                        min_vals.append(alu_stmt.min)
                        extents.append(alu_stmt.extent)
                        for_types.append(alu_stmt.kind)
                        alu_stmt = alu_stmt.body

                    if isinstance(alu_stmt.value, (tvm.tir.expr.Add, tvm.tir.expr.Mul)):
                        alu_lhs = alu_stmt.value.a
                        alu_rhs = alu_stmt.value.b
                        if isinstance(alu_lhs, tvm.tir.expr.Cast):
                            alu_lhs = alu_lhs.value
                        if isinstance(alu_rhs, tvm.tir.expr.Cast):
                            alu_rhs = alu_rhs.value
                        if isinstance(alu_lhs, tvm.tir.expr.IntImm):
                            return None
                        if isinstance(alu_rhs, tvm.tir.expr.IntImm):
                            return None
                        dst_idx = alu_stmt.index
                        dst_var = alu_stmt.buffer_var
                        dst_coeff = tvm.arith.detect_linear_equation(dst_idx, loop_vars)
                        lhs_coeff = tvm.arith.detect_linear_equation(alu_lhs.index, loop_vars)
                        rhs_coeff = tvm.arith.detect_linear_equation(alu_rhs.index, loop_vars)
                        lhs_equal = True
                        rhs_equal = True
                        for i, coef in enumerate(dst_coeff):
                            if not tvm.ir.structural_equal(coef, lhs_coeff[i]):
                                lhs_equal = False
                            if not tvm.ir.structural_equal(coef, rhs_coeff[i]):
                                rhs_equal = False

                        if lhs_equal is False and rhs_equal is False:
                            assert alu_lhs.buffer_var.same_as(dst_var)
                            assert alu_rhs.buffer_var.same_as(dst_var)

                            # Create a modified MUL stmt and a new MOV stmt
                            new_alu_lhs = tvm.tir.expr.Load(alu_lhs.dtype, dst_var,
                                                            dst_idx, alu_lhs.predicate)
                            if isinstance(alu_stmt.value, tvm.tir.expr.Mul):
                                new_alu_value = tvm.tir.expr.Mul(new_alu_lhs, alu_rhs)
                            else:
                                new_alu_value = tvm.tir.expr.Add(new_alu_lhs, alu_rhs)
                            new_alu_stmt = tvm.tir.stmt.Store(dst_var, new_alu_value,
                                                              dst_idx, alu_stmt.predicate)

                            mov_stmt = tvm.tir.stmt.Store(dst_var, alu_lhs,
                                                          dst_idx, alu_stmt.predicate)

                            for i in range(len(extents)-1, -1, -1):
                                new_alu_stmt = tvm.tir.stmt.For(loop_vars[i],
                                                                min_vals[i],
                                                                extents[i],
                                                                for_types[i],
                                                                new_alu_stmt)
                                mov_stmt = tvm.tir.stmt.For(loop_vars[i],
                                                            min_vals[i],
                                                            extents[i],
                                                            for_types[i],
                                                            mov_stmt)

                            new_alu_stmt = tvm.tir.stmt.AttrStmt(stmt.body.node,
                                                                 stmt.body.attr_key,
                                                                 stmt.body.value,
                                                                 new_alu_stmt)
                            mov_stmt = tvm.tir.stmt.AttrStmt(stmt.body.node,
                                                             stmt.body.attr_key,
                                                             stmt.body.value,
                                                             mov_stmt)
                            alu_stmt_vta = _do_fold(new_alu_stmt)
                            mov_stmt_vta = _do_fold(mov_stmt)
                            alu_body = tvm.tir.stmt.AttrStmt(stmt.node, stmt.attr_key,
                                                             stmt.value, alu_stmt_vta)
                            mov_body = tvm.tir.stmt.AttrStmt(stmt.node, stmt.attr_key,
                                                             stmt.value, mov_stmt_vta)
                            newbody = tvm.tir.stmt_seq(*([mov_body]+[alu_body]))
                            return newbody
            return None

        def _find_CommRed(op):
            if isinstance(op, tvm.tir.stmt.SeqStmt) and len(op.seq) > 3:
                for i in range(len(op.seq) - 1):
                    d_i = op.seq[i]
                    d_j = op.seq[i+1]
                    suppress_alu = None

                    while isinstance(d_i, (tvm.tir.stmt.AttrStmt, tvm.tir.stmt.For)):
                        if isinstance(d_i, tvm.tir.stmt.AttrStmt) and _match_pragma(d_i, "alu"):
                            suppress_alu = d_i
                        d_i = d_i.body # descend lower
                    if not isinstance(d_i, tvm.tir.stmt.Store):
                        continue
                    while isinstance(d_j, (tvm.tir.stmt.AttrStmt, tvm.tir.stmt.For)):
                        d_j = d_j.body
                    if not isinstance(d_j, tvm.tir.stmt.Store):
                        continue

                    # now we have two stores in back-to-back attr/for-loop nests
                    if d_i.buffer_var.same_as(d_j.buffer_var) and \
                       isinstance(d_i.value, tvm.tir.expr.IntImm) and \
                       isinstance(d_j.value.a, tvm.tir.expr.Load) and \
                       d_j.value.a.buffer_var.same_as(d_i.buffer_var) and \
                       suppress_alu is not None:
                        comm_suppress_init.append(suppress_alu) # remove alu pragma and assignment
                        comm_rewrite.append(d_j) # rewrite only the store statement at the core
                        break

        def _clip_Mutator_(stmt):
            if isinstance(stmt, tvm.tir.stmt.AttrStmt) and \
                isinstance(stmt.body, tvm.tir.stmt.AttrStmt):
                alu_stmt = stmt.body.body
                loop_vars = []
                min_vals = []
                extents = []
                for_types = []

                if isinstance(alu_stmt, tvm.tir.stmt.For):
                    while isinstance(alu_stmt, tvm.tir.stmt.For):
                        # Collect for-loop structure
                        loop_vars.append(alu_stmt.loop_var)
                        min_vals.append(alu_stmt.min)
                        extents.append(alu_stmt.extent)
                        for_types.append(alu_stmt.kind)
                        alu_stmt = alu_stmt.body
                    # find nested max/mins with imm ranges
                    if isinstance(alu_stmt.value, tvm.tir.expr.Max) and\
                       isinstance(alu_stmt.value.a, tvm.tir.expr.Min) and\
                       isinstance(alu_stmt.value.b, tvm.tir.expr.IntImm) and\
                       isinstance(alu_stmt.value.a.b, tvm.tir.expr.IntImm):
                        max_arg = alu_stmt.value.b.value
                        min_arg = alu_stmt.value.a.b.value
                        if not env.ENABLE_INSTRUCTION_CLP or max_arg != -1 * min_arg:
                            # decompose into max and min
                            new_min = tvm.tir.stmt.Store(alu_stmt.buffer_var, alu_stmt.value.a,
                                                         alu_stmt.index, alu_stmt.predicate)
                            new_max_value_a = tvm.tir.expr.Load(alu_stmt.value.a.dtype,
                                                                alu_stmt.buffer_var,
                                                                alu_stmt.index,
                                                                alu_stmt.predicate)
                            new_max_value = tvm.tir.expr.Max(new_max_value_a, alu_stmt.value.b)
                            new_max = tvm.tir.stmt.Store(alu_stmt.buffer_var, new_max_value,
                                                         alu_stmt.index, alu_stmt.predicate)

                            for i in range(len(extents)-1, -1, -1):
                                new_min = tvm.tir.stmt.For(loop_vars[i], min_vals[i], extents[i],
                                                           for_types[i], new_min)
                                new_max = tvm.tir.stmt.For(loop_vars[i], min_vals[i], extents[i],
                                                           for_types[i], new_max)

                            new_min = tvm.tir.stmt.AttrStmt(stmt.body.node,
                                                            stmt.body.attr_key,
                                                            stmt.body.value,
                                                            new_min)
                            new_max = tvm.tir.stmt.AttrStmt(stmt.body.node,
                                                            stmt.body.attr_key,
                                                            stmt.body.value,
                                                            new_max)
                            min_body = tvm.tir.stmt.AttrStmt(stmt.node, stmt.attr_key,
                                                             stmt.value, _do_fold(new_min))
                            max_body = tvm.tir.stmt.AttrStmt(stmt.node, stmt.attr_key,
                                                             stmt.value, _do_fold(new_max))
                            newbody = tvm.tir.stmt_seq(*([min_body]+[max_body]))
                            return newbody
            return None # change nothing

        # track if a CommReducer optimization can be implemented for VTA
        comm_suppress_init.clear() # reset
        comm_rewrite.clear() # reset
        tvm.tir.stmt_functor.post_order_visit(func.body, _find_CommRed)
        clip_body = tvm.tir.stmt_functor.ir_transform(func.body, _clip_Mutator_,
                                                      None, ["tir.AttrStmt"])
        return func.with_body(tvm.tir.stmt_functor.ir_transform(clip_body, _alu_Mutator_,
                                                                _do_fold, ["tir.AttrStmt"]))

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.vta.InjectALUIntrin"
    )
