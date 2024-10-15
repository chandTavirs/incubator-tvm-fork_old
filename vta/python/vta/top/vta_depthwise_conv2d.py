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

"""Depthwise Conv2D operator declaration and schedule registration for VTA."""

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from .utils import is_packed_layout
from ..environment import get_env

@autotvm.register_topi_compute("depthwise_conv2d_packed.vta")
def depthwise_conv2d_packed(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """Packed depthwise conv2d function"""
    if not is_packed_layout(layout):
        raise topi.InvalidShapeError()
    assert dilation == (1, 1)
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data

    env = get_env()
    assert env.BATCH == 1 # Schedule works only for VTA Batch = 1
    _ = cfg

    hstride, wstride = strides
    dshape = topi.utils.get_const_tuple(data.shape)
    kshape = topi.utils.get_const_tuple(kernel.shape)
    ch_mul = kernel.shape[1]*env.BATCH
    assert ch_mul == 1

    oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // hstride + 1)
    owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // wstride + 1)
    oshape = (dshape[0], dshape[1], oheight, owidth, dshape[4], dshape[5])
    temp_shape = (dshape[0], dshape[1], oheight, owidth, kshape[2], kshape[3], dshape[4], dshape[5])

    d_i = te.reduce_axis((0, kshape[2]), name='d_i')
    d_j = te.reduce_axis((0, kshape[3]), name='d_j')

    # Element-wise multiplication
    temp = te.compute(
        temp_shape,
        lambda b_o, c_o, i, j, ti, tj, b_i, c_i:
        pad_data[b_o, c_o, i*hstride + ti, j*wstride + tj, b_i, c_i].astype(out_dtype) *
        kernel[c_o, 0, ti, tj, c_i, b_i].astype(out_dtype),
        name="product", tag="product")

    out = te.compute(
        oshape,
        lambda b_o, c_o, i, j, b_i, c_i: te.sum(
            temp[b_o, c_o, i, j, d_i, d_j, b_i, c_i],
            axis=[d_i, d_j]), name="res", tag="sum")

    return out

@autotvm.register_topi_schedule('depthwise_conv2d_packed.vta')
def schedule_depthwise_conv2d_packed(cfg, outs):
    """Schedule packed depthwise conv2d"""
    assert len(outs) == 1
    output = outs[0]
    s = te.create_schedule(output.op)

    # Extract stages
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    sum_ops = []
    prod_ops = []

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
        elif op.tag == "sum":
            sum_ops.append(op)
            _traverse(op.input_tensors[0].op)
        else:
            assert op.tag == "product"
            prod_ops.append(op)

    _traverse(output.op)
    assert len(sum_ops) == 1
    assert len(prod_ops) == 1
    prod_stage = prod_ops[0].output(0)
    sum_stage = sum_ops[0].output(0)

    data, kernel = prod_stage.op.input_tensors

    # Extract bias placeholder op
    # bias_stage = ewise_ops[2].output(0)
    # _, bias = bias_stage.op.input_tensors

    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None

    env = get_env()

    # Setup loads
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.acc_scope)
    else:
        cdata = s.cache_read(data, env.acc_scope, [prod_stage])
    ckernel = s.cache_read(kernel, env.acc_scope, [prod_stage])
    s[prod_stage].set_scope(env.acc_scope)
    s[sum_stage].set_scope(env.acc_scope)

    # cache read input
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(
            s.cache_read(tensor, env.acc_scope, [consumer]))

    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)

    # Reorder and unroll axes for sum stage
    s_bo, s_co, s_i, s_j, s_bi, s_ci = s[sum_stage].op.axis
    s_ki, s_kj = s[sum_stage].op.reduce_axis
    s[sum_stage].reorder(s_bo, s_co, s_i, s_j, s_ki, s_kj, s_bi, s_ci)
    s[sum_stage].unroll(s_ki)
    s[sum_stage].unroll(s_kj)

    # Define tiling
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    cfg.define_split('tile_b', x_bo, num_outputs=2)
    cfg.define_split('tile_h', x_i, num_outputs=2)
    cfg.define_split('tile_w', x_j, num_outputs=2)
    cfg.define_split('tile_co', x_co, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    # cfg.define_knob("h_nthread", [1, 2])

    # Apply tiling
    x_bo0, x_bo1 = cfg['tile_b'].apply(s, output, x_bo)
    x_i0, x_i1 = cfg['tile_h'].apply(s, output, x_i)
    x_j0, x_j1 = cfg['tile_w'].apply(s, output, x_j)
    x_co0, x_co1 = cfg['tile_co'].apply(s, output, x_co)
    s[output].reorder(x_bo0, x_i0, x_co0, x_j0, x_bo1, x_co1, x_i1, x_j1, x_bi, x_ci)

    # Set compute locations
    store_pt = x_j0
    s[prod_stage].compute_at(s[output], store_pt)
    s[sum_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    # Set ALU scopes
    _, p_co, p_h, _, _, _, _, _ = s[prod_stage].op.axis
    s[prod_stage].pragma(p_h, env.alu)
    s[sum_stage].pragma(s_bi, env.alu)

    # Move load scopes
    s[cdata].compute_at(s[prod_stage], p_co)
    s[ckernel].compute_at(s[prod_stage], p_co)

    # Use VTA Instructions
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    s[output].pragma(x_bi, env.dma_copy)

    return s
