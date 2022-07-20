"""Pooling operator declaration and schedule registration for VTA."""

# Modified by contributors from Intel Labs

import math
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from .utils import is_packed_layout
from ..environment import get_env

def get_factors(x):
    assert x >= 1
    factors = []
    for i in range(1, x+1):
        if x % i == 0:
            factors.append(i)
    return factors

def get_shifts_and_addsubs(x, num_bits):
    """ Determine the sequence of shifts and adds/subs to approximate division
        We have num_bits overall to work with
    """
    assert x >= 1 # a fraction <= 1
    frac = float(1)/float(x)
    div = len(bin(x)[:1:-1]) - 1 # number of bits
    sel = []
    remain = frac

    for i in range(div, num_bits):
        if remain >= 2**(-1*i): # do we get closer to zero
            remain -= 2**(-1*i)
            sel.append(i)
    print(num_bits, "bit approximation of 1 /", x, "has error:", remain)
    # post-process sel array to incremental mode
    if len(sel) > 1:
        cur = sel[0]
        for i in range(1, len(sel)):
            sel[i] = sel[i] - cur
            cur += sel[i]
    return sel

# scaling
def get_scaling_bits(min_a, max_a, b, full_width):
    max_val = (abs(min_a) + max_a) * b
    scaling_factor = math.floor((1 << (full_width - 1)) / max_val)
    scaling_bits = len(bin(scaling_factor)[:1:-1]) - 1
    print("Division approximation upscales input by",
          scaling_bits, "based on summation of", b, "values between", min_a, "and", max_a)
    return scaling_bits

@autotvm.register_topi_compute("pooling_packed.vta")
def pooling_packed(cfg, data, kernel, stride, padding, pool_type,
                   ceil_mode, layout, count_include_pad,
                   in_range=None, out_range=None):
    """ Packed pooling function."""

    if in_range is None:
        in_range = [-128, 127]
    if out_range is None:
        out_range = [-128, 127]

    _ = cfg
    _ = ceil_mode
    _ = count_include_pad

    if not is_packed_layout(layout):
        raise topi.InvalidShapeError()

    if pool_type == "avg": # only support global average pool for now (1 output per filter)
        oshape = [data.shape[0], data.shape[1], 1, 1, data.shape[4], data.shape[5]]
        x = te.reduce_axis((0, data.shape[2]), name="x")
        y = te.reduce_axis((0, data.shape[3]), name="y")
        res = te.compute(oshape, lambda i, j, ox, oy, k, m:
                         te.sum(data[i, j, x, y, k, m], axis=[x, y]), name="res")

        # input and output ranges
        in_bits = len(bin(in_range[1]-in_range[0]+1)[:1:-1]) - 1
        out_bits = len(bin(out_range[1]-out_range[0]+1)[:1:-1]) - 1

        assert in_range[1] >= in_range[0], "Input range must contain >= 1 value"
        assert out_range[1] >= out_range[0], "Output range must contain >= 1 value"
        assert (1 << in_bits) == in_range[1]-in_range[0]+1, "In range must span power-of-two"
        assert (1 << out_bits) == out_range[1]-out_range[0]+1, "Out range must span power-of-two"
        assert in_bits >= out_bits, "Input range must be at least as large as output range"

        # approximate the division operation
        divisor = (data.shape[2] * data.shape[3]).value

        sbits = get_scaling_bits(in_range[0], in_range[1], divisor, 32)
        sel_vec = get_shifts_and_addsubs(divisor, 32)

        if in_range[0] < 0: # if there could be negative inputs
            res = res + abs(in_range[0])*divisor # make the sum positive

        res = te.compute(oshape, lambda i, j, ox, oy, k, m:
                         res[i, j, ox, oy, k, m] << sbits, name="scale_up")
        scratch = te.compute(oshape, lambda i, j, ox, oy, k, m: 0, name="zero") # init temp buffer
        for ziter in sel_vec: # shift and accumulate loop
            res = topi.right_shift(res, ziter)
            scratch = scratch + res
        # scale down, but take into account potential output range restriction vs. input range
        scratch = te.compute(oshape, lambda i, j, ox, oy, k, m:
                             scratch[i, j, ox, oy, k, m] >> (sbits + in_bits - out_bits),
                             name="scale_down")
        if out_range[0] < 0:
            scratch = scratch + out_range[0] # subtract back to output range if needed
        return scratch
    # max pool is much simpler, just use the topi code
    res = topi.nn.pool(data, kernel, stride, padding, pool_type, layout=layout)
    return res

# @autotvm.register_topi_schedule("pooling_packed.vta")
# def schedule_pooling_packed(cfg, outs, layout=None):
#     """Schedule packed pooling"""
#     assert len(outs) == 1
#     env = get_env()
#     _ = cfg
#     _ = layout
#
#     output = outs[0]
#     assert "int" in output.op.input_tensors[0].dtype
#     s = te.create_schedule(output.op)
#
#     def traverse_ops(op, pad_ops, pool_ops, div_ops, inps, seen):
#         if op in seen:
#             return pad_ops, pool_ops, div_ops, inps, seen
#         seen.append(op)
#         if isinstance(op, tvm.te.PlaceholderOp):
#             inps.append(op) # end of the chain
#             return pad_ops, pool_ops, div_ops, inps, seen
#         if isinstance(op, tvm.te.ComputeOp):
#             if isinstance(op.body, tvm.ir.container.Array) and \
#                isinstance(op.body[0], tvm.tir.expr.Cast):
#                 assert op == output.op
#             elif isinstance(op.body, tvm.ir.container.Array) and \
#                isinstance(op.body[0], tvm.tir.expr.Reduce):
#                 assert len(pool_ops) == 0
#                 pool_ops.append(op)
#             elif "pad" in op.name:
#                 assert len(pad_ops) == 0
#                 pad_ops.append(op)
#             else: # a decomposed division op
#                 div_ops.append(op)
#         else:
#             print("Unknown:", type(op))
#
#         for i in op.input_tensors: # recursive call for all inputs
#             pad_ops, pool_ops, div_ops, inps, seen = traverse_ops(i.op, pad_ops, pool_ops,
#                                                                   div_ops, inps, seen)
#         return pad_ops, pool_ops, div_ops, inps, seen
#
#     # order of ops before output is: inp, [pad], pool, [div], output
#     pad_ops = []
#     pool_ops = []
#     div_ops = []
#     inps = []
#     seen = []
#     pad_ops, pool_ops, div_ops, inps, seen = traverse_ops(output.op, pad_ops, pool_ops,
#                                                           div_ops, inps, seen)
#
#     assert len(inps) == 1
#     inp = inps[0]
#
#     if pad_ops != []:
#         pad_op = pad_ops[0]
#     else:
#         pad_op = None
#
#     if pool_ops != []:
#         pool_op = pool_ops[0]
#     else:
#         pool_op = None
#
#     p_i, p_j, p_oh, p_ow, p_x, p_y = s[pool_op].op.axis
#     p_kh, p_kw = s[pool_op].op.reduce_axis
#
#     # start with assumption that entire frame, with BATCH * BLOCK_OUT components
#     # per pixel, resides in the scratchpad
#     acc_size = int(env.ACC_BUFF_SIZE//(env.ACC_WIDTH/8))
#     scratch_axis = 1
#
#     if len(div_ops) == 0:
#         extra_space_factor = 1
#     else:
#         extra_space_factor = 2 # need another copy of output as temp storage
#
#     scratch_size = np.prod(inp.shape[scratch_axis+1:]) + \
#       extra_space_factor * np.prod(output.shape[scratch_axis+1:])
#
#     if scratch_size > acc_size:
#         if len(div_ops) == 0: # max pool
#             print("Splitting into row to reduce scratchpad utilization")
#             #print("Out height factors:", get_factors(output.shape[scratch_axis+1].value))
#
#             scratch_axis += 1 # descend one level lower
#             inps_per_out = int(round(inp.shape[scratch_axis].value/
#                                      output.shape[scratch_axis].value))
#             scratch_size = inps_per_out * np.prod(inp.shape[scratch_axis+1:]) \
#                          + np.prod(output.shape[scratch_axis+1:]) # the row axis is gone
#             if scratch_size > acc_size: #Cannot fit single row of pooling I/O in scratchpad
#                 print("Splitting into single element to further reduce scratchpad utilization")
#                 scratch_axis += 1 # descend one level: now a single output pixel
#                 scratch_size = inps_per_out * inps_per_out * np.prod(inp.shape[scratch_axis+1:]) \
#                              + np.prod(output.shape[scratch_axis+1:]) # the row/col axes are gone
#             #todo: split height axis first according to factors, then compute_at into new axis
#         else: # global average pool
#             print("Cannot split global average pool frame any further, ERROR")
#             return s
#
#     print("Acc utilization:", scratch_size, "elems")
#     print("Acc size:", acc_size, "elems")
#
#     # scratchpad and compute at the desired level
#     output_store_pt = s[output].op.axis[scratch_axis]
#     output_dma_pt = s[output].op.axis[scratch_axis+1]
#
#     for diter in div_ops: # average pooling operation div sequence
#         s[diter].set_scope(env.acc_scope)
#         s[diter].pragma(s[diter].op.axis[0], env.alu)
#         s[diter].compute_at(s[output], output_store_pt)
#
#     s[pool_op].reorder(p_kh, p_kw, p_i, p_j, p_oh, p_ow, p_x, p_y)
#     s[pool_op].compute_at(s[output], output_store_pt)
#
#     #print("Unrolling kernel of height:", p_kh.dom.extent, "width:", p_kw.dom.extent)
#     s[pool_op].unroll(p_kh)
#     s[pool_op].unroll(p_kw)
#
#     #ic_out, ic_inn = s[res_conv].split(ic, factor=ic_block)
#     # set acc scope and alu pragma
#     s[pool_op].set_scope(env.acc_scope)
#     s[pool_op].pragma(s[pool_op].op.axis[0], env.alu)
#
#     if pad_op is None:
#         cdata = s.cache_read(inps[0].output(0), env.acc_scope, pool_op)
#     else:
#         cdata = pad_op.output(0)
#         s[pad_op].set_scope(env.acc_scope)
#
#     s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
#     s[cdata].compute_at(s[output], output_store_pt)
#
#     s[output].pragma(output_dma_pt, env.dma_copy)
#     return s

@autotvm.register_topi_schedule("pooling_packed.vta")
def schedule_pooling_packed(cfg, outs, layout=None):
    """Schedule packed pooling"""
    assert len(outs) == 1
    env = get_env()
    _ = cfg
    _ = layout

    output = outs[0]
    assert "int" in output.op.input_tensors[0].dtype
    s = te.create_schedule(output.op)

    def traverse_ops(op, pad_ops, pool_ops, div_ops, inps, seen):
        if op in seen:
            return pad_ops, pool_ops, div_ops, inps, seen
        seen.append(op)
        if isinstance(op, tvm.te.PlaceholderOp):
            inps.append(op) # end of the chain
            return pad_ops, pool_ops, div_ops, inps, seen
        if isinstance(op, tvm.te.ComputeOp):
            if isinstance(op.body, tvm.ir.container.Array) and \
               isinstance(op.body[0], tvm.tir.expr.Cast):
                assert op == output.op
            elif isinstance(op.body, tvm.ir.container.Array) and \
               isinstance(op.body[0], tvm.tir.expr.Reduce):
                assert len(pool_ops) == 0
                pool_ops.append(op)
            elif "pad" in op.name:
                assert len(pad_ops) == 0
                pad_ops.append(op)
            else: # a decomposed division op
                div_ops.append(op)
        else:
            print("Unknown:", type(op))

        for i in op.input_tensors: # recursive call for all inputs
            pad_ops, pool_ops, div_ops, inps, seen = traverse_ops(i.op, pad_ops, pool_ops,
                                                                  div_ops, inps, seen)
        return pad_ops, pool_ops, div_ops, inps, seen

    # order of ops before output is: inp, [pad], pool, [div], output
    pad_ops = []
    pool_ops = []
    div_ops = []
    inps = []
    seen = []
    pad_ops, pool_ops, div_ops, inps, seen = traverse_ops(output.op, pad_ops, pool_ops,
                                                          div_ops, inps, seen)

    assert len(inps) == 1
    inp = inps[0]

    if pad_ops != []:
        pad_op = pad_ops[0]
    else:
        pad_op = None

    if pool_ops != []:
        pool_op = pool_ops[0]
    else:
        pool_op = None

    #pool2d_stage = pool_op.output(0)

    ##### space definition begin #####
    b, c_o, x_i, x_j, p_x, p_y = s[pool_op].op.axis
    p_kh, p_kw = s[pool_op].op.reduce_axis
    cfg.define_split("tile_b", b, num_outputs=2)
    cfg.define_split("tile_h", x_i, num_outputs=2)
    cfg.define_split("tile_w", x_j, num_outputs=2)
    cfg.define_split("tile_co", c_o, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    cfg.define_knob("h_nthread", [1, 2])

    # p_i, p_j, p_oh, p_ow, p_x, p_y = s[pool_op].op.axis
    # p_kh, p_kw = s[pool_op].op.reduce_axis

    if pad_op is None:
        cdata = s.cache_read(inps[0].output(0), env.acc_scope, pool_op)
    else:
        cdata = pad_op.output(0)
        s[pad_op].set_scope(env.acc_scope)

    s[pool_op].set_scope(env.acc_scope)
    s[pool_op].pragma(s[pool_op].op.axis[0], env.alu)

    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg["tile_co"].apply(s, output, x_co)
    x_i0, x_i1 = cfg["tile_h"].apply(s, output, x_i)
    x_j0, x_j1 = cfg["tile_w"].apply(s, output, x_j)
    x_bo0, x_bo1 = cfg['tile_b'].apply(s, output, x_bo)
    s[output].reorder(x_bo0, x_i0, x_co0, x_j0, x_bo1, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0

    s[pool_op].compute_at(s[output], store_pt)


    # set all compute scopes
    # s[pool_op].reorder(p_kh, p_kw, b, c_o, x_i, x_j, p_x, p_y)



    # virtual threading along output channel axes
    if cfg["oc_nthread"].val > 1:
        _, v_t = s[output].split(x_co0, factor=cfg["oc_nthread"].val)
        s[output].reorder(v_t, x_bo0)
        s[output].bind(v_t, te.thread_axis(env.VTHREAD_NAME))

    # virtual threading along spatial rows
    if cfg["h_nthread"].val > 1:
        _, v_t = s[output].split(x_i0, factor=cfg["h_nthread"].val)
        s[output].reorder(v_t, x_bo0)
        s[output].bind(v_t, te.thread_axis(env.VTHREAD_NAME))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[pool_op].op.axis
    d_i, d_j = s[pool_op].op.reduce_axis
    #s[pool_op].reorder(x_bo, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci)
    s[pool_op].reorder(d_i, d_j, x_bo, x_co, x_i, x_j, x_bi, x_ci)

    s[pool_op].unroll(p_kh)
    s[pool_op].unroll(p_kw)

    # Batch Tiling
    batch_idx = 0
    tile_b = cfg['tile_b'].size
    if tile_b[1] != 1:
        d_bo, _, _, _, _, _ = s[cdata].op.axis
        s[cdata].unroll(d_bo)
        batch_idx = 1

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[batch_idx], env.dma_copy)
    s[cdata].compute_at(s[output], store_pt)
    #s[pool2d_stage].tensorize(x_bi, env.alu)
    s[output].pragma(x_co1, env.dma_copy)

    # # start with assumption that entire frame, with BATCH * BLOCK_OUT components
    # # per pixel, resides in the scratchpad
    # acc_size = int(env.ACC_BUFF_SIZE//(env.ACC_WIDTH/8))
    # scratch_axis = 1
    #
    # if len(div_ops) == 0:
    #     extra_space_factor = 1
    # else:
    #     extra_space_factor = 2 # need another copy of output as temp storage

    # scratch_size = np.prod(inp.shape[scratch_axis+1:]) + \
    #   extra_space_factor * np.prod(output.shape[scratch_axis+1:])
    #
    # if scratch_size > acc_size:
    #     if len(div_ops) == 0: # max pool
    #         print("Splitting into row to reduce scratchpad utilization")
    #         #print("Out height factors:", get_factors(output.shape[scratch_axis+1].value))
    #
    #         scratch_axis += 1 # descend one level lower
    #         inps_per_out = int(round(inp.shape[scratch_axis].value/
    #                                  output.shape[scratch_axis].value))
    #         scratch_size = inps_per_out * np.prod(inp.shape[scratch_axis+1:]) \
    #                      + np.prod(output.shape[scratch_axis+1:]) # the row axis is gone
    #         if scratch_size > acc_size: #Cannot fit single row of pooling I/O in scratchpad
    #             print("Splitting into single element to further reduce scratchpad utilization")
    #             scratch_axis += 1 # descend one level: now a single output pixel
    #             scratch_size = inps_per_out * inps_per_out * np.prod(inp.shape[scratch_axis+1:]) \
    #                          + np.prod(output.shape[scratch_axis+1:]) # the row/col axes are gone
    #         #todo: split height axis first according to factors, then compute_at into new axis
    #     else: # global average pool
    #         print("Cannot split global average pool frame any further, ERROR")
    #         return s
    #
    # print("Acc utilization:", scratch_size, "elems")
    # print("Acc size:", acc_size, "elems")

    # scratchpad and compute at the desired level
    # output_store_pt = s[output].op.axis[scratch_axis]
    # output_dma_pt = s[output].op.axis[scratch_axis+1]

    # for diter in div_ops: # average pooling operation div sequence
    #     s[diter].set_scope(env.acc_scope)
    #     s[diter].pragma(s[diter].op.axis[0], env.alu)
    #     s[diter].compute_at(s[output], output_store_pt)

    # s[pool_op].reorder(p_kh, p_kw, p_i, p_j, p_oh, p_ow, p_x, p_y)
    # s[pool_op].compute_at(s[output], output_store_pt)

    #print("Unrolling kernel of height:", p_kh.dom.extent, "width:", p_kw.dom.extent)
    # s[pool_op].unroll(p_kh)
    # s[pool_op].unroll(p_kw)

    #ic_out, ic_inn = s[res_conv].split(ic, factor=ic_block)
    # set acc scope and alu pragma
    # s[pool_op].set_scope(env.acc_scope)


    # if pad_op is None:
    #     cdata = s.cache_read(inps[0].output(0), env.acc_scope, pool_op)
    # else:
    #     cdata = pad_op.output(0)
    #     s[pad_op].set_scope(env.acc_scope)



    # s[output].pragma(output_dma_pt, env.dma_copy)
    return s