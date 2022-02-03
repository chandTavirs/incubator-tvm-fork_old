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
"""Conv2D operator declaration and schedule registration for VTA."""
import os

import itertools
from functools import reduce
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from .utils import is_packed_layout
from ..environment import get_env

def finp(tilevec, wklparams, env):
    """
    Function to compute utilized space in input scratchpad

    Parameters
    ---------------
    tilevec : list
        Tiling parameter candidate

    wklparams : dict
        Workload parameters

    env: vta.environment.Environment
        VTA environment parameters

    Returns
    ---------------
    inpbits : int
        Utilized input scratchpad (in bits)
    """
    tile_b, _, tile_h, tile_w, tile_ci, oc_nthread, h_nthread = tilevec
    w_b = wklparams['w_b']
    i_h, i_w = wklparams['i_h'], wklparams['i_w']
    k_h, k_w = wklparams['k_h'], wklparams['k_w']
    s_h, s_w = wklparams['s_h'], wklparams['s_w']
    p_h, p_w = wklparams['p_h'], wklparams['p_w']
    f_i = wklparams['f_i']
    inpuse = ((w_b/env.BATCH/tile_b) * (f_i/env.BLOCK_IN/tile_ci) *
              (((i_h/tile_h + 2*p_h - k_h)//s_h)*s_h + k_h) *
              (((i_w/tile_w + 2*p_w - k_w)//s_w)*s_w + k_w) *
              env.BATCH * env.BLOCK_IN * oc_nthread * h_nthread)
    inpbits = inpuse * env.INP_WIDTH
    return int(inpbits)

def fwgt(tilevec, wklparams, env):
    """
    Function to compute utilized space in weight scratchpad

    Parameters
    ---------------
    tilevec : list
        Tiling parameter candidate

    wklparams : dict
        Workload parameters

    env: vta.environment.Environment
        VTA environment parameters

    Returns
    ---------------
    wgtbits : int
        Utilized weight scratchpad (in bits)
    """
    _, tile_co, _, _, tile_ci, oc_nthread, h_nthread = tilevec
    f_i, f_o = wklparams['f_i'], wklparams['f_o']
    k_h, k_w = wklparams['k_h'], wklparams['k_w']
    wgtuse = ((f_o * f_i * k_h * k_w)/(tile_co * tile_ci) * oc_nthread * h_nthread)
    wgtbits = wgtuse * env.WGT_WIDTH
    return int(wgtbits)

def facc(tilevec, wklparams, env):
    """
    Function to compute utilized space in accumulator scratchpad

    Parameters
    ---------------
    tilevec : list
        Tiling parameter candidate

    wklparams : dict
        Workload parameters

    env: vta.environment.Environment
        VTA environment parameters

    Returns
    ---------------
    accbits : int
        Utilized accumulator scratchpad (in bits)
    """
    tile_b, tile_co, tile_h, tile_w, _, oc_nthread, h_nthread = tilevec
    w_b = wklparams['w_b']
    f_o = wklparams['f_o']
    o_h, o_w = wklparams['o_h'], wklparams['o_w']
    accuse = ((w_b * f_o * o_h * o_w)/
              (tile_b * tile_co * tile_h * tile_w) +
              (f_o*w_b)/(tile_b*tile_co)) * oc_nthread * h_nthread
    accbits = accuse * env.ACC_WIDTH
    return int(accbits)

def cons(tilevec, wklparams, env, tag, vthread, acc_avoid, height_tile, width_tile):
    """
    Function to check constraints during tiling parameter search
    Returns False, if any constraint is violated
    Else, returns the cost, depending on tag
    If tag = 'buff' return total unutilized scratchpad bits
    If tag = 'load' return total DRAM bytes loaded

    Parameters
    ---------------
    tilevec : list
        Tiling parameter candidate

    wklparams : dict
        Workload parameters

    env: vta.environment.Environment
        VTA environment parameters

    tag : str
        Tag for defining cost

    vthread: bool
        If true, mandatory virtual threading
        If false, optional threading

    acc_avoid: bool
        Get acc_avoid from tileparam_search

    height_tile: bool
        Get height_tile from tileparam_search

    width_tile: bool
        Get width_tile from tileparam_search

    Returns
    ---------------
    cost : int
        Cost of tiling parameters, if constraint unviolated
        Else, return is False
    """
    tile_b, tile_co, tile_h, tile_w, tile_ci, oc_nthread, h_nthread = tilevec
    # Support any one virtual threading
    if oc_nthread == 2 and h_nthread == 2:
        return False
    # Check mandatory or optional virtual threading
    if vthread:
        if oc_nthread == 1 and h_nthread == 1:
            return False

    # Avoid tiling candidate choices that cause acc_idx writing
    # in two consecutive cycles
    # Related to the VerifyDep function in UopKernel class
    if acc_avoid:
        co_bound = wklparams['f_o']//env.BLOCK_OUT
        h_bound = wklparams['o_h']
        tile_co_in = co_bound//tile_co
        tile_h_in = h_bound//tile_h
        # This condition will cause acc_index to be same within 2 cycles
        if tile_co_in*tile_h_in <= 2 and wklparams['k_h'] != 1:
            return False

    # Check tiling inner width dimension (NCHW)
    if not width_tile:
        if tile_w != 1:
            return False

    # Check tiling height dimension (NCHW)
    if not height_tile:
        if tile_h != 1:
            return False

    # Constraints to avoid likely operators
    if tile_co % oc_nthread != 0:
        return False
    if tile_h % h_nthread != 0:
        return False

    # Avoid compiler limitations on memory fetch
    if wklparams['f_i']//env.BLOCK_IN != tile_ci and wklparams['s_h'] != 0:
        return False

    # Avoid double threading if tile_ci == 1
    # Without outer loop of tile_ci, double threading is incorrect
    # Purpose of double threading is to overlap GEMM and LOAD
    # Without outer loop of tile_ci, GEMM is followed by ALU
    if tile_ci == 1:
        if h_nthread == 2 or oc_nthread == 2:
            return False

    cost = 0
    inpbits = finp(tilevec, wklparams, env)
    wgtbits = fwgt(tilevec, wklparams, env)
    accbits = facc(tilevec, wklparams, env)

    # Scratchpad sizes
    inpsize = env.INP_BUFF_SIZE * 8
    wgtsize = env.WGT_BUFF_SIZE * 8
    accsize = env.ACC_BUFF_SIZE * 8

    # Check input scratchpad fit
    inpfit = inpsize - inpbits
    if inpfit < 0:
        return False
    if tag == 'buff':
        cost += inpfit
    else:
        dthread_fac = 1
        if oc_nthread == 2:
            dthread_fac = 0.5
        cost += (tile_b * (tile_h/h_nthread) * (tile_co/oc_nthread) *
                 tile_w * tile_ci * inpbits) * dthread_fac

    # Check weight scratchpad fit
    wgtfit = wgtsize - wgtbits
    if wgtfit < 0:
        return False
    if tag == 'buff':
        cost += wgtfit
    else:
        dthread_fac = 1
        if h_nthread == 2:
            dthread_fac = 0.5
        cost += (tile_b * (tile_h/h_nthread) * (tile_co/oc_nthread) *
                 tile_w * tile_ci * wgtbits) * dthread_fac

    # Check acc scratchpad fit
    accfit = accsize - accbits
    if accfit < 0:
        return False
    if tag == 'buff':
        cost += accfit
    else:
        cost += tile_b * tile_h * tile_w * wklparams['f_o'] * env.ACC_WIDTH

    # Check uop scratchpad fit
    uop_width = 1 << env.LOG_UOP_WIDTH
    uop_elem_bytes = uop_width//8
    uop_depth = env.UOP_BUFF_SIZE/uop_elem_bytes

    b_bound = wklparams['w_b']//env.BATCH
    co_bound = wklparams['f_o']//env.BLOCK_OUT
    h_bound = wklparams['o_h']
    w_bound = wklparams['o_w']
    ci_bound = wklparams['f_i']//env.BLOCK_IN

    uop_loop_order = []
    uop_loop_order.append(ci_bound//tile_ci)
    uop_loop_order.append(b_bound//tile_b)
    uop_loop_order.append(w_bound//tile_w)
    uop_loop_order.append(wklparams['k_h'])
    uop_loop_order.append(wklparams['k_w'])
    uop_loop_order.append(co_bound//tile_co)
    uop_loop_order.append(h_bound//tile_h)

    # Calculate kernel sequence size
    seq_size = 0
    uop_kernel_size = 1
    for val in uop_loop_order:
        if val != 1:
            seq_size += 1
        if seq_size > 2:
            uop_kernel_size *= val

    # Calculate uop fit
    if uop_kernel_size > uop_depth:
        return False

    return int(cost)

def find_factors(n):
    return list(set(reduce(list.__add__,
                           ([i, n//i] for i in range(1, int(n**0.5) + 1) if not n % i))))

def tileparam_search(wkl, env, tag='load', vthread=True, acc_avoid=True,
                     height_tile=True, width_tile=False):
    """
    Function to search for tiling parameters

    Parameters
    -------------
    wkl : tvm.ir.container.Array
        Workload container from conv2d stage

    env: vta.environment.Environment
        VTA environment parameters

    tag : str
        Default: 'load' - minimizes DRAM load transfer
        'buff' - maximizes scratchpad usage

    vthread : bool
        Default: True - enforces virtual threading
        False - optional virtual threading

    acc_avoid : bool
        Default: True - avoids acc_index writing in 2 cycles in a row
        False - Doesn't enforce specific parameters
        To keep this false, VerifyDep needs to be bypassed at runtime.cc

    height_tile : bool
        Default: True - enables tiling in the height dimension
        False - disables tiling in the height dimension

    width_tile : bool
        Default: False - disables tiling in the width dimension
        True - enables tiling in the width dimension

    Returns
    --------------
    soln : list
        Optimal tiling parameters
    """
    # Collect parameters
    wklparams = {}
    wklparams['w_b'] = int(wkl[1][1][0]) * env.BATCH
    wklparams['i_h'] = int(wkl[1][1][2])
    wklparams['i_w'] = int(wkl[1][1][3])
    wklparams['k_h'] = int(wkl[2][1][2])
    wklparams['k_w'] = int(wkl[2][1][3])
    wklparams['f_i'] = int(wkl[2][1][1] * wkl[2][1][5])
    wklparams['f_o'] = int(wkl[2][1][0] * wkl[2][1][4])
    wklparams['s_h'] = int(wkl[3][0])
    wklparams['s_w'] = int(wkl[3][1])
    wklparams['p_h'] = int(wkl[4][0])
    wklparams['p_w'] = int(wkl[4][1])
    wklparams['o_h'] = ((wklparams['i_h'] + 2*wklparams['p_h'] - wklparams['k_h'])//
                        wklparams['s_h'] + 1)
    wklparams['o_w'] = ((wklparams['i_w'] + 2*wklparams['p_w'] - wklparams['k_w'])//
                        wklparams['s_w'] + 1)

    # Create candidates
    # Tiling on 'width' axis is avoided, virtual thread limit of 2
    bounds = [wklparams['w_b']//env.BATCH, wklparams['f_o']//env.BLOCK_OUT,
              wklparams['o_h'], wklparams['o_w'], wklparams['f_i']//env.BLOCK_IN,
              2, 2]
    combinations = list(itertools.product(*[find_factors(bounds[i]) for i in range(len(bounds))]))
    values = []

    # First, search with threading (double buffering) enabled
    for combination in combinations:
        cost = cons(combination, wklparams, env, tag, vthread, acc_avoid, height_tile, width_tile)
        if cost:
            values.append((combination, cost))
    # Second, search with threading disabled
    if not values:
        for combination in combinations:
            cost = cons(combination, wklparams, env, tag, False, acc_avoid, height_tile, width_tile)
            if cost:
                values.append((combination, cost))
    # Third, search with both width tiling and threading enabled
    if not values:
        for combination in combinations:
            cost = cons(combination, wklparams, env, tag, vthread, acc_avoid, False, True)
            if cost:
                values.append((combination, cost))
    # Fourth, search with width tiling enabled and threading disabled
    if not values:
        for combination in combinations:
            cost = cons(combination, wklparams, env, tag, False, acc_avoid, False, True)
            if cost:
                values.append((combination, cost))
    # Fifth, search with both height and width tiling disabled and threading enabled
    if not values:
        for combination in combinations:
            cost = cons(combination, wklparams, env, tag, vthread, acc_avoid, False, False)
            if cost:
                values.append((combination, cost))
    # Sixth, search with both height and width tiling disabled and threading disabled
    if not values:
        for combination in combinations:
            cost = cons(combination, wklparams, env, tag, False, acc_avoid, False, False)
            if cost:
                values.append((combination, cost))

    # Use fallback if TPS fails to search valid parameters
    if not values:
        return None

    best = [c for c, v in values if v == min([v for c, v in values])]
    soln = [None] * len(bounds)
    for i, _ in enumerate(bounds):
        soln[i] = ([best[0][i], bounds[i]//best[0][i]])
    return soln

@autotvm.register_topi_compute("conv2d_packed.vta")
def conv2d_packed(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """ Packed conv2d function."""
    if not is_packed_layout(layout):
        raise topi.InvalidShapeError()
    assert dilation == (1, 1)

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, data.shape[4], kernel.shape[4])

    ishape = topi.utils.get_const_tuple(data.shape)
    kshape = topi.utils.get_const_tuple(kernel.shape)
    d_i = te.reduce_axis((0, kshape[2]), name="d_i")
    d_j = te.reduce_axis((0, kshape[3]), name="d_j")
    k_o = te.reduce_axis((0, ishape[1]), name="k_o")
    k_i = te.reduce_axis((0, ishape[-1]), name="k_i")
    hstride, wstride = strides
    res = te.compute(
        oshape,
        lambda b_o, c_o, i, j, b_i, c_i: te.sum(
            pad_data[b_o, k_o, i * hstride + d_i, j * wstride + d_j, b_i, k_i].astype(out_dtype)
            * kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i],
        ),
        name="res",
        tag="conv2d_dense",
    )

    cfg.add_flop(
        2
        * np.prod(topi.utils.get_const_tuple(oshape))
        * kshape[2]
        * kshape[3]
        * ishape[1]
        * ishape[-1]
    )

    return res


@autotvm.register_topi_schedule("conv2d_packed.vta")
def schedule_conv2d_packed(cfg, outs):
    """Schedule packed conv2d"""
    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert "int" in output.op.input_tensors[0].dtype

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
            assert op.tag == "conv2d_dense"
            conv2d_res.append(op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)
    s = te.create_schedule(output.op)

    ##### space definition begin #####
    b, c_o, x_i, x_j, _, _ = s[conv2d_stage].op.axis
    c_i, _, _, _ = s[conv2d_stage].op.reduce_axis
    cfg.define_split("tile_b", b, num_outputs=2)
    cfg.define_split("tile_h", x_i, num_outputs=2)
    cfg.define_split("tile_w", x_j, num_outputs=2)
    cfg.define_split("tile_ci", c_i, num_outputs=2)
    cfg.define_split("tile_co", c_o, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    cfg.define_knob("h_nthread", [1, 2])
    ###### space definition end ######

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None

    env = get_env()

    # setup pad
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.inp_scope)
    else:
        cdata = s.cache_read(data, env.inp_scope, [conv2d_stage])
    ckernel = s.cache_read(kernel, env.wgt_scope, [conv2d_stage])
    s[conv2d_stage].set_scope(env.acc_scope)

    # cache read input
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(s.cache_read(tensor, env.acc_scope, [consumer]))

    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)

    for op in const_ops:
        s[op].compute_inline()

    # Tiling Parameter Search
    if cfg.is_fallback and env.ENABLE_TPS:
        wkl = conv2d_res[0].attrs['workload']
        tiles = tileparam_search(wkl, env, acc_avoid=False, vthread=bool(env.ENABLE_VTHREAD))
        if tiles is not None:
            tile_b, tile_co, tile_h, tile_w, tile_ci, oc_nthread, h_nthread = tiles
            if env.ENABLE_TILING_WRITEOUT:
                filepath = os.path.join(os.environ['TVM_HOME'], "ash", "tps.log")
                config_key = "x".join([str(env.BATCH), str(env.BLOCK_IN), str(env.BLOCK_OUT)])
                config_key = "_".join([config_key, "i8w8a32"])
                config_key = "_".join([config_key, str(env.LOG_UOP_BUFF_SIZE)])
                config_key = "_".join([config_key, str(env.LOG_INP_BUFF_SIZE)])
                config_key = "_".join([config_key, str(env.LOG_WGT_BUFF_SIZE)])
                config_key = "_".join([config_key, str(env.LOG_ACC_BUFF_SIZE)])
                f = open(filepath, "a+")
                f.write('{"input": ["ext_dev -device=vta -keys=cpu -model=sim_')
                f.write(config_key)
                f.write('", "')
                f.write(str(wkl[0]))
                f.write('", [')
                outstr = str(wkl[1])
                for i in range(2, len(wkl)-2):
                    outstr = ', '.join([outstr, str(wkl[i])])
                f.write(outstr)
                f.write(', "')
                f.write(str(wkl[-2]))
                f.write('", "')
                f.write(str(wkl[-1]))
                f.write('"], {}]')
                f.write(', "config": {"index": 1, "code_hash": null, "entity": [')
                outstr = ', '.join(['["tile_b"', '"sp"', str(tile_b)])
                outstr = ''.join([outstr, ']'])
                outstr = ', '.join([outstr, '["tile_h"', '"sp"', str(tile_h)])
                outstr = ''.join([outstr, ']'])
                outstr = ', '.join([outstr, '["tile_w"', '"sp"', str(tile_w)])
                outstr = ''.join([outstr, ']'])
                outstr = ', '.join([outstr, '["tile_ci"', '"sp"', str(tile_ci)])
                outstr = ''.join([outstr, ']'])
                outstr = ', '.join([outstr, '["tile_co"', '"sp"', str(tile_co)])
                outstr = ''.join([outstr, ']'])
                outstr = ', '.join([outstr, '["oc_nthread"', '"ot"', str(oc_nthread[0])])
                outstr = ''.join([outstr, ']'])
                outstr = ', '.join([outstr, '["h_nthread"', '"ot"', str(h_nthread[0])])
                outstr = ''.join([outstr, ']]}'])
                f.write(outstr)
                f.write(', "result": [1, 0, 1, 1], "version": 0.2')
                f.write(', "tvm_version": "0.7.dev0"}')
                f.write('\n')
                f.close()
            cfg['tile_co'] = tvm.autotvm.task.space.SplitEntity(tile_co)
            cfg['tile_ci'] = tvm.autotvm.task.space.SplitEntity(tile_ci)
            cfg['tile_h'] = tvm.autotvm.task.space.SplitEntity(tile_h)
            cfg['tile_w'] = tvm.autotvm.task.space.SplitEntity(tile_w)
            cfg['oc_nthread'] = tvm.autotvm.task.space.OtherOptionEntity(oc_nthread[0])
            cfg['h_nthread'] = tvm.autotvm.task.space.OtherOptionEntity(h_nthread[0])
            cfg['tile_b'] = tvm.autotvm.task.space.SplitEntity(tile_b)
            print("Replacing fallback parameters with optimal tiling parameters" \
            " for DRAM load minimization.")

    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg["tile_co"].apply(s, output, x_co)
    x_i0, x_i1 = cfg["tile_h"].apply(s, output, x_i)
    x_j0, x_j1 = cfg["tile_w"].apply(s, output, x_j)
    x_bo0, x_bo1 = cfg['tile_b'].apply(s, output, x_bo)
    s[output].reorder(x_bo0, x_i0, x_co0, x_j0, x_bo1, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

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

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

    k_o, _ = cfg["tile_ci"].apply(s, conv2d_stage, k_o)
    s[cdata].compute_at(s[conv2d_stage], k_o)
    s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Batch Tiling
    batch_idx = 0
    tile_b = cfg['tile_b'].size
    if tile_b[1] != 1:
        d_bo, _, _, _, _, _ = s[cdata].op.axis
        s[cdata].unroll(d_bo)
        batch_idx = 1

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[batch_idx], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    s[conv2d_stage].tensorize(x_bi, env.gemm)
    s[output].pragma(x_co1, env.dma_copy)

    return s
