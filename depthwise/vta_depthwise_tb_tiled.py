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

# Created my contributors from Intel Labs

"""Simple Python-based testbench for running depthwise conv stride 1"""

import ctypes
import tvm
import tvm.topi.testing
import vta
from vta.testing import simulator
import numpy as np
from collections import namedtuple
from functools import partial
from itertools import product, accumulate
import argparse
import json

# ***** constants

# buffer copy directions
VTA_MEMCPY_H2D = 1
VTA_MEMCPY_D2H = 2
VTA_MEMCPY_D2D = 3

# scratchpads
VTA_MEM_ID_UOP = 0
VTA_MEM_ID_WGT = 1
VTA_MEM_ID_INP = 2
VTA_MEM_ID_ACC = 3
VTA_MEM_ID_OUT = 4
VTA_MEM_ID_ACC_8BIT = 5

# GEMM opcodes
VTA_GEMM_OPCODE_RESET = 0
VTA_GEMM_OPCODE_NORMAL = 1
VTA_GEMM_OPCODE_DEPTHWISE_STRIDE1 = 2
VTA_GEMM_OPCODE_DEPTHWISE_STRIDE2 = 3

# ALU opcodes
VTA_ALU_OPCODE_MIN = 0
VTA_ALU_OPCODE_MAX = 1
VTA_ALU_OPCODE_ADD = 2
VTA_ALU_OPCODE_SHR = 3
VTA_ALU_OPCODE_MUL = 4

# arch stages
kNoneStage = 0
kLoadStage = 1
kComputeStage = 2
kStoreStage = 3

# debug flags
VTA_DEBUG_DUMP_INSN = 2
VTA_DEBUG_DUMP_UOP = 4
VTA_DEBUG_SKIP_READ_BARRIER = 8
VTA_DEBUG_SKIP_WRITE_BARRIER = 16
VTA_DEBUG_FORCE_SERIAL = 32

# prototypes
FINIT_FUNC_SIG = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

# ***** VTA setup
rt = simulator.LIBS[0]
#simulator.debug_mode(0) # execute functionality
simulator.clear_stats()
# print(simulator.stats())
cmd = rt.VTATLSCommandHandle()
#rt.VTASetDebugMode(cmd, VTA_DEBUG_FORCE_SERIAL + VTA_DEBUG_DUMP_UOP + VTA_DEBUG_DUMP_INSN)
rt.VTASetDebugMode(cmd, VTA_DEBUG_DUMP_UOP + VTA_DEBUG_DUMP_INSN)

# Workload
env = vta.get_env()

Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "channel_multiplier",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

wkls = [
        ("depthwise_stride1.small", Workload(env.BATCH, 5, 5, 32, 1, 3, 3, 1, 1, 1, 1)),
        ("depthwise_stride2.small", Workload(env.BATCH, 5, 5, 32, 1, 3, 3, 1, 1, 2, 2)),
        ('mobilenet.D1',  Workload(env.BATCH, 112, 112, 32, 1, 3, 3, 1, 1, 1, 1)),
        ('mobilenet.D2',  Workload(env.BATCH, 112, 112, 64, 1, 3, 3, 1, 1, 2, 2)),
        ('mobilenet.D3',  Workload(env.BATCH, 56, 56, 128, 1, 3, 3, 1, 1, 1, 1)),
        ('mobilenet.D4',  Workload(env.BATCH, 56, 56, 128, 1, 3, 3, 1, 1, 2, 2)),
        ('mobilenet.D5',  Workload(env.BATCH, 28, 28, 256, 1, 3, 3, 1, 1, 1, 1)),
        ('mobilenet.D6',  Workload(env.BATCH, 28, 28, 256, 1, 3, 3, 1, 1, 2, 2)),
        ('mobilenet.D7',  Workload(env.BATCH, 14, 14, 512, 1, 3, 3, 1, 1, 1, 1)),
        ('mobilenet.D8',  Workload(env.BATCH, 14, 14, 512, 1, 3, 3, 1, 1, 2, 2)),
        ('mobilenet.D9',  Workload(env.BATCH, 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1)),
]

def balanced_tiling( n, t):
    k = (n + t - 1)//t

    hi = (n + k - 1)//k

    over = hi*k - n

    hi_lst = [hi] * (k-over)
    lo_lst = [hi-1] * (over)

    lst = hi_lst + lo_lst

    assert n == sum(lst)

    return lst

def skewed_tiling( n, t):
    k = (n + t - 1)//t

    hi = t

    over = hi*k - n

    hi_lst = [hi] * (k-1)
    lo_lst = [hi-over] * (1)

    lst = hi_lst + lo_lst

    assert n == sum(lst)
    assert all( x > 0 for x in lst)

    return lst

def gen_offsets( lst):
    return [0] + list(accumulate(lst))[:-1]

def test_skewed_tiling():
    n = 112
    for t in range(1, n+1):
        lst = skewed_tiling(n, t)
        assert lst[:-1] == ([lst[0]]*(len(lst[:-1])))

def test_balanced_tiling():
    n = 112
    for t in range(1, n+1):
        lst = balanced_tiling(n, t)
        assert all( x in {lst[0], lst[0]-1} for x in lst)
        assert all( x >= y for (x,y) in zip(lst[:-1],lst[1:]))

def dw_tiling_algorithm(wl, buffering, ow_tile_in=None, oh_tile_in=None):

    def inp_tile(out_tile, wl):
        oh_tile, ow_tile = out_tile
        ih_tile_padded = (oh_tile - 1) * wl.hstride + wl.hkernel
        iw_tile_padded = (ow_tile - 1) * wl.wstride + wl.wkernel
        return ih_tile_padded, iw_tile_padded
    
    #eventually this function will be an algorithm that generates the tiling
    # input to this algorithm will be: wkl--> dimensions of a depthwise layer, vta_config--> vta hardware parameters

    # tiling of channels must be equal to env.BLOCK_OUT: in future may have to take care of situation when #of channels < env.BLOCK_OUT
    c_tile = env.BLOCK_OUT 

    # original output size
    #
    # for stride 1, kernel = 3, pad = 1 => h + 2 - 3 + 1 = h
    # for stride 2, h even (h=2*hh),  kernel = 3, pad = 1 => (2*hh     + 2 - 3) // 2 + 1 = hh-1 + 1 == hh
    # for stride 2, h odd (h=2*hh+1), kernel = 3, pad = 1 => (2*hh + 1 + 2 - 3) // 2 + 1 = hh + 1

    oheight = (wl.height + 2*wl.hpad - wl.hkernel)//wl.hstride + 1
    owidth  = (wl.width  + 2*wl.wpad - wl.wkernel)//wl.wstride + 1

    # Assume even sizes with stride of two
    if wl.hstride == 2:
        assert oheight % 2 == 0
    if wl.wstride == 2:
        assert owidth % 2 == 0

    print( wl)
    print( f'INP_BUFF_SIZE {env.INP_BUFF_SIZE} {env.INP_BUFF_SIZE // env.BLOCK_IN}')
    print( f'ACC_BUFF_SIZE {env.ACC_BUFF_SIZE} {env.ACC_BUFF_SIZE // env.BLOCK_OUT // 4}')
    print( f'IM2COL_LINE_SIZE {env.IM2COL_LINE_SIZE}')

    def check_if_tiling_fits( oh_tile, ow_tile, wl):
        ih_tile_padded, iw_tile_padded = inp_tile([oh_tile, ow_tile], wl)
        inp_remaining = env.INP_BUFF_SIZE - ih_tile_padded * iw_tile_padded * env.BLOCK_IN * buffering
        # the + 1 accounts for bias storage
        out_remaining = env.ACC_BUFF_SIZE // 4 - (oh_tile * ow_tile + 1) * env.BLOCK_OUT * buffering
        print( f'inp_remaining: {inp_remaining} out_remaining: {out_remaining}')
        return inp_remaining >= 0 and out_remaining >= 0

    def check_width( oh_tile, ow_tile, wl):
        ih_tile_padded, iw_tile_padded = inp_tile([oh_tile, ow_tile], wl)
        line_size_remaining = env.IM2COL_LINE_SIZE - iw_tile_padded
        print( f'line_size_remaining: {line_size_remaining}')
        return line_size_remaining >= 0

    # Reduce scratchpad size by 2x, first in height then in width
    def scheme1():
        oh_tile, ow_tile = oheight, owidth

        while True:
            if check_if_tiling_fits( oh_tile, ow_tile, wl):
                break
            oh_tile //= 2
            if check_if_tiling_fits( oh_tile, ow_tile, wl):
                break
            ow_tile //= 2

        while True:
            if check_width( oh_tile, ow_tile, wl):
                break
            ow_tile //= 2

        return oh_tile, ow_tile

    def scheme2():
        oh_tile, ow_tile = oheight, owidth

        while True:
            if check_width( oh_tile, ow_tile, wl):
                break
            ow_tile //= 2

        while True:
            if check_if_tiling_fits( oh_tile, ow_tile, wl):
                break
            oh_tile //= 2
            if oh_tile % 2 != 0:
                break

        while True:
            if check_if_tiling_fits( oh_tile, ow_tile, wl):
                break
            ow_tile //= 2

        return oh_tile, ow_tile

    def scheme3():
        oh_tile, ow_tile = oheight, owidth

        def factors( x):
            """Very dumb"""
            result = []
            for i in range(1,x+1):
                if x % i == 0:
                    result.append(i)
            return result

        factors_oheight = factors(oheight)
        factors_owidth = factors(owidth)
        print( f'factors of oheight: {oheight} = {factors_oheight}')
        print( f'factors of owidth: {owidth} = {factors_owidth}')

        legal = []
        for (fh, fw) in product( factors_oheight, factors_owidth):
            if check_width( fh, fw, wl) and check_if_tiling_fits( fh, fw, wl):
                legal.append( (fh,fw))
                
        def cost( p):
            oh_tile, ow_tile = p
            ih_tile_padded, iw_tile_padded = inp_tile([oh_tile, ow_tile], wl)
            
            nh = oheight // oh_tile
            nw = owidth // ow_tile

            return (-((2*nh+oheight)*(2*nw+owidth) + nh*nw*7), ow_tile)


        sorted_legal = list(sorted( legal, key=cost))

        print( [(p,cost(p)) for p in sorted_legal])
        return sorted_legal[-1]


    #oh_tile, ow_tile = scheme1()
    # oh_tile, ow_tile = scheme2()

    if oh_tile_in is not None:
        assert ow_tile_in is not None
        oh_tile, ow_tile = oh_tile_in, ow_tile_in
    else:
        oh_tile, ow_tile = scheme3()

    ih_tile_padded, iw_tile_padded = inp_tile([oh_tile, ow_tile], wl)

    # print("ih_tile_padded = ", ih_tile_padded, "; iw_tile_padded = ", iw_tile_padded)
    # print("oh_tile = ", oh_tile, "; ow_tile = ", ow_tile, "; c_tile = ", c_tile)

    tiling_params = {'IHT': ih_tile_padded, 'IWT': iw_tile_padded, 'OHT': oh_tile, 'OWT': ow_tile, 'CT': c_tile}
    print(tiling_params)

    #constraints 
    assert c_tile == env.BLOCK_OUT
    assert wl.hkernel*wl.wkernel <= env.BLOCK_IN
    assert (wl.in_filter * wl.channel_multiplier) % c_tile == 0

    # we can fix this if it is overly restrictive (just need to run with a smaller block in last row or column
    #assert oheight % oh_tile == 0
    #assert owidth % ow_tile == 0

    assert iw_tile_padded <= env.IM2COL_LINE_SIZE
    assert iw_tile_padded >= wl.wkernel
    assert ih_tile_padded >= wl.hkernel
    assert check_if_tiling_fits( oh_tile, ow_tile, wl)

    return tiling_params

# running depthwise layer
# Set to 1 for single buffering and 2 for double buffering
def run_depthwise(wkl, buffering=2, oh_tile_in=None, ow_tile_in=None, out_file=None):
    # Variable shapes
    wl_name, wl = wkl
    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)    # why channel multiplier is not in input shape?
    w_orig_shape = (wl.in_filter, wl.channel_multiplier, wl.hkernel, wl.wkernel)
    w_shape = (wl.in_filter*wl.channel_multiplier, env.BLOCK_IN, 1, 1)
    b_shape = (wl.batch, wl.in_filter * wl.channel_multiplier, 1, 1)
    a_pack = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_IN,
            wl.height, wl.width,
            env.BATCH, env.BLOCK_IN)
    w_pack = (wl.in_filter//env.BLOCK_OUT, env.BLOCK_IN//env.BLOCK_IN,
            1, 1,
            env.BLOCK_OUT, env.BLOCK_IN)
    b_pack = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_OUT,
            1, 1,
            env.BATCH, env.BLOCK_OUT)

    # Calculate output shape
    oheight = (wl.height + 2*wl.hpad - wl.hkernel)//wl.hstride + 1
    owidth = (wl.width + 2*wl.wpad - wl.wkernel)//wl.wstride + 1
    r_shape = (wl.batch, wl.in_filter*wl.channel_multiplier, oheight, owidth)
    r_pack = (a_pack[0], w_pack[0], oheight, owidth, a_pack[4], w_pack[4])

    # ***** numpy inputs
    np.random.seed(0)
    a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
    w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
    b_min, b_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
    a_vta = np.random.randint(a_min, a_max, size=a_pack).astype(np.int8)
    w_vta = np.random.randint(w_min, w_max, size=w_pack).astype(np.int8)
    b_vta = np.random.randint(b_min, b_max, size=b_pack).astype(np.int32)

    # Data packing for VTA 
    # ctypes void pointers don't respect transpose operators, DRAM layout don't change
    # Still retaining weight and bias transpose, as height and width are 1
    a_np = a_vta.transpose((0, 4, 1, 5, 2, 3)).reshape(a_shape)
    w_vta[:, :, :, :, :, 9:] = 0
    w_np_vta = w_vta.transpose((0, 4, 1, 5, 2, 3)).reshape(w_shape)
    w_np = np.zeros(w_orig_shape).reshape(wl.in_filter*wl.channel_multiplier,
                                          wl.hkernel*wl.wkernel, 1, 1)
    w_np = w_np_vta[:, :9, :, ]
    w_np = w_np.reshape(w_orig_shape)
    b_np = b_vta.transpose((0, 4, 1, 5, 2, 3)).reshape(b_shape)

    if wl.hpad == 0:
        pad = 'VALID'
    else:
        pad = 'SAME'
    r_ref = tvm.topi.testing.depthwise_conv2d_python_nchw(
        a_np.astype(env.acc_dtype),
        w_np.astype(env.acc_dtype),
        (wl.hstride, wl.wstride),
        pad,
    ).astype(env.acc_dtype)
    r_vta = np.zeros(r_pack, dtype = np.int8)

    # ***** allocate and copy vta arrays
    a_vta_dram = rt.VTABufferAlloc(a_vta.size * a_vta.dtype.itemsize)
    w_vta_dram = rt.VTABufferAlloc(w_vta.size * w_vta.dtype.itemsize)
    b_vta_dram = rt.VTABufferAlloc(b_vta.size * b_vta.dtype.itemsize)
    r_vta_dram = rt.VTABufferAlloc(r_vta.size * r_vta.dtype.itemsize)

    rt.VTABufferCopy(a_vta.ctypes.data_as(ctypes.c_void_p), 0,
                    ctypes.c_void_p(a_vta_dram), 0,
                    a_vta.size * a_vta.dtype.itemsize,
                    VTA_MEMCPY_H2D)

    rt.VTABufferCopy(w_vta.ctypes.data_as(ctypes.c_void_p), 0,
                    ctypes.c_void_p(w_vta_dram), 0,
                    w_vta.size * w_vta.dtype.itemsize,
                    VTA_MEMCPY_H2D)

    rt.VTABufferCopy(b_vta.ctypes.data_as(ctypes.c_void_p), 0,
                    ctypes.c_void_p(b_vta_dram), 0,
                    b_vta.size * b_vta.dtype.itemsize,
                    VTA_MEMCPY_H2D)

    rt.VTABufferCopy(r_vta.ctypes.data_as(ctypes.c_void_p), 0,
                    ctypes.c_void_p(r_vta_dram), 0,
                    r_vta.size * r_vta.dtype.itemsize,
                    VTA_MEMCPY_H2D)

    #obtaining tiling parameters
    tiling_params = dw_tiling_algorithm(wl, buffering, ow_tile_in, oh_tile_in)
    c_tile = tiling_params['CT']
    ih_tile_padded = tiling_params['IHT']
    iw_tile_padded = tiling_params['IWT']
    oh_tile = tiling_params['OHT']
    ow_tile = tiling_params['OWT']

    inp_single_buffer_sz = ih_tile_padded*iw_tile_padded
    out_single_buffer_sz = oh_tile*ow_tile
    wgt_single_buffer_sz = 1


    print( f'inp_single_buffer_sz: {inp_single_buffer_sz}')
    print( f'out_single_buffer_sz: {out_single_buffer_sz}')
    print( f'wgt_single_buffer_sz: {wgt_single_buffer_sz}')


    # ***** definition of the compute instructions       
    def gemm_kernel(inp_parity, wgt_parity, ow_tile, ih_tile_padded, iw_tile_padded, a):
        rt.VTAUopLoopBegin(ih_tile_padded, ow_tile, # extent = 7, dst_factor = 5
                    iw_tile_padded, 0) # src_factor = 7, wgt_factor = 0
        rt.VTAUopLoopBegin(iw_tile_padded, 1, # extent = 7, dst_factor = 1
                    1, 0) # src_factor = 1, wgt_factor = 0
        rt.VTAUopPush(0, VTA_GEMM_OPCODE_DEPTHWISE_STRIDE1 if wl.hstride == 1 else VTA_GEMM_OPCODE_DEPTHWISE_STRIDE2,
                      out_single_buffer_sz*inp_parity, # dest
                      inp_single_buffer_sz*inp_parity, # inp
                      wgt_single_buffer_sz*wgt_parity, # wgt
                      0, 0, # don't care opcode, don't care imm
                      0) # imm = don't care
        rt.VTAUopLoopEnd()
        rt.VTAUopLoopEnd()
        return 0

    def alu_add(inp_parity, wgt_parity, oh_tile, ow_tile, a):
        rt.VTAUopLoopBegin(oh_tile*ow_tile, 1, # extent = 25, dst_factor = 1
                    0, 0) # src_factor = 1, wgt_factor = 0
        rt.VTAUopPush(1, 0, # ALU mode with no reset
                      out_single_buffer_sz*inp_parity, # dest
                      out_single_buffer_sz*buffering+wgt_parity, # src
                      0, # wgt (not used)
                      VTA_ALU_OPCODE_ADD, 0, # opcode and imm used?
                      0) # imm (not used)
        rt.VTAUopLoopEnd()
        return 0

    def alu_shr(inp_parity, oh_tile, ow_tile, a):
        rt.VTAUopLoopBegin(oh_tile*ow_tile, 1, # extent = 25, dst_factor = 1
                    1, 0) # src_factor = 1, wgt_factor = 0
        rt.VTAUopPush(1, 0, # ALU mode with no reset
                out_single_buffer_sz*inp_parity, out_single_buffer_sz*inp_parity, 0, # dest, src, wgt (not used)
                VTA_ALU_OPCODE_SHR, 1, # opcode and imm used?
                8) # imm
        rt.VTAUopLoopEnd()
        return 0

    def alu_min(inp_parity, oh_tile, ow_tile, a):
        rt.VTAUopLoopBegin(oh_tile*ow_tile, 1, # extent = 25, dst_factor = 1
                    1, 0) # src_factor = 1, wgt_factor = 0
        rt.VTAUopPush(1, 0, # ALU mode with no reset
                out_single_buffer_sz*inp_parity, out_single_buffer_sz*inp_parity, 0, # dest, src, wgt (not used)
                VTA_ALU_OPCODE_MIN, 1, # opcode and imm used?
                127) # imm = 127
        rt.VTAUopLoopEnd()
        return 0

    def alu_max(inp_parity, oh_tile, ow_tile, a):
        rt.VTAUopLoopBegin(oh_tile*ow_tile, 1, # extent = 25, dst_factor = 1
                    1, 0) # src_factor = 1, wgt_factor = 0
        rt.VTAUopPush(1, 0, # ALU mode with no reset
                out_single_buffer_sz*inp_parity, out_single_buffer_sz*inp_parity, 0, # dest, src, wgt (not used)
                VTA_ALU_OPCODE_MAX, 1, # opcode and imm used?
                0) # imm = 0
        rt.VTAUopLoopEnd()
        return 0

    # ***** load instructions
    # wl.height, wl.width, oheight, owidth etc get replaced by their corresponding tile size 
    # not sure what params to change to offset the darm address so that the correct chunk from a_vta is loaded


    oc_max = (wl.in_filter*wl.channel_multiplier) // c_tile
    oh_max = (oheight + oh_tile - 1) // oh_tile
    ow_max = (owidth + ow_tile - 1)  // ow_tile


    # choose between balanced or skewed tiling
    oh_lst = balanced_tiling(oheight, oh_tile)
    ow_lst = balanced_tiling(owidth,  ow_tile)

    print(f'oh_lst: {oh_lst}')
    print(f'ow_lst: {ow_lst}')

    oh_offset_lst = gen_offsets(oh_lst)
    ow_offset_lst = gen_offsets(ow_lst)

    assert len(oh_lst) == oh_max
    assert len(ow_lst) == ow_max


    for oc in range(oc_max):
        block_count = ow_max*oh_max*oc

        # print("oc = ", oc)

        wgt_parity = oc % buffering

        #weight is loaded once with the channel loop 
        if block_count >= buffering:
            rt.VTADepPop(cmd,kComputeStage,kLoadStage)

        rt.VTALoadBuffer2D(cmd, w_vta_dram, oc,
            1, 1, 1, # 1 row, 1 column, 1 stride
            0, 0, 0, 0, # no padding on each side
            0, # is pad min value
            wgt_parity, VTA_MEM_ID_WGT) # wgt_parity

        # Bias loading
        rt.VTALoadBuffer2D(cmd, b_vta_dram, oc,
                        1, 1, 1, # 1 row, 1 column, 1 stride
                        0, 0, 0, 0, # no padding on each side
                        0, # is pad min value
                        out_single_buffer_sz*buffering+wgt_parity, VTA_MEM_ID_ACC) # 


        for oh, ow in product(range(oh_max),range(ow_max)):

            block_count = ow + ow_max*(oh + oh_max*oc)
            inp_parity = block_count % buffering

            #height computations
            oh_offset = oh_offset_lst[oh]
            oh_tile_local = oh_lst[oh]

            hpad_before = wl.hpad if oh == 0 else 0
            hpad_after  = wl.hpad if oh == oh_max - 1 and wl.hstride == 1 else 0

            ih_offset = 0 if oh == 0 else wl.hstride*oh_offset_lst[oh] - wl.hpad
            ih_tile_padded_local = oh_tile_local*wl.hstride + (1 if wl.hstride == 2 else 2)
            ih_tile_local = ih_tile_padded_local - hpad_before - hpad_after

            #width computations
            ow_offset = ow_offset_lst[ow]
            ow_tile_local = ow_lst[ow]

            wpad_before = wl.wpad if ow == 0 else 0
            wpad_after  = wl.wpad if ow == ow_max - 1 and wl.wstride == 1 else 0

            iw_offset = 0 if ow == 0 else wl.wstride*ow_offset_lst[ow] - wl.wpad
            iw_tile_padded_local = ow_tile_local*wl.wstride + (1 if wl.wstride == 2 else 2)
            iw_tile_local = iw_tile_padded_local - wpad_before - wpad_after


            # *****input load instruction
            if block_count >= buffering and (oh != 0 or ow != 0):
                rt.VTADepPop(cmd,kComputeStage,kLoadStage)

            rt.VTALoadBuffer2D(cmd, a_vta_dram, oc*wl.height*wl.width+iw_offset+wl.width*ih_offset,
                               iw_tile_local, ih_tile_local, wl.width,
                               wpad_before, hpad_before, wpad_after, hpad_after, # padding on each side
                               0, # is pad min value
                               inp_single_buffer_sz*inp_parity, VTA_MEM_ID_INP)

            rt.VTADepPush(cmd, kLoadStage, kComputeStage);


            # ******compute instructions
            rt.VTADepPop(cmd, kLoadStage, kComputeStage);
            if block_count >= buffering:
                rt.VTADepPop(cmd, kStoreStage, kComputeStage);                        
            rt.VTAPushGEMMOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(partial(gemm_kernel, inp_parity, wgt_parity, ow_tile_local, ih_tile_padded_local, iw_tile_padded_local)), ctypes.POINTER(ctypes.c_int)(), 0)

            rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(partial(alu_shr, inp_parity, oh_tile_local, ow_tile_local)), ctypes.POINTER(ctypes.c_int)(), 0)
            rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(partial(alu_add, inp_parity, wgt_parity, oh_tile_local, ow_tile_local)), ctypes.POINTER(ctypes.c_int)(), 0)
            rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(partial(alu_min, inp_parity, oh_tile_local, ow_tile_local)), ctypes.POINTER(ctypes.c_int)(), 0)
            rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(partial(alu_max, inp_parity, oh_tile_local, ow_tile_local)), ctypes.POINTER(ctypes.c_int)(), 0)
            rt.VTADepPush(cmd, kComputeStage, kStoreStage);

            max_block_count = oc_max*oh_max*ow_max

            if block_count < max_block_count - buffering:
                rt.VTADepPush(cmd, kComputeStage, kLoadStage);

            # ***** store instructions
            rt.VTADepPop(cmd, kComputeStage, kStoreStage);
            rt.VTAStoreBuffer2D(cmd, out_single_buffer_sz*inp_parity, VTA_MEM_ID_OUT, # src
                                r_vta_dram, oc*owidth*oheight+ow_offset+owidth*oh_offset, # dest
                                ow_tile_local, oh_tile_local, owidth) # 25 row, 1 column, 25 stride
            if block_count < max_block_count - buffering:
                rt.VTADepPush(cmd, kStoreStage, kComputeStage);

    # ***** run program
    rt.VTASynchronize(cmd, 1000000)
    print(simulator.stats())

    if out_file is not None:
        with open(out_file, 'wt') as fp:
            d = dict(simulator.stats())
            d['oh_lst'] = oh_lst
            d['ow_lst'] = ow_lst
            json.dump( d, fp=fp, indent=2)


    # ***** print arrays/stats
    r_ref = r_ref >> env.WGT_WIDTH
    r_ref += b_np
    r_ref = np.clip(r_ref, 0, (1 << env.OUT_WIDTH - 1) -1)
    r_ref = r_ref.astype(env.out_dtype)

    rt.VTABufferCopy(ctypes.c_void_p(r_vta_dram), 0,
                    r_vta.ctypes.data_as(ctypes.c_void_p), 0,
                    r_vta.size * r_vta.dtype.itemsize,
                    VTA_MEMCPY_D2H)
    r_vta = r_vta.transpose((0, 4, 1, 5, 2, 3)).reshape(r_shape)

    correct = np.allclose(r_ref, r_vta)
    print(wl_name)
    print("PASSED" if correct else "FAILED")
    print("**************")

    # ***** teardown
    rt.VTABufferFree(a_vta_dram)
    rt.VTABufferFree(w_vta_dram)
    rt.VTABufferFree(b_vta_dram)
    rt.VTABufferFree(r_vta_dram)
    simulator.clear_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run depthwise workloads a measure results")
    parser.add_argument( "-w", "--workload", type=str, help="Name of workload")
    parser.add_argument( "-b", "--buffering", type=int, default=2, help="Using single (1) or double (2) buffering.")
    parser.add_argument( "--oh_tile", type=int, help="Set the size of the vertical tiles.")
    parser.add_argument( "--ow_tile", type=int, help="Set the size of the horizontal tiles.")
    parser.add_argument( "-o", "--out_file", type=str, help="Filename for stats.")
    args = parser.parse_args()
    wkl_tbl = dict(wkls)
    wkl = wkl_tbl[args.workload]
    run_depthwise( (args.workload, wkl), buffering=args.buffering, oh_tile_in=args.oh_tile, ow_tile_in=args.ow_tile, out_file=args.out_file)
    rt.VTARuntimeShutdown()
        
