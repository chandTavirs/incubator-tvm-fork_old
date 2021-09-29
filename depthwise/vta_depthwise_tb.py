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
# simulator.debug_mode(0) # execute functionality
simulator.clear_stats()
# print(simulator.stats())
cmd = rt.VTATLSCommandHandle()
rt.VTASetDebugMode(cmd, VTA_DEBUG_FORCE_SERIAL + VTA_DEBUG_DUMP_UOP + VTA_DEBUG_DUMP_INSN)

# Workload
env = vta.get_env()

Workload = namedtuple(
    "DepthwiseConv2DWorkload",
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

wkls = [("depthwise_stride1.small", Workload(env.BATCH, 5, 5, 16, 1, 3, 3, 1, 1, 1, 1)),
       ("depthwise_stride2.small", Workload(env.BATCH, 5, 5, 16, 1, 3, 3, 1, 1, 2, 2)),
]

def run_depthwise(wkl, count=10):
    for _ in range(count):
        # Variable shapes
        wl_name, wl = wkl
        a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
        w_orig_shape = (wl.in_filter, wl.channel_multiplier, wl.hkernel, wl.wkernel)
        w_shape = (wl.in_filter*wl.channel_multiplier, 16, 1, 1)
        b_shape = (wl.batch, wl.in_filter * wl.channel_multiplier, 1, 1)
        a_pack = (wl.batch//env.BATCH, wl.in_filter//env.BLOCK_IN,
                wl.height, wl.width,
                env.BATCH, env.BLOCK_IN)
        w_pack = (wl.in_filter//env.BLOCK_OUT, 16//env.BLOCK_IN,
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
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min, b_max = 0 - 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2), 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2)
        a_vta = np.random.randint(a_min, a_max, size=a_pack).astype(np.int8)
        w_np = np.random.randint(w_min, w_max, size=w_orig_shape).astype(np.int8)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(np.int32)
        a_np = a_vta.transpose((0, 4, 1, 5, 2, 3)).reshape(a_shape)
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


        # Data packing for VTA 
        # ctypes void pointers don't respect transpose operators, DRAM layout don't change
        # Still retaining weight and bias transpose, as height and width are 1
        w_np = w_np.reshape(wl.in_filter, wl.hkernel*wl.wkernel, 1, 1)
        w_np_vta = np.zeros(w_shape).astype(np.int8)
        w_np_vta[:, :9, :, :] = w_np
        w_vta = w_np_vta.reshape(
            w_pack[0], env.BLOCK_OUT,
            w_pack[1], env.BLOCK_IN,
            w_pack[2], w_pack[3]
            ).transpose((0, 2, 4, 5, 1 ,3))
        b_vta = b_np.reshape(
            b_shape[0]//env.BATCH, env.BATCH,
            b_shape[1]//env.BLOCK_OUT, env.BLOCK_OUT,
            b_shape[2], b_shape[3]
            ).transpose((0, 2, 4, 5, 1, 3))
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

        # ***** load instructions

        if wl.hpad:
            rt.VTALoadBuffer2D(cmd, a_vta_dram, 0,
                            wl.height, wl.width, wl.width,
                            wl.hpad, wl.hpad, wl.wpad, wl.wpad, # padding on each side
                            0, # is pad min value
                            0, VTA_MEM_ID_INP)
        else:
            rt.VTALoadBuffer2D(cmd, a_vta_dram, 0,
                            wl.height*wl.width, 1, wl.height*wl.width,
                            wl.hpad, wl.hpad, wl.wpad, wl.wpad, # padding on each side
                            0, # is pad min value
                            0, VTA_MEM_ID_INP)

        rt.VTALoadBuffer2D(cmd, w_vta_dram, 0,
                        1, 1, 1, # 1 row, 1 column, 1 stride
                        0, 0, 0, 0, # no padding on each side
                        0, # is pad min value
                        0, VTA_MEM_ID_WGT)

        rt.VTALoadBuffer2D(cmd, b_vta_dram, 0,
                        1, 1, 1, # 1 row, 1 column, 0 stride
                        0, 0, 0, 0, # no padding on each side
                        0, # is pad min value
                        oheight*owidth, VTA_MEM_ID_ACC)

        # ***** compute instructions

        def gemm_reset(a):
            rt.VTAUopLoopBegin(oheight, 1, # extent = 5, dst_factor = 1
                        0, 0) # src_factor = 0, wgt_factor = 0
            rt.VTAUopLoopBegin(owidth, oheight, # extent = 5, dst_factor = 5
                        0, 0) # src_factor = 0, wgt_factor = 0
            rt.VTAUopPush(0, 1, # Gemm mode with reset
                        0, 0, 0, # dest = 0, inp = 0, wgt = 0
                        0, 0,
                        0)
            rt.VTAUopLoopEnd()
            rt.VTAUopLoopEnd()
            return 0

        def gemm_kernel(a):
            rt.VTAUopLoopBegin(wl.height+2*wl.hpad, oheight, # extent = 7, dst_factor = 5
                        wl.width+2*wl.wpad, 0) # src_factor = 7, wgt_factor = 0
            rt.VTAUopLoopBegin(wl.width+2*wl.wpad, 1, # extent = 7, dst_factor = 1
                        1, 0) # src_factor = 1, wgt_factor = 0
            if wl.hstride == 1:
                rt.VTAUopPush(0, 2, # Gemm mode with depthwise stride 1
                            0, 0, 0, # dest = 0, inp = 0, wgt = 0
                            0, 0, # don't care opcode, don't care imm
                            0) # imm = don't care
            elif wl.wstride == 2:
                rt.VTAUopPush(0, 3, # Gemm mode with depthwise stride 2
                            0, 0, 0, # dest = 0, inp = 0, wgt = 0
                            0, 0, # don't care opcode, don't care imm
                            0) # imm = don't care
            rt.VTAUopLoopEnd()
            rt.VTAUopLoopEnd()
            return 0

        # rt.VTAPushGEMMOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(gemm_reset), ctypes.POINTER(ctypes.c_int)(), 0)
        rt.VTAPushGEMMOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(gemm_kernel), ctypes.POINTER(ctypes.c_int)(), 0)

        def alu_add(a):
            rt.VTAUopLoopBegin(oheight*owidth, 1, # extent = 25, dst_factor = 1
                        0, 0) # src_factor = 1, wgt_factor = 0
            rt.VTAUopPush(1, 0, # ALU mode with no reset
                    0, oheight*owidth, 0, # dest = 0, src = 25, wgt = don't care
                    VTA_ALU_OPCODE_ADD, 0, # opcode and imm used?
                    0) # imm = 0
            rt.VTAUopLoopEnd()
            return 0

        def alu_shr(a):
            rt.VTAUopLoopBegin(oheight*owidth, 1, # extent = 25, dst_factor = 1
                        1, 0) # src_factor = 1, wgt_factor = 0
            rt.VTAUopPush(1, 0, # ALU mode with no reset
                    0, 0, 0, # dest = 0, src = 0, wgt = don't care
                    VTA_ALU_OPCODE_SHR, 1, # opcode and imm used?
                    8) # imm = 8
            rt.VTAUopLoopEnd()
            return 0

        def alu_min(a):
            rt.VTAUopLoopBegin(oheight*owidth, 1, # extent = 25, dst_factor = 1
                        1, 0) # src_factor = 1, wgt_factor = 0
            rt.VTAUopPush(1, 0, # ALU mode with no reset
                    0, 0, 0, # dest = 1, src = 0, wgt = don't care
                    VTA_ALU_OPCODE_MIN, 1, # opcode and imm used?
                    127) # imm = 127
            rt.VTAUopLoopEnd()
            return 0

        def alu_max(a):
            rt.VTAUopLoopBegin(oheight*owidth, 1, # extent = 25, dst_factor = 1
                        1, 0) # src_factor = 1, wgt_factor = 0
            rt.VTAUopPush(1, 0, # ALU mode with no reset
                    0, 0, 0, # dest = 0, src = 0, wgt = don't care
                    VTA_ALU_OPCODE_MAX, 1, # opcode and imm used?
                    0) # imm = 0
            rt.VTAUopLoopEnd()
            return 0

        rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(alu_shr), ctypes.POINTER(ctypes.c_int)(), 0)
        rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(alu_add), ctypes.POINTER(ctypes.c_int)(), 0)
        rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(alu_min), ctypes.POINTER(ctypes.c_int)(), 0)
        rt.VTAPushALUOp(ctypes.byref(ctypes.c_int()), FINIT_FUNC_SIG(alu_max), ctypes.POINTER(ctypes.c_int)(), 0)

        # ***** store instructions

        rt.VTAStoreBuffer2D(cmd, 0, VTA_MEM_ID_OUT, # src
                            r_vta_dram, 0, # dest
                            oheight*owidth, 1, oheight*owidth) # 25 row, 1 column, 25 stride

        # ***** run program
        print("Running VTA program\n")
        rt.VTASynchronize(cmd, 10000)
        print(simulator.stats())

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
        print("PASSED") if correct else print("FAILED")
        print("**************")

        # ***** teardown
        rt.VTABufferFree(a_vta_dram)
        rt.VTABufferFree(w_vta_dram)
        rt.VTABufferFree(b_vta_dram)
        rt.VTABufferFree(r_vta_dram)
        simulator.clear_stats()

if __name__ == "__main__":
    for wkl in wkls:
        run_depthwise(wkl, 1)
    rt.VTARuntimeShutdown()
