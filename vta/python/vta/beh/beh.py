'''VTA Behavioral Simulation model using HeteroCL'''

# Modified by contributors from Intel Labs

import numpy as np
import heterocl as hcl
import tvm
from .instructions import store, gemm, alu, load, finish
from . import config, state

DRAM_ARRAY = tvm.nd.array(np.zeros(config.DRAM_SIZE, dtype=np.uint8))

# instr_struct = hcl.Struct({'opcode': hcl.UInt(3), 'unused': hcl.UInt(5)})
# ts = hcl.Struct({"opcode": hcl.Int(8), "fb": hcl.Fixed(8, 2), "fc": hcl.Float()})

def core(instr, uop_mem, inp_mem, wgt_mem, acc_mem, out_mem, dram):
    '''VTA core'''
    opcode = hcl.scalar(instr[3:0], name='opcode')
    #hcl.print(opcode, 'opcode: %d\n')
    with hcl.Stage("S"):
        with hcl.if_(opcode == load.VTA_OPCODE_LOAD):
            #hcl.print(0, "Load Instruction\n")
            load.load(instr, uop_mem, inp_mem, wgt_mem, acc_mem, dram)
        with hcl.elif_(opcode == store.VTA_OPCODE_STORE):
            #hcl.print(0, "Store Instruction\n")
            store.store(instr, out_mem, dram)
        with hcl.elif_(opcode == alu.VTA_OPCODE_ALU):
            #hcl.print(0, "ALU Instruction\n")
            alu.alu(instr, uop_mem, acc_mem, out_mem)
        with hcl.elif_(opcode == gemm.VTA_OPCODE_GEMM):
            #hcl.print(0, "GEMM Instruction\n")
            gemm.gemm(instr, uop_mem, inp_mem, wgt_mem, acc_mem, out_mem)
        with hcl.elif_(opcode == finish.VTA_OPCODE_FINISH):
            trace_mgr.Event("EXE", "FIN  %016lx%016lx\n", (instr[128:64], instr[64:0]))
            trace_mgr.Event("RET", "FIN  %016lx%016lx\n", (instr[128:64], instr[64:0]))

def vta(instr_phy_addr, instr_count, uop_mem, inp_mem, wgt_mem, acc_mem, out_mem, dram):
    '''VTA BSIM Shell'''
    ratio = 128 // 8
    with hcl.Stage("S"):
        base = hcl.scalar(instr_phy_addr, name='instr_phy_addr')
        with hcl.for_(0, hcl.scalar(instr_count, name='instr_count'), name="i") as i:
            #hcl.print(i, '\n# Instruction: %d\n')
            trace_mgr.Event("INSN_IDX", "%4x\n", i)
            inbytes = hcl.compute((ratio,), lambda x: dram[base+x], 'insn_bytes', dtype=dram.dtype)
            insn = hcl.pack(inbytes, name='insn', factor=ratio)
            core(insn[0], uop_mem, inp_mem, wgt_mem, acc_mem, out_mem, dram)
            base.v = base.v + ratio

def top(target=None):
    '''top module of the vta'''
    hcl.init()
    instr_phy_addr = hcl.placeholder((), name="instr_phy_addr", dtype=hcl.UInt(32))
    instr_count = hcl.placeholder((), name="instr_count", dtype=hcl.UInt(32))
    inputs = [instr_phy_addr, instr_count, state.UOP_MEM, state.INP_MEM, state.WGT_MEM,
              state.ACC_MEM, state.OUT_MEM, state.DRAM]
    s = hcl.create_schedule(inputs, vta)
    # print('-- printing lower:')
    # print(hcl.lower(s))
    m = hcl.build(s, target)
    # print(m.get_source('ll'))
    # print('-- printing model:\n', m)
    return m

def beh_model(insn_phy_addr, insn_count):
    '''behavioral simulation model to be called by the bsim_driver'''
    f = top()
    uop_mem = hcl.asarray(np.zeros(state.UOP_MEM.shape), dtype=state.UOP_MEM.dtype)
    inp_mem = hcl.asarray(np.zeros(state.INP_MEM.shape), dtype=state.INP_MEM.dtype)
    wgt_mem = hcl.asarray(np.zeros(state.WGT_MEM.shape), dtype=state.WGT_MEM.dtype)
    acc_mem = hcl.asarray(np.zeros(state.ACC_MEM.shape), dtype=state.ACC_MEM.dtype)
    out_mem = hcl.asarray(np.zeros(state.OUT_MEM.shape), dtype=state.OUT_MEM.dtype)

    #print('bsim model: executing...')
    f(insn_phy_addr, insn_count, uop_mem, inp_mem, wgt_mem, acc_mem, out_mem, DRAM_ARRAY)
    #print('bsim model: done...')
