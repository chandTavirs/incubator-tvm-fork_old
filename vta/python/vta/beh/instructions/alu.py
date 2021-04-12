'''modeling ALU instruction'''

# Modified by contributors from Intel Labs

import heterocl as hcl
# from .. import config, utils

# Opcode, ALU encoding
VTA_OPCODE_ALU = 4

# ALU opcode, unary min op
VTA_ALU_OPCODE_MIN = 0
# ALU opcode, unary max op
VTA_ALU_OPCODE_MAX = 1
# ALU opcode, binary add op
VTA_ALU_OPCODE_ADD = 2
# ALU opcode, shift right by immediate op
VTA_ALU_OPCODE_SHR = 3
# ALU opcode, clp between -immediate and immediate
VTA_ALU_OPCODE_CLP = 4
# ALU opcode, mov from src/imm into dest
VTA_ALU_OPCODE_MOV = 5

ENCODING = {
    'name': 'alu',
    'type': 'alu',
    'opcode': VTA_OPCODE_ALU
}

# alu_struct = utils.make_struct(encoding)

# for (i = 0; i < iter_out; i++) {
#   for (j = 0; j < iter_in; j++) {
#     for (k = uop_bgn; k < uop_end; k++) {
#       // Read micro op
#       uop_T uop = uop_mem[k];
#       // Read in memory indices
#       acc_idx_T dst_idx = uop.dst_idx;
#       inp_idx_T src_idx = uop.inp_idx;
#       // Update those indices with the following affine functions
#       dst_idx += iter_in * dst_factor_in + iter_out * dst_factor_out;
#       src_idx += iter_in * src_factor_in + iter_out * src_factor_out;
#       // Perform ALU operation
#       if (use_imm) {
#         acc_mem[dst_idx] = alu_op(alu_opcode, acc_mem[dst_idx], imm);
#       } else {
#         acc_mem[dst_idx] = alu_op(alu_opcode, acc_mem[dst_idx], acc_mem[src_idx]);
#       }

def decode_uop(uop):
    acc_idx = hcl.scalar(uop[11:0], dtype=hcl.UInt(16))  # dst 11-bit
    inp_idx = hcl.scalar(uop[22:11], dtype=hcl.UInt(16))
    wgt_idx = hcl.scalar(uop[32:22], dtype=hcl.UInt(16))
    return acc_idx, inp_idx, wgt_idx

def alu(instr, uop_mem, acc_mem, out_mem):
    '''alu instruction'''
    uop_bgn = hcl.scalar(instr[21:8], name="uop_bgn")
    uop_end = hcl.scalar(instr[35:21], name="uop_end")
    iter_out = hcl.scalar(instr[49:35], name="iter_out")
    iter_in = hcl.scalar(instr[63:49], name="iter_in")
    # empty = hcl.scalar(instr[64:63], name="empty")
    dst_factor_out = hcl.scalar(instr[75:64], name="dst_factor_out")
    dst_factor_in = hcl.scalar(instr[86:75], name="dst_factor_in")
    src_factor_out = hcl.scalar(instr[97:86], name="src_factor_out")
    src_factor_in = hcl.scalar(instr[108:97], name="src_factor_in")
    alu_opcode = hcl.scalar(instr[111:108], name="alu_opcode") # update to 3 bits
    use_imm = hcl.scalar(instr[112:111], name="use_imm", dtype=hcl.UInt(1))
    imm = hcl.scalar(instr[128:112], name="imm")

    # Compute blocks such as the following one that are only used for tracing
    # can be made conditional to the trace being enabled.
    if trace_mgr.Enabled():
        with hcl.if_(alu_opcode.v == VTA_ALU_OPCODE_MIN):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("EXE", "MINI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("EXE", "MIN  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_MAX):
            with hcl.if_(use_imm == 1):
                trace_mgr.Event("EXE", "MAXI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("EXE", "MAX  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_ADD):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("EXE", "ADDI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("EXE", "ADD  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_SHR):
            with hcl.if_(instr[127] == 1):
                with hcl.if_(use_imm.v == 1):
                    trace_mgr.Event("EXE", "SHLI %016lx%016lx\n", (instr[128:64], instr[64:0]))
                with hcl.else_():
                    trace_mgr.Event("EXE", "SHL  %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                with hcl.if_(use_imm.v == 1):
                    trace_mgr.Event("EXE", "SHRI %016lx%016lx\n", (instr[128:64], instr[64:0]))
                with hcl.else_():
                    trace_mgr.Event("EXE", "SHR  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_CLP):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("EXE", "CLPI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("EXE", "CLP  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_MOV):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("EXE", "MOVI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("EXE", "MOV  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.else_():
            trace_mgr.Event("EXE", "CNOP %016lx%016lx\n")

    trace_mgr.Event("ALU_LOOP", "%04x %04x %04x %04x\n",
                    (iter_out, iter_in, uop_bgn, uop_end))

    _, nrows, ncols = acc_mem.shape
    def fmutate(i, j, k):  # i <--iter_out   j <-- iter_in   k <-- 0 to (uop_end-uop_bgn)
        k += uop_bgn
        uop = uop_mem[k]
        acc_idx, inp_idx, _ = decode_uop(uop)
        acc_idx += j * hcl.cast(hcl.UInt(16), dst_factor_in.v) + \
                   i * hcl.cast(hcl.UInt(16), dst_factor_out.v)
        inp_idx += j * hcl.cast(hcl.UInt(16), src_factor_in.v) + \
                   i * hcl.cast(hcl.UInt(16), src_factor_out.v)
        dst_tensor, src_tensor = acc_mem[acc_idx], acc_mem[inp_idx]
        trace_mgr.Event("ALU_ITR", "%04x %04x %04x %03x", (i, j, k, acc_idx))

        with hcl.for_(0, nrows) as x:
            with hcl.for_(0, ncols) as y:
                # Comment out for bug with 32 bit signed immediate.
                #src = hcl.select(use_imm.v == 1, hcl.cast(hcl.Int(32), imm),
                src = hcl.select(use_imm.v == 1, hcl.cast(hcl.Int(16), imm),
                                 hcl.cast(hcl.Int(32), src_tensor[x][y]))
                dst = hcl.cast(hcl.Int(32), dst_tensor[x][y])
                with hcl.if_(alu_opcode.v == VTA_ALU_OPCODE_MIN):
                    dst_tensor[x][y] = hcl.select(dst <= src, dst_tensor[x][y], src)
                with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_MAX):
                    dst_tensor[x][y] = hcl.select(dst >= src, dst_tensor[x][y], src)
                with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_ADD):
                    dst_tensor[x][y] = dst + src
                with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_SHR):
                    dst_tensor[x][y] = hcl.select(src >= 0, dst >> src, dst << -src)
                with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_CLP):
                    dst_tensor[x][y] = hcl.select(dst >= (-1 * src),
                                                  hcl.select(dst <= src, dst_tensor[x][y], src),
                                                  (-1 * src))
                with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_MOV):
                    dst_tensor[x][y] = src
                with hcl.else_():
                    hcl.print(alu_opcode, 'ERROR: unknown alu_opcode: %d\n')
                    dst_tensor[x][y] = 0
                trace_mgr.Event("+ALU_ITR", ' %08x', dst_tensor[x][y])
        trace_mgr.Event("+ALU_ITR", "\n")

        def fmutate_out(row, col):
            out_mem[acc_idx][row][col] = hcl.cast(out_mem.dtype, acc_mem[acc_idx][row][col])
        hcl.mutate((nrows, ncols), fmutate_out)

    with hcl.Stage('alu'):
        domain = (hcl.cast(hcl.UInt(32), iter_out.v), hcl.cast(hcl.UInt(32), iter_in.v),
                  (hcl.cast(hcl.UInt(32), uop_end.v-uop_bgn.v)))
        hcl.mutate(domain, fmutate)

    if trace_mgr.Enabled():
        with hcl.if_(alu_opcode.v == VTA_ALU_OPCODE_MIN):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("RET", "MINI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("RET", "MIN  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_MAX):
            with hcl.if_(use_imm == 1):
                trace_mgr.Event("RET", "MAXI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("RET", "MAX  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_ADD):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("RET", "ADDI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("RET", "ADD  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_SHR):
            with hcl.if_(instr[127] == 1):
                with hcl.if_(use_imm.v == 1):
                    trace_mgr.Event("RET", "SHLI %016lx%016lx\n", (instr[128:64], instr[64:0]))
                with hcl.else_():
                    trace_mgr.Event("RET", "SHL  %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                with hcl.if_(use_imm.v == 1):
                    trace_mgr.Event("RET", "SHRI %016lx%016lx\n", (instr[128:64], instr[64:0]))
                with hcl.else_():
                    trace_mgr.Event("RET", "SHR  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_CLP):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("RET", "CLPI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("RET", "CLP  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(alu_opcode.v == VTA_ALU_OPCODE_MOV):
            with hcl.if_(use_imm.v == 1):
                trace_mgr.Event("RET", "MOVI %016lx%016lx\n", (instr[128:64], instr[64:0]))
            with hcl.else_():
                trace_mgr.Event("RET", "MOV  %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.else_():
            trace_mgr.Event("RET", "CNOP %016lx%016lx\n")
