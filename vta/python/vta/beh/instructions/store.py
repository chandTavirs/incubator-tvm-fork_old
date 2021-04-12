'''modeling STORE instruction'''

# Modified by contributors from Intel Labs

import heterocl as hcl

# Opcode, store encoding
VTA_OPCODE_STORE = 1

ENCODING = {
    'name': 'store',
    'type': 'memory',
    'opcode': VTA_OPCODE_STORE
}

# store_struct = utils.make_struct(encoding)
def store(instr, out_mem, dram):
    '''store instruction'''
    sram_base = hcl.scalar(instr[25:9], name="sram_base")
    dram_base = hcl.scalar(instr[57:25], name="dram_base")
    y_size = hcl.scalar(instr[80:64], name="y_size")
    x_size = hcl.scalar(instr[96:80], name="x_size")
    x_stride = hcl.scalar(instr[112:96], name="x_stride")

    #hcl.print(sram_base, "- sram_base: 0x%x\n")
    #hcl.print(dram_base, "- dram_base: 0x%x\n")
    #hcl.print(y_size, "- y_size: %d\n")
    #hcl.print(x_size, "- x_size: %d\n")
    with hcl.if_(x_size.v == 0):
        trace_mgr.Event("EXE", "SNOP %016lx%016lx\n", (instr[128:64], instr[64:0]))
        trace_mgr.Event("RET", "SNOP %016lx%016lx\n", (instr[128:64], instr[64:0]))
    with hcl.else_():
        trace_mgr.Event("EXE", "SOUT %016lx%016lx\n", (instr[128:64], instr[64:0]))
        store_2d(sram_base, dram_base, x_size, y_size, x_stride, out_mem, dram)
        trace_mgr.Event("RET", "SOUT %016lx%016lx\n", (instr[128:64], instr[64:0]))

def store_2d(sram_base, dram_base, x_size, y_size, x_stride, sram, dram):
    '''sub-block of store'''
    _, nrows, ncols = sram.shape
    #hcl.print((nrows, ncols), "- nRows: %d nCols: %d\n")
    # ratio = hcl.get_bitwidth(sram.dtype) // hcl.get_bitwidth(dram.dtype)
    with hcl.Stage("store"):
        def fmutate(y, x):
            tile = sram[sram_base + y*x_size + x]
            dram_idx = (dram_base + y*x_stride + x) * nrows * ncols
            def move_tile(row, col):
                dram[dram_idx + row*ncols + col] = tile[row][col]
            hcl.mutate((nrows, ncols), move_tile)
        hcl.mutate((y_size, x_size), fmutate, 'store_data')
        def trace_sram(y, x):
            sram_idx = sram_base + x_size * y + x
            def trace_vector(row, col):
                trace_mgr.Event("+ST_SRAM", " %02x", sram[sram_idx][row][col])
            trace_mgr.Event("ST_SRAM", "%03x", sram_idx)
            hcl.mutate((nrows, ncols), trace_vector, name='trace_vector')
            trace_mgr.Event("+ST_SRAM", "\n")
        if trace_mgr.Enabled():
            hcl.mutate((y_size, x_size), trace_sram, 'trace_sram')
