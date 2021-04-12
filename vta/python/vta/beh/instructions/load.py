'''modeling LOAD instruction'''

# Modified by contributors from Intel Labs

import heterocl as hcl

# Opcode, load encoding
VTA_OPCODE_LOAD = 0

# Mem ID constant, uop memory
VTA_MEM_ID_UOP = 0
# Mem ID constant, weight memory
VTA_MEM_ID_WGT = 1
# Mem ID constant, input memory
VTA_MEM_ID_INP = 2
# Mem ID constant, accumulator/bias memory
VTA_MEM_ID_ACC = 3
# Mem ID constant, output store buffer
# VTA_MEM_ID_OUT = 4

ENCODING = {
    'name': 'load',
    'type': 'memory',
    'opcode': VTA_OPCODE_LOAD,
}
#    'top':  load,
    # 'memory_type': {
    #     VTA_MEM_ID_UOP: load_uop,
	# 	VTA_MEM_ID_WGT: load_wgt,
	# 	VTA_MEM_ID_INP: load_inp,
	# 	VTA_MEM_ID_ACC: load_acc
	# }

# load_struct = utils.make_struct(ENCODING)
def load(instr, uop_mem, inp_mem, wgt_mem, acc_mem, dram):
    '''load module'''
    memory_type = hcl.scalar(instr[9:7], name="memory_type").v
    sram_base = hcl.scalar(instr[25:9], name="sram_base")
    dram_base = hcl.scalar(instr[57:25], name="dram_base")
    is_min_pad_value = hcl.scalar(instr[58:57], name="is_pad_min_value")
    # 6 bits empty
    y_size = hcl.scalar(instr[80:64], name="y_size")
    x_size = hcl.scalar(instr[96:80], name="x_size")
    x_stride = hcl.scalar(instr[112:96], name="x_stride")
    y_pad_0 = hcl.scalar(instr[116:112], name="y_pad_0")
    y_pad_1 = hcl.scalar(instr[120:116], name="y_pad_1")
    x_pad_0 = hcl.scalar(instr[124:120], name="x_pad_0")
    x_pad_1 = hcl.scalar(instr[128:124], name="x_pad_1")

    #hcl.print((sram_base, dram_base), "- sram_base: 0x%x dram_base: 0x%x\n")
    #hcl.print((y_size, x_size, x_stride), "- y_size: %d x_size: %d x_stride: %d\n")
    #hcl.print((y_pad_0, y_pad_1, x_pad_0, x_pad_1), "- y_pads %d, %d, x_pads %d, %d\n")

    with hcl.Stage("load"):
        with hcl.if_(x_size.v == 0):
            trace_mgr.Event("EXE", "LNOP %016lx%016lx\n", (instr[128:64], instr[64:0]))
            trace_mgr.Event("RET", "LNOP %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(memory_type == VTA_MEM_ID_UOP):
            trace_mgr.Event("EXE", "LUOP %016lx%016lx\n", (instr[128:64], instr[64:0]))
            load_uop(sram_base, dram_base, x_size, uop_mem, dram)
            trace_mgr.Event("RET", "LUOP %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(memory_type == VTA_MEM_ID_WGT):
            trace_mgr.Event("EXE", "LWGT %016lx%016lx\n", (instr[128:64], instr[64:0]))
            load_2d(sram_base, dram_base, x_size, y_size, x_stride,
                    y_pad_0, y_pad_1, x_pad_0, x_pad_1, wgt_mem, dram, is_min_pad_value)
            trace_mgr.Event("RET", "LWGT %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(memory_type == VTA_MEM_ID_ACC):
            trace_mgr.Event("EXE", "LACC %016lx%016lx\n", (instr[128:64], instr[64:0]))
            load_2d(sram_base, dram_base, x_size, y_size, x_stride,
                    y_pad_0, y_pad_1, x_pad_0, x_pad_1, acc_mem, dram, is_min_pad_value)
            trace_mgr.Event("RET", "LACC %016lx%016lx\n", (instr[128:64], instr[64:0]))
        with hcl.elif_(memory_type == VTA_MEM_ID_INP):
            trace_mgr.Event("EXE", "LINP %016lx%016lx\n", (instr[128:64], instr[64:0]))
            load_2d(sram_base, dram_base, x_size, y_size, x_stride,
                    y_pad_0, y_pad_1, x_pad_0, x_pad_1, inp_mem, dram, is_min_pad_value)
            trace_mgr.Event("RET", "LINP %016lx%016lx\n", (instr[128:64], instr[64:0]))

def load_uop(sram_base, dram_base, x_size, uop_mem, dram):
    '''load uop submodule'''
    ratio, remainder = divmod(hcl.get_bitwidth(uop_mem.dtype), hcl.get_bitwidth(dram.dtype))
    assert remainder == 0, 'we get into trouble'
    with hcl.Stage("load_uop"):
        with hcl.for_(0, x_size, name='x') as x:
            sram_idx = sram_base + x
            dram_idx = (dram_base + x) * ratio
            burst = hcl.compute((ratio,), lambda i: dram[dram_idx+i], name="burst",
                                dtype=dram.dtype)
            burst = hcl.pack(burst, name="uop", factor=ratio)
            uop_mem[sram_idx] = burst[0]
            trace_mgr.Event("LD_SRAM", "%03x %08x\n", (sram_idx, uop_mem[sram_idx]))

def load_2d(sram_base, dram_base, x_size, y_size, x_stride,
            y_pad_0, y_pad_1, x_pad_0, x_pad_1, sram, dram, is_min_pad_value):
    '''
    Load 2D from DRAM into SRAM according to decoded instruction.
    x_pad_0         x_pad_1
    |<-->|          |<->|
    000000000000000000000 | y_pad_0
    000000000000000000000 |
    000000xxxxxxxxxx00000 -
    000000xxxxxxxxxx00000 | y_size
    000000xxxxxxxxxx00000 -
    000000000000000000000 | y_pad_1
    000000000000000000000 |
          |<------>|
           x_size
    '''
    _, nrows, ncols = sram.shape
    #hcl.print((nelem, nrows, ncols), "- shape: nElem= %d, nRows = %d, nCols = %d\n")
    sram_bits, dram_bits = hcl.get_bitwidth(sram.dtype), hcl.get_bitwidth(dram.dtype)
    ratio, remainder = divmod(sram_bits, dram_bits)
    assert remainder == 0, 'we get into trouble'
    trace_fmt = " %08x" if sram_bits == 32 else " %02x"
    with hcl.Stage("load_2d"):
        y_tot = hcl.cast(hcl.UInt(32), y_size.v + y_pad_0.v + y_pad_1.v)
        x_tot = hcl.cast(hcl.UInt(32), x_size.v + x_pad_0.v + x_pad_1.v)
        pad_val = hcl.select(is_min_pad_value.v == 1,\
                        hcl.cast(hcl.Int(16), 1 << (sram_bits - 1)), 0)
        def pad_top(y, x):
            sram_idx = sram_base + x_tot * y + x
            def clear(row, col):
                sram[sram_idx][row][col] = pad_val
            hcl.mutate((nrows, ncols), clear, name='pad_clear')
        def pad_left(y, x):
            sram_idx = sram_base + x_tot * (y_pad_0 + y) + x
            def clear(row, col):
                sram[sram_idx][row][col] = pad_val
            hcl.mutate((nrows, ncols), clear, name='pad_clear')
        def load_data(y, x):
            elem_bytes = nrows * ncols * ratio
            dram_idx = (dram_base + y * x_stride + x) * elem_bytes
            sram_idx = sram_base + x_tot * (y_pad_0 + y) + x + x_pad_0
            tile = hcl.compute((nrows, ncols*ratio),
                               lambda r, c: dram[dram_idx + r*ncols + c],
                               dtype=dram.dtype)
            tile = hcl.pack(tile, axis=1, factor=ratio)
            def copy_tile(row, col):
                sram[sram_idx][row][col] = tile[row][col]
            hcl.mutate((nrows, ncols), copy_tile)
        def pad_right(y, x):
            sram_idx = sram_base + x_tot * (y_pad_0 + y) + x + x_pad_0 + x_size
            def clear(row, col):
                sram[sram_idx][row][col] = pad_val
            hcl.mutate((nrows, ncols), clear, name='pad_clear')
        def pad_bottom(y, x):
            sram_idx = sram_base + x_tot * (y_pad_0 + y_size + y) + x
            def clear(row, col):
                sram[sram_idx][row][col] = pad_val
            hcl.mutate((nrows, ncols), clear, name='pad_clear')
        def trace_sram(y, x):
            sram_idx = sram_base + x_tot * y + x
            def trace_vector(row, col):
                trace_mgr.Event("+LD_SRAM", trace_fmt, sram[sram_idx][row][col])
            trace_mgr.Event("LD_SRAM", "%03x", sram_idx)
            hcl.mutate((nrows, ncols), trace_vector, name='trace_vector')
            trace_mgr.Event("+LD_SRAM", "\n")

        hcl.mutate((y_pad_0, x_tot), pad_top, 'pad_top')
        hcl.mutate((y_size, x_pad_0), pad_left, 'pad_left')
        hcl.mutate((y_size, x_size), load_data, 'load_data')
        hcl.mutate((y_size, x_pad_1), pad_right, 'pad_right')
        hcl.mutate((y_pad_1, x_tot), pad_bottom, 'pad_bottom')
        if trace_mgr.Enabled():
            hcl.mutate((y_tot, x_tot), trace_sram, 'trace_sram')
