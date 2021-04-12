'''instruction encoding definition of VTA for bsim'''

# Modified by contributors from Intel Labs

VTA_LOG_BUS_WIDTH = 6
VTA_LOG_INP_WIDTH = 3
VTA_LOG_WGT_WIDTH = 3
VTA_LOG_ACC_WIDTH = 5
VTA_LOG_OUT_WIDTH = 3
VTA_LOG_BATCH = 0
VTA_LOG_BLOCK = 4
VTA_LOG_BLOCK_IN = VTA_LOG_BLOCK
VTA_LOG_BLOCK_OUT = VTA_LOG_BLOCK
VTA_LOG_UOP_BUFF_SIZE = 15
VTA_LOG_INP_BUFF_SIZE = 15
VTA_LOG_WGT_BUFF_SIZE = 18
VTA_LOG_ACC_BUFF_SIZE = 17

# Memory bus width
VTA_BUS_WIDTH = (1 << VTA_LOG_BUS_WIDTH)

# log2 of instruction data type width
VTA_LOG_INS_WIDTH = 7
# Instruction data type width
VTA_INS_WIDTH = (1 << VTA_LOG_INS_WIDTH)
# log2 of micro op data type width
VTA_LOG_UOP_WIDTH = 5
# Micro Op data type width
VTA_UOP_WIDTH = (1 << VTA_LOG_UOP_WIDTH)
# Weight data type width
VTA_WGT_WIDTH = (1 << VTA_LOG_WGT_WIDTH)
# Input data type width
VTA_INP_WIDTH = (1 << VTA_LOG_INP_WIDTH)
# Output data type width
VTA_OUT_WIDTH = (1 << VTA_LOG_OUT_WIDTH)
# Accumulator data type width
VTA_ACC_WIDTH = (1 << VTA_LOG_ACC_WIDTH)

# Batch size (corresponds to A in (A,B)x(B,C) mat mult)*/
VTA_BATCH = (1 << VTA_LOG_BATCH)
# Blocking factor of inner most loop (corresponds to B in (A,B)x(B,C) mat mult)
VTA_BLOCK_IN = (1 << VTA_LOG_BLOCK_IN)
# Blocking factor of the outer loop (corresponds to C in (A,B)x(B,C) mat mult)
VTA_BLOCK_OUT = (1 << VTA_LOG_BLOCK_OUT)

# On-chip micro-op buffer size in B
VTA_UOP_BUFF_SIZE = (1 << VTA_LOG_UOP_BUFF_SIZE)
# On-chip weight buffer size in B
VTA_WGT_BUFF_SIZE = (1 << VTA_LOG_WGT_BUFF_SIZE)
# On-chip activation buffer size in B
VTA_INP_BUFF_SIZE = (1 << VTA_LOG_INP_BUFF_SIZE)
# On-chip accumulator buffer size in B
VTA_ACC_BUFF_SIZE = (1 << VTA_LOG_ACC_BUFF_SIZE)

# Input vector size in bits
VTA_INP_MATRIX_WIDTH = (VTA_INP_WIDTH * VTA_BATCH * VTA_BLOCK_IN)
# Weight vector size in bits
VTA_WGT_MATRIX_WIDTH = (VTA_WGT_WIDTH * VTA_BLOCK_OUT * VTA_BLOCK_IN)
# Accumulator vector size in bits
VTA_ACC_MATRIX_WIDTH = (VTA_ACC_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
# Output vector size in bits
VTA_OUT_MATRIX_WIDTH = (VTA_OUT_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)

# Ratio between input matrix size and axi width
INP_MAT_AXI_RATIO = (VTA_INP_MATRIX_WIDTH / VTA_BUS_WIDTH)
# Ratio between weight matrix size and axi width
WGT_MAT_AXI_RATIO = (VTA_WGT_MATRIX_WIDTH / VTA_BUS_WIDTH)
# Ratio between accumulator matrix size and axi width
ACC_MAT_AXI_RATIO = (VTA_ACC_MATRIX_WIDTH / VTA_BUS_WIDTH)
# Ratio between output matrix size and axi width
OUT_MAT_AXI_RATIO = (VTA_OUT_MATRIX_WIDTH / VTA_BUS_WIDTH)

# Size of instruction buffer element in B
VTA_INS_ELEM_BYTES = (VTA_INS_WIDTH / 8)
# Size of uop buffer element in B*/
VTA_UOP_ELEM_BYTES = (VTA_UOP_WIDTH / 8)
# Size of activation buffer element in B*/
VTA_INP_ELEM_BYTES = (VTA_INP_MATRIX_WIDTH / 8)
# Size of weight buffer element in B*/
VTA_WGT_ELEM_BYTES = (VTA_WGT_MATRIX_WIDTH / 8)
# Size of accumulator buffer element in B*/
VTA_ACC_ELEM_BYTES = (VTA_ACC_MATRIX_WIDTH / 8)
# Size of output buffer element in B*/
VTA_OUT_ELEM_BYTES = (VTA_OUT_MATRIX_WIDTH / 8)

# On-chip micro-op buffer depth
VTA_UOP_BUFF_DEPTH = (VTA_UOP_BUFF_SIZE / VTA_UOP_ELEM_BYTES)
# log2 of on-chip micro-op buffer depth
VTA_LOG_UOP_BUFF_DEPTH = (VTA_LOG_UOP_BUFF_SIZE - VTA_LOG_UOP_WIDTH + 3)
# On-chip weight buffer depth
VTA_WGT_BUFF_DEPTH = (VTA_WGT_BUFF_SIZE / VTA_WGT_ELEM_BYTES)
# log2 of weight micro-op buffer depth
VTA_LOG_WGT_BUFF_DEPTH = \
  (VTA_LOG_WGT_BUFF_SIZE - VTA_LOG_BLOCK_OUT - VTA_LOG_BLOCK_IN - VTA_LOG_WGT_WIDTH + 3)
# On-chip activation buffer depth
VTA_INP_BUFF_DEPTH = (VTA_INP_BUFF_SIZE / VTA_INP_ELEM_BYTES)
# log2 of activation micro-op buffer depth
VTA_LOG_INP_BUFF_DEPTH = \
  (VTA_LOG_INP_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_IN - VTA_LOG_INP_WIDTH + 3)
# On-chip accumulator buffer depth
VTA_ACC_BUFF_DEPTH = (VTA_ACC_BUFF_SIZE / VTA_ACC_ELEM_BYTES)
# log2 of on-chip accumulator buffer depth
VTA_LOG_ACC_BUFF_DEPTH = \
    (VTA_LOG_ACC_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_OUT - VTA_LOG_ACC_WIDTH + 3)

# Instruction opcode field bitwidth
VTA_OPCODE_BIT_WIDTH = 3
# ALU opcode field bitwidth
VTA_ALU_OPCODE_BIT_WIDTH = 3




# Opcode, finish encoding
VTA_OPCODE_FINISH = 3


# Memory type field bitwidth
VTA_MEMOP_ID_BIT_WIDTH = 2
# Load/Store Instruction, DRAM address width*/
VTA_MEMOP_SRAM_ADDR_BIT_WIDTH = 16
# Load/Store Instruction, DRAM address width*/
VTA_MEMOP_DRAM_ADDR_BIT_WIDTH = 32
# Load/Store Instruction, transfer size width*/
VTA_MEMOP_SIZE_BIT_WIDTH = 16
# Load/Store Instruction, stride size width*/
VTA_MEMOP_STRIDE_BIT_WIDTH = 16
# Load/Store Instruction, padding width*/
VTA_MEMOP_PAD_BIT_WIDTH = 4
# Load/Store Instruction, padding value encoding width*/
VTA_MEMOP_PAD_VAL_BIT_WIDTH = 2
# GEMM/ALU Instruction, loop max iter bits
VTA_LOOP_ITER_WIDTH = 14
# ALU Instruction, immediate bitwidth*/
VTA_ALUOP_IMM_BIT_WIDTH = 16
# ALU Instruction, shift arg bitwidth*/
VTA_SHR_ARG_BIT_WIDTH = (VTA_LOG_ACC_WIDTH)
# ALU Instruction, multiply arg bitwidth*/
VTA_MUL_ARG_BIT_WIDTH = 8

# GEMM Micro-op start position of the acc_idx field
VTA_UOP_GEM_0_0 = 0
# GEMM Micro-op end position of the acc_idx field
VTA_UOP_GEM_0_1 = (VTA_UOP_GEM_0_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
# GEMM Micro-op start position of the inp_idx field
VTA_UOP_GEM_1_0 = (VTA_UOP_GEM_0_1 + 1)
# GEMM Micro-op end position of the inp_idx field
VTA_UOP_GEM_1_1 = (VTA_UOP_GEM_1_0 + VTA_LOG_INP_BUFF_DEPTH - 1)
# GEMM Micro-op start position of the wgt_idx field
VTA_UOP_GEM_2_0 = (VTA_UOP_GEM_1_1 + 1)
# GEMM Micro-op end position of the wgt_idx field
VTA_UOP_GEM_2_1 = (VTA_UOP_GEM_2_0 + VTA_LOG_WGT_BUFF_DEPTH - 1)

# GEMM Micro-op start position of the acc_idx field
VTA_UOP_ALU_0_0 = 0
# GEMM Micro-op end position of the acc_idx field
VTA_UOP_ALU_0_1 = (VTA_UOP_ALU_0_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
# GEMM Micro-op start position of the inp_idx field
VTA_UOP_ALU_1_0 = (VTA_UOP_ALU_0_1 + 1)
# GEMM Micro-op end position of the inp_idx field
VTA_UOP_ALU_1_1 = (VTA_UOP_ALU_1_0 + VTA_LOG_INP_BUFF_DEPTH - 1)

# instruction format requirements:
# - first element must be a tuple
# - at most one dictionary (for union purpose)
INSTRUCTION_FMT = (
    ('opcode', VTA_OPCODE_BIT_WIDTH),
    ('pop_prev_dep', 1),
    ('pop_next_dep', 1),
    ('push_prev_dep', 1),
    ('push_next_dep', 1),
    {
        'memory': (
            #Source/destination SRAM for store/load instruction
            ('memory_type', VTA_MEMOP_ID_BIT_WIDTH),
            # SRAM base address (pointer to memory elem type)
            ('sram_base', VTA_MEMOP_SRAM_ADDR_BIT_WIDTH),
            # DRAM base address (pointer to memory elem type)
            ('dram_base', VTA_MEMOP_DRAM_ADDR_BIT_WIDTH),
            ('empty', 7), # ?? C bitfield packing hack?
            # 2D access pattern: y-size
            ('y_size', VTA_MEMOP_SIZE_BIT_WIDTH),
            # 2D access pattern: x-size (in terms of memory elements)
            ('x_size', VTA_MEMOP_SIZE_BIT_WIDTH),
            # 2D access pattern: x-stride (in terms of memory elements)
            ('x_stride', VTA_MEMOP_STRIDE_BIT_WIDTH),
            # 2D access pattern: start padding along y dimension
            ('y_pad_0', VTA_MEMOP_PAD_BIT_WIDTH),
            # 2D access pattern: end padding along y dimension
            ('y_pad_1', VTA_MEMOP_PAD_BIT_WIDTH),
            # 2D access pattern: start padding along x dimension
            ('x_pad_0', VTA_MEMOP_PAD_BIT_WIDTH),
            # 2D access pattern: end padding along x dimension
            ('x_pad_1', VTA_MEMOP_PAD_BIT_WIDTH)
        ),
        'alu' : (
            # Reset register
            ('reset_reg', 1),
            # Micro-op begin address
            ('uop_bgn', VTA_LOG_UOP_BUFF_DEPTH),
            # Micro-op end address
            ('uop_end', VTA_LOG_UOP_BUFF_DEPTH + 1),
            # Iterations in the outer uop execution loop
            ('iter_out', VTA_LOOP_ITER_WIDTH),
            # Iterations in the inner uop execution loop
            ('iter_in', VTA_LOOP_ITER_WIDTH),
            ('empty', 1), # ?? C bitfield packing hack?
            # Outer loop accumulator memory destination index factor
            ('dst_factor_out', VTA_LOG_ACC_BUFF_DEPTH),
            # Inner loop accumulator memory destination index factor
            ('dst_factor_in', VTA_LOG_ACC_BUFF_DEPTH),
            # Outer loop accumulator memory source index factor
            ('src_factor_out', VTA_LOG_INP_BUFF_DEPTH),
            # Inner loop accumulator memory source index factor
            ('src_factor_in', VTA_LOG_INP_BUFF_DEPTH),
            # ALU opcode
            ('alu_opcode', VTA_ALU_OPCODE_BIT_WIDTH),
            # Use immediate is true
            ('use_imm', 1),
            # Immediate value: allow negative value
            ('imm', VTA_ALUOP_IMM_BIT_WIDTH)
        ),
        'gemm': (
            # Reset register
            ('reset_reg', 1),
            # Micro-op begin address
            ('uop_bgn', VTA_LOG_UOP_BUFF_DEPTH),
            # Micro-op end address
            ('uop_end', VTA_LOG_UOP_BUFF_DEPTH + 1),
            # Iterations in the outer uop execution loop
            ('iter_out', VTA_LOOP_ITER_WIDTH),
            # Iterations in the inner uop execution loop
            ('iter_in', VTA_LOOP_ITER_WIDTH),
            ('empty', 1), # ?? C bitfield packing hack?
            # Outer loop accumulator memory index factor
            ('dst_factor_out', VTA_LOG_ACC_BUFF_DEPTH),
            # Inner loop accumulator memory index factor
            ('dst_factor_in', VTA_LOG_ACC_BUFF_DEPTH),
            # Outer loop input memory index factor
            ('src_factor_out', VTA_LOG_INP_BUFF_DEPTH),
            # Inner loop input memory index factor
            ('src_factor_in', VTA_LOG_INP_BUFF_DEPTH),
            # Outer loop weight memory index factor
            ('wgt_factor_out', VTA_LOG_WGT_BUFF_DEPTH),
            # Inner loop weight memory index factor
            ('wgt_factor_in', VTA_LOG_WGT_BUFF_DEPTH)
        )
    }
)
