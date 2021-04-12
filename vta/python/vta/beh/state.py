'''internal state of the machine'''

# Modified by contributors from Intel Labs

import heterocl as hcl
from .config import env
from . import config

INP_MEM = hcl.placeholder((env.INP_BUFF_SIZE, env.BATCH, env.BLOCK_IN,),
                          name='inp_mem', dtype=hcl.UInt(config.DATA_WIDTH))

WGT_MEM = hcl.placeholder((env.WGT_BUFF_SIZE, env.BLOCK_IN, env.BLOCK_OUT,),
                          name='wgt_mem', dtype=hcl.UInt(config.DATA_WIDTH))

ACC_MEM = hcl.placeholder((env.ACC_BUFF_SIZE, env.BATCH, env.BLOCK_OUT,),
                          name='acc_mem', dtype=hcl.UInt(config.LONG_DATA_WIDTH))

UOP_MEM = hcl.placeholder((env.UOP_BUFF_SIZE,),
                          name='uop_mem', dtype=hcl.UInt(config.UOP_WIDTH))

OUT_MEM = hcl.placeholder((env.OUT_BUFF_SIZE, env.BATCH, env.BLOCK_OUT,),
                          name='out_mem', dtype=hcl.UInt(config.DATA_WIDTH))

DRAM = hcl.placeholder((config.DRAM_SIZE,), name='DRAM', dtype=hcl.UInt(8))
