'''configuration of bsim'''

# Modified by contributors from Intel Labs

from ..environment import get_env

env = get_env() # pylint: disable=invalid-name

LOG_DATA_WIDTH = 3
LOG_LONG_DATA_WIDTH = 5  # TODO: remove me

LOG_UOP_WIDTH = 5
UOP_WIDTH = 1 << LOG_UOP_WIDTH # 32 bits in 4 bytes: [0,2] acc_idx, [3:5] inp_idx, [6:10] wgt_idx

LOG_DRAM_SIZE = 28
DRAM_SIZE = 1 << LOG_DRAM_SIZE # large memory

DATA_WIDTH = 1 << LOG_DATA_WIDTH # 8 bits in 1 byte
LONG_DATA_WIDTH = 1 << LOG_LONG_DATA_WIDTH # 32 bits in 4 bytes
