'''behavorial simulation model'''

# Modified by contributors from Intel Labs

from . import config, utils, state
from .beh import DRAM_ARRAY, beh_model
from .instructions import load, store, gemm, alu
