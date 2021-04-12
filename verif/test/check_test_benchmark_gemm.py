"""
Verification Test Module.
Wrapper for vta/tests/python/integration/test_benchmark_gemm.py.
Accepts all targets but cpu.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_benchmark_gemm')

# May also parametrize batch, channel, block.
dims = [(2,16,16),(2,32,32),(128,128,128)]
targets = [t for t in selected_targets if t != 'cpu']
skip_test_module(not targets, 'NO_AVAILABLE_TARGETS')
items = ('E2E', 'GEMM', 'ALU', 'LD_INP', 'LD_WGT', 'ST_OUT')
params = [(i, batch, channel, block, t) for t in targets for i in items for (batch,channel,block) in dims]

@pytest.mark.parametrize('item,batch,channel,block,target', params)
def test_benchmark_gemm(item, batch, channel, block, target, testid):
  with config_target(target):
    from .test_benchmark_gemm import test_gemm
    with config_trace(*testid):
      test_gemm(item,batch=batch,channel=channel,block=block)

