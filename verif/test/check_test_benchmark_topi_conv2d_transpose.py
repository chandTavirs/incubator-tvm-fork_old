"""
Verification Test Module.
Wrapper of vta/tests/python/integration/test_benchmark_topi_conv2d_transpose.py.
Accepts all targets.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, t2d, t2t, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_benchmark_topi_conv2d_transpose')

targets = selected_targets
skip_test_module(not targets, 'NO_AVAILABLE_TARGETS')

items = (
    'DCGAN.CT1',
    'DCGAN.CT2',
    'DCGAN.CT3',
)
params = [(i, t) for t in targets for i in items]

@pytest.mark.parametrize('workload,target', params)
def test_conv2d_transpose(workload, target, testid):
  with config_target(t2t[target]):
    from .test_benchmark_topi_conv2d_transpose import test_conv2d_transpose
    with config_trace(*testid):
      test_conv2d_transpose(device=t2d[target], workload=workload)
