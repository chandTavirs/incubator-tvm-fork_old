"""
Verification Test Module.
Wrapper for vta/tests/python/integration/test_benchmark_topi_depthwise_conv2d.py.
Accepts all targets.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, t2d, t2t, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_benchmark_topi_depthwise_conv2d')
excluded_targets = ['bsim']
targets = [ t for t in selected_targets if t not in excluded_targets]
items = (
    'mobilenet.D1',
    'mobilenet.D2',
    'mobilenet.D3',
    'mobilenet.D4',
    'mobilenet.D5',
    'mobilenet.D6',
    'mobilenet.D7',
    'mobilenet.D8',
    'mobilenet.D9',
)
params = [(i, t) for t in targets for i in items]

@pytest.mark.parametrize('workload,target', params)
def test_depthwise_conv2d(workload, target, testid):
  with config_target(t2t[target]):
    from .test_benchmark_topi_depthwise_conv2d import test_conv2d
    with config_trace(*testid):
      test_conv2d(device=t2d[target], workload=workload)
