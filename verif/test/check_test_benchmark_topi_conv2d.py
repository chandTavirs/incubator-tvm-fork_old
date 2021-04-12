"""
Verification Test Module.
Wrapper for vta/tests/python/integration/test_benchmark_topi_conv2d.py.
Accepts all targets.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, t2d, t2t, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_benchmark_topi_conv2d')

items = (
    'resnet-18.C2',
    'resnet-18.C3',
    'resnet-18.C4',
    'resnet-18.C5',
    'resnet-18.C6',
    'resnet-18.C7',
    'resnet-18.C8',
    'resnet-18.C9',
    'resnet-18.C10',
    'resnet-18.C11',
)
params = [(i, t) for t in selected_targets for i in items]

@pytest.mark.parametrize('workload,target', params)
def test_conv2d(workload, target, testid):
  with config_target(t2t[target]):
    from .test_benchmark_topi_conv2d import test_conv2d
    with config_trace(*testid):
      test_conv2d(device=t2d[target], workload=workload)
