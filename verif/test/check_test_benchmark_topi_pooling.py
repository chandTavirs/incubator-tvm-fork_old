"""
Verification Test Module.
Wrapper for vta/tests/python/integration/test_benchmark_topi_pooling.py.
Accepts all targets.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, t2d, t2t, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_benchmark_topi_pooling')

items = (
    'pool.max.small',
    'pool.max.pad',
    'pool.max.medium',
    'pool.max.resnet',
    'pool.avg.tiny',
    'pool.avg.small',
    'pool.avg.medium',
    'pool.avg.resnet',
)
params = [(i, t) for t in selected_targets for i in items]

@pytest.mark.parametrize('workload,target', params)
def test_pooling(workload, target, testid):
  with config_target(t2t[target]):
    from .test_benchmark_topi_pooling import test_pooling
    with config_trace(*testid):
      test_pooling(device=t2d[target], workload=workload)
