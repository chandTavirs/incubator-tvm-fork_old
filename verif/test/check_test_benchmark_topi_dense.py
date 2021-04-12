"""
Verification Test Module.
Wrapper for vta/tests/python/integration/test_benchmark_topi_dense.py.
Accepts all targets but cpu.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_benchmark_topi_dense')

targets = [t for t in selected_targets if t != 'cpu']
skip_test_module(not targets, 'NO_AVAILABLE_TARGETS')

@pytest.mark.parametrize('target', targets)
def test_dense(target, testid):
  with config_target(target):
    from .test_benchmark_topi_dense import test_gemm
    with config_trace(*testid):
      test_gemm()
