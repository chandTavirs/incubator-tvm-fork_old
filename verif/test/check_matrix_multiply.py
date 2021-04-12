"""
Verification Test Module.
Wrapper for vta/tutorials/matrix_multiply.py.
Accepts all targets but cpu.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.matrix_multiply')

bfactor = (1,2,4)
ofactor = (16,32)
ifactor = (16,32)

params = [(b, o, i, t) for t in selected_targets
                       for b in bfactor
                       for o in ofactor
                       for i in ifactor]

@pytest.mark.parametrize('b,o,i,t', params)
def test_matrix_multiply(b, o, i, t, testid):
  with config_target(t):
    import vta, vta.testing
    # Pass parameters via module.
    vta.OFACTOR, vta.IFACTOR, vta.BFACTOR = o, i, b
    with config_trace(*testid):
      from .matrix_multiply import host
