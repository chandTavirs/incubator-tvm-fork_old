"""
Verification Test Module.
Wrapper for vta/tests/python/unittest/test_vta_insn.py.
Accepts all targets but cpu.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   skip_test_module, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.test_vta_insn')

targets = [t for t in selected_targets if t != 'cpu']
skip_test_module(not targets, 'NO_AVAILABLE_TARGETS')
alu_items = ('SHL', 'SHR', 'MAX', 'MAXI', 'ADD', 'ADDI', 'MOV', 'MOVI')
alu_params = [(i, t) for t in targets for i in alu_items]
pad_items = ('Y0', 'Y1', 'X0', 'X1', 'ALL')
pad_params = [(i, t) for t in targets for i in pad_items]
gem_items = ('DEFAULT', 'SMT')
gem_params = [(i, t) for t in targets for i in gem_items]

@pytest.mark.parametrize('target', targets)
def test_vta_store_insn(target, testid):
  with config_target(target):
    from .test_vta_insn import test_save_load_out
    with config_trace(*testid):
      test_save_load_out()

@pytest.mark.parametrize('item,target', pad_params)
def test_vta_load_insn(item, target, testid):
  with config_target(target):
    from .test_vta_insn import test_padded_load
    with config_trace(*testid):
      test_padded_load(item)

@pytest.mark.parametrize('item,target', gem_params)
def test_vta_gemm_insn(item, target, testid):
  with config_target(target):
    from .test_vta_insn import test_gemm
    with config_trace(*testid):
      test_gemm(item)

@pytest.mark.parametrize('item,target', alu_params)
def test_vta_alu_insn(item, target, testid):
  with config_target(target):
    from .test_vta_insn import test_alu
    with config_trace(*testid):
      test_alu(item)

@pytest.mark.parametrize('target', targets)
def test_vta_relu(target, testid):
  with config_target(target):
    from .test_vta_insn import test_relu
    with config_trace(*testid):
      test_relu()

@pytest.mark.parametrize('target', targets)
def test_vta_shift_and_scale(target, testid):
  with config_target(target):
    from .test_vta_insn import test_shift_and_scale
    with config_trace(*testid):
      test_shift_and_scale()
