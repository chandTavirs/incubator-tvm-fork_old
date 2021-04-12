"""
Verification Utility Module.
"""

# Modified by contributors from Intel Labs

import os, json, pytest
from glob import glob

def root():
  fp = os.path.dirname(os.path.abspath(__file__))
  fp = f"{fp[0:fp.rfind('/verif/verif')]}"
  return fp

def home():
  return f'{root()}/verif'

def trace_stem(node: str):
  return f'{home()}/work/{node}'

def trace_path(node: str, mode: str):
  return f'{trace_stem(node)}.{mode}'

def trace_modes():
  fn = f'{home()}/test/trace_modes.json'
  modes = []
  with open(fn, 'r') as fp:
    modes = list(json.load(fp).keys())
  return modes

def sample_targets():
  samples = glob(f'{root()}/3rdparty/vta-hw/config/*_target.json')
  return {t[t.rfind('/')+1:t.rfind('_target')] for t in samples}

def skip_test_module(expr, msg=''):
  if expr:
    pytest.skip(msg, allow_module_level=True)

def is_remote():
  return os.sys.argv and os.sys.argv[0].find('rpc_server') != -1
