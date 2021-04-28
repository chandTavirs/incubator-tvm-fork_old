"""
Test Context.
"""

# Modified by contributors from Intel Labs

import json
from os import system, environ, chdir
from os.path import abspath, dirname, exists
from shutil import copy, move
from contextlib import contextmanager
from .test_utils import home, root

use_env_target = True

@contextmanager
def config_target(target):
  '''
  Target Configuration Context.
  Create a context for the target configuration of a test.
  Note that because the configuration is file based, only
  one target is allowed to be used at a time and therefore
  parallel testing is only allowed if within one target.
  Use the environment variable VTA_TARGET to work around this
  limitation enabling the loading of ${VTA_TARGET}_target.json
  files directly.
  '''
  cfg_dir = f'{root()}/3rdparty/vta-hw/config'
  vta_file = f'{cfg_dir}/vta_config.json'
  vta_save = f'{cfg_dir}/vta_config.save'
  tgt_file = f'{cfg_dir}/{target}_target.json'
  if target == 'de10nano' or target == 'pynq':
    fl = f'{home()}/test/fpga_targets.json'
    if exists(fl):
      with open(fl) as fp:
        fpga_targets = json.load(fp)
        if 'VTA_RPC_HOST' not in environ:
          environ['VTA_RPC_HOST'] = fpga_targets[target]['host']
        if 'VTA_RPC_PORT' not in environ:
          environ['VTA_RPC_PORT'] = fpga_targets[target]['port']
    assert 'VTA_RPC_HOST' in environ, "Must set VTA_RPC_HOST environment variable"
    assert 'VTA_RPC_PORT' in environ, "Must set VTA_RPC_PORT environment variable"
  try:
    chdir(f'{home()}/work')
    if use_env_target:
      if 'VTA_TARGET' not in environ:
        environ['VTA_TARGET'] = target
      yield tgt_file
    else:
      copy(vta_file, vta_save)
      copy(tgt_file, vta_file)
      yield vta_file
  except:
    raise
  finally:
    if not use_env_target:
      move(vta_save, vta_file)

if __name__ == '__main__':
  with config_target('tsim') as cfg:
    system('cat ' + cfg)
