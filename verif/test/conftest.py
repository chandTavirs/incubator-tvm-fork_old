"""
Verification pytest configuration.
"""

# Modified by contributors from Intel Labs

import os, pytest, verif

def pytest_addoption(parser):
  parser.addoption("--mode", choices=verif.trace_modes(), default='log',
    help="Trace mode (default: %(default)s)."
  )
  parser.addoption("--targets", nargs='+',
    choices=verif.trace_targets(),
    default=['bsim', 'fsim', 'tsim'],
    help="Accelerator targets (default: %(default)s)."
  )
  parser.addoption("--seed", 
    type=lambda x: int(x,0),
    default=int(os.environ.get("VTA_RND_SEED", "0xCafeFace"), 16),
    help="Random seed (hex) (default: %(default)x)."
  )

def pytest_configure(config):
  verif.selected_mode = config.getoption('--mode')
  verif.selected_targets = config.getoption('--targets')
  verif.random_seed = config.getoption('--seed')

@pytest.fixture
def testid(request):
  return (request.node.name, 
          request.config.getoption('--mode'))

#@pytest.fixture(scope='function')
#def targets(request):
#  return request.config.getoption('--targets')

#def pytest_collection_modifyitems(session, config, items):
#  for item in items:
#    item._nodeid = verif.rename_trace_node(item.nodeid)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
  # Run only after all other hooks.
  outcome = yield
  rep = outcome.get_result()
  # If the test has failed save the complete report.
  if rep.when == "call" and verif.selected_mode != 'quiet':
    if not rep.skipped:
      if rep.capstdout:
        fn = f'{verif.home()}/work/{item.name}.out'
        with open(fn, 'w') as fp:
          fp.write(rep.capstdout)
      if rep.capstderr:
        fn = f'{verif.home()}/work/{item.name}.err'
        with open(fn, 'w') as fp:
          fp.write(rep.capstderr)
      if rep.caplog:
        fn = f'{verif.home()}/work/{item.name}.log'
        with open(fn, 'w') as fp:
          fp.write(rep.caplog)
    fn = f'{verif.home()}/work/{item.name}.fail'
    if rep.passed:
      if os.path.exists(fn):
        os.remove(fn)
    elif rep.failed:
      with open(fn, 'w') as fp:
        fp.write(rep.longreprtext)
