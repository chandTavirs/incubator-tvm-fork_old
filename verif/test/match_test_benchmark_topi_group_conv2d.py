"""
Verification Test Module.
Check for equivalence between abstraction models of test
vta/tests/python/integration/test_benchmark_topi_group_conv2d.py
Accepts all targets.
Note: Unfortunately pytest-depends does not work well with pytest-testmon
in that if a test dependency is not collected because not changed pytest-depends
will skip the dependent test instead or running. Also pytest-depend does not
work well with pytest-xdist, what a mess of plugins.
"""

# Modified by contributors from Intel Labs

import os, pytest
from verif import selected_targets

parent = 'verif/test_benchmark_topi_group_conv2d'
t = selected_targets
pairs = [(t[i], t[j]) for i in range(0, len(t)-1) for j in range(i+1, len(t))]
workloads = (
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

@pytest.mark.parametrize('workload', [
  pytest.param(w, marks=pytest.mark.depends(
    on=[f'{parent}[{w}-{t}]' for t in selected_targets])
  ) for w in workloads
])
def match_test_group_conv2d(workload, testid):
  '''
  Compare the trace of each pair of models.
  '''
  def compare_models(m, n):
    print('\nl:', m)
    print('r:', n, '\n')
    rs = os.system(f'cmp {m} {n}')
    assert rs == 0

  stem, mode = testid[0][:-1].replace('match_', 'work/'), testid[1]
  for m, n in pairs:
    m, n = f'{stem}-{m}].{mode}', f'{stem}-{n}].{mode}'
    assert os.path.exists(m) and os.path.exists(n)
    print(m, n)
    compare_models(m, n)
   
