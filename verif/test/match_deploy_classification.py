"""
Verification Test Module.
Check for equivalence between abstraction models of test
vta/tutorials/frontend/deploy_classification.py.
Accepts all targets.
Note: Unfortunately pytest-depends does not work well with pytest-testmon
in that if a test dependency is not collected because not changed pytest-depends
will skip the dependent test instead or running. Also pytest-depend does not
work well with pytest-xdist, what a mess of plugins.
"""

# Modified by contributors from Intel Labs

import os, pytest
from verif import selected_targets

workloads = (
   #'resnet18_v1', # tiger cat,    Egyptian cat, tabby cat,    lynx,         weasel
    'resnet18_v2', # Egyptian cat, tiger cat,    tabby cat,    bucket,       corn
   #'resnet34_v1', # tennis ball,  tiger cat,    tabby cat,    Egyptian cat, ping-pong ball
    'resnet34_v2', # tabby cat,    tiger cat,    Egyptian cat, lynx,         ping-pong ball
    'resnet50_v2', # tiger cat,    Egyptian cat, tabby cat,    laptop,       Pembroke
    'resnet101_v2' # Egyptian cat, Chihuahua,    tabby cat,    tiger cat,    partridge
)
parent = 'verif/test/check_deploy_classification.py::test_deploy_classification'
t = selected_targets
pairs = [(t[i], t[j]) for i in range(0, len(t)-1) for j in range(i+1, len(t))]
params = [(w, l, r) for w in workloads for l, r in pairs \
  if not (w == 'resnet101_v2' and (l == 'pynq' or r == 'pynq'))]

@pytest.mark.parametrize('workload,left,right', [
  pytest.param(w, l, r, marks=pytest.mark.depends(
    on=[f'{parent}[{w}-{t}]' for t in (l, r)])
    ) for w, l, r in params
])
def match_deploy_classification(workload, left, right, testid):
  '''
  Compare the output of each pair of models.
  '''
  def compare_models(m, n):
    print('\nl:', m)
    print('r:', n, '\n')
    m = [l for l in open(m).readlines() if l.startswith('#')]
    n = [l for l in open(n).readlines() if l.startswith('#')]
    assert m and n
    for u, v in zip(m, n):
      print(f'{u[4:].rstrip():>32s} == {v[4:].rstrip():<32s}')
      assert u == v

  stem = f'work/test_deploy_classification[{workload}'
  m, n = f'{stem}-{left}].out', f'{stem}-{right}].out'
  assert os.path.exists(m) and os.path.exists(n)
  compare_models(m, n)
   
