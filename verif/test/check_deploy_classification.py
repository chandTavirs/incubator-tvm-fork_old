"""
Verification Test Module.
Wrapper for vta/tutorials/frontend/deploy_classification.py.
Accepts all targets.
"""

# Modified by contributors from Intel Labs

import pytest
from verif import (config_target, config_trace, selected_targets,
                   t2t, random_seed)
from numpy.random import seed

seed(random_seed)
pytest.register_assert_rewrite('test.deploy_classification')

items = (
   #'resnet18_v1', # tiger cat,    Egyptian cat, tabby cat,    lynx,         weasel
    'resnet18_v2', # Egyptian cat, tiger cat,    tabby cat,    bucket,       corn
   #'resnet34_v1', # tennis ball,  tiger cat,    tabby cat,    Egyptian cat, ping-pong ball
    'resnet34_v2', # tabby cat,    tiger cat,    Egyptian cat, lynx,         ping-pong ball
    'resnet50_v2', # tiger cat,    Egyptian cat, tabby cat,    laptop,       Pembroke
    'resnet101_v2', # Egyptian cat, Chihuahua,    tabby cat,    tiger cat,    partridge
    'mobilenet1.0' # tiger cat,   Egyptian cat, tabby cat,    lynx,         Persian cat
)
params = [(i, t) for t in selected_targets for i in items]

@pytest.mark.parametrize('workload,target', params)
def test_deploy_classification(workload, target, testid):
  with config_target(t2t[target]):
    import vta, vta.testing
    vta.DEVICE = target if target == 'cpu' else 'vta'
    vta.WORKLOAD = workload
    with config_trace(*testid):
      from .deploy_classification import inference_time
