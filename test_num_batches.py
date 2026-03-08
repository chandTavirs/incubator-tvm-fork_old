#!/usr/bin/env python3
import torch
import torch.nn as nn
import tvm
from tvm import relay

# Simple model with BatchNorm
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        return self.bn(x)

model = SimpleModel().eval()
input_data = torch.randn(1, 16, 7, 7)
traced = torch.jit.trace(model, input_data)

print("State dict keys:")
for k, v in traced.state_dict().items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\nGraph:")
print(traced.graph)

print("\nGraph nodes:")
for node in traced.graph.nodes():
    print(f"\n  Node: {node.kind()}")
    print(f"    Inputs: {[inp.debugName() for inp in node.inputs()]}")
    print(f"    Outputs: {[out.debugName() for out in node.outputs()]}")

# Convert to Relay
shape_list = [('input', (1, 16, 7, 7))]
print("\n\nConverting to Relay...")
try:
    mod, params = relay.frontend.from_pytorch(traced, shape_list)
    print("SUCCESS!")
    print("\nRelay module:")
    print(mod.astext(show_meta_data=False))
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

