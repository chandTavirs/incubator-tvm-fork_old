#!/usr/bin/env python3
import torch
import torch.nn as nn
import tvm
from tvm import relay
import sys
import os

# Add VTA Python path
vta_python_path = '/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/python'
if vta_python_path not in sys.path:
    sys.path.insert(0, vta_python_path)

from vta.top.graphpack import graph_pack

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

# Convert to Relay
shape_list = [('input', (1, 16, 7, 7))]
print("Converting to Relay...")
mod, params = relay.frontend.from_pytorch(traced, shape_list)
print("SUCCESS!\n")

# Try graph packing
print("Attempting graph pack...")
try:
    relay_prog = graph_pack(
        mod["main"],
        1,  # bfactor
        16, # blockin
        16, # blockout
        8,  # weight_bits
        start_name=None,
        stop_name=None,
        device_annot=False,
    )
    print("Graph pack SUCCESS!")
except Exception as e:
    print(f"Graph pack FAILED: {e}")
    import traceback
    traceback.print_exc()

