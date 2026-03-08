#!/usr/bin/env python3
"""Minimal test to reproduce the quantization error."""

import sys
sys.path.insert(0, "/home/srchand/Desktop/research/OFA_Obfs")

import torch
import tvm
from tvm import relay

# Clear cached imports
sys.modules.pop("ofa_base_models", None)
from ofa_base_models import OFADynamicResnetAllMod

import json

print("="*80)
print("MINIMAL QUANTIZATION TEST")
print("="*80)

# Load model
print("\n1. Loading OFA model...")
net = OFADynamicResnetAllMod()
model_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
checkpoint = torch.load(model_path, map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state = checkpoint['model_state_dict']
else:
    state = checkpoint
net.load_state_dict(state, strict=False)
print("   Model loaded successfully")

# Load arch config
print("\n2. Loading architecture...")
arch_config_json_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
with open(arch_config_json_path, 'r') as f:
    data = json.load(f)

# Get first architecture
if isinstance(data, dict) and 'architectures' in data:
    arch = data['architectures'][0]['architecture']
elif isinstance(data, list):
    arch = data[0]['architecture'] if 'architecture' in data[0] else data[0]
else:
    arch = list(data.values())[0]

print(f"   Architecture: {list(arch.keys())}")

# Set active subnet
print("\n3. Setting active subnet...")
net.set_active_subnet(arch)
net.precompute_active_weights(arch)
net.eval()
print("   Subnet configured")

# Convert to Relay
print("\n4. Converting to Relay IR...")
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)

with torch.no_grad():
    scripted_model = torch.jit.trace(net, input_data).eval()

shape_list = [("input0", input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(f"   Conversion complete. Parameters: {len(params)}")

# Run InferType
print("\n5. Running InferType...")
try:
    mod = relay.transform.InferType()(mod)
    print("   InferType successful")
except Exception as e:
    print(f"   InferType FAILED: {e}")
    sys.exit(1)

# Try quantization
print("\n6. Attempting quantization...")
print("   This is where the error occurs...")
try:
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod_quantized = relay.quantize.quantize(mod, params=params)
    print("\n" + "="*80)
    print("SUCCESS! Quantization completed without errors")
    print("="*80)
except Exception as e:
    print("\n" + "="*80)
    print("QUANTIZATION FAILED")
    print("="*80)
    print(f"Error type: {type(e).__name__}")
    print(f"\nError message:")
    print(str(e))
    print("="*80)
    sys.exit(1)

