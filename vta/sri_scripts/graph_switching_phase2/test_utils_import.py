#!/usr/bin/env python3
"""Quick test to verify utils import works."""

import sys
sys.path.insert(0, '/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts')

# Test the import
from multi_runtime_executor import MultiRuntimeExecutor, CompiledModelInfo
from tvm.contrib import utils
import tvm

print("✓ Imports successful")

# Test tempdir
temp = utils.tempdir()
print(f"✓ utils.tempdir() works: {temp}")
print(f"✓ temp.relpath() works: {temp.relpath('test.tar')}")

# Test creating executor with remote
from tvm import rpc
remote = rpc.connect('10.42.0.188', 9091)
ctx = remote.ext_dev(0)

print(f"✓ Remote connected: {remote}")
print(f"✓ Context: {ctx}")

executor = MultiRuntimeExecutor(ctx, remote)
print(f"✓ MultiRuntimeExecutor created")
print(f"✓ Temp dir: {executor._temp_dir}")

print("\n✅ All tests passed!")

