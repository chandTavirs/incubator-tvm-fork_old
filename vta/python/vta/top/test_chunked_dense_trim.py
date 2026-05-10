import sys
import os
import tvm
from tvm import relay
import numpy as np

# Ensure repo root is on path so we can import the local tvm.vta package
# Find repo root by walking up until we see a 'tvm' directory sibling
candidate = os.path.abspath(os.path.dirname(__file__))
repo_root = None
for _ in range(10):
    parent = os.path.dirname(candidate)
    if parent == candidate:
        break
    if os.path.isdir(os.path.join(parent, 'tvm')):
        repo_root = parent
        break
    candidate = parent

if repo_root is None:
    # Fallback to a conservative guess
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import graphpack via file path to avoid package import issues
import importlib.util
graphpack_path = os.path.join(repo_root, 'tvm', 'vta', 'python', 'vta', 'top', 'graphpack.py')
spec = importlib.util.spec_from_file_location('graphpack_local', graphpack_path)
graphpack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphpack)

# This test constructs a fake pattern similar to chunked packed dense concatenate
# We will simulate: data -> (possibly wrappers) -> concatenate(dense(...), dense(...)) -> reshape(target_newshape)

def make_dense_call(input_var, weight_var):
    # simple dense call
    return relay.nn.dense(input_var, weight_var)


def build_and_pack():
    # Create fake data and weights
    # data: shape (rows, cols) but we'll shape them so that reshape path will be triggered
    rows = 1024
    units = 25 * 25
    data = relay.var("data", shape=(rows, units), dtype="int32")
    w1 = relay.var("w1", shape=(units, units), dtype="int8")
    w2 = relay.var("w2", shape=(units, units), dtype="int8")

    d1 = make_dense_call(data, w1)
    d2 = make_dense_call(data, w2)

    # concatenate as chunked parts (simulate concatenate(Tuple([...])) form
    tup = relay.Tuple([d1, d2])
    concat = relay.op.concatenate(tup, axis=0)

    # target reshape: simulate 4D transform-matrix shape
    target_newshape = (32, 25, 25, 25)  # example shape matching conditions
    resh = relay.op.reshape(concat, newshape=target_newshape)

    fn = relay.Function([data, w1, w2], resh)

    # run graph_pack
    packed = graphpack.graph_pack(fn, bfactor=1, blockin=1, blockout=1, weight_bits=8)
    print("Packed type:", type(packed))
    print(packed)

if __name__ == '__main__':
    build_and_pack()




