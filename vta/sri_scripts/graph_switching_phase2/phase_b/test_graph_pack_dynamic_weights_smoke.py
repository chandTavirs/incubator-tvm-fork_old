"""Smoke test for graph_pack_dynamic_weights.

This test creates a tiny Relay graph and runs the dynamic graph pack API.
It validates import, execution, and return-status contract.
"""

from __future__ import absolute_import, print_function

import tvm
from tvm import relay
from vta.top import graph_pack_dynamic_weights


def build_tiny_func():
    data = relay.var("data", shape=(1, 3, 8, 8), dtype="float32")
    weight = relay.var("weight", shape=(8, 3, 3, 3), dtype="float32")
    conv = relay.nn.conv2d(data, weight, channels=8, kernel_size=(3, 3), padding=(1, 1))
    out = relay.nn.relu(conv)
    func = relay.Function([data, weight], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod["main"]


def main():
    expr = build_tiny_func()
    packed_expr, used_graph_pack, reason = graph_pack_dynamic_weights(
        expr,
        bfactor=1,
        blockin=1,
        blockout=1,
        weight_bits=8,
        pack_all=True,
        allow_fallback=True,
        return_status=True,
    )

    assert isinstance(packed_expr, relay.Function)
    assert isinstance(used_graph_pack, bool)
    assert reason is None or isinstance(reason, str)

    mode = "graph_pack" if used_graph_pack else "fallback"
    print("graph_pack_dynamic_weights smoke test passed")
    print("mode:", mode)
    if reason:
        print("reason:", reason)


if __name__ == "__main__":
    main()

