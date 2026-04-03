"""Smoke test for dynamic dense quantization feature gate.

Validates that quantize_with_dynamic_weights:
1) rejects skip_dense_layer=False when gate is disabled,
2) accepts/runs when gate is enabled.
"""

from __future__ import absolute_import, print_function

import os
import numpy as np
import tvm
from tvm import relay

from quantize_dynamic_weights import quantize_with_dynamic_weights


def build_dynamic_dense_to_conv_mod():
    data = relay.var("input0", shape=(1, 1, 5, 5), dtype="float32")
    base_w = relay.var("pool_base_weight", shape=(1, 1, 3, 3), dtype="float32")
    tm = relay.var("pool_transform_matrix", shape=(9, 9), dtype="float32")

    w_flat = relay.reshape(base_w, newshape=(1, 9))
    w_flat_t = relay.nn.dense(w_flat, tm)
    # Include multiply in the derivation chain to cover WEIGHT-kind propagation.
    w_flat_t = relay.multiply(w_flat_t, relay.const(1.0, "float32"))
    w = relay.reshape(w_flat_t, newshape=(1, 1, 3, 3))

    y = relay.nn.conv2d(data, w, channels=1, kernel_size=(3, 3), padding=(1, 1))
    func = relay.Function([data, base_w, tm], y)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    params = {
        "pool_base_weight": tvm.nd.array(np.random.randn(1, 1, 3, 3).astype("float32")),
        "pool_transform_matrix": tvm.nd.array(np.random.randn(9, 9).astype("float32")),
    }
    dynamic_vars = ["pool_base_weight", "pool_transform_matrix"]
    return mod, params, dynamic_vars


def main():
    mod, params, dynamic_vars = build_dynamic_dense_to_conv_mod()

    # Gate disabled: should fail fast.
    os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
    failed_as_expected = False
    with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[], skip_dense_layer=False):
        try:
            quantize_with_dynamic_weights(mod, params, dynamic_weight_var_names=dynamic_vars)
        except ValueError as err:
            failed_as_expected = "TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX" in str(err)

    assert failed_as_expected, "Expected guard failure when gate is disabled"

    # Gate enabled: should pass guard and run quantization.
    os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = "1"
    try:
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[], skip_dense_layer=False):
            mod_q = quantize_with_dynamic_weights(mod, params, dynamic_weight_var_names=dynamic_vars)
        assert isinstance(mod_q, tvm.ir.IRModule)
        print(mod_q.astext(show_meta_data=False))
    finally:
        os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)

    print("dynamic dense quant gate smoke test passed")


if __name__ == "__main__":
    main()


