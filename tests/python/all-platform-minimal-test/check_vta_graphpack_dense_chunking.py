import numpy as np

import tvm
from tvm import relay


def _count_op(expr, op_name: str) -> int:
    op = relay.op.get(op_name)
    count = 0

    class _V(relay.ExprVisitor):
        def visit_call(self, call):
            nonlocal count
            if call.op == op:
                count += 1
            super().visit_call(call)

    _V().visit(expr)
    return count


def test_graphpack_dense_chunking_inserted_for_large_batch():
    # Large batch to trigger chunking: after packing, B_outer = batch // bfactor.
    # Use moderate units/in_dim to keep test runtime tiny.
    # ACC_BUFF_SIZE in this workspace is 131072, and with bfactor=1 we need batch > 131072
    # to force chunking.
    batch = 262144
    in_dim = 32
    units = 64

    # graph_pack only knows how to start packing from a 4D NCHW tensor.
    # Use a tiny conv2d to enter the packed region, then batch_flatten -> dense.
    data = relay.var("data", shape=(batch, in_dim, 1, 1), dtype="int8")
    w_conv = relay.var("w_conv", shape=(in_dim, in_dim, 1, 1), dtype="int8")
    w_dense = relay.var("w_dense", shape=(units, in_dim), dtype="int8")

    conv = relay.nn.conv2d(
        data,
        w_conv,
        strides=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        channels=in_dim,
        kernel_size=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="int32",
    )
    # Cast back to int8 so graph_pack's int32 conv2d case doesn't early-exit.
    conv_i8 = relay.cast(conv, "int8")
    x = relay.nn.batch_flatten(conv_i8)
    y = relay.nn.dense(x, w_dense, units=units, out_dtype="int32")
    y = relay.nn.relu(y)
    func = relay.Function([data, w_conv, w_dense], y)

    from vta.top.graphpack import graph_pack

    packed = graph_pack(
        func,
        bfactor=1,
        blockin=16,
        blockout=16,
        weight_bits=8,
        start_name="nn.conv2d",
        stop_name="nn.relu",
        start_name_idx=None,
        stop_name_idx=None,
        count_meta=False,
    )

    packed = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(packed))["main"]

    print(packed.astext(show_meta_data=False))

    # Chunking path concatenates slices along axis 0.
    assert _count_op(packed.body, "concatenate") >= 1
    assert _count_op(packed.body, "nn.dense") >= 2


if __name__ == "__main__":
    test_graphpack_dense_chunking_inserted_for_large_batch()





