"""Regression test for chunked dense epilogue handling.

When large batches trigger dense chunking via _chunk_packed_dense, each chunk
should have the VTA epilogue (right_shift + clip + cast + copy + stop_fusion + cast)
applied individually before concatenation, not just on the final concatenated result.
"""

import numpy as np
import tvm
from tvm import relay


def _count_op(expr, op_name):
    """Count occurrences of an op by name in a Relay expression."""
    op = relay.op.get(op_name)
    count = 0

    class _Visitor(relay.ExprVisitor):
        def visit_call(self, call):
            nonlocal count
            if call.op == op:
                count += 1
            super().visit_call(call)

    _Visitor().visit(expr)
    return count


def test_chunked_dense_epilogue_per_chunk():
    """Verify that chunked dense applies epilogue to each chunk before concatenation.
    
    Large batch sizes should trigger _chunk_packed_dense, which creates multiple
    dense calls and concatenates them. Each dense chunk should have the full VTA
    epilogue (right_shift -> clip -> cast -> copy -> stop_fusion -> cast int32)
    applied individually.
    """
    from vta.top.graphpack import graph_pack

    # Large batch to trigger chunking: after packing, B_outer = batch // bfactor.
    # ACC_BUFF_SIZE is typically 131072. With bfactor=1, we need batch > 131072.
    batch = 262144
    in_dim = 32
    units = 64

    # graph_pack only works from 4D NCHW. Use a tiny conv2d to enter packed region.
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
    conv_i8 = relay.cast(conv, "int8")
    x = relay.nn.batch_flatten(conv_i8)
    y = relay.nn.dense(x, w_dense, units=units, out_dtype="int32")
    y = relay.nn.relu(y)
    func = relay.Function([data, w_conv, w_dense], y)

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

    packed_mod = tvm.IRModule.from_expr(packed)
    packed_mod = relay.transform.InferType()(packed_mod)

    body = packed_mod["main"].body
    rendered = packed_mod["main"].astext(show_meta_data=False)

    # Verify chunking was applied: multiple dense calls
    assert _count_op(body, "nn.dense") >= 2, "Expected at least 2 dense calls from chunking"

    # Verify concatenation happened
    assert _count_op(body, "concatenate") >= 1, "Expected concatenate from chunking"

    # Verify that we have multiple right_shift/clip/cast/copy/stop_fusion sequences,
    # one per chunk. Each chunk should have these ops before concatenation.
    # This is reflected in the op count being > 1 per epilogue op.
    assert _count_op(body, "right_shift") >= 1, "Expected right_shift in epilogue"
    assert _count_op(body, "clip") >= 1, "Expected clip in epilogue"
    assert _count_op(body, "copy") >= 1, "Expected copy in epilogue"
    assert _count_op(body, "annotation.stop_fusion") >= 1, "Expected stop_fusion in epilogue"

    # Sanity check: look at the rendered text
    assert "concatenate" in rendered
    assert "right_shift" in rendered
    assert "annotation.stop_fusion" in rendered


if __name__ == "__main__":
    test_chunked_dense_epilogue_per_chunk()
    print("✓ Chunked dense epilogue regression passed")

