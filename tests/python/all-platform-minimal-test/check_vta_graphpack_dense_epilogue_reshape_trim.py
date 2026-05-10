import tvm
from tvm import relay


def test_graphpack_trim_handles_right_shift_wrapped_dense():
    # Minimal pattern from dynamic weight path:
    # reshape([-1, 25]) -> nn.dense(units=None) -> right_shift/clip/cast/copy/stop_fusion/cast
    # -> reshape([32, 8, 5, 5]).
    data = relay.var("data", shape=(1, 8, 56, 56), dtype="int8")
    w_start = relay.var("w_start", shape=(8, 8, 1, 1), dtype="int8")
    w_base = relay.var("w_base", shape=(32, 8, 7, 7), dtype="int8")
    tm_7to5 = relay.var("tm_7to5", shape=(25, 25), dtype="int8")

    starter = relay.nn.conv2d(
        data,
        w_start,
        channels=8,
        kernel_size=(1, 1),
        padding=(0, 0),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="int32",
    )
    starter_i8 = relay.cast(starter, "int8")

    w_5x5 = relay.strided_slice(w_base, begin=[0, 0, 1, 1], end=[32, 8, 6, 6])
    flat = relay.reshape(w_5x5, newshape=[-1, 25])
    dense = relay.nn.dense(flat, tm_7to5, units=None, out_dtype="int32")

    # Dense epilogue wrappers that must be unwrapped by graph_pack reshape logic.
    shifted = relay.op.tensor.right_shift(dense, relay.const(8, "int32"))
    clipped = relay.clip(shifted, a_min=-127.0, a_max=127.0)
    cast_i8 = relay.cast(clipped, "int8")
    copied = relay.copy(cast_i8)
    stopped = relay.annotation.stop_fusion(copied)
    back_i32 = relay.cast(stopped, "int32")

    kernel = relay.reshape(back_i32, newshape=[32, 8, 5, 5])
    y = relay.nn.conv2d(
        starter_i8,
        kernel,
        channels=32,
        kernel_size=(5, 5),
        padding=(2, 2),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="int32",
    )
    y = relay.nn.relu(y)

    func = relay.Function([data, w_start, w_base, tm_7to5], y)

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

    # This used to fail with reshape incompatibility (e.g., 6400 vs 8192).
    packed_mod = tvm.IRModule.from_expr(packed)
    packed_mod = relay.transform.InferType()(packed_mod)
    text = packed_mod["main"].astext(show_meta_data=False)
    assert "reshape" in text
    assert "nn.dense" in text


if __name__ == "__main__":
    test_graphpack_trim_handles_right_shift_wrapped_dense()
    print("graphpack dense epilogue reshape-trim regression passed")

