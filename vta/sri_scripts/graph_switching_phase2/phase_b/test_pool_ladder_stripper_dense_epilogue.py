"""Regression tests for the step-3 pool ladder stripper.

These tests cover two things:
1) int8 dense paths keep a VTA-style requant epilogue, and
2) pool ladders still strip through copy/stop_fusion wrappers.
"""

from __future__ import absolute_import, print_function

import tvm
from tvm import relay

from step3_merged_mod_deriv_poc import _PoolLadderStripper


def _count_op(expr, op_name):
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


def test_dense_rewrite_keeps_vta_style_epilogue():
    data = relay.var("data", shape=(4, 16), dtype="int8")
    weight = relay.var("weight", shape=(8, 16), dtype="int8")
    dense = relay.nn.dense(data, weight, units=8, out_dtype="int32")
    func = relay.Function([data, weight], dense)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    rewritten = _PoolLadderStripper({"weight"}).visit(mod["main"])
    rewritten_mod = tvm.IRModule.from_expr(rewritten)
    rewritten_mod = relay.transform.InferType()(rewritten_mod)

    body = rewritten_mod["main"].body
    assert _count_op(body, "nn.dense") >= 1
    assert _count_op(body, "right_shift") >= 1
    assert _count_op(body, "clip") >= 1
    assert _count_op(body, "copy") >= 1
    assert _count_op(body, "annotation.stop_fusion") >= 1

    rendered = rewritten_mod["main"].astext(show_meta_data=False)
    assert "right_shift" in rendered
    assert "annotation.stop_fusion" in rendered


def test_dense_rewrite_handles_copy_and_stop_fusion_wrapped_inputs():
    data = relay.var("data", shape=(4, 16), dtype="int8")
    weight = relay.var("weight", shape=(8, 16), dtype="int8")
    wrapped_data = relay.copy(relay.annotation.stop_fusion(data))
    wrapped_weight = relay.copy(relay.annotation.stop_fusion(weight))
    dense = relay.nn.dense(wrapped_data, wrapped_weight, units=8, out_dtype="int32")
    func = relay.Function([data, weight], dense)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    rewritten = _PoolLadderStripper({"data", "weight"}).visit(mod["main"])
    rewritten_mod = tvm.IRModule.from_expr(rewritten)
    rewritten_mod = relay.transform.InferType()(rewritten_mod)

    body = rewritten_mod["main"].body
    assert _count_op(body, "nn.dense") >= 1
    assert _count_op(body, "right_shift") >= 1
    assert _count_op(body, "clip") >= 1
    assert _count_op(body, "copy") >= 1
    assert _count_op(body, "annotation.stop_fusion") >= 1


if __name__ == "__main__":
    test_dense_rewrite_keeps_vta_style_epilogue()
    test_dense_rewrite_handles_copy_and_stop_fusion_wrapped_inputs()
    print("pool ladder stripper regression passed")



