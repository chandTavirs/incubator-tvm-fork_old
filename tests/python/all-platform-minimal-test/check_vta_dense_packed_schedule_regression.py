from __future__ import absolute_import, print_function

import os
import sys

import tvm
from tvm import autotvm
from tvm import te
from tvm import topi

TVM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VTA_PYTHON = os.path.join(TVM_ROOT, "vta", "python")
if VTA_PYTHON not in sys.path:
    sys.path.insert(0, VTA_PYTHON)

import vta


def test_dense_packed_schedule_regression_lowers():
    env = vta.get_env()
    log_file = os.path.join(
        TVM_ROOT,
        "vta",
        "sri_scripts",
        "logs",
        "tuning_logs",
        "vta_1x16x16",
        "candidate_set",
        "dense_packed.vta.log",
    )

    data_shape = (256, 1, 1, 16)
    weight_shape = (1, 1, 16, 16)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)

    with autotvm.apply_history_best(log_file):
        with tvm.target.vta():
            res = vta.top.dense_packed(data, weight, None, env.acc_dtype)
            res = topi.right_shift(res, 8)
            res = topi.cast(res, env.out_dtype)
            sch = vta.top.schedule_dense_packed([res])
            lowered = tvm.lower(sch, [data, weight, res], simple_mode=True)

    assert lowered is not None


if __name__ == "__main__":
    test_dense_packed_schedule_regression_lowers()



