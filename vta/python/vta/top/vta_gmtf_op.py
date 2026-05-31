# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# Modified by contributors from Intel Labs
"""Python-side registration for vta.gmtf_dense_small/large Relay ops.

These ops are registered in C++ (src/relay/op/nn/nn.cc) with custom type
relations that accept the GMTF packed weight layouts.  This module:
  - Exposes Python make functions for emitting the ops in graphpack.py
  - Registers VTA strategies that dispatch to the GMTF TOPI compute+schedule
"""

import tvm
from tvm import relay
from tvm.relay.op.op import OpStrategy
from tvm.relay.op import strategy as _strat

from .vta_gmtf_dense import (
    dense_pack_gmtf_small,
    schedule_dense_pack_gmtf_small,
    dense_pack_gmtf_large,
    schedule_dense_pack_gmtf_large,
)


# ---------------------------------------------------------------------------
# Python make helpers  (thin wrappers around the C++ global functions)
# ---------------------------------------------------------------------------

def gmtf_dense_small(data, weight, out_dtype="int32"):
    """Emit vta.gmtf_dense_small: GEMM_Mat_Trf small-mode (9x9 transform).

    data:   (n_batch, 1, BATCH, BLOCK_IN)
    weight: (1, 1, BLOCK_OUT, BLOCK_IN)
    out:    (n_batch, 1, BATCH, BLOCK_OUT)
    """
    return tvm.get_global_func("relay.op.nn._make.gmtf_dense_small")(
        data, weight, tvm.runtime.DataType(out_dtype)
    )


def gmtf_dense_large(data, weight, out_dtype="int32"):
    """Emit vta.gmtf_dense_large: GEMM_Mat_Trf large-mode (25x25 transform).

    data:   (n_batch, 2, BATCH, BLOCK_IN)
    weight: (2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)
    out:    (n_batch, 2, BATCH, BLOCK_OUT)
    """
    return tvm.get_global_func("relay.op.nn._make.gmtf_dense_large")(
        data, weight, tvm.runtime.DataType(out_dtype)
    )


# ---------------------------------------------------------------------------
# Strategy registration for vta target
# ---------------------------------------------------------------------------

def _gmtf_small_strategy(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        _strat.wrap_compute_dense(dense_pack_gmtf_small),
        _strat.wrap_topi_schedule(schedule_dense_pack_gmtf_small),
        name="dense_pack_gmtf_small.vta",
    )
    return strategy


def _gmtf_large_strategy(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        _strat.wrap_compute_dense(dense_pack_gmtf_large),
        _strat.wrap_topi_schedule(schedule_dense_pack_gmtf_large),
        name="dense_pack_gmtf_large.vta",
    )
    return strategy


# Register as FTVMStrategy attributes — bypasses the GenericFunc requirement
# since these ops have no default CPU fallback and are VTA-only.
tvm.ir.register_op_attr("vta.gmtf_dense_small", "FTVMStrategy", _gmtf_small_strategy)
tvm.ir.register_op_attr("vta.gmtf_dense_large", "FTVMStrategy", _gmtf_large_strategy)
