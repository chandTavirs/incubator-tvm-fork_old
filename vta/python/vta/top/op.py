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
# pylint: disable=unused-argument, ungrouped-imports

# Modified by contributors from Intel Labs

"""Namespace for supporting Relay operators on VTA."""
from __future__ import absolute_import as _abs

import tvm
from tvm import topi

from tvm.relay.op import op as reg
from tvm.relay.op import strategy as _strategy
from tvm.relay.op.op import OpPattern, OpStrategy

from .utils import is_packed_layout
from .vta_conv2d import conv2d_packed, schedule_conv2d_packed
from .vta_conv2d_transpose import conv2d_transpose_packed, schedule_conv2d_transpose_packed
from .vta_dense import dense_packed, schedule_dense_packed
from .vta_depthwise_conv2d import depthwise_conv2d_packed, schedule_depthwise_conv2d_packed
from .vta_pooling import pooling_packed, schedule_pooling_packed
from ..environment import get_env

# override to force partition at copy
reg.register_pattern("copy", OpPattern.INJECTIVE, level=15)

@_strategy.conv2d_strategy.register("vta")
def conv2d_strategy_vta(attrs, inputs, out_type, target):
    """conv2d vta strategy"""
    strategy = OpStrategy()
    kernel = inputs[1]
    dilation = topi.utils.get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout

    assert dilation == (1, 1), "support for dilation limited to (1, 1)"
    if is_packed_layout(layout):
        if groups == 1:
            env = get_env()
            assert env.LOG_INP_WIDTH == 3, "only support 8bit inp for now"
            assert env.LOG_WGT_WIDTH == 3, "only support 8bit wgt for now"
            assert kernel.dtype == "int8"

            strategy.add_implementation(
                _strategy.wrap_compute_conv2d(conv2d_packed, True),
                _strategy.wrap_topi_schedule(schedule_conv2d_packed),
                name="conv2d_packed.vta",
            )
        else:  # depthwise_conv2d
            strategy.add_implementation(
                _strategy.wrap_compute_conv2d(depthwise_conv2d_packed, True),
                _strategy.wrap_topi_schedule(schedule_depthwise_conv2d_packed),
                name="depthwise_conv2d_packed.vta",
            )
        return strategy

    # If it's not packed, run on ARM CPU
    arm_tgt = tvm.target.arm_cpu(target.model)
    return _strategy.arm_cpu.conv2d_strategy_arm_cpu(attrs, inputs, out_type, arm_tgt)


@_strategy.conv2d_transpose_strategy.register("vta")
def conv2d_transpose_strategy_vta(attrs, inputs, out_type, target):
    """conv2d_transpose vta strategy"""
    dilation = topi.utils.get_const_tuple(attrs.dilation)
    layout = attrs.data_layout
    assert dilation == (1, 1), "support for dilation limited to (1, 1)"

    if is_packed_layout(layout):
        strategy = OpStrategy()
        strategy.add_implementation(
            _strategy.wrap_compute_conv2d_transpose(conv2d_transpose_packed),
            _strategy.wrap_topi_schedule(schedule_conv2d_transpose_packed),
            name="conv2d_transpose_packed.vta",
        )
        return strategy

    # If it's not packed, run on ARM CPU
    arm_tgt = tvm.target.arm_cpu(target.model)
    return _strategy.arm_cpu.conv2d_transpose_strategy_arm_cpu(attrs, inputs, out_type, arm_tgt)


@_strategy.dense_strategy.register("vta")
def dense_strategy_vta(attrs, inputs, out_type, target):
    """dense vta strategy"""
    if len(inputs[0].shape) == 4:  # this implies the layout is packed
        strategy = OpStrategy()
        strategy.add_implementation(
            _strategy.wrap_compute_dense(dense_packed),
            _strategy.wrap_topi_schedule(schedule_dense_packed),
            name="dense_packed.vta",
        )
        return strategy
    # If it's not packed, run on ARM CPU
    arm_tgt = tvm.target.arm_cpu(target.model)
    return _strategy.x86.dense_strategy_cpu(attrs, inputs, out_type, arm_tgt)

def wrap_compute_max_pool2d(topi_compute):
    """Wrap max_pool2d topi compute"""
    def _compute_max_pool2d(attrs, inputs, out_type):
        pool_size = topi.utils.get_const_tuple(attrs.pool_size)
        strides = topi.utils.get_const_tuple(attrs.strides)

        p_small = topi.utils.get_const_tuple(attrs.padding)
        if len(p_small) == 1:
            (p_a, p_b, p_c, p_d) = (p_small, p_small, p_small, p_small)
        elif len(p_small) == 2:
            (p_a, p_b) = p_small
            (p_c, p_d) = p_small
        elif len(p_small) == 4:
            (p_a, p_b, p_c, p_d) = p_small
        else:
            (p_a, p_b, p_c, p_d) = (-1, -1, -1, -1) # cause downstream error

        padding = (p_a, p_b, p_c, p_d)
        layout = attrs.layout
        ceil_mode = attrs.ceil_mode
        count_include_pad = True
        assert layout == "NCHW"
        shape = inputs[0].shape
        if len(shape) == 6:
            env = get_env()
            layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_OUT)
        args = [inputs[0], pool_size, strides, padding, 'max', ceil_mode,
                layout, count_include_pad]
        return [topi_compute(*args)]
    return _compute_max_pool2d

def wrap_compute_global_avg_pool2d(topi_compute):
    """Wrap global_avg_pool2d topi compute"""
    def _compute_global_avg_pool2d(attrs, inputs, out_type):
        shape = inputs[0].shape # height and width remain the same regardless of packing
        assert attrs.layout == "NCHW"
        if len(shape) == 6:
            env = get_env()
            layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_OUT)
            args = [inputs[0], [shape[2], shape[3]], [1, 1], [0, 0, 0, 0], 'avg',
                    False, layout, True, [0, 2047]] # input range assumptions
            return [topi_compute(*args)]
        args = [inputs[0], [shape[2], shape[3]], [1, 1], [0, 0, 0, 0], 'avg',
                False, attrs.layout, True] # no input range assumptions
        return [topi_compute(*args)]
    return _compute_global_avg_pool2d

def pool_strategy_vta(attrs, inputs, out_type, target):
    """pool_strategy_vta"""
    strategy = OpStrategy()
    shape = inputs[0].shape

    if len(shape) == 6: # VTA hardware
        env = get_env()
        assert inputs[0].dtype == "int32", "VTA pooling only supports 32-bit input data"
        assert env.LOG_ACC_WIDTH == 5, "VTA pooling only supports 32-bit accumulator"
        if isinstance(attrs, tvm.relay.op.op_attrs.GlobalPool2DAttrs):
            strategy.add_implementation(
                wrap_compute_global_avg_pool2d(pooling_packed),
                _strategy.wrap_topi_schedule(schedule_pooling_packed),
                name="packed_global_avg_pool2d_strategy.vta")
        elif isinstance(attrs, tvm.relay.op.op_attrs.MaxPool2DAttrs):
            strategy.add_implementation(
                wrap_compute_max_pool2d(pooling_packed),
                _strategy.wrap_topi_schedule(schedule_pooling_packed),
                name="packed_max_pool2d_strategy.vta")
    else: # unpacked, so ARM cpu
        if isinstance(attrs, tvm.relay.op.op_attrs.GlobalPool2DAttrs):
            strategy.add_implementation(
                wrap_compute_global_avg_pool2d(topi.nn.pool), _strategy.schedule_pool,
                name="unpacked_global_avg_pool2d_strategy.vta")
        elif isinstance(attrs, tvm.relay.op.op_attrs.MaxPool2DAttrs):
            strategy.add_implementation(
                wrap_compute_max_pool2d(topi.nn.pool), _strategy.schedule_pool,
                name="unpacked_max_pool2d_strategy.vta")
    return strategy

reg.get("nn.global_avg_pool2d").get_attr("FTVMStrategy").register(pool_strategy_vta, "vta")
reg.get("nn.max_pool2d").get_attr("FTVMStrategy").register(pool_strategy_vta, "vta")
