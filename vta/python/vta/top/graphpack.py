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
# pylint: disable=unused-argument

# Modified by contributors from Intel Labs

"""A Relay implementation of graph packing."""

import numpy as np
import tvm
from tvm import relay
from tvm import topi
from tvm.relay import op, transform
from tvm.relay import ExprMutator

from ..environment import get_env


def run_opt_pass(expr, opt_pass):
    """Exectue a relay pass."""
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    # print(mod.astext(show_meta_data=False))
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def _to_shape(shape):
    """convert shape into tuple."""
    return tuple(int(sh) for sh in shape)


def _pack_batch_channel(data, dshape, bfactor, block, typetrack):
    """Pack the data channel dimension."""
    assert int(dshape[0]) % bfactor == 0
    if int(dshape[1]) % block == 0:
        newshape = (int(dshape[0]) // bfactor, bfactor,
                    int(dshape[1]) // block, block,
                    int(dshape[2]), int(dshape[3]))
        data = op.reshape(data, newshape=newshape)
    else:
        channel_pad = block - int(dshape[1]) % block
        data = op.nn.pad(data, [[0, 0], [0, channel_pad], [0, 0], [0, 0]])
        newshape = (int(dshape[0]) // bfactor, bfactor,
                    int(dshape[1]) // block + 1, block,
                    int(dshape[2]), int(dshape[3]))
        data = op.reshape(data, newshape=newshape)
    data = op.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    if typetrack:
        newaxes = [newshape[0], newshape[2],
                   newshape[4], newshape[5],
                   newshape[1], newshape[3]]
        data._checked_type_ = tvm.ir.tensor_type.TensorType(newaxes)
    return data

def _pack_batch_channel_dense(data, dshape, bfactor, block):
    """Pack the data channel dimension for dense operator.
    """
    assert int(dshape[0]) % bfactor == 0
    if int(dshape[1]) % block == 0:
        newshape = (int(dshape[0]) // bfactor, bfactor,
                    int(dshape[1]) // block, block)
        data = op.reshape(data, newshape=newshape)
    else:
        channel_pad = block - int(dshape[1]) % block
        data = op.nn.pad(data, [[0, 0], [0, channel_pad]])
        newshape = (int(dshape[0]) // bfactor, bfactor,
                    int(dshape[1]) // block + 1, block)
        data = op.reshape(data, newshape=newshape)
    data = op.transpose(
        data, axes=(0, 2, 1, 3))
    return data


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _dense_acc_tile_ok(b_tiles, co_tiles, env):
    """Return True if the accumulator tile fits in VTA local.acc_buffer."""
    return int(b_tiles) * int(co_tiles) <= int(env.ACC_BUFF_SIZE)


def _chunk_packed_dense(
    packed_data,
    packed_weight,
    out_dtype,
    *,
    b_outer,
    k_outer,
    b_inner,
    k_inner,
    c_outer,
    c_inner,
):
    """Chunk packed dense along the packed batch axis to satisfy VTA ACC buffer.

    packed_data shape  = [B_outer, K_outer, B_inner, K_inner]
    packed_weight shape= [C_outer, K_outer, C_inner, K_inner]
    output shape       = [B_outer, C_outer, B_inner, C_inner]

    We split B_outer into slices, run dense per slice, then concatenate along axis 0.
    Each chunk's dense output is wrapped in the VTA epilogue (right_shift + clip + cast
    + copy + stop_fusion + cast back to int32) before concatenation, ensuring consistent
    requantization per chunk rather than on the entire concatenated result.
    """
    env = get_env()

    # The caller provides compile-time constant packed shapes. Keep this helper side-effect free.
    b_outer = int(b_outer)
    k_outer = int(k_outer)
    b_inner = int(b_inner)
    k_inner = int(k_inner)
    c_outer = int(c_outer)
    c_inner = int(c_inner)

    # Conservative on-chip accumulator constraint from schedule (tile_b_inner * tile_co_inner).
    # In current schedule tile_co_inner is forced to 1, so we bound by ACC buffer directly.
    max_b_outer_per_call = int(env.ACC_BUFF_SIZE) // 2
    if max_b_outer_per_call <= 0:
        return op.nn.dense(packed_data, packed_weight, out_dtype=out_dtype)

    if b_outer <= max_b_outer_per_call:
        return op.nn.dense(packed_data, packed_weight, out_dtype=out_dtype)

    def _apply_vta_dense_epilogue(dense_i32):
        """Wrap a dense int32 output in the VTA requantization epilogue."""
        shifted = relay.op.tensor.right_shift(
            dense_i32, relay.Constant(tvm.nd.array(np.array(8, dtype="int32"))))
        clipped = relay.op.tensor.clip(shifted, -127., 127.)
        cast_i8 = relay.op.transform.cast(clipped, "int8")
        copied = relay.Call(op.op.get('copy'), [cast_i8])
        stopped = relay.annotation.stop_fusion(copied)
        return relay.op.transform.cast(stopped, "int32")

    parts = []
    for begin in range(0, b_outer, max_b_outer_per_call):
        end = min(b_outer, begin + max_b_outer_per_call)
        data_slice = op.strided_slice(
            packed_data,
            begin=[begin, 0, 0, 0],
            end=[end, k_outer, b_inner, k_inner],
            strides=[1, 1, 1, 1],
        )
        dense_chunk = op.nn.dense(data_slice, packed_weight, out_dtype="int32")
        epilogued_chunk = _apply_vta_dense_epilogue(dense_chunk)
        parts.append(epilogued_chunk)

    return op.concatenate(parts, axis=0)

def _unpack_batch_channel(data, old_shape, block, typetrack):
    """Unpack the data channel dimension.
    """
    if len(old_shape) == 4:
        data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
        data = op.reshape(data, newshape=old_shape)
    else:
        data = op.transpose(data, axes=(0, 2, 1, 3))
        if int(old_shape[1]) % block == 0:
            padding = 0
        else:
            padding = block - int(old_shape[1]) % block
        new_shape = (old_shape[0], old_shape[1] + padding)
        data = op.reshape(data, newshape=new_shape)
        data = op.strided_slice(data, begin=[0, 0], end=old_shape)
    if typetrack:
        data._checked_type_ = tvm.ir.tensor_type.TensorType(old_shape)
    return data


def _const_shape_match(data, dshape, block, typetrack):
    """ Pad the constant if the shape[0] not divisible by blockout.
    """
    assert len(dshape) == 3 or len(dshape) == 1
    pad_width = int(dshape[0]) % block
    if pad_width != 0:
        pad_width = block - pad_width
        if len(dshape) == 3:
            data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0]])
            dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2]])
        else:
            data = op.nn.pad(data, [[0, pad_width]])
            dshape = tuple([dshape[0] + pad_width])
    if typetrack:
        data._checked_type_ = tvm.ir.tensor_type.TensorType(dshape)
    return data, dshape

def _weight_shape_match(data, dshape, channels, blockout, blockin, typetrack):
    """ Pad the weight if the shape[0] not divisible by blockout.
        Pad the weight if the shape[1] not divisible by blockin
    """
    assert len(dshape) == 4
    pad_width = int(dshape[0]) % blockout
    channels_pad = int(channels) % blockout
    in_channel_width = int(dshape[1]) % blockin
    if pad_width != 0:
        pad_width = blockout - pad_width
        data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2], dshape[3]])
    if in_channel_width != 0:
        in_channel_width = blockin - in_channel_width
        data = op.nn.pad(data, [[0, 0], [0, in_channel_width], [0, 0], [0, 0]])
        dshape = tuple([dshape[0], dshape[1] + in_channel_width, dshape[2], dshape[3]])
    if channels_pad != 0:
        channels = channels + (blockout - channels_pad)
    if typetrack:
        data._checked_type_ = tvm.ir.tensor_type.TensorType(dshape)

    return data, dshape, channels

def _weight_shape_match_dense(data, dshape, units, blockout, blockin):
    """ Pad the weight if the shape[0] not divisible by blockout.
        Pad the weight if the shape[1] not divisible by blockin
    """
    assert len(dshape) == 2
    if units is None:
        # Relay allows nn.dense(..., units=None); infer from weight rows [units, in_dim].
        units = int(dshape[0])
    pad_width = int(dshape[0]) % blockout
    units_pad = int(units) % blockout
    in_feat_width = int(dshape[1]) % blockin
    if pad_width != 0:
        pad_width = blockout - pad_width
        data = op.nn.pad(data, [[0, pad_width], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1]])
    if in_feat_width != 0:
        in_feat_width = blockin - in_feat_width
        data = op.nn.pad(data, [[0, 0], [0, in_feat_width]])
        dshape = tuple([dshape[0], dshape[1] + in_feat_width])
    if units_pad != 0:
        units = units + (blockout - units_pad)

    return data, dshape, units

def _weight_shape_match_transpose(data, dshape, channels, cfactor_out):
    """Pad the weight if the shape[1] not divisible by cfactor_out."""
    assert len(dshape) == 4
    pad_width = int(dshape[1]) % cfactor_out
    channels_pad = int(channels) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, 0], [0, pad_width], [0, 0], [0, 0]])
        dshape = tuple(dshape[0], [dshape[1] + pad_width, dshape[2], dshape[3]])

    if channels_pad != 0:
        channels = channels + (cfactor_out - channels_pad)

    return data, dshape, channels

def _pack_weight(data, dshape, blockout, blockin, typetrack):
    """Pack the weight into packed format."""
    assert len(dshape) == 4
    assert int(dshape[0]) % blockout == 0
    assert int(dshape[1]) % blockin == 0
    newshape = (int(dshape[0]) // blockout, blockout,
                int(dshape[1]) // blockin, blockin,
                int(dshape[2]), int(dshape[3]))
    data = op.reshape(data, newshape=newshape)
    data = op.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    if typetrack:
        newaxes = [newshape[0], newshape[2],
                   newshape[4], newshape[5],
                   newshape[1], newshape[3]]
        data._checked_type_ = tvm.ir.tensor_type.TensorType(newaxes)

    return data

def _pack_weight_dense(data, dshape, blockout, blockin):
    """Pack the dense weight into packed format.
    """
    assert len(dshape) == 2
    assert int(dshape[0]) % blockout == 0
    assert int(dshape[1]) % blockin == 0
    newshape = (int(dshape[0]) // blockout, blockout,
                int(dshape[1]) // blockin, blockin)
    data = op.reshape(data, newshape=newshape)
    data = op.transpose(
        data, axes=(0, 2, 1, 3))

    return data

def _pack_weight_conv2d_transpose(data, dshape, cfactor):
    """Pack the weight into packed format."""
    dshape = _to_shape(dshape)
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = op.reshape(
        data,
        newshape=(
            dshape[0] // cfactor,
            cfactor,
            dshape[1] // cfactor,
            cfactor,
            dshape[2],
            dshape[3],
        ),
    )
    data = op.transpose(data, axes=(2, 0, 4, 5, 3, 1))
    return data


def _pack_const(data, dshape, dtype, bfactor, block, typetrack):
    """Pack a constant parameter."""
    dshape = _to_shape(dshape)
    assert len(dshape) == 3
    # assert dshape[0] % cfactor == 0
    if dshape[0] % block != 0:
        pad_width = block - dshape[0]
        data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2]])
    data = op.reshape(data,
                      newshape=(dshape[0] // block,
                                block, dshape[1],
                                dshape[2], 1))
    data = op.transpose(
        data, axes=(0, 2, 3, 4, 1))

    # broadcast batch dimension to bfactor
    newshape = (dshape[0]//block, dshape[1], dshape[2], bfactor, block)
    data = op.broadcast_to(data, shape=newshape)
    if typetrack:
        data._checked_type_ = tvm.ir.tensor_type.TensorType(newshape)
    return data

def _pack_const_dense(data, dshape, dtype, bfactor, block):
    """Pack a constant parameter.
    """
    dshape = _to_shape(dshape)
    assert len(dshape) == 1
    data = op.reshape(data,
                      newshape=(dshape[0] // block,
                                block, 1))
    data = op.transpose(data, axes=(0, 2, 1))

    # broadcast batch dimension to bfactor
    newshape = (dshape[0]//block, bfactor, block)
    data = op.broadcast_to(data, shape=newshape)
    return data

def _get_tensor_shape(node):
    """Get node shape."""
    if isinstance(node.checked_type, relay.ty.TensorType):
        return _to_shape(node.checked_type.shape)
    return []


def _get_tensor_type(node):
    """Get node type."""
    if isinstance(node.checked_type, relay.ty.TensorType):
        return node.checked_type.dtype
    return "float32"


def _operator_idx_inc(expr, count_meta, operator_current_idx):
    """Increase operator index"""
    if isinstance(expr, relay.expr.Constant):
        operator_current_idx = operator_current_idx + 1 if count_meta else operator_current_idx
    else:
        operator_current_idx = operator_current_idx + 1
    return operator_current_idx


class ExprDeviceAnnot(ExprMutator):
    """Visitor to perform graph annotation on an AST.

    Parameters
    ----------
    start: int
        the start location to mark run on vta (inclusive)
    end: int
        the end location to mark run on vta (exclusive)

    Returns
    ---------
    None
    """

    def __init__(self, start=-1, end=-1):
        self.ext_ctx = tvm.context("ext_dev")
        self.cpu_ctx = tvm.context("cpu")
        self.cast = op.op.get("cast")
        self.counter = -1
        self.start = start
        self.end = end
        super().__init__()

    def visit_call(self, call):
        """ Visit the children. """
        # First visit the children.
        args = [self.visit(arg) for arg in call.args]

        self.counter += 1
        if self.counter == self.start:
            ret = relay.Call(call.op, args, call.attrs)
            ret = relay.annotation.on_device(ret, self.ext_ctx)
            return ret

        if self.counter == self.end:
            ret = relay.Call(call.op, args, call.attrs)
            ret = relay.annotation.on_device(ret, self.cpu_ctx)
            return ret

        if self.counter > self.start and self.counter < self.end:
            ret = relay.Call(call.op, args, call.attrs)

            # skip the float op, i.e., float->int cast
            if self.is_float_op(call):
                return ret

            return relay.annotation.on_device(ret, self.ext_ctx)

        return relay.Call(self.visit(call.op), args, call.attrs)

    def is_float_op(self, call):
        """check if this op belongs to a float op
        in general, float op's odtype is float;
        a special case is float->int cast, which follow this op sequence:
        multiply(float) -> round(float) -> clip(float) -> cast(int);
        """
        args = call.args
        odtype = _get_tensor_type(call)

        if odtype == "float32":
            return True

        if call.op == self.cast:
            idtype = _get_tensor_type(args[0])
            if idtype == "float32":
                return True

        return False


class ExprLocator(ExprMutator):
    """Visitor to locate op on an AST."""

    def __init__(self):
        self.counter = -1
        self.op2nodes = {}
        super().__init__()

    def visit_call(self, call):
        """ Visit the children. """
        # First visit the children.
        args = [self.visit(arg) for arg in call.args]

        odtype = _get_tensor_type(call)
        self.counter += 1
        if (call.op, odtype) in self.op2nodes:
            self.op2nodes[(call.op, odtype)].append(self.counter)
        else:
            self.op2nodes[(call.op, odtype)] = [self.counter]

        return relay.Call(self.visit(call.op), args, call.attrs)


class ExprPack(ExprMutator):
    """Visitor to perform graph packing on an AST."""

    def __init__(self, bfactor, blockin, blockout, weight_bits):
        self.bfactor = bfactor
        self.blockin = blockin
        self.blockout = blockout
        # self.typetrack = False
        self.typetrack = False
        self.is_packed = False
        if self.blockin != self.blockout:
            self.typetrack = True
        self.weight_bits = weight_bits
        self.start_pack = False
        # Cache Operator the algorithm matches against.
        self.bitpack_start = op.op.get("annotation.bitpack_start")
        self.bitpack_end = op.op.get("annotation.bitpack_end")
        self.conv2d = op.op.get("nn.conv2d")
        self.conv2d_transpose = op.op.get("nn.conv2d_transpose")
        self.add = op.op.get("add")
        self.multiply = op.op.get("multiply")
        self.bias_add = op.op.get("nn.bias_add")
        self.pad = op.op.get("nn.pad")
        self.upsampling = op.op.get("nn.upsampling")
        self.reshape = op.op.get("reshape")
        self.strided_slice = op.op.get("strided_slice")
        self.global_avg_pool2d = op.op.get("nn.global_avg_pool2d")
        self.max_pool2d = op.op.get("nn.max_pool2d")
        self.dense = op.op.get("nn.dense")
        self.number_of_conv2d = 0
        super().__init__()

    def _resolve_effective_rank(self, data):
        """Best-effort rank inference for visited expressions.

        Returns an int rank when it can be proven, otherwise None so callers
        can conservatively keep the op unchanged.
        """
        try:
            shape = _get_tensor_shape(data)
            if shape:
                return len(shape)
        except ValueError:
            pass

        probe = data
        for _ in range(8):
            if not isinstance(probe, relay.Call) or not isinstance(probe.op, tvm.ir.Op):
                break
            op_name = probe.op.name
            if op_name == "annotation.stop_fusion" and len(probe.args) == 1:
                probe = probe.args[0]
                continue
            if op_name in ("copy", "cast", "clip", "round") and len(probe.args) == 1:
                probe = probe.args[0]
                continue
            if op_name == "transpose":
                return len(probe.attrs.axes)
            if op_name == "reshape":
                return len(probe.attrs.newshape)
            if op_name == "strided_slice":
                return len(probe.attrs.end)
            if op_name == "right_shift" and len(probe.args) == 1:
                probe = probe.args[0]
                continue
            break

        return None

    def visit_call(self, call):
        """ Visit the children. """
        # First visit the children.
        oshape = _get_tensor_shape(call)
        odtype = _get_tensor_type(call)
        input_types = [arg.checked_type for arg in call.args]
        args = [self.visit(arg) for arg in call.args]

        # Start and stop cases.
        if call.op == self.bitpack_start:
            assert not self.start_pack
            self.start_pack = True
            self.is_packed = True
            return _pack_batch_channel(args[0], oshape, self.bfactor, self.blockin, self.typetrack)
        if call.op == self.bitpack_end:
            if self.start_pack:
                self.start_pack = False
                data = args[0]
                data_shape = _get_tensor_shape(call.args[0])
                if self.is_packed:
                    self.is_packed = False
                    return _unpack_batch_channel(data, data_shape, self.blockout, self.typetrack)
                return data
        if self.start_pack:
            # if odtype == 'float32':
            #     pseudo_break=True
            # Operator cases
            if call.op == self.conv2d and odtype == "int32":
                self.number_of_conv2d += 1
                assert 8 % self.weight_bits == 0
                w_lanes = 8 // self.weight_bits
                if call.attrs.groups == 1:
                    data_layout = "NCHW%dn%dc" % (self.bfactor, self.blockin)
                    kernel_layout = "OIHW%do%di" % (self.blockout, self.blockin)
                else:
                    data_layout = "NCHW%dn%dc" % (self.bfactor, self.blockout)
                    kernel_layout = "OIHW%do%di" % (self.blockout, self.bfactor)
                out_layout = "NCHW%dn%dc" % (self.bfactor, self.blockout)
                data, weight = args
                data_shape = _to_shape(input_types[0].shape)
                kernel_shape = _to_shape(input_types[1].shape)
                channels = call.attrs.channels
                if call.attrs.groups != 1:
                    data_cast = relay.op.transform.cast(data, "int32")
                    data = relay.Call(op.op.get('copy'), [data_cast])
                    weight_cast = relay.op.transform.cast(weight, "int32")
                    weight = relay.Call(op.op.get('copy'), [weight_cast])
                if self.typetrack:
                    data = _unpack_batch_channel(data, data_shape, self.blockout, self.typetrack)
                    if call.attrs.groups == 1:
                        data = _pack_batch_channel(data, data_shape, self.bfactor,
                                                   self.blockin, self.typetrack)
                    else:
                        data = _pack_batch_channel(data, data_shape, self.bfactor,
                                                   self.blockout, self.typetrack)
                if call.attrs.groups == 1:
                    weight, kernel_shape, channels = _weight_shape_match(weight,
                                                                         kernel_shape,
                                                                         channels,
                                                                         self.blockout,
                                                                         self.blockin,
                                                                         self.typetrack)
                    kernel = _pack_weight(weight, kernel_shape, self.blockout,
                                          self.blockin, self.typetrack)
                    groups = call.attrs.groups
                else:
                    weight, kernel_shape, channels = _weight_shape_match(weight,
                                                                         kernel_shape,
                                                                         channels,
                                                                         self.blockout,
                                                                         self.bfactor,
                                                                         self.typetrack)
                    kernel = _pack_weight(weight, kernel_shape, self.blockout,
                                          self.bfactor, self.typetrack)
                    groups = channels.value
                # insert bit packing when necessary
                if w_lanes != 1:
                    assert 8 % w_lanes == 0
                    kernel = op.bitpack(kernel, lanes=w_lanes)

                conv2d = op.nn.conv2d(
                    data,
                    kernel,
                    strides=call.attrs.strides,
                    padding=call.attrs.padding,
                    dilation=call.attrs.dilation,
                    groups=groups,
                    channels=channels,
                    kernel_size=call.attrs.kernel_size,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    out_layout=out_layout,
                    out_dtype=call.attrs.out_dtype,
                )
                if self.typetrack:
                    newshape = [oshape[0]//self.bfactor, oshape[1]//self.blockout,
                                oshape[2], oshape[3],
                                self.bfactor, self.blockout]
                    conv2d._checked_type_ = tvm.ir.tensor_type.TensorType(newshape)
                return conv2d

            if call.op == self.conv2d_transpose and odtype == "int32":
                self.number_of_conv2d += 1
                assert 8 % self.weight_bits == 0
                w_lanes = 8 // self.weight_bits
                if self.start_pack:
                    data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                    kernel_layout = "IOHW%di%do" % (self.cfactor, self.cfactor)
                    data, weight = args
                    data_shape = _to_shape(input_types[0].shape)
                    kernel_shape = _to_shape(input_types[1].shape)
                    channels = call.attrs.channels
                    weight, kernel_shape, channels = _weight_shape_match_transpose(
                        weight, kernel_shape, channels, self.cfactor
                    )
                    kernel = _pack_weight_conv2d_transpose(weight, kernel_shape, self.cfactor)
                    conv2d = op.nn.conv2d_transpose(
                        data,
                        kernel,
                        strides=call.attrs.strides,
                        padding=call.attrs.padding,
                        dilation=call.attrs.dilation,
                        groups=call.attrs.groups,
                        channels=call.attrs.channels,
                        kernel_size=call.attrs.kernel_size,
                        data_layout=data_layout,
                        kernel_layout=kernel_layout,
                        output_padding=call.attrs.output_padding,
                        out_dtype=call.attrs.out_dtype,
                    )
                return conv2d
            if call.op == self.add and \
                    tuple(input_types[0].shape) == tuple(input_types[1].shape):
                if not self.typetrack:
                    pass
                else:
                    arg1, arg2 = args
                    arg1_shape = _get_tensor_shape(args[0])
                    arg2_shape = _get_tensor_shape(args[1])
                    if arg1_shape[-1] != self.blockout:
                        arg1 = _unpack_batch_channel(arg1, input_types[0].shape,
                                                     self.blockout, self.typetrack)
                        arg1 = _pack_batch_channel(arg1, input_types[0].shape,
                                                   self.bfactor, self.blockout,
                                                   self.typetrack)
                    if arg2_shape[-1] != self.blockout:
                        arg2 = _unpack_batch_channel(arg2, input_types[1].shape,
                                                     self.blockout, self.typetrack)
                        arg2 = _pack_batch_channel(arg2, input_types[1].shape,
                                                   self.bfactor, self.blockout,
                                                   self.typetrack)
                    addnode = relay.Call(self.add, [arg1, arg2])
                    newshape = [oshape[0]//self.bfactor, oshape[1]//self.blockout,
                                oshape[2], oshape[3],
                                self.bfactor, self.blockout]
                    addnode._checked_type_ = tvm.ir.tensor_type.TensorType(newshape)
                    return addnode
            elif call.op == self.add and len(input_types[1].shape) == 3:
                data, const = args
                if self.typetrack:
                    data_shape = _get_tensor_shape(data)
                    if data_shape[-1] != self.blockout:
                        data = _unpack_batch_channel(data, input_types[0].shape,
                                                     self.blockout, self.typetrack)
                        data = _pack_batch_channel(data, input_types[0].shape,
                                                   self.bfactor, self.blockout, self.typetrack)
                const, input_shape = _const_shape_match(const,
                                                        input_types[1].shape,
                                                        self.blockout, self.typetrack)
                const = _pack_const(const,
                                    _to_shape(input_shape),
                                    input_types[1].dtype,
                                    self.bfactor,
                                    self.blockout,
                                    self.typetrack)
                addnode = relay.Call(self.add, [data, const])
                newshape = [oshape[0]//self.bfactor, oshape[1]//self.blockout,
                            oshape[2], oshape[3],
                            self.bfactor, self.blockout]
                addnode._checked_type_ = tvm.ir.tensor_type.TensorType(newshape)
                return addnode
            elif call.op == self.add and len(input_types[1].shape) == 1:
                data, const = args
                assert data.op.name == "nn.dense" # check if bias addition for dense
                if self.typetrack:
                    data_shape = _get_tensor_shape(data)
                    if data_shape[-1] != self.blockout:
                        data = _unpack_batch_channel(data, input_types[0].shape,
                                                     self.blockout, self.typetrack)
                        data = _pack_batch_channel(data, input_types[0].shape,
                                                   self.bfactor, self.blockout, self.typetrack)
                const, input_shape = _const_shape_match(const,
                                                        input_types[1].shape,
                                                        self.blockout, self.typetrack)
                const = _pack_const_dense(const,
                                          _to_shape(input_shape),
                                          input_types[1].dtype,
                                          self.bfactor,
                                          self.blockout)
                addnode = relay.Call(self.add, [data, const])
                # Pull in shift and clip changes inside VTA dense schedule
                # This ensures that values are scaled to int8 range before store
                dshift = relay.op.tensor.right_shift(
                    addnode, relay.expr.Constant(tvm.nd.array(np.array(8, dtype="int32"))))
                dclip = relay.op.tensor.clip(dshift, -127., 127.)
                dcast = relay.op.transform.cast(dclip, "int8")
                dcopy = relay.Call(op.op.get('copy'), [dcast])
                dfuse = relay.annotation.stop_fusion(dcopy)
                return dfuse
            elif call.op == self.multiply and \
                    tuple(input_types[0].shape) == tuple(input_types[1].shape):
                pass
            elif call.op == self.multiply and len(input_types[1].shape) == 3:
                data, const = args
                if self.typetrack:
                    data = _unpack_batch_channel(data, input_types[0].shape,
                                                 self.blockout, self.typetrack)
                    data = _pack_batch_channel(data, input_types[0].shape,
                                               self.bfactor, self.blockout, self.typetrack)
                const = _pack_const(const,
                                    _to_shape(input_types[1].shape),
                                    input_types[1].dtype,
                                    self.bfactor,
                                    self.blockout,
                                    self.typetrack)
                productnode = relay.Call(self.multiply, [data, const])
                newshape = [oshape[0]//self.bfactor, oshape[1]//self.blockout,
                            oshape[2], oshape[3],
                            self.bfactor, self.blockout]
                productnode._checked_type_ = tvm.ir.tensor_type.TensorType(newshape)
                return productnode
            elif call.op == self.bias_add:
                data, bias = args
                bias = _pack_const(bias,
                                   _to_shape(input_types[1].shape),
                                   input_types[1].dtype,
                                   self.bfactor,
                                   self.blockout,
                                   self.typetrack)
                return relay.Call(self.add, [data, bias])
            elif call.op == op.op.get("cast") and input_types[0].dtype == "int32":
                cast = relay.Call(op.op.get("cast"), [args[0]], call.attrs)
                return relay.Call(op.op.get("copy"), [cast])
            elif call.op == self.global_avg_pool2d:
                davgpool = relay.Call(self.global_avg_pool2d, [args[0]], call.attrs) # 0-2048 range
                dcast = relay.op.transform.cast(davgpool, "int8") # fuse cast for VTA transfer out
                dfuse = relay.annotation.stop_fusion(dcast) # do not fuse any subsequent ops
                dbig = relay.op.transform.cast(dfuse, "int32") # still -128,127 range
                x = relay.op.tensor.add(
                    dbig, relay.Constant(tvm.nd.array(np.array([128], dtype="int32")))) # 0,256
                y = relay.op.tensor.left_shift(
                    x, relay.Constant(tvm.nd.array(np.array([3], dtype="int32")))) # 0, 2047
                # Unpack to avoid batch_flatten smash the inner batch dimension
                if self.is_packed:
                    y_unpacked = _unpack_batch_channel(y, oshape, self.blockout, self.typetrack)
                    self.is_packed = False
                return y_unpacked
            elif call.op == self.max_pool2d:
                dcast = relay.op.transform.cast(args[0], "int32")
                dmp = relay.Call(self.max_pool2d, [dcast], call.attrs)
                dcast2 = relay.op.transform.cast(dmp, "int8") # fuse an int8 cast
                return relay.annotation.stop_fusion(dcast2) # not fuse any subsequent ops
            elif call.op == self.dense and odtype == "int32":
                data, weight = args
                # Dynamic weight-transform dense ops use units=None (e.g. 25x25, 9x9
                # OFA kernels). Packing them here introduces padded inner-block outputs
                # that break subsequent fixed-size reshapes in the transform pipeline.
                # Keep packing enabled for units=None; reshape handling below trims
                # padded packed-dense outputs back to logical kernel sizes.

                data_shape = _to_shape(input_types[0].shape)
                kernel_shape = _to_shape(input_types[1].shape)
                units = call.attrs.units

                if len(data_shape) == 2:
                    packed_data = _pack_batch_channel_dense(data, data_shape, self.bfactor,
                                                            self.blockin)
                else:
                    return relay.Call(self.dense, [data, weight], call.attrs)

                self.is_packed = True

                if len(kernel_shape) == 2:
                    weight, kernel_shape, units = _weight_shape_match_dense(
                        weight,
                        kernel_shape,
                        units,
                        self.blockout,
                        self.blockin,
                    )
                    kernel = _pack_weight_dense(weight, kernel_shape, self.blockout, self.blockin)
                else:
                    return relay.Call(self.dense, [packed_data, weight], call.attrs)

                # If packed batch outer axis is too large, dense_packed.vta produces no legal
                # schedule (local.acc_buffer / VTA_MAX_XFER overflow). Split into smaller calls.
                # Packed shapes are statically determined from original 2D shapes.
                b_outer = int(data_shape[0]) // int(self.bfactor)
                k_outer = _ceil_div(int(data_shape[1]), int(self.blockin))
                c_outer = int(units) // int(self.blockout)
                dense = _chunk_packed_dense(
                    packed_data,
                    kernel,
                    call.attrs.out_dtype,
                    b_outer=b_outer,
                    k_outer=k_outer,
                    b_inner=int(self.bfactor),
                    k_inner=int(self.blockin),
                    c_outer=c_outer,
                    c_inner=int(self.blockout),
                )
                if self.typetrack:
                    newshape = [oshape[0]//self.bfactor, units//self.blockout,
                                self.bfactor, self.blockout]
                    dense._checked_type_ = tvm.ir.tensor_type.TensorType(newshape)
                return dense
            elif call.op == self.pad:
                pad_width = call.attrs.pad_width
                if len(pad_width) == 6:
                    pass
                elif len(pad_width) == 4:
                    (data,) = args
                    new_pad_width = []
                    new_pad_width.extend(pad_width)
                    for _ in range(2):
                        new_pad_width.append([0, 0])
                    return op.nn.pad(data, pad_value=call.attrs.pad_value, pad_width=new_pad_width)
            elif call.op == self.upsampling:
                (data,) = args
                scale_h = call.attrs.scale_h
                scale_w = call.attrs.scale_w
                data_layout = "NCHW%dn%dc" % (self.bfactor, self.blockin)
                method = call.attrs.method
                align_corners = call.attrs.align_corners
                return op.nn.upsampling(data, scale_h, scale_w, data_layout, method, align_corners)
            elif call.op == self.reshape:
                (data,) = args
                target_newshape = [int(x) for x in call.attrs.newshape]

                # Dynamic transform-matrix path can feed padded dense output
                # (e.g. logical 25 padded to 32). Trim to logical units before reshape.
                if len(target_newshape) == 4 and \
                   target_newshape[-1] in (3, 5, 7) and \
                   target_newshape[-2] == target_newshape[-1] and \
                   all(dim > 0 for dim in target_newshape):
                    probe = data
                    # Unwrap common single-arg wrappers (stop_fusion/copy/cast/...) to reach
                    # the underlying producer. For concatenate-of-dense we need to detect
                    # the pattern where probe is a concatenate of dense calls.
                    def _unwrap_to_base(node):
                        """Descend through common wrapper ops to reach the primary data producer.

                        This unwraps:
                        - unary wrappers with one arg: stop_fusion, copy, cast, clip, round, right_shift,
                          reshape, transpose, strided_slice, nn.pad
                        - binary-ish ops where one arg is a constant (e.g., multiply by constant,
                          add constant). In those cases we follow the non-constant arg.
                        - stops on concatenate and Tuple nodes (they are handled separately).
                        """
                        p = node
                        seen = set()
                        while True:
                            # Prevent pathological loops
                            if id(p) in seen:
                                break
                            seen.add(id(p))

                            if isinstance(p, relay.Call) and isinstance(p.op, tvm.ir.Op):
                                opname = p.op.name
                                # Unary wrappers
                                if opname in (
                                    "annotation.stop_fusion",
                                    "copy",
                                    "cast",
                                    "clip",
                                    "round",
                                    "reshape",
                                    "transpose",
                                    "strided_slice",
                                    "nn.pad",
                                ) and len(p.args) == 1:
                                    p = p.args[0]
                                    continue

                                # Bit-shifts in quant epilogues are effectively unary for provenance:
                                # always follow the data operand (arg0).
                                if opname in ("right_shift", "left_shift") and len(p.args) == 2:
                                    p = p.args[0]
                                    continue

                                # Binary ops with a constant operand: follow the non-constant input
                                if opname in (
                                    "multiply",
                                    "add",
                                    "subtract",
                                    "divide",
                                    "right_shift",
                                    "left_shift",
                                ) and len(p.args) == 2:
                                    left, right = p.args
                                    is_left_const = isinstance(left, relay.Constant)
                                    is_right_const = isinstance(right, relay.Constant)
                                    if is_left_const and not is_right_const:
                                        p = right
                                        continue
                                    if is_right_const and not is_left_const:
                                        p = left
                                        continue

                            # Stop unwrapping for other node types (including concatenate / Tuple)
                            break
                        return p

                    probe = _unwrap_to_base(probe)

                    def _contains_dense(node):
                        found = [False]

                        class _DenseFinder(relay.ExprVisitor):
                            def visit_call(self, c):
                                if isinstance(c.op, tvm.ir.Op) and c.op.name == "nn.dense":
                                    found[0] = True
                                super().visit_call(c)

                        _DenseFinder().visit(node)
                        return found[0]

                    dense_like = False
                    # Case 1: direct dense producer
                    if isinstance(probe, relay.Call) and probe.op == self.dense:
                        dense_like = True
                    # Case 2: concatenate of dense parts (from _chunk_packed_dense)
                    elif isinstance(probe, relay.Call) and isinstance(probe.op, tvm.ir.Op) and probe.op.name == "concatenate":
                        # The concatenate may receive its inputs as multiple args or as a single
                        # Tuple node (common in Relay: concatenate(Tuple([...]))). Normalize to
                        # a flat list of elements to inspect.
                        concat_elems = []
                        if len(probe.args) == 1 and hasattr(probe.args[0], "fields"):
                            # probe.args[0] is a Tuple node
                            concat_elems = list(probe.args[0].fields)
                        else:
                            concat_elems = list(probe.args)

                        # Ensure every concatenated element unwraps to a dense call
                        all_dense = True
                        for a in concat_elems:
                            base = _unwrap_to_base(a)
                            if not (isinstance(base, relay.Call) and base.op == self.dense):
                                all_dense = False
                                break
                        if all_dense:
                            dense_like = True
                    elif hasattr(probe, "fields"):
                        # probe is a bare Tuple (relay.Tuple) of elements (no concatenate call).
                        concat_elems = list(probe.fields)
                        all_dense = True
                        for a in concat_elems:
                            base = _unwrap_to_base(a)
                            if not (isinstance(base, relay.Call) and base.op == self.dense):
                                all_dense = False
                                break
                        if all_dense:
                            dense_like = True

                    # Fallback for wrapped dynamic-weight patterns where direct unwrap
                    # does not land exactly on a dense call.
                    if not dense_like and _contains_dense(data):
                        dense_like = True

                    rows = int(np.prod(target_newshape[:-2]))
                    logical_units = int(target_newshape[-2] * target_newshape[-1])

                    src_shape = None
                    try:
                        src_shape = [int(x) for x in input_types[0].shape]
                    except Exception:  # pylint: disable=broad-except
                        src_shape = None

                    # In ANF the reshape input is often a Var, so producer-based unwrap
                    # may miss dense provenance. Use typed shape gating as fallback:
                    # - unpacked dense: [rows, padded_units]
                    # - packed dense:   [rows, co_outer, b_inner, co_inner]
                    should_trim = False
                    if src_shape is not None:
                        if len(src_shape) == 2 and src_shape[0] == rows and src_shape[1] >= logical_units:
                            should_trim = True
                        elif len(src_shape) == 4 and src_shape[0] == rows:
                            packed_units = int(src_shape[1]) * int(src_shape[2]) * int(src_shape[3])
                            if packed_units >= logical_units:
                                should_trim = True

                    if dense_like or should_trim:
                        data = op.reshape(data, newshape=[rows, -1])
                        dense_trimmed = op.strided_slice(
                            data,
                            begin=[0, 0],
                            end=[rows, logical_units],
                            strides=[1, 1],
                        )
                        return op.reshape(dense_trimmed, newshape=target_newshape)

                # Keep original packed activation unpack behavior for 4D tensors.
                if len(input_types[0].shape) != 4:
                    return relay.Call(self.reshape, [data], call.attrs)

                data_rank = self._resolve_effective_rank(data)
                if data_rank != 6:
                    return relay.Call(self.reshape, [data], call.attrs)
                data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
                return op.reshape(data, [int(x) for x in input_types[0].shape])
            # elif call.op.name == "concatenate":
            #     concat = True
            elif call.op == self.strided_slice:
                (data,) = args
                orig_begin = call.attrs.begin
                orig_end = call.attrs.end

                # Rewrite to packed indices only when the effective slice input is rank-6.
                # Keep rank-4 weight-transform slices untouched.
                data_rank = self._resolve_effective_rank(data)

                # If rank info is stale (common around stop_fusion), recover packed activation
                # slices from pack-state plus N-axis slice bounds, while leaving weight slices 4D.
                if data_rank != 6 and len(orig_begin) == 4 and len(orig_end) == 4:
                    looks_like_activation_batch_slice = (
                        int(orig_begin[0]) >= 0 and int(orig_end[0]) <= self.bfactor
                    )
                    if self.is_packed and looks_like_activation_batch_slice:
                        data_rank = 6

                if data_rank != 6:
                    return relay.Call(self.strided_slice, [data], call.attrs)

                # Original indices in NCHW format
                if len(orig_begin) != 4 or len(orig_end) != 4:
                    return relay.Call(self.strided_slice, [data], call.attrs)

                # Packed format is [N//bfactor, C//block, H, W, bfactor, block]
                # Calculate the block indices for begin and end
                begin_batch_block = orig_begin[0] // self.bfactor
                begin_channel_block = orig_begin[1] // self.blockout

                # Ceil-div for end indices to capture partial blocks.
                end_batch_block = (orig_end[0] + self.bfactor - 1) // self.bfactor
                end_channel_block = (orig_end[1] + self.blockout - 1) // self.blockout

                begin = [begin_batch_block, begin_channel_block, orig_begin[2], orig_begin[3], 0, 0]
                end = [end_batch_block, end_channel_block, orig_end[2], orig_end[3], self.bfactor, self.blockout]
                strides = [1, 1, 1, 1, 1, 1]
                return op.strided_slice(data, begin, end, strides)

        callnode = relay.Call(self.visit(call.op), args, call.attrs)
        if self.typetrack:
            callnode._checked_type_ = tvm.ir.tensor_type.TensorType(oshape)
        return callnode

class BT(Exception):
    pass
def get_subgraph(expr, start_name, stop_name, start_name_idx, stop_name_idx, count_meta):
    """We assume stop_name only appears once for simplicity.
    This constraint will be lifted in the future.
    bitpack_start and bitpack_end are both inclusive.
    """
    bitpack_start = op.op.get("annotation.bitpack_start")
    bitpack_end = op.op.get("annotation.bitpack_end")
    anf = run_opt_pass(expr, transform.ToANormalForm())
    operator_current_idx = 0

    def _recursion(anf, start_found, stop_found, operator_current_idx):
        """Helper to obtain the subgraph."""
        if isinstance(anf, relay.Function):
            return relay.Function(
                anf.params,
                _recursion(anf.body, start_found, stop_found, operator_current_idx),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, relay.expr.Let):
            value = anf.value
            if isinstance(value, relay.expr.Call):
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name == start_name and not start_found:
                        if operator_current_idx == start_name_idx or start_name_idx is None:
                            value = relay.expr.Call(bitpack_start, [value])
                            start_found = True
                    elif value.op.name == stop_name:
                        if operator_current_idx == stop_name_idx or stop_name_idx is None:
                            raise BT()

            operator_current_idx = _operator_idx_inc(value, count_meta, operator_current_idx)

            try:
                return relay.expr.Let(
                    anf.var,
                    value,
                    _recursion(anf.body, start_found, stop_found, operator_current_idx),
                )
            except BT:
                assert start_found
                assert not stop_found
                stop_found = True
                value = relay.expr.Call(bitpack_end, [value])
                # todo: check anf.body has no more stop_name beside that one
                return relay.expr.Let(anf.var, value, anf.body)
        else:
            assert start_found
            assert stop_found
            return anf

    annotated = _recursion(anf, False, False, operator_current_idx)
    return run_opt_pass(annotated, transform.ToGraphNormalForm())

def graph_pack(
    expr,
    bfactor,
    blockin,
    blockout,
    weight_bits,
    start_name="nn.max_pool2d",
    stop_name="nn.global_avg_pool2d",
    start_name_idx=None,
    stop_name_idx=None,
    count_meta=False,
    device_annot=False,
    annot_start_name="nn.conv2d",
    annot_end_name="annotation.stop_fusion",
):
    """Pack the graph into batch&channel packed format.

    Parameters
    ----------
    expr : relay.Expr
       The input program.

    bfactor : int
       The packing factor in batch

    blockin : int
       The packing factor in channel_in

    blockout : int
        The packing factor in channel_out

    weight_bits: int
        The bit-width of the weights.

    start_name: str, optional
       Start packing from certain known node when start_name_idx is None.

    stop_name: str, optional
       Stop packing from certain known node when stop_name_idx is None.

    start_name_idx: int, optional
        When start_name_idx not None, start packing only when node name equal start_name
        and node idx equals start_name_idx.

    stop_name_idx: int, optional
        When stop_name_idx not None, stop packing only when node name equal stop_name
        and node index equals stop_name_idx.

    count_meta:boolean, optional
        When count_meta is False, the operator increase logic would not count the meta that have
        the type 'relay.expr.Constant', start_name_idx and stop_name_idx follow the index from
        'expr.astext(show_meta_data=False)'. When count_meta is True, the operator increase
        logic would count the meta.

    device_annot: boolean, optional
        if we want to annoate the device_type

    annot_start_name: str, optional
        device annotation start node, from which we mark the nodes as `ext_dev`

    annot_end_name: str, optional
        device annotation end node, after which we mark the nodes as 'cpu'

    Returns
    -------
    expr : Expr
        The transformed expression.
    """
    assert isinstance(expr, relay.Function)
    assert (
        (start_name != stop_name)
        or (start_name_idx is None != stop_name_idx is None)
        or (not (start_name_idx is None and stop_name_idx is None))
        or (start_name_idx < stop_name_idx)
    )
    expr = get_subgraph(expr, start_name, stop_name, start_name_idx, stop_name_idx, count_meta)
    try:
        expr = run_opt_pass(expr, transform.InferType())
    except tvm.TVMError as err:
        if "Input tensor shape and reshaped shape are not compatible" not in str(err):
            raise
        expr = _rewrite_prepack_dense_reshape_mismatch(expr)
        expr = run_opt_pass(expr, transform.InferType())
    packer = ExprPack(
        bfactor, blockin,
        blockout, weight_bits)
    expr = packer.visit(expr)
    assert not packer.start_pack
    try:
        expr = run_opt_pass(expr, transform.InferType())
    except tvm.TVMError as err:
        if "Input tensor shape and reshaped shape are not compatible" not in str(err):
            raise
        expr = _rewrite_prepack_dense_reshape_mismatch(expr)
        expr = run_opt_pass(expr, transform.InferType())

    if device_annot:
        expr_locator = ExprLocator()
        expr_locator.visit(expr)

        annot_start = op.op.get(annot_start_name)
        start = expr_locator.op2nodes[(annot_start, "int32")][0]

        annot_end = op.op.get(annot_end_name)
        # we mark the next op to the last stop_fusion on cpu device
        end = expr_locator.op2nodes[(annot_end, "int8")][-1] + 1

        device_annot = ExprDeviceAnnot(start=start, end=end)
        expr = device_annot.visit(expr)
        return run_opt_pass(expr, transform.InferType())

    return expr


def _rewrite_prepack_dense_reshape_mismatch(expr):
    """Trim padded dense outputs before static reshape to avoid pre-pack InferType failures."""

    class _Fix(ExprMutator):
        def _unwrap_to_base(self, node):
            p = node
            seen = set()
            while True:
                if id(p) in seen:
                    break
                seen.add(id(p))
                if isinstance(p, relay.Call) and isinstance(p.op, tvm.ir.Op):
                    opname = p.op.name
                    if opname in (
                        "annotation.stop_fusion",
                        "copy",
                        "cast",
                        "clip",
                        "round",
                        "reshape",
                        "transpose",
                        "strided_slice",
                        "nn.pad",
                    ) and len(p.args) == 1:
                        p = p.args[0]
                        continue
                    if opname in ("right_shift", "left_shift") and len(p.args) == 2:
                        p = p.args[0]
                        continue
                    if opname in (
                        "multiply",
                        "add",
                        "subtract",
                        "divide",
                        "right_shift",
                        "left_shift",
                    ) and len(p.args) == 2:
                        left, right = p.args
                        is_left_const = isinstance(left, relay.Constant)
                        is_right_const = isinstance(right, relay.Constant)
                        if is_left_const and not is_right_const:
                            p = right
                            continue
                        if is_right_const and not is_left_const:
                            p = left
                            continue
                break
            return p

        def visit_call(self, call):
            call = super().visit_call(call)
            if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "reshape"):
                return call
            if len(call.args) != 1:
                return call

            target_newshape = [int(x) for x in call.attrs.newshape]
            if any(dim <= 0 for dim in target_newshape):
                return call

            probe = self._unwrap_to_base(call.args[0])

            dense_like = isinstance(probe, relay.Call) and isinstance(probe.op, tvm.ir.Op) and probe.op.name == "nn.dense"
            if not dense_like:
                found = [False]

                class _DenseFinder(relay.ExprVisitor):
                    def visit_call(self, c):
                        if isinstance(c.op, tvm.ir.Op) and c.op.name == "nn.dense":
                            found[0] = True
                        super().visit_call(c)

                _DenseFinder().visit(call.args[0])
                dense_like = found[0]
            if not dense_like:
                return call

            rows = None
            logical_units = None
            if len(target_newshape) == 4 and target_newshape[-1] in (3, 5, 7) and target_newshape[-2] == target_newshape[-1]:
                rows = int(np.prod(target_newshape[:-2]))
                logical_units = int(target_newshape[-2] * target_newshape[-1])
            elif len(target_newshape) == 2 and target_newshape[1] in (9, 25, 49):
                rows = int(target_newshape[0])
                logical_units = int(target_newshape[1])

            if rows is None or logical_units is None or rows <= 0 or logical_units <= 0:
                return call

            normalized = op.reshape(call.args[0], newshape=[rows, -1])
            trimmed = op.strided_slice(
                normalized,
                begin=[0, 0],
                end=[rows, logical_units],
                strides=[1, 1],
            )
            return op.reshape(trimmed, newshape=target_newshape)

    return _Fix().visit(expr)


def _collect_op_sequence(expr, count_meta=False):
    """Collect (op_name, op_idx) from ANF using graph_pack indexing semantics."""
    anf = run_opt_pass(expr, transform.ToANormalForm())
    operator_current_idx = 0
    op_seq = []

    def _recursion(node, current_idx):
        if isinstance(node, relay.Function):
            return _recursion(node.body, current_idx)
        if isinstance(node, relay.expr.Let):
            value = node.value
            if isinstance(value, relay.expr.Call) and isinstance(value.op, tvm.ir.Op):
                op_seq.append((value.op.name, current_idx))
            current_idx = _operator_idx_inc(value, count_meta, current_idx)
            return _recursion(node.body, current_idx)
        return current_idx

    _recursion(anf, operator_current_idx)
    return op_seq


def graph_pack_dynamic_weights(
    expr,
    bfactor,
    blockin,
    blockout,
    weight_bits,
    start_name="nn.max_pool2d",
    stop_name="nn.global_avg_pool2d",
    start_name_idx=None,
    stop_name_idx=None,
    count_meta=False,
    device_annot=False,
    annot_start_name="nn.conv2d",
    annot_end_name="annotation.stop_fusion",
    pack_all=True,
    allow_fallback=True,
    return_status=False,
):
    """Dynamic-weight-aware graph packing wrapper.

    This API keeps legacy graph_pack behavior by default, but can auto-select a
    full-graph packing range (first op -> last op) for dynamic-weight quantized
    modules where weight-derivation subgraphs (e.g. dense/reshape/slice) must be
    included in the same packed region.

    Parameters mirror graph_pack plus:
    - pack_all: discover start/stop op indices from ANF and pack full graph.
    - allow_fallback: on known dynamic graph_pack shape mismatch, return unpacked expr.
    - return_status: if True, return (expr, used_graph_pack, fallback_reason).
    """
    assert isinstance(expr, relay.Function)

    dyn_start_name = start_name
    dyn_stop_name = stop_name
    dyn_start_idx = start_name_idx
    dyn_stop_idx = stop_name_idx

    if pack_all:
        op_seq = _collect_op_sequence(expr, count_meta=count_meta)
        if not op_seq:
            if return_status:
                return expr, False, "No operator sequence found to pack"
            return expr
        dyn_start_name, dyn_start_idx = op_seq[0]
        dyn_stop_name, dyn_stop_idx = op_seq[-1]

    used_graph_pack = True
    fallback_reason = None
    try:
        packed = graph_pack(
            expr,
            bfactor,
            blockin,
            blockout,
            weight_bits,
            start_name=dyn_start_name,
            stop_name=dyn_stop_name,
            start_name_idx=dyn_start_idx,
            stop_name_idx=dyn_stop_idx,
            count_meta=count_meta,
            device_annot=device_annot,
            annot_start_name=annot_start_name,
            annot_end_name=annot_end_name,
        )
    except Exception as err:
        msg = str(err)
        if allow_fallback and "axes has 6 elements" in msg and "data.ndim = 4" in msg:
            packed = expr
            used_graph_pack = False
            fallback_reason = "graph_pack transpose rank mismatch on dynamic graph"
        elif allow_fallback and "Input tensor shape and reshaped shape are not compatible" in msg:
            packed = expr
            used_graph_pack = False
            fallback_reason = "graph_pack reshape shape mismatch from chunked dense on dynamic graph"
        else:
            raise

    if return_status:
        return packed, used_graph_pack, fallback_reason
    return packed

