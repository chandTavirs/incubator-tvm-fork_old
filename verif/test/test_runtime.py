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

"""Testing runtime using topi group conv2d operator for VTA"""

import json
import os

import pytest
import numpy as np
from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.contrib import utils
from tvm import topi
import tvm.topi.testing
import vta
import vta.testing
from vta.testing import simulator


Workload = namedtuple("GroupConv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter', 'groups',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

# Get batch info from env
env = vta.get_env()

# Mobilenet (grouped variant) workloads
mobilenet_wkls = [
    ('mobilenet.D1', Workload(env.BATCH, 112, 112,   32,   32,  2, 3, 3, 1, 1, 1, 1)),
    ('mobilenet.D2', Workload(env.BATCH, 112, 112,   64,   64,  4, 3, 3, 1, 1, 2, 2)),
    ('mobilenet.D3', Workload(env.BATCH,  56,  56,  128,  128,  8, 3, 3, 1, 1, 1, 1)),
    ('mobilenet.D4', Workload(env.BATCH,  56,  56,  128,  128,  8, 3, 3, 1, 1, 2, 2)),
    ('mobilenet.D5', Workload(env.BATCH,  28,  28,  256,  256, 16, 3, 3, 1, 1, 1, 1)),
    ('mobilenet.D6', Workload(env.BATCH,  28,  28,  256,  256, 16, 3, 3, 1, 1, 2, 2)),
    ('mobilenet.D7', Workload(env.BATCH,  14,  14,  512,  512, 32, 3, 3, 1, 1, 1, 1)),
    ('mobilenet.D8', Workload(env.BATCH,  14,  14,  512,  512, 32, 3, 3, 1, 1, 2, 2)),
    ('mobilenet.D9', Workload(env.BATCH,   7,  7,  1024, 1024, 64, 3, 3, 1, 1, 1, 1)),
]

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x

def run_runtime(env, remote, wl, target,
                     check_correctness=True, print_ir=False,
                     samples=4):

    # Workload assertions
    assert wl.hpad == wl.wpad

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
        fcompute = topi.nn.group_conv2d_nchw
        fschedule = topi.generic.schedule_group_conv2d_nchw
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
        fcompute = vta.top.group_conv2d_packed
        fschedule = vta.top.schedule_group_conv2d_packed

    # Derive shapes depending upon packing
    CI_G = wl.in_filter // wl.groups
    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, CI_G, wl.hkernel, wl.wkernel)
    b_shape = (wl.batch, wl.out_filter, 1, 1)
    if data_pack:
        data_shape = (wl.batch//env.BATCH, max(wl.in_filter//env.BLOCK_IN, wl.groups),
                      wl.height, wl.width, env.BATCH, env.BLOCK_IN)
        kernel_shape = (max(wl.out_filter//env.BLOCK_OUT, wl.groups), max(CI_G//env.BLOCK_IN, 1),
                        wl.hkernel, wl.wkernel, env.BLOCK_OUT, env.BLOCK_IN)
        bias_shape = (wl.batch//env.BATCH, max(wl.out_filter//env.BLOCK_OUT, wl.groups),
                      1, 1, env.BATCH, env.BLOCK_OUT)
    else:
        data_shape = a_shape
        kernel_shape = w_shape
        bias_shape = b_shape
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
    padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))

    # Define base computation schedule
    with target:
        res = fcompute(
            data, kernel, (wl.hstride, wl.wstride), padding, (1, 1),
            wl.groups, env.acc_dtype)
        res = topi.right_shift(res, 8)
        res = topi.add(res, bias)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)
        # Derive base schedule
        s = fschedule([res])
        if print_ir:
            print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))

    # Derive number of ops
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    num_ops = 2 * wl.batch * fout_height * fout_width * wl.hkernel * wl.wkernel * \
        wl.out_filter * wl.in_filter // wl.groups

    def get_ref_data():
        # derive min max for act, wgt, and bias types (max non inclusive)
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min, b_max = 0 - 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2), 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2)
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(data.dtype)
        w_np = np.random.randint(w_min, w_max, size=w_shape).astype(kernel.dtype)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(env.acc_dtype)
        r_np = topi.testing.conv2d_nchw_python(
            a_np.astype(env.acc_dtype), w_np.astype(env.acc_dtype),
            (wl.hstride, wl.wstride), wl.hpad, wl.groups).astype(env.acc_dtype)
        return a_np, w_np, b_np, r_np

    # Data in original format
    data_np, kernel_np, bias_np, res_ref = get_ref_data()
    if data_pack:
        data_zeros = np.zeros((wl.batch, max(wl.in_filter//env.BLOCK_IN, wl.groups)*env.BLOCK_IN,
                                wl.height, wl.width)).astype(data.dtype)
        kernel_zeros = np.zeros((max(wl.out_filter//env.BLOCK_OUT, wl.groups)*env.BLOCK_OUT,
                                 max(CI_G//env.BLOCK_IN, 1)*env.BLOCK_IN,
                                 wl.hkernel, wl.wkernel)).astype(kernel.dtype)
        bias_zeros = np.zeros((wl.batch, max(wl.out_filter//env.BLOCK_OUT, wl.groups)*env.BLOCK_OUT,
                                1, 1)).astype(env.acc_dtype)

        data_idx = max(env.BLOCK_IN, CI_G)
        ker_idx = max(env.BLOCK_OUT, CI_G)
        for gr in range(wl.groups):
            data_zeros[:, data_idx*gr:data_idx*gr+CI_G, :, :] = data_np[:, CI_G*gr:CI_G*(gr+1), :, :]
            kernel_zeros[ker_idx*gr:ker_idx*gr+CI_G, :CI_G, :, :] = kernel_np[CI_G*gr:CI_G*(gr+1), :, :, :]
            bias_zeros[:, ker_idx*gr:ker_idx*gr+CI_G, :, :] = bias_np[:, CI_G*gr:CI_G*(gr+1), :, :]

        data_np = data_zeros.reshape(
            wl.batch//env.BATCH, env.BATCH,
            max(wl.in_filter//env.BLOCK_IN, wl.groups), env.BLOCK_IN,
            wl.height, wl.width).transpose((0, 2, 4, 5, 1, 3))
        kernel_np = kernel_zeros.reshape(
            max(wl.out_filter//env.BLOCK_OUT, wl.groups), env.BLOCK_OUT,
            max(CI_G//env.BLOCK_IN, 1), env.BLOCK_IN,
            wl.hkernel, wl.wkernel).transpose((0, 2, 4, 5, 1, 3))
        bias_np = bias_zeros.reshape(
            wl.batch//env.BATCH, env.BATCH,
            max(wl.out_filter//env.BLOCK_OUT, wl.groups), env.BLOCK_OUT,
            1, 1).transpose((0, 2, 4, 5, 1, 3))
        
    # Build
    if "vta" in target.keys:
        mod = vta.build(s, [data, kernel, bias, res],
                        target=target,
                        target_host=env.target_host,
                        name="conv2d")
    else:
        mod = tvm.build(s, [data, kernel, bias, res],
                        target=target,
                        target_host=env.target_host,
                        name="conv2d")

    # Start tracing from runtime on.
    if env.TARGET in ["de10nano", "pynq"]:
      vta.trace_init(remote)

    temp = utils.tempdir()
    mod.save(temp.relpath("conv2d.o"))
    remote.upload(temp.relpath("conv2d.o"))
    f = remote.load_module("conv2d.o")
    ctx = remote.context(str(target))

    #res_np = np.zeros(topi.utils.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, ctx)      # INP
    kernel_arr = tvm.nd.array(kernel_np, ctx)  # WGT FSIM divergence in input data
    #bias_arr = tvm.nd.array(bias_np, ctx)      # ACC
    #res_arr = tvm.nd.array(res_np, ctx)        # OUT or here

    correct = np.allclose(kernel_np, kernel_arr.asnumpy())

    # In vta sim mode, collect simulator runtime statistics
    stats = {}
    if env.TARGET in ["sim", "tsim", "bsim"]:
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            if env.TARGET == "sim":
                remote.get_function("vta.simulator.profiler_clear")()
            elif env.TARGET == "tsim":
                remote.get_function("vta.tsim.profiler_clear")()
            #f(data_arr, kernel_arr, bias_arr, res_arr)
            if env.TARGET == "sim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            elif env.TARGET == "tsim":
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            #f(data_arr, kernel_arr, bias_arr, res_arr)
            stats = simulator.stats()
    else:
        #f(data_arr, kernel_arr, bias_arr, res_arr)
        if env.TARGET in ["de10nano", "pynq"]:
            vta.trace_done(remote)

    status = "PASSED" if correct else "FAILED"
    print(status)
    if "arm_cpu" in target.keys:
        device = "CPU"
    elif "vta" in target.keys:
        device = "VTA"

    return correct, stats

@pytest.mark.parametrize("device", ["vta", "arm_cpu"])
def test_runtime(device, workload=None):
    def _run(env, remote):
        if device == "vta":
            target = env.target
            if env.TARGET not in ["sim", "tsim", "bsim"]:
                assert tvm.runtime.enabled("rpc")
        elif device == "arm_cpu":
            target = env.target_vta_cpu
        with autotvm.tophub.context(target): # load pre-tuned schedule parameters
            for name, wl in mobilenet_wkls:
                if workload is not None and name != workload:
                    continue
                print(name, wl)
                run_runtime(env, remote, wl, target)
    vta.testing.run(_run)

if __name__ == "__main__":
    test_conv2d(device="arm_cpu")
    test_conv2d(device="vta")
