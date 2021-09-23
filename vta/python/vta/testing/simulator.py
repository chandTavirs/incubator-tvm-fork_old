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

"""Utilities to start simulator."""
import ctypes
import json
import tvm
from ..environment import get_env
from ..libinfo import find_libvta


def _load_sw():
    """Load hardware library for simulator."""

    env = get_env()
    lib_driver_map = {
        'sim':  'libvta_fsim',
        'tsim': 'libvta_tsim',
        'bsim': 'libvta_bsim'
        }
    if env.TARGET not in lib_driver_map:
        return []

    lib_driver_name = lib_driver_map[env.TARGET]

    # Load driver library
    lib_driver = find_libvta(lib_driver_name, optional=True)
    assert lib_driver
    try:
        libs = [ctypes.CDLL(lib_driver[0], ctypes.RTLD_GLOBAL)]
    except OSError:
        return []

    if env.TARGET == "tsim":
        lib_hw = find_libvta("libvta_hw", optional=True)
        assert lib_hw  # make sure to make in ${VTA_HW_PATH}/hardware/chisel
        try:
            f = tvm.get_global_func("vta.tsim.init")
            m = tvm.runtime.load_module(lib_hw[0], "vta-tsim")
            f(m)
            # return lib_hw
        except OSError:
            return []
    elif env.TARGET == 'bsim':
        # pylint: disable=import-outside-toplevel
        from .. import beh
        f = tvm.runtime.convert(beh.beh_model)
        set_behavioral_model = tvm.get_global_func("vta.bsim.set_behavioral_model", False)
        set_behavioral_model(f, beh.DRAM_ARRAY)

    return libs


def enabled():
    """Check if simulator is enabled."""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    return f is not None


def clear_stats():
    """Clear profiler statistics."""
    env = get_env()
    if env.TARGET == "sim":
        f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    elif env.TARGET == 'bsim':
        f = None
    else:
        f = tvm.get_global_func("vta.tsim.profiler_clear", True)
    if f:
        f()


def stats():
    """Get profiler statistics

    Returns
    -------
    stats : dict
        Current profiler statistics
    """
    env = get_env()
    if env.TARGET == "sim":
        x = tvm.get_global_func("vta.simulator.profiler_status")()
    elif env.TARGET == 'bsim':
        x = "{}"
        #raise Exception('stats on bsim not implemented')
    else:
        x = tvm.get_global_func("vta.tsim.profiler_status")()
    return json.loads(x)


# debug flag to skip execution.
DEBUG_SKIP_EXEC = 1


def debug_mode(flag):
    """Set debug mode
    Paramaters
    ----------
    flag : int
        The debug flag, 0 means clear all flags.
    """
    tvm.get_global_func("vta.simulator.profiler_debug_mode")(flag)


LIBS = _load_sw()
