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

"""VTA RPC client function"""
import os
import logging

from .environment import get_env
from .bitstream import download_bitstream, get_bitstream_path


def reconfig_runtime(remote):
    """Reconfigure remote runtime based on current hardware spec.

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session
    """
    env = get_env()
    freconfig = remote.get_function("tvm.contrib.vta.reconfig_runtime")
    freconfig(env.pkg.cfg_json)


def program_fpga(remote, bitstream=None):
    """Upload and program bistream

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session

    bitstream : str, optional
        Path to a local bistream file. If unset, tries to download from cache server.
    """
    if bitstream:
        assert os.path.isfile(bitstream)
    else:
        bitstream = get_bitstream_path()
        if not os.path.isfile(bitstream):
            download_bitstream()

    fprogram = remote.get_function("tvm.contrib.vta.init")
    remote.upload(bitstream)
    fprogram(os.path.basename(bitstream))


def trace_init(remote):
    # pylint: disable=broad-except
    """Initialize the remote trace engine

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session
    """
    try:
        trace_mgr.init_remote_trace(remote)
    except Exception as exception:
        logging.warning("Trace manager: %s", str(exception))


def trace_done(remote):
    # pylint: disable=import-outside-toplevel
    # pylint: disable=broad-except
    """Destruct the remote trace engine

    Parameters
    ----------
    remote : RPCSession
        The TVM RPC session
    """
    import builtins
    try:
        builtins.trace_mgr.done_remote_trace(remote)
        del builtins.trace_mgr
    except Exception as exception:
        logging.warning("Trace manager: %s", str(exception))


def start_power_monitor(remote, interval=1e-6):
    fstart_pm = remote.get_function("tvm.contrib.vta.start_power_monitor")
    fstart_pm(interval)

def stop_power_monitor(remote, file_name='/home/xilinx/power_expts/test.csv'):
    fstop_pm = remote.get_function("tvm.contrib.vta.stop_power_monitor")
    fstop_pm(file_name)

def start_ro_monitor(remote):
    fstart_pm = remote.get_function("tvm.contrib.vta.start_ro_monitor")
    fstart_pm()

def stop_ro_monitor(remote, file_number):
    fstop_pm = remote.get_function("tvm.contrib.vta.stop_ro_monitor")
    fstop_pm(file_number)

def reset_ro_monitor(remote):
    fstart_pm = remote.get_function("tvm.contrib.vta.reset_ro_monitor")
    fstart_pm()