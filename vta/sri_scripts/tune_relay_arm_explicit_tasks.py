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
"""
Auto-tuning Explicitly Specified Tasks
=======================================

This script allows you to explicitly specify which tasks to tune instead of
extracting them from a network. Useful for tuning specific dense and conv2d
operations with particular shapes.
"""

import os
import sys
import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
# Import topi to register the compute and schedule functions
from tvm import topi

#################################################################
# Explicitly Define Tasks to Tune
#################################################################

def create_explicit_tasks(target):
    """
    Create a list of tasks to tune explicitly.

    Returns:
        List of Task objects to tune
    """
    tasks = []

    # Dense tasks for (10, 512)
    dense_512_args = (('TENSOR', (1, 512), 'float32'), ('TENSOR', (10, 512), 'float32'), None, 'float32')

    tasks.append(autotvm.task.create(
        'dense_nopack.x86',
        args=dense_512_args,
        target=target
    ))

    tasks.append(autotvm.task.create(
        'dense_pack.x86',
        args=dense_512_args,
        target=target
    ))

    # Dense tasks for (10, 256)
    dense_256_args = (('TENSOR', (1, 256), 'float32'), ('TENSOR', (10, 256), 'float32'), None, 'float32')

    tasks.append(autotvm.task.create(
        'dense_nopack.x86',
        args=dense_256_args,
        target=target
    ))

    tasks.append(autotvm.task.create(
        'dense_pack.x86',
        args=dense_256_args,
        target=target
    ))

    # Conv2d task with (1, 3, 224, 224) input and (128, 3, 7, 7) weights
    conv2d_args = (
        ('TENSOR', (1, 3, 224, 224), 'float32'),
        ('TENSOR', (128, 3, 7, 7), 'float32'),
        (2, 2),  # stride
        (3, 3, 3, 3),  # padding
        (1, 1),  # dilation
        'float32'
    )

    tasks.append(autotvm.task.create(
        'conv2d_nchw_spatial_pack.arm_cpu',
        args=conv2d_args,
        target=target
    ))

    return tasks


#################################################################
# Tuning Function
#################################################################

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    """Tune a list of tasks and save the results to a log file."""
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        print("=" * 80)
        print("Tuning Task %d/%d: %s" % (i + 1, len(tasks), str(tsk)))
        print("=" * 80)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        print(f"Tuning with {tsk_trial} trials (config space size: {len(tsk.config_space)})")

        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
    print("\n" + "=" * 80)
    print(f"Tuning completed! Best results saved to: {log_filename}")
    print("=" * 80)


#################################################################
# Main Tuning Function
#################################################################

def tune_explicit_tasks(tuning_opt, target):
    """Create explicit tasks and tune them."""
    print("Creating explicit tasks...")
    tasks = create_explicit_tasks(target)

    print(f"\nFound {len(tasks)} tasks to tune:")
    for i, task in enumerate(tasks):
        print(f"  {i+1}. {task.name}")
        print(f"     Workload: {task.workload[0]}")
        if 'dense' in task.name:
            print(f"     Input shape: {task.args[0][1]}, Weight shape: {task.args[1][1]}")
        elif 'conv2d' in task.name:
            print(f"     Input shape: {task.args[0][1]}, Weight shape: {task.args[1][1]}")
            print(f"     Stride: {task.args[2]}, Padding: {task.args[3]}, Dilation: {task.args[4]}")
        print()

    # run tuning tasks
    print("Starting tuning process...")
    tune_tasks(tasks, **tuning_opt)


#################################################################
# Configuration and Execution
#################################################################

if __name__ == "__main__":
    # Target configuration
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")

    # Device key for RPC tracker
    device_key = "zcu104"

    # Set this to True if you use android phone
    use_android = False

    # Tuning options
    log_file = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/arm_cpu_explicit_tasks_%s.log" % device_key

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1000,
        "early_stopping": 800,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
            runner=autotvm.RPCRunner(
                device_key,
                host="127.0.0.1",
                port=9190,
                number=5,
                timeout=10,
            ),
        ),
    }

    # Run tuning
    tune_explicit_tasks(tuning_option, target)
