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
import time

import copy
import multiprocessing
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm import MeasureInput, MeasureResult
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel, _same_task_identity

from test_autotvm_common import get_sample_task, get_sample_records


def test_fit():
    task, target = get_sample_task()
    records = get_sample_records(n=500)

    base_model = XGBoostCostModel(task, feature_type="itervar", loss_type="rank")
    base_model.fit_log(records, plan_size=32)

    upper_model = XGBoostCostModel(task, feature_type="itervar", loss_type="rank")
    upper_model.load_basemodel(base_model)

    xs = np.arange(10)
    ys = np.arange(10)

    upper_model.fit(xs, ys, plan_size=32)


def fit_spawn():
    assert multiprocessing.get_start_method(False) == "spawn"
    test_fit()


def test_fit_spawn():
    # Subprocesses inherit the spawn method of their parents
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=test_fit)
    p.start()
    p.join()


def test_tuner():
    task, target = get_sample_task()
    records = get_sample_records(n=100)

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.load_history(records)


def test_same_task_identity_filter():
    task, target = get_sample_task()

    # Matching task/workload/target should be accepted.
    match_task = copy.deepcopy(task)
    match_task.config_space.code_hash = "cafecafe"
    match_inp = MeasureInput(target, match_task, match_task.config_space.get(0))
    match_inp.config.code_hash = "cafecafe"
    assert _same_task_identity(match_inp, match_task)

    # Mismatched workload should be rejected even if the task name is the same.
    mismatch_task, _ = get_sample_task(n=64)
    mismatch_task.config_space.code_hash = "cafecafe"
    mismatch_conf = mismatch_task.config_space.get(0)
    mismatch_conf.code_hash = "cafecafe"
    mismatch_inp = MeasureInput(target, mismatch_task, mismatch_conf)
    assert not _same_task_identity(mismatch_inp, match_task)

    # Mismatched code hash should also be rejected when both hashes are known.
    hash_task = copy.deepcopy(task)
    hash_task.config_space.code_hash = "cafecafe"
    hash_inp = MeasureInput(target, hash_task, hash_task.config_space.get(0))
    hash_inp.config.code_hash = "dbffdbff"
    assert not _same_task_identity(hash_inp, hash_task)


if __name__ == "__main__":
    test_fit()
    test_fit_spawn()
    test_tuner()
    test_same_task_identity_filter()
