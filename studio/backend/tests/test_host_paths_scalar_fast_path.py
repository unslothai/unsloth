# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact-type scalar shortcut in the redaction walk must not change a single output."""

import enum
import math
from collections import namedtuple

import pytest

from hub.utils import host_paths
from hub.utils.host_paths import (
    _redact,
    _restore,
    cache_reference,
    redact_host_paths,
    redact_inventory_host_paths,
)
from models.training import TrainingRunMetrics, TrainingStatus

HOME = "/home/alice/.cache/huggingface/hub/models--x--y/snapshots/abc"
WIN = "C:\\Users\\alice\\models\\llama"


@pytest.mark.parametrize(
    "value", [None, True, False, 0, -3, 2**80, 1.5, float("inf"), "plain", HOME, WIN, ""]
)
def test_exact_scalars_pass_through_as_the_same_object(value):
    for redact in (redact_host_paths, redact_inventory_host_paths):
        assert redact(value, via_api_key = True) is value
    assert _redact(value, redact_ambiguous_path = True) is value


def test_nan_and_numeric_lists_pass_through():
    nan = float("nan")
    values = [1, 2.5, None, True, nan, "x"]
    out = redact_host_paths({"loss": values, "steps": (1, 2, 3)}, via_api_key = True)
    assert out["steps"] == (1, 2, 3) and type(out["steps"]) is tuple
    assert out["loss"] is not values
    assert [type(v) for v in out["loss"]] == [int, float, type(None), bool, float, str]
    assert all(a is b for a, b in zip(out["loss"], values))
    assert math.isnan(out["loss"][4])


def test_restore_keeps_numbers_and_still_rewrites_text():
    known = {HOME: "ref:handle"}
    payload = [1, 2.0, None, False, f"loading {HOME}", {"loss": [0.5, 0.25]}]
    out = _restore(payload, known)
    assert out == [1, 2.0, None, False, "loading ref:handle", {"loss": [0.5, 0.25]}]
    assert out[0] is payload[0] and out[1] is payload[1]


def test_mixed_payload_matches_the_full_walk():
    # Expected values were produced by the walk before the shortcut existed.
    payload = {
        "message": f"Loading model {HOME}/model.safetensors",
        "warnings": [f"could not read {WIN}\\cfg.json", 3, None, {"local_path": HOME}],
        "model_name": HOME,
        "output_dir": "/srv/out/run1",
        "dataset_path": "/data/x",
        "scanned_dirs": ["/a", "/b"],
        "local_datasets": ["/data/a.jsonl", "hf/ds", 7],
        "path": HOME,
        "metric_history": {"steps": [1, 2], "loss": [0.5, 0.25], "lr": [2e-4, 1e-4]},
        "nested": [[1, [2.5, HOME]], ("x", 4)],
        "repo_path": HOME,
    }
    out = redact_host_paths(payload, via_api_key = True)
    assert out == {
        "message": "Loading model <path>",
        "warnings": ["could not read <path>", 3, None, {"local_path": ""}],
        "model_name": cache_reference(HOME),
        "output_dir": cache_reference("/srv/out/run1"),
        "dataset_path": "",
        "scanned_dirs": [],
        "local_datasets": [cache_reference("/data/a.jsonl"), "hf/ds", 7],
        "path": HOME,
        "metric_history": {"steps": [1, 2], "loss": [0.5, 0.25], "lr": [2e-4, 1e-4]},
        "nested": [[1, [2.5, HOME]], ("x", 4)],
        "repo_path": "",
        "cache_ref": cache_reference(HOME),
    }
    inventory = redact_inventory_host_paths({"path": HOME, "size": 12}, via_api_key = True)
    assert inventory == {"path": "", "size": 12}


def test_training_status_model_redacts_and_keeps_history():
    status = TrainingStatus(
        job_id = "job-1",
        phase = "training",
        is_training_running = True,
        message = f"Loading {HOME}",
        warnings = [f"bad {WIN}\\x.json"],
        details = {"output_dir": "/srv/out/run1"},
        metric_history = {"steps": [1, 2, 3], "loss": [3.0, 2.0, 1.0]},
    )
    out = redact_host_paths(status, via_api_key = True)
    assert out["message"] == "Loading <path>"
    assert out["warnings"] == ["bad <path>"]
    assert out["details"] == {"output_dir": cache_reference("/srv/out/run1")}
    assert out["metric_history"] == {"steps": [1, 2, 3], "loss": [3.0, 2.0, 1.0]}
    metrics = redact_host_paths(TrainingRunMetrics(step_history = [1, 2]), via_api_key = True)
    assert metrics["step_history"] == [1, 2] and metrics["loss_history"] == []


class _DumpInt(int):
    def model_dump(self):
        return {"local_path": HOME, "value": int(self)}


class _DumpStr(str):
    def model_dump(self):
        return {"repo_path": HOME}


class _Flag(enum.IntEnum):
    ON = 1


class _MyFloat(float):
    pass


def test_subclasses_still_take_the_full_walk():
    assert redact_host_paths(_DumpInt(5), via_api_key = True) == {"local_path": "", "value": 5}
    assert redact_host_paths([_DumpStr("x")], via_api_key = True) == [
        {"repo_path": "", "cache_ref": cache_reference(HOME)}
    ]
    assert _restore([_DumpInt(2)], {HOME: "ref:h"}) == [{"local_path": "ref:h", "value": 2}]
    flag, number = _Flag.ON, _MyFloat(1.5)
    out = redact_host_paths([flag, number], via_api_key = True)
    assert out[0] is flag and out[1] is number
    assert type(_DumpInt(1)) not in host_paths._REDACT_PASSTHROUGH_TYPES


def test_named_tuple_rebuild_is_unchanged():
    Row = namedtuple("Row", "output_dir loss")
    out = redact_host_paths(Row("/srv/out/run1", 0.5), via_api_key = True)
    assert type(out) is Row and out == Row("/srv/out/run1", 0.5)


def test_browser_callers_get_the_payload_back():
    payload = {"local_path": HOME, "loss": [1.0]}
    assert redact_host_paths(payload, via_api_key = False) is payload
