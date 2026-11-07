# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unstable step's NaN or inf must not make /api/train/activations unrenderable."""

import json
import math

from storage import studio_db


def test_non_finite_activation_values_read_back_as_null():
    studio_db.create_run("run-nan", "unsloth/test", "data", "{}", "2026-10-16T00:00:00Z", 10)
    studio_db.insert_activation_record(
        "run-nan",
        3,
        float("nan"),
        {"0": {"mean": float("inf"), "values": [1.0, float("-inf"), float("nan")]}},
        {"q_proj": float("nan")},
        {"q_proj": 0.5},
    )
    (record,) = studio_db.get_activation_records("run-nan")
    assert record["loss"] is None
    assert record["layers"] == {"0": {"mean": None, "values": [1.0, None, None]}}
    assert record["grad_norms"] == {"q_proj": None}
    assert record["lora_norms"] == {"q_proj": 0.5}
    json.dumps(record, allow_nan = False)
    assert not math.isnan(record["lora_norms"]["q_proj"])
