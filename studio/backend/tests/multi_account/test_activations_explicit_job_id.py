# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/activations?job_id= reads the caller's own studio.db, so naming another account's run returns nothing."""

import routes.training as tr
from storage.studio_db import create_run, insert_activation_record, upsert_activation_metadata
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
JOB = "job_alice_capture"
META = {
    "model_name": "unsloth/test",
    "num_layers": 1,
    "hidden_size": 2,
    "intermediate_size": 2,
    "captured_channels": [0, 1],
    "capture_interval": 1,
    "max_channels": 2,
    "capture_mlp_out": False,
    "capture_gradients": True,
    "capture_lora_norms": False,
    "lora_targets": [],
    "created_at": "2026-01-01T00:00:00Z",
}


def _read(account, subject):
    return run_as(account, tr.get_activations, JOB, None, None, subject)


def test_a_foreign_job_id_reads_nothing(isolated_auth):
    run_as(ALICE, create_run, JOB, "unsloth/test", "data", "{}", "2026-01-01T00:00:00Z", 1)
    run_as(ALICE, upsert_activation_metadata, JOB, META)
    run_as(
        ALICE,
        insert_activation_record,
        JOB,
        0,
        1.5,
        {"0": {"mean_abs": [1.0, 2.0], "mean": [0.5, -0.5]}},
        None,
        None,
    )
    assert _read(BOB, "bob") == {"metadata": None, "records": []}
    own = _read(ALICE, "alice")
    assert own["metadata"]["model_name"] == "unsloth/test"
    assert [r["step"] for r in own["records"]] == [0]
