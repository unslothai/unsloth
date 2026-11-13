# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from models.training import TrainingStartRequest
from routes import openai_fine_tuning

BODIES = json.loads(
    (Path(__file__).parent / "fixtures" / "training_api_job_bodies.json").read_text(
        encoding = "utf-8"
    )
)
FILE_ID = "file-0123456789abcdef0123456789abcdef"
OPENAI_FIELDS = {"model", "hyperparameters", "seed", "suffix"}


@pytest.fixture(autouse = True)
def uploaded_file(monkeypatch, tmp_path):
    uploaded = tmp_path / f"{FILE_ID}.jsonl"
    uploaded.write_text(
        '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}\n',
        encoding = "utf-8",
    )
    monkeypatch.setattr(openai_fine_tuning, "_find_file", lambda file_id: uploaded)
    return uploaded


def _request(body: dict) -> TrainingStartRequest:
    create = openai_fine_tuning.FineTuningJobCreate.model_validate(
        {**body, "training_file": FILE_ID}
    )
    return TrainingStartRequest.model_validate(openai_fine_tuning._training_config(create))


@pytest.mark.parametrize("name", sorted(BODIES))
def test_generated_body_is_accepted_with_its_settings(name, uploaded_file):
    body = BODIES[name]
    request = _request(body)
    assert request.model_name == body["model"]
    assert request.local_datasets == [str(uploaded_file)]
    assert request.num_epochs == body["hyperparameters"]["n_epochs"]
    assert request.batch_size == body["hyperparameters"]["batch_size"]
    assert request.random_seed == body["seed"]
    assert request.project_name == body.get("suffix")

    assert float(request.learning_rate) == float(body["learning_rate"])
    extras = {
        key: value
        for key, value in body.items()
        if key not in OPENAI_FIELDS and key != "learning_rate"
    }
    assert extras
    for key, value in extras.items():
        assert getattr(request, key) == value, key
    assert request.training_type == body.get("training_type", "LoRA/QLoRA")
    assert request.format_type == body.get("format_type", "auto")
    assert request.hf_token is None and request.hf_dataset is None


@pytest.mark.parametrize("key", ["model_name", "local_datasets", "project_name", "not_a_setting"])
def test_a_key_outside_the_contract_is_refused(key):
    with pytest.raises(HTTPException) as caught:
        _request({**BODIES["lora"], key: "x"})
    assert caught.value.status_code == 400
    assert caught.value.detail == f"Unknown parameter(s): {key}"
