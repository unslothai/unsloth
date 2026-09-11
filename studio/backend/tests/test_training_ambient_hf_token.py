# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import routes.training as tr
from models import TrainingStartRequest

_BACKEND = Path(__file__).resolve().parent.parent

_HF_TOKEN_ENV = (
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_OIDC_RESOURCE",
    "HF_TOKEN_PATH",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
)


class _Backend:
    current_job_id = None

    def __init__(self):
        self.kwargs = None

    def is_training_active(self):
        return False

    def start_training(
        self,
        job_id,
        *,
        before_spawn = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.current_job_id = job_id
        return True


@pytest.mark.parametrize(
    "via_api_key,request_token,probe_token,allow_ambient",
    [
        (True, None, False, False),
        (True, "hf_caller", "hf_caller", False),
        (False, None, None, True),
        (False, "hf_caller", "hf_caller", True),
    ],
)
def test_start_gives_the_model_preflight_only_the_callers_token(
    monkeypatch, via_api_key, request_token, probe_token, allow_ambient
):
    backend = _Backend()
    probed = []

    def _probe(model_name, hf_token):
        probed.append(hf_token)
        return None

    monkeypatch.setattr(tr, "get_training_backend", lambda: backend)
    monkeypatch.setattr(tr, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(tr, "_diffusion_gpu_admission", contextlib.nullcontext)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(tr, "_remote_untrainable_model_format", _probe)
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)

    request = TrainingStartRequest(
        model_name = "unsloth/tiny-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_token = request_token,
        load_in_4bit = False,
        trust_remote_code = True,
    )
    response = asyncio.run(
        tr.start_training(request = request, current_subject = "alice", via_api_key = via_api_key)
    )

    assert response.status == "queued", response
    assert probed == [probe_token]
    assert backend.kwargs["allow_ambient"] is allow_ambient


def test_worker_config_carries_the_ambient_policy():
    from core.training.training import _build_training_worker_config

    assert _build_training_worker_config({"model_name": "org/model"})["allow_ambient"] is True
    config = _build_training_worker_config({"model_name": "org/model", "allow_ambient": False})
    assert config["allow_ambient"] is False


_WORKER_PROBE = textwrap.dedent(
    """
    import json
    import queue
    import sys

    from utils.native_path_leases import run_without_native_path_secret

    run_without_native_path_secret(
        "core.training.worker",
        "run_training_process",
        {},
        event_queue = queue.Queue(),
        stop_queue = queue.Queue(),
        config = json.loads(sys.argv[1]),
    )

    from huggingface_hub import get_token
    from huggingface_hub.utils import build_hf_headers

    print(json.dumps({"token": get_token(), "header": build_hf_headers().get("authorization")}))
    """
)


@pytest.mark.parametrize(
    "allow_ambient,hf_token,expected_token",
    [
        (False, "", None),
        (False, "hf_caller", "hf_caller"),
        (True, "", "hf_operator_env"),
    ],
)
def test_training_worker_holds_only_the_callers_credential(
    tmp_path, allow_ambient, hf_token, expected_token
):
    (tmp_path / "token").write_text("hf_operator_file")
    env = {key: value for key, value in os.environ.items() if key not in _HF_TOKEN_ENV}
    env.update(
        HF_HOME = str(tmp_path),
        HF_TOKEN = "hf_operator_env",
        HUGGING_FACE_HUB_TOKEN = "hf_operator_env",
        HF_HUB_OFFLINE = "1",
        PYTHONPATH = os.pathsep.join(filter(None, (str(_BACKEND), env.get("PYTHONPATH")))),
    )
    config = {
        "model_name": "org/model",
        "model_format": "gguf",
        "hf_token": hf_token,
        "allow_ambient": allow_ambient,
    }

    out = subprocess.run(
        [sys.executable, "-c", _WORKER_PROBE, json.dumps(config)],
        cwd = str(_BACKEND),
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )

    assert out.returncode == 0, out.stderr[-2000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen["token"] == expected_token
    assert seen["header"] == (f"Bearer {expected_token}" if expected_token else None)
