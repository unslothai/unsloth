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
        (True, " hf_caller\n", "hf_caller", False),
        (False, None, None, True),
        (False, "hf_caller", "hf_caller", True),
        (False, " hf_caller\n", "hf_caller", True),
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
    assert backend.kwargs["hf_token"] == (probe_token or "")


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize(
    "via_api_key,request_token,authorized,refused",
    [
        (True, None, False, True),
        (True, "hf_no_access", False, True),
        (True, "hf_caller", True, False),
        (False, None, False, False),
    ],
)
def test_private_cached_dataset_requires_caller_authorization(
    monkeypatch, tmp_path, offline, via_api_key, request_token, authorized, refused
):
    from fastapi import HTTPException
    from hub.utils import dataset_cache, hf_tokens

    backend = _Backend()
    monkeypatch.setattr(tr, "get_training_backend", lambda: backend)
    monkeypatch.setattr(tr, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(tr, "_diffusion_gpu_admission", contextlib.nullcontext)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: offline)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(
        tr,
        "_reject_untrainable_model_request",
        lambda request, *a: tr._ModelPreflightResult(request.model_name, None, None),
    )
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda repo_id: True)
    monkeypatch.setattr(
        dataset_cache, "training_dataset_cache_pin", lambda *a, **k: (str(tmp_path), "rev")
    )
    monkeypatch.setattr(hf_tokens, "_explicit_token_reaches_repo", lambda *a, **k: authorized)

    request = TrainingStartRequest(
        model_name = "unsloth/tiny-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_token = request_token,
        hf_dataset = "org/private-dataset",
        dataset_known_cached = True,
        load_in_4bit = False,
    )
    start = tr.start_training(request = request, current_subject = "alice", via_api_key = via_api_key)
    if refused:
        with pytest.raises(HTTPException) as error:
            asyncio.run(start)
        assert error.value.detail["code"] == "hf_dataset_access_denied"
        assert backend.kwargs is None
    else:
        assert asyncio.run(start).status == "queued"


@pytest.mark.parametrize("token,refused", [(False, True), ("hf_no_access", True), (None, False)])
def test_snapshot_cached_during_the_metadata_probe_is_authorized(
    monkeypatch, tmp_path, token, refused
):
    from fastapi import HTTPException
    from core.training import training as training_module
    from hub.utils import hf_tokens

    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    (tmp_path / "model.safetensors").write_bytes(b"cached weights")
    monkeypatch.setattr("hub.utils.hf_cache_state.iter_repo_cache_dirs", lambda *a, **k: iter(()))
    monkeypatch.setattr(training_module, "_resolve_model_snapshot", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(hf_tokens, "_explicit_token_reaches_repo", lambda *a, **k: False)

    def denied(*args):
        raise tr._hf_preflight_error(422, "hf_model_access_denied", "Denied")

    monkeypatch.setattr(tr, "_remote_untrainable_model_format", denied)
    request = TrainingStartRequest(
        model_name = "org/private-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )
    if refused:
        with pytest.raises(HTTPException) as error:
            tr._reject_untrainable_model_request(request, hf_token = token)
        assert error.value.detail["code"] == "hf_model_access_denied"
    else:
        result = tr._reject_untrainable_model_request(request, hf_token = token)
        assert result.cached_model_pin == ("org/private-model", str(tmp_path))


def test_dataset_cached_after_the_first_scan_is_not_pinned(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from hub.utils import dataset_cache, hf_tokens

    backend = _Backend()
    scans = []

    def appears_after_first_scan(*args, **kwargs):
        scans.append(args)
        return (str(tmp_path), "rev") if len(scans) > 1 else (None, None)

    monkeypatch.setattr(tr, "get_training_backend", lambda: backend)
    monkeypatch.setattr(tr, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(tr, "_diffusion_gpu_admission", contextlib.nullcontext)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: True)
    monkeypatch.setattr(
        tr,
        "_reject_untrainable_model_request",
        lambda request, *a: tr._ModelPreflightResult(request.model_name, None, None),
    )
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda repo_id: bool(scans))
    monkeypatch.setattr(dataset_cache, "training_dataset_cache_pin", appears_after_first_scan)
    monkeypatch.setattr(hf_tokens, "_explicit_token_reaches_repo", lambda *a, **k: False)

    request = TrainingStartRequest(
        model_name = "unsloth/tiny-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/private-dataset",
        dataset_known_cached = True,
        load_in_4bit = False,
    )
    with pytest.raises(HTTPException):
        asyncio.run(tr.start_training(request = request, current_subject = "alice", via_api_key = True))
    assert backend.kwargs is None


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


@pytest.mark.parametrize(
    "allow_ambient,token,expected",
    [
        (False, "", None),
        (False, "hf_caller", "Bearer hf_caller"),
        (True, "", "Bearer hf_operator_probe"),
    ],
)
def test_parent_gpu_probe_uses_only_authorized_token(monkeypatch, allow_ambient, token, expected):
    from core.training import training as training_module
    from utils.hardware import hardware
    from huggingface_hub import hf_api

    class Captured(BaseException):
        pass

    seen = []

    class Session:
        def get(self, url, **kwargs):
            seen.append(kwargs.get("headers", {}).get("authorization"))
            raise Captured

    monkeypatch.setenv("HF_TOKEN", "hf_operator_probe")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    monkeypatch.setattr(training_module, "should_use_mlx_training_backend", lambda **kwargs: False)
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(
        hardware, "_resolve_model_identifier_for_gpu_estimate", lambda name, hf_token = None: name
    )
    monkeypatch.setattr(hf_api, "get_session", lambda: Session())
    backend = training_module.TrainingBackend()
    with pytest.raises(Captured):
        backend.start_training(
            "credential-probe",
            model_name = "org/public-model",
            training_type = "LoRA/QLoRA",
            hf_token = token,
            allow_ambient = allow_ambient,
            load_in_4bit = False,
        )
    assert seen == [expected]


@pytest.mark.parametrize(
    "cached_path", ["fallback", "known", "offline", "resume", "public_metadata"]
)
@pytest.mark.parametrize("token", [False, "hf_no_access"])
def test_private_cached_model_requires_caller_authorization(
    monkeypatch, tmp_path, cached_path, token
):
    from fastapi import HTTPException
    from core.training import training as training_module
    from hub.utils import hf_tokens

    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    (tmp_path / "model.safetensors").write_bytes(b"cached weights")
    monkeypatch.setattr(training_module, "_resolve_model_snapshot", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.iter_repo_cache_dirs", lambda *a, **k: iter([tmp_path])
    )
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.latest_snapshot_from_cache_path", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(tr, "hf_env_offline", lambda: cached_path == "offline")
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(hf_tokens, "_explicit_token_reaches_repo", lambda *a, **k: False)

    def denied(*args):
        raise tr._hf_preflight_error(422, "hf_model_access_denied", "Denied")

    monkeypatch.setattr(
        tr,
        "_remote_untrainable_model_format",
        (lambda *a: None) if cached_path == "public_metadata" else denied,
    )
    request = TrainingStartRequest(
        model_name = "org/private-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        model_known_cached = cached_path == "known",
        resume_from_checkpoint = str(tmp_path) if cached_path == "resume" else None,
        model_snapshot_path = str(tmp_path) if cached_path == "resume" else None,
    )
    with pytest.raises(HTTPException) as error:
        tr._reject_untrainable_model_request(request, hf_token = token)
    assert error.value.detail["code"] == "hf_model_access_denied"


@pytest.mark.parametrize("token,authorized", [(None, False), (False, True), ("hf_caller", True)])
def test_authorized_cached_models_remain_available(monkeypatch, tmp_path, token, authorized):
    from core.training import training as training_module
    from hub.utils import hf_tokens

    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    (tmp_path / "model.safetensors").write_bytes(b"cached weights")
    monkeypatch.setattr(training_module, "_resolve_model_snapshot", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.iter_repo_cache_dirs", lambda *a, **k: iter([tmp_path])
    )
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_explicit_token_reaches_repo", lambda *a, **k: authorized)
    request = TrainingStartRequest(
        model_name = "org/model",
        model_known_cached = True,
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )
    result = tr._reject_untrainable_model_request(request, hf_token = token)
    assert result.model_name == "org/model"


def test_chat_coexistence_preserves_anonymous_token(monkeypatch):
    from routes.training_vram import can_keep_chat_during_training
    from utils import hardware

    class Captured(BaseException):
        pass

    seen = []

    def select(*args, **kwargs):
        seen.append(kwargs["hf_token"])
        raise Captured

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "auto_select_gpu_ids", select)
    with pytest.raises(Captured):
        can_keep_chat_during_training(
            model_name = "org/model",
            hf_token = False,
            training_type = "LoRA/QLoRA",
            load_in_4bit = False,
            batch_size = 1,
            max_seq_length = 128,
            lora_rank = 16,
            target_modules = None,
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            gpu_ids = None,
        )
    assert seen == [False]
