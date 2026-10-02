# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from core.systemone.catalog import LAYA_REPO
from models.training import TrainingStartRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse = True)
def hub_cache(monkeypatch, tmp_path):
    from hub.utils import hf_cache_state
    from utils import hf_cache_settings

    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(root))
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kwargs: [root])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    return root


@pytest.fixture
def route():
    spec = importlib.util.spec_from_file_location(
        "decision_training_route", _BACKEND_ROOT / "routes" / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._hub_unreachable = lambda: False
    return module


def _request(**overrides) -> TrainingStartRequest:
    return TrainingStartRequest(
        **{
            "model_name": LAYA_REPO,
            "training_type": "LoRA/QLoRA",
            "format_type": "auto",
            "hf_dataset": "org/decisions",
            "is_decision": True,
            **overrides,
        }
    )


def _laya_folder(folder: Path) -> Path:
    for name in ("encoder", "tokenizer"):
        (folder / name).mkdir(parents = True)
    for name in ("rl_agent_config.json", "model.safetensors"):
        (folder / name).write_text("{}", encoding = "utf-8")
    return folder


def _cache_laya(hub_cache: Path, subfolder = None) -> Path:
    repo = hub_cache / "models--convaiinnovations--laya"
    snapshot = repo / "snapshots" / "rev"
    _laya_folder(snapshot / subfolder if subfolder else snapshot)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev", encoding = "utf-8")
    return repo


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


def _started_config(route, request) -> dict:
    captured = {}
    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: captured.update(kwargs) or True,
    )
    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "_preflight_hf_dataset_request", return_value = None),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
    ):
        response = asyncio.run(route.start_training(request, current_subject = "test-user"))
    assert response.status == "queued", response
    return captured


@pytest.mark.parametrize(
    "subfolder",
    [
        "*",
        "**",
        "[mt]*",
        "?ultilingual",
        " multilingual",
        "multilingual ",
        "encoder",
        "~",
        "C:",
        "D:evil",
        "multi\x00lingual",
        pytest.param("m" * 5000, id = "5000 characters"),
        "",
        ".",
        "..",
        "../other",
        "multilingual/../..",
        "..\\other",
    ],
)
def test_subfolder_must_be_one_of_the_laya_checkpoints(route, subfolder):
    with pytest.raises(HTTPException) as refused:
        route._validate_decision_request(_request(model_subfolder = subfolder))

    assert refused.value.status_code == 400
    assert "Invalid checkpoint subfolder" in refused.value.detail


@pytest.mark.parametrize("subfolder", [None, "multilingual", "typed-decisions"])
def test_catalog_checkpoints_are_accepted(route, subfolder):
    route._validate_decision_request(_request(model_subfolder = subfolder))


def test_other_sources_take_a_plain_folder_name_only(route, tmp_path):
    route._validate_decision_request(_request(model_name = str(tmp_path), model_subfolder = "v2"))
    with pytest.raises(HTTPException):
        route._validate_decision_request(_request(model_name = str(tmp_path), model_subfolder = "v*"))


@pytest.mark.parametrize(
    ("request_overrides", "expected"),
    [
        ({"training_type": "Continued Pretraining"}, "continued pretraining is not available"),
        ({"resume_from_checkpoint": "outputs/run"}, "cannot be resumed"),
        ({"dataset_streaming": True, "max_steps": 10}, "dataset_streaming"),
        ({"load_in_4bit": True}, "QLoRA is not available"),
        ({"use_dora": True}, "DoRA and LoftQ are not available"),
        ({"use_loftq": True}, "DoRA and LoftQ are not available"),
    ],
)
def test_start_refuses_what_the_recipe_cannot_run(route, request_overrides, expected):
    refusing = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: pytest.fail("backend should not start"),
    )
    with (
        patch.object(route, "get_training_backend", return_value = refusing),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        pytest.raises(HTTPException) as refused,
    ):
        asyncio.run(
            route.start_training(_request(**request_overrides), current_subject = "test-user")
        )

    assert refused.value.status_code == 400
    assert expected in refused.value.detail


@pytest.mark.parametrize(
    ("request_overrides", "requested"),
    [
        ({}, True),
        ({"training_type": "Full Finetuning", "load_in_4bit": True}, True),
        ({"load_in_4bit": False}, False),
    ],
)
def test_decision_models_train_in_16_bit(route, request_overrides, requested):
    request = _request(**request_overrides)
    assert request.load_in_4bit is requested

    route._validate_decision_request(request)

    assert request.load_in_4bit is False


def test_a_managed_account_is_told_decision_training_is_owner_only(route, monkeypatch):
    from utils import account_context

    monkeypatch.setattr(account_context, "is_owner_context", lambda: False)
    monkeypatch.setattr(route, "managed_account", lambda: True)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(route.start_training(_request(), current_subject = "managed-user"))

    assert refused.value.status_code == 403
    assert "Only the Studio owner" in refused.value.detail


def test_python_older_than_3_10_cannot_train_decision_models(route, monkeypatch):
    request = _request()
    with monkeypatch.context() as patched:
        patched.setattr(route.sys, "version_info", (3, 9, 18))
        with pytest.raises(HTTPException) as refused:
            route._validate_decision_request(request)

    assert refused.value.status_code == 400
    assert "Python 3.10" in refused.value.detail


@pytest.mark.parametrize(
    ("training_type", "learning_rate"),
    [("LoRA/QLoRA", 8e-4), ("Full Finetuning", 2.5e-5)],
)
def test_api_runs_without_hyperparameters_get_the_laya_recipe(
    route, tmp_path, training_type, learning_rate
):
    base = _laya_folder(tmp_path / "laya")
    config = _started_config(route, _request(model_name = str(base), training_type = training_type))

    assert float(config["learning_rate"]) == learning_rate
    assert config["load_in_4bit"] is False
    assert (config["num_epochs"], config["max_steps"], config["warmup_steps"]) == (4, 0, 0)
    assert (config["batch_size"], config["gradient_accumulation_steps"]) == (8, 8)
    assert (config["optim"], config["lr_scheduler_type"]) == ("adamw_torch", "cosine")
    assert config["weight_decay"] == 0.01
    assert config["gradient_checkpointing"] == "unsloth"
    assert (config["lora_r"], config["lora_alpha"], config["lora_dropout"]) == (64, 64, 0.0)


def test_api_runs_keep_the_hyperparameters_they_set(route, tmp_path):
    base = _laya_folder(tmp_path / "laya")
    config = _started_config(
        route,
        _request(
            model_name = str(base),
            training_type = "Full Finetuning",
            learning_rate = "1e-5",
            batch_size = 2,
            optim = "adamw_8bit",
            lora_r = 16,
        ),
    )

    assert float(config["learning_rate"]) == 1e-5
    assert (config["batch_size"], config["optim"], config["lora_r"]) == (2, "adamw_8bit", 16)
    assert config["gradient_accumulation_steps"] == 8


@pytest.mark.parametrize("subfolder", [None, "multilingual"])
@pytest.mark.parametrize("hub_state", ["offline", "unreachable"])
def test_a_cached_laya_checkpoint_starts_without_the_hub(
    route, hub_cache, monkeypatch, subfolder, hub_state
):
    # Only the requested checkpoint is cached, as the Decision API leaves it.
    repo = _cache_laya(hub_cache, subfolder)
    if hub_state == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        route._hub_unreachable = lambda: True
    request = _request(
        model_subfolder = subfolder, model_known_cached = True, model_local_path = str(repo)
    )

    result = route._reject_untrainable_model_request(request)

    assert result.model_name == LAYA_REPO
    assert result.cached_model_pin is None


@pytest.mark.parametrize(("hub_state", "status"), [("offline", 409), ("unreachable", 503)])
def test_an_uncached_laya_checkpoint_still_needs_the_hub(
    route, hub_cache, monkeypatch, hub_state, status
):
    _cache_laya(hub_cache, "multilingual")
    if hub_state == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        route._hub_unreachable = lambda: True

    with pytest.raises(HTTPException) as refused:
        route._reject_untrainable_model_request(_request(model_subfolder = "typed-decisions"))

    assert refused.value.status_code == status


def test_a_cached_decision_run_reports_no_download(hub_cache):
    from core.training.training import _apply_cache_pins, _build_training_worker_config

    repo = _cache_laya(hub_cache, "multilingual")
    config = _build_training_worker_config(
        {
            "model_name": LAYA_REPO,
            "model_subfolder": "multilingual",
            "is_decision": True,
            "model_known_cached": True,
            "model_local_path": str(repo),
        }
    )

    _apply_cache_pins(config)

    assert config["cache_pin_warnings"] == []
    assert config["model_snapshot_path"] is None


@pytest.mark.parametrize(
    ("files", "subfolder", "accepted"),
    [
        (["config.json", "model.safetensors"], None, False),
        (["rl_agent_config.json", "model.safetensors", "encoder/config.json"], None, True),
        (["v2/rl_agent_config.json", "v2/model.safetensors"], "v2", True),
        (["v2/rl_agent_config.json", "v2/model.safetensors"], None, False),
        (["rl_agent_config.json", "model.safetensors"], "v2", False),
    ],
)
def test_a_hub_repo_must_be_a_decision_model(route, files, subfolder, accepted):
    from utils.models import model_config

    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = name) for name in files])
    request = _request(model_name = "org/model", model_subfolder = subfolder)
    with (
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
        patch.object(model_config, "_hub_model_info", return_value = info),
    ):
        if accepted:
            route._reject_untrainable_model_request(request)
            return
        with pytest.raises(HTTPException) as refused:
            route._reject_untrainable_model_request(request)

    assert refused.value.status_code == 400
    assert refused.value.detail["code"] == "training_remote_model_not_decision"


def test_a_local_checkpoint_is_checked_at_its_subfolder(route, tmp_path):
    root = tmp_path / "laya"
    _laya_folder(root / "multilingual")
    request = _request(model_name = str(root), model_subfolder = "multilingual")

    assert route._reject_untrainable_model_request(request).model_name == str(root.resolve())

    with pytest.raises(HTTPException) as refused:
        route._reject_untrainable_model_request(_request(model_name = str(root)))
    assert refused.value.detail["code"] == "training_local_model_not_decision"


def test_a_local_llm_is_not_trained_as_a_decision_model(route, tmp_path):
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "config.json").write_text("{}", encoding = "utf-8")
    (llm / "model.safetensors").write_bytes(b"x")

    with pytest.raises(HTTPException) as refused:
        route._reject_untrainable_model_request(_request(model_name = str(llm)))

    assert refused.value.status_code == 400
    assert refused.value.detail["code"] == "training_local_model_not_decision"


def test_a_hub_checkpoint_downloads_under_the_stall_watchdog(monkeypatch, tmp_path):
    from core.systemone import laya_runtime
    from core.training import worker
    from utils import hf_xet_fallback

    events, watchdogs, downloads = [], [], []

    def start_watchdog(**kwargs):
        watchdogs.append({**kwargs, "stop": threading.Event()})
        return watchdogs[-1]["stop"]

    def checkpoint_dir(checkpoint, **kwargs):
        downloads.append(
            (
                checkpoint.source,
                checkpoint.subfolder,
                os.environ.get("HF_TOKEN"),
                watchdogs[-1]["stop"].is_set(),
            )
        )
        watchdogs[-1]["on_stall"]("No download progress for 30s")
        return tmp_path

    monkeypatch.setattr(hf_xet_fallback, "start_watchdog", start_watchdog)
    monkeypatch.setattr(laya_runtime, "_checkpoint_dir", checkpoint_dir)
    # Set first so monkeypatch removes what the download leaves behind.
    monkeypatch.setenv("HF_TOKEN", "")
    queue = SimpleNamespace(put = events.append)

    worker._download_decision_checkpoint(
        queue,
        {"model_name": LAYA_REPO, "model_subfolder": "multilingual", "hf_token": "hf_private"},
    )

    assert [w["repo_ids"] for w in watchdogs] == [[LAYA_REPO]]
    assert downloads == [(LAYA_REPO, "multilingual", "hf_private", False)]
    assert [e["type"] for e in events] == [
        "status",
        "model_load_started",
        "stall",
        "model_load_completed",
    ]
    assert watchdogs[0]["stop"].is_set()

    events.clear()
    worker._download_decision_checkpoint(queue, {"model_name": str(tmp_path)})
    assert events == [] and len(downloads) == 1


@pytest.mark.skipif(sys.platform == "darwin", reason = "Apple Silicon trains with MLX")
def test_decision_worker_uses_one_gpu_and_no_llm_kernel_installs(monkeypatch, tmp_path):
    import multiprocessing as mp

    from core.training.training import _build_training_worker_config
    from core.training.worker import run_training_process

    # Offline, so an LLM-path flash-attn install reports itself instead of downloading.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    not_laya = tmp_path / "not-laya"
    not_laya.mkdir()
    config = _build_training_worker_config(
        {
            "model_name": str(not_laya),
            "training_type": "LoRA/QLoRA",
            "is_decision": True,
            "max_seq_length": 32768,
        }
    )
    config.update(resolved_gpu_ids = [0, 1], device_backend = "cuda")
    context = mp.get_context("spawn")
    events, stops = context.Queue(), context.Queue()
    process = context.Process(
        target = run_training_process,
        kwargs = {"event_queue": events, "stop_queue": stops, "config": config},
    )
    process.start()
    received = []
    try:
        while not received or received[-1]["type"] not in ("complete", "error"):
            received.append(events.get(timeout = 300))
    finally:
        process.join(60)

    statuses = [e["message"] for e in received if e["type"] == "status"]
    assert [e["message"] for e in received if e["type"] == "warning"] == [
        "Decision models train on one GPU; using GPU 0."
    ]
    assert "Importing Unsloth..." in statuses
    assert not any("flash-attn" in status for status in statuses)
    assert received[-1]["type"] == "error"
