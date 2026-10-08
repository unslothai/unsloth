# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import json
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
    route, device, tmp_path, training_type, learning_rate
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


def test_api_runs_keep_the_hyperparameters_they_set(route, device, tmp_path):
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


def test_a_cached_sibling_subfolder_does_not_stand_in_for_the_requested_one(hub_cache, monkeypatch):
    from utils.models import model_config

    _cache_laya(hub_cache, "multilingual")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_config, "cache_reads_authorized", lambda *args, **kwargs: True)

    assert model_config.decision_layout(LAYA_REPO, subfolder = "multilingual") == "laya"
    assert model_config.decision_layout(LAYA_REPO) == "laya"
    # Only multilingual is cached, so the English checkpoint is not a decision model the cache can serve.
    assert model_config.decision_layout(LAYA_REPO, subfolder = "typed-decisions") is None


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


@pytest.mark.parametrize("root_format", ["adapter", "gguf"])
def test_a_hub_subfolder_checkpoint_is_not_judged_by_the_repo_root(route, root_format):
    from utils.models import model_config

    files = ["adapter_config.json", "v2/rl_agent_config.json", "v2/model.safetensors"]
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = name) for name in files])
    with (
        patch.object(route, "_remote_untrainable_model_format", return_value = root_format),
        patch.object(model_config, "_hub_model_info", return_value = info),
    ):
        route._reject_untrainable_model_request(
            _request(model_name = "org/model", model_subfolder = "v2")
        )


def test_a_local_checkpoint_is_checked_at_its_subfolder(route, tmp_path):
    root = tmp_path / "laya"
    _laya_folder(root / "multilingual")
    request = _request(model_name = str(root), model_subfolder = "multilingual")

    assert route._reject_untrainable_model_request(request).model_name == str(root.resolve())

    with pytest.raises(HTTPException) as refused:
        route._reject_untrainable_model_request(_request(model_name = str(root)))
    assert refused.value.detail["code"] == "training_local_model_not_decision"


def _llm_folder(folder: Path) -> Path:
    folder.mkdir(parents = True)
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64}
    (folder / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (folder / "model.safetensors").write_bytes(b"x")
    return folder


def test_a_local_llm_is_only_a_decision_checkpoint_when_validated_as_an_llm(route, tmp_path):
    llm = _llm_folder(tmp_path / "llm")

    # A caller's layout claim is not what makes an LLM pass: its validated layout is.
    with pytest.raises(HTTPException) as refused:
        route._reject_untrainable_model_request(
            _request(model_name = str(llm), decision_layout = "laya")
        )
    assert refused.value.status_code == 400
    assert refused.value.detail["code"] == "training_local_model_not_decision"


def test_an_llm_trains_as_a_decision_model_with_a_new_clef_head(route, device, tmp_path):
    from utils.models.model_config import load_llm_decision_defaults

    llm = _llm_folder(tmp_path / "llm")
    request = _request(model_name = str(llm), decision_layout = "laya")
    route._validate_decision_request(request)
    assert request.decision_layout == "llm"
    assert route._reject_untrainable_model_request(request).model_name == str(llm.resolve())

    config = _started_config(route, _request(model_name = str(llm), load_in_4bit = True))
    recipe = load_llm_decision_defaults()
    assert config["decision_layout"] == "llm"
    # QLoRA works for an LLM, unlike Laya.
    assert config["load_in_4bit"] is True
    assert float(config["learning_rate"]) == float(recipe["training"]["learning_rate"])
    assert config["max_seq_length"] == recipe["training"]["max_seq_length"]
    assert (config["batch_size"], config["gradient_accumulation_steps"]) == (
        recipe["training"]["batch_size"],
        recipe["training"]["gradient_accumulation_steps"],
    )
    assert (config["lora_r"], config["lora_alpha"]) == (
        recipe["lora"]["lora_r"],
        recipe["lora"]["lora_alpha"],
    )

    full = _request(model_name = str(llm), training_type = "Full Finetuning")
    route._validate_decision_request(full)
    assert full.learning_rate == route._DECISION_FULL_FINETUNING_LR

    with pytest.raises(HTTPException) as refused:
        route._validate_decision_request(_request(model_name = str(llm), use_dora = True))
    assert "DoRA" in refused.value.detail


def test_a_laya_subfolder_or_catalog_repo_never_becomes_an_llm_run(route, device):
    from utils.models import model_config

    # Offline or without access the layout is unknown; a subfolder or Laya's repo still means Laya.
    with patch.object(model_config, "decision_layout", return_value = None):
        for request in (
            _request(model_name = "org/decisions", model_subfolder = "multilingual"),
            _request(model_name = LAYA_REPO, model_subfolder = "multilingual"),
            _request(model_name = LAYA_REPO),
        ):
            route._validate_decision_request(request)
            assert request.decision_layout == "laya", request.model_name


@pytest.mark.parametrize("kind", ["cpu", "xpu"])
def test_llm_decision_training_needs_an_nvidia_or_amd_gpu(route, device, tmp_path, kind):
    from core.systemone.catalog import CLEF_NEEDS_GPU

    device(kind)
    with pytest.raises(HTTPException) as refused:
        route._validate_decision_request(_request(model_name = str(_llm_folder(tmp_path / "l"))))
    assert (refused.value.status_code, refused.value.detail) == (400, CLEF_NEEDS_GPU)


def test_a_plain_request_carries_no_decision_layout(route):
    request = _request(is_decision = False, model_name = "org/llm", decision_layout = "llm")
    route._validate_decision_request(request)
    assert request.decision_layout is None


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
    # An LLM that gets a new head downloads through FastModel in the trainer.
    worker._download_decision_checkpoint(
        queue, {"model_name": "unsloth/Qwen3.5-0.8B", "decision_layout": "llm"}
    )
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


@pytest.fixture
def device(monkeypatch):
    from utils.hardware import hardware

    def use(kind):
        value = {
            "cuda": hardware.DeviceType.CUDA,
            "cpu": hardware.DeviceType.CPU,
            "xpu": hardware.DeviceType.XPU,
        }[kind]
        monkeypatch.setattr(hardware, "get_device", lambda: value)
        monkeypatch.setattr(hardware, "DEVICE", value)

    use("cuda")
    return use


def _clef_folder(folder: Path) -> Path:
    folder.mkdir(parents = True)
    for name in ("config.json", "joint_head.safetensors", "joint_head_config.json"):
        (folder / name).write_text("{}", encoding = "utf-8")
    return folder


def test_api_clef_runs_get_the_clef_recipe_and_keep_qlora(route, device, tmp_path):
    config = _started_config(route, _request(model_name = str(_clef_folder(tmp_path / "clef"))))

    assert config["decision_layout"] == "clef"
    assert config["load_in_4bit"] is True
    assert float(config["learning_rate"]) == 2e-4
    assert (config["batch_size"], config["gradient_accumulation_steps"]) == (4, 2)
    assert (config["lora_r"], config["lora_alpha"]) == (16, 16)
    assert config["max_seq_length"] == 4096

    laya = _started_config(route, _request(model_name = str(_laya_folder(tmp_path / "laya"))))
    assert (laya["decision_layout"], laya["load_in_4bit"]) == ("laya", False)


def test_api_clef_full_finetuning_gets_the_full_finetuning_rate(route, device, tmp_path):
    clef = str(_clef_folder(tmp_path / "clef"))
    config = _started_config(route, _request(model_name = clef, training_type = "Full Finetuning"))

    assert float(config["learning_rate"]) == 2.5e-5


def test_a_caller_cannot_claim_a_layout_or_a_clef_subfolder(route, device, tmp_path):
    clef = _clef_folder(tmp_path / "clef")
    laya = _laya_folder(tmp_path / "laya")
    claimed = _request(model_name = str(laya), decision_layout = "clef")
    route._validate_decision_request(claimed)
    assert claimed.decision_layout == "laya"

    with pytest.raises(HTTPException) as refused:
        route._validate_decision_request(_request(model_name = str(clef), model_subfolder = "v2"))
    assert "model_subfolder" in refused.value.detail


def test_a_decision_run_is_planned_on_the_one_gpu_it_uses(route, device, tmp_path):
    request = _request(model_name = str(_laya_folder(tmp_path / "laya")), gpu_ids = [2, 3])
    route._validate_decision_request(request)
    assert request.gpu_ids == [2]


def test_a_local_clef_in_a_subfolder_is_detected_as_clef(route, device, tmp_path):
    _clef_folder(tmp_path / "parent" / "clef")
    request = _request(model_name = str(tmp_path / "parent"), model_subfolder = "clef")
    with pytest.raises(HTTPException) as refused:
        route._validate_decision_request(request)
    assert "Clef repos hold one checkpoint" in refused.value.detail


def test_a_hub_clef_repo_is_a_decision_model(route, device):
    from utils.models import model_config

    files = [
        "config.json",
        "joint_head.safetensors",
        "joint_head_config.json",
        "model-00001-of-00004.safetensors",
    ]
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = name) for name in files])
    request = _request(model_name = "Cloudflare/clef-flash")
    with (
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
        patch.object(model_config, "_hub_model_info", return_value = info),
    ):
        route._validate_decision_request(request)
        assert request.decision_layout == "clef"
        route._reject_untrainable_model_request(request)
        assert model_config.decision_layout("Cloudflare/clef-flash") == "clef"


@pytest.mark.parametrize("kind", ["cpu", "xpu"])
def test_clef_training_is_refused_without_an_nvidia_or_amd_gpu(route, device, tmp_path, kind):
    from core.systemone.catalog import CLEF_NEEDS_GPU

    device(kind)
    with pytest.raises(HTTPException) as refused:
        route._validate_decision_request(_request(model_name = str(_clef_folder(tmp_path / "c"))))
    assert (refused.value.status_code, refused.value.detail) == (400, CLEF_NEEDS_GPU)
    laya = _request(model_name = str(_laya_folder(tmp_path / "laya")))
    route._validate_decision_request(laya)
    assert laya.decision_layout == "laya"
