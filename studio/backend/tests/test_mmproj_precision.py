# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Projector precision: BF16 on ROCm and pre-Ampere NVIDIA, F16 elsewhere; MiMo-VL never F16."""

from __future__ import annotations

import sys
import threading
import types as _types
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend, _pick_mmproj  # noqa: E402
from hub.utils.gguf_plan import preferred_mmproj_sibling  # noqa: E402
from utils.models import model_config  # noqa: E402
from utils.models.model_config import (  # noqa: E402
    MMPROJ_ORDER_BF16,
    MMPROJ_ORDER_NO_F16,
    detect_mmproj_file,
    device_mmproj_precision,
    mmproj_f16_unsafe,
    mmproj_order,
)

MIMO_LISTING = [
    "MiMo-VL-7B-RL-UD-Q4_K_XL.gguf",
    "mmproj-BF16.gguf",
    "mmproj-F16.gguf",
    "mmproj-F32.gguf",
]
QWEN_LISTING = [
    "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
    "mmproj-BF16.gguf",
    "mmproj-F16.gguf",
    "mmproj-F32.gguf",
]


@pytest.fixture
def device(monkeypatch):
    def _set(precision):
        monkeypatch.setattr(model_config, "device_mmproj_precision", lambda: precision)

    _set("f16")
    return _set


@pytest.mark.parametrize(
    "name",
    [
        "unsloth/MiMo-VL-7B-RL-GGUF",
        "bartowski/XiaomiMiMo_MiMo-VL-7B-RL-2508-GGUF",
        "XiaomiMiMo/MiMo-VL-7B-RL-GGUF",
        "MiMo-VL-7B-RL-UD-Q4_K_XL.gguf",
    ],
)
def test_mimo_vl_is_f16_unsafe(name):
    assert mmproj_f16_unsafe(name)


@pytest.mark.parametrize(
    "name",
    [
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF",
        "jedisct1/MiMo-7B-RL-GGUF",
        "unsloth/gemma-3-4b-it-GGUF",
        "",
        None,
    ],
)
def test_other_models_are_not_f16_unsafe(name):
    assert not mmproj_f16_unsafe(name)


@pytest.mark.parametrize(
    "system, backend, caps, expected",
    [
        ("Darwin", "metal", {}, "f16"),
        ("Linux", "rocm", {}, "bf16"),
        ("Windows", "rocm", {}, "bf16"),
        ("Linux", "cuda", {0: 75}, "bf16"),
        ("Linux", "cuda", {0: 70}, "bf16"),
        ("Linux", "cuda", {0: 86, 1: 75}, "bf16"),
        ("Linux", "cuda", {0: 80}, "f16"),
        ("Linux", "cuda", {0: 89}, "f16"),
        ("Windows", "cuda", {0: 120}, "f16"),
        ("Linux", "cuda", {}, "f16"),
        ("Linux", "vulkan", {}, "f16"),
        ("Linux", "cpu", {}, "f16"),
        ("Linux", None, {}, "f16"),
    ],
)
def test_device_precision(monkeypatch, system, backend, caps, expected):
    from utils.hardware import hardware

    monkeypatch.setattr(model_config.platform, "system", lambda: system)
    monkeypatch.setattr(hardware, "_installed_llama_backend", lambda: backend)
    monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: caps))
    device_mmproj_precision.cache_clear()
    try:
        assert device_mmproj_precision() == expected
    finally:
        device_mmproj_precision.cache_clear()


def test_device_precision_probe_failure_keeps_f16(monkeypatch):
    from utils.hardware import hardware

    def boom():
        raise RuntimeError("no marker")

    monkeypatch.setattr(model_config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_installed_llama_backend", boom)
    device_mmproj_precision.cache_clear()
    try:
        assert device_mmproj_precision() == "f16"
    finally:
        device_mmproj_precision.cache_clear()


@pytest.mark.parametrize("precision", ["f16", "bf16"])
def test_mimo_vl_order_ignores_device(device, precision):
    device(precision)
    assert mmproj_order("unsloth/MiMo-VL-7B-RL-GGUF") == MMPROJ_ORDER_NO_F16


def test_other_model_order_follows_device(device):
    assert mmproj_order("unsloth/Qwen2.5-VL-7B-Instruct-GGUF") is None
    device("bf16")
    assert mmproj_order("unsloth/Qwen2.5-VL-7B-Instruct-GGUF") == MMPROJ_ORDER_BF16


def test_default_pick_still_prefers_f16():
    assert _pick_mmproj(MIMO_LISTING) == "mmproj-F16.gguf"


@pytest.mark.parametrize(
    "listing, expected",
    [
        (MIMO_LISTING, "mmproj-BF16.gguf"),
        (["mmproj-F16.gguf", "mmproj-F32.gguf"], "mmproj-F32.gguf"),
        (["mmproj-F16.gguf"], "mmproj-F16.gguf"),
        (
            [
                "mmproj-XiaomiMiMo_MiMo-VL-7B-RL-2508-bf16.gguf",
                "mmproj-XiaomiMiMo_MiMo-VL-7B-RL-2508-f16.gguf",
            ],
            "mmproj-XiaomiMiMo_MiMo-VL-7B-RL-2508-bf16.gguf",
        ),
        (["mmproj-MiMo-VL-7B-RL_BF16.gguf"], "mmproj-MiMo-VL-7B-RL_BF16.gguf"),
    ],
)
def test_no_f16_order(listing, expected):
    assert _pick_mmproj(listing, order = MMPROJ_ORDER_NO_F16) == expected


@pytest.mark.parametrize(
    "listing, expected",
    [
        (QWEN_LISTING, "mmproj-BF16.gguf"),
        (["mmproj-F16.gguf", "mmproj-F32.gguf"], "mmproj-F16.gguf"),
        (["mmproj-F32.gguf"], "mmproj-F32.gguf"),
    ],
)
def test_bf16_device_order(listing, expected):
    assert _pick_mmproj(listing, order = MMPROJ_ORDER_BF16) == expected


def _siblings(names):
    return [SimpleNamespace(rfilename = n) for n in names]


def test_hub_plan(device):
    assert preferred_mmproj_sibling(_siblings(MIMO_LISTING)).rfilename == "mmproj-BF16.gguf"
    assert preferred_mmproj_sibling(_siblings(QWEN_LISTING)).rfilename == "mmproj-F16.gguf"
    device("bf16")
    assert preferred_mmproj_sibling(_siblings(QWEN_LISTING)).rfilename == "mmproj-BF16.gguf"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * 32)
    return path


def _local_pick(tmp_path, listing):
    for n in listing:
        _touch(tmp_path / n)
    return Path(detect_mmproj_file(str(tmp_path / listing[0]))).name


def test_local_folder(device, tmp_path):
    assert _local_pick(tmp_path / "mimo", MIMO_LISTING) == "mmproj-BF16.gguf"
    # F16 vs F32 tie on stem length and the order is the filesystem's, so the default case uses two.
    assert _local_pick(tmp_path / "qwen", QWEN_LISTING[:3]) == "mmproj-F16.gguf"
    device("bf16")
    assert _local_pick(tmp_path / "qwen_bf16", QWEN_LISTING) == "mmproj-BF16.gguf"


def _download_mmproj_call(hf_repo):
    backend = object.__new__(LlamaCppBackend)
    backend._cancel_event = threading.Event()
    seen = {}

    def fake_companion(**kw):
        seen.update(kw)
        return None

    backend._download_companion_gguf = fake_companion
    backend._download_mmproj(hf_repo = hf_repo)
    return seen


def test_download_skips_cached_f16_reuse_for_mimo_vl(device):
    seen = _download_mmproj_call("unsloth/MiMo-VL-7B-RL-GGUF")
    assert seen["reuse_snapshot_sibling"] is False
    assert seen["pick"](MIMO_LISTING) == "mmproj-BF16.gguf"


def test_download_unchanged_for_other_models(device):
    seen = _download_mmproj_call("unsloth/Qwen2.5-VL-7B-Instruct-GGUF")
    assert seen["reuse_snapshot_sibling"] is True
    assert seen["pick"] is _pick_mmproj


def test_download_bf16_device_keeps_cached_reuse(device):
    device("bf16")
    seen = _download_mmproj_call("unsloth/Qwen2.5-VL-7B-Instruct-GGUF")
    assert seen["reuse_snapshot_sibling"] is True
    assert seen["pick"](QWEN_LISTING) == "mmproj-BF16.gguf"


def test_local_folder_pairs_the_checkpoint_before_precision(device, tmp_path):
    device("bf16")
    listing = [
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen2.5-VL-7B-Instruct-F16.gguf",
        "mmproj-Qwen2-VL-2B-Instruct-BF16.gguf",
    ]
    assert _local_pick(tmp_path, listing) == "mmproj-Qwen2.5-VL-7B-Instruct-F16.gguf"


def test_download_detects_mimo_vl_from_the_weight_name(device):
    backend = object.__new__(LlamaCppBackend)
    backend._cancel_event = threading.Event()
    seen = {}
    backend._download_companion_gguf = lambda **kw: seen.update(kw)
    backend._download_mmproj(
        hf_repo = "someone/vision-gguf", near_path = "/c/snap/MiMo-VL-7B-RL-Q4_K_M.gguf"
    )
    assert seen["reuse_snapshot_sibling"] is False
    assert seen["pick"](MIMO_LISTING) == "mmproj-BF16.gguf"


def test_llama_update_job_clears_the_device_precision(monkeypatch):
    from utils import llama_cpp_update as upd

    monkeypatch.setattr(upd._flow, "run_chained_update", lambda *a, **k: None)
    device_mmproj_precision.cache_clear()
    device_mmproj_precision()
    assert device_mmproj_precision.cache_info().currsize == 1
    upd._operation_lock.acquire()
    upd._run_claimed_job([])
    assert device_mmproj_precision.cache_info().currsize == 0


def _download_with_cached(monkeypatch, tmp_path, cached_name):
    from core.inference import llama_cpp

    cached = _touch(tmp_path / cached_name)
    monkeypatch.setattr(llama_cpp, "_companion_snapshot_sibling", lambda near, pick: str(cached))
    backend = object.__new__(LlamaCppBackend)
    backend._cancel_event = threading.Event()
    seen = {}
    backend._download_companion_gguf = lambda **kw: seen.update(kw)
    got = backend._download_mmproj(
        hf_repo = "unsloth/MiMo-VL-7B-RL-GGUF", near_path = str(tmp_path / "w.gguf")
    )
    return got, seen


def test_mimo_vl_reuses_a_cached_bf16_projector(device, monkeypatch, tmp_path):
    got, seen = _download_with_cached(monkeypatch, tmp_path, "mmproj-BF16.gguf")
    assert Path(got).name == "mmproj-BF16.gguf" and not seen


def test_mimo_vl_does_not_reuse_a_cached_f16_projector(device, monkeypatch, tmp_path):
    got, seen = _download_with_cached(monkeypatch, tmp_path, "mmproj-F16.gguf")
    assert got is None and seen["reuse_snapshot_sibling"] is False


def test_estimate_and_loader_share_the_picker(device):
    from core.inference.llama_cpp import _mmproj_picker
    assert (
        _mmproj_picker("unsloth/MiMo-VL-7B-RL-GGUF", "w.gguf")(MIMO_LISTING) == "mmproj-BF16.gguf"
    )
    assert _mmproj_picker("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "w.gguf") is _pick_mmproj


def test_remote_pick_pairs_the_checkpoint_before_precision(device):
    from core.inference.llama_cpp import _mmproj_picker

    device("bf16")
    listing = ["ModelA-Q4_K_M.gguf", "mmproj-ModelA-F16.gguf", "mmproj-ModelB-BF16.gguf"]
    assert (
        _mmproj_picker("org/collection-GGUF", "/c/snap/ModelA-Q4_K_M.gguf")(listing)
        == "mmproj-ModelA-F16.gguf"
    )


F16_WEIGHT_LISTING = [
    "MiMo-VL-7B-F16.gguf",
    "mmproj-MiMo-VL-7B-F16.gguf",
    "mmproj-MiMo-VL-7B-BF16.gguf",
]


def test_f16_weight_does_not_pair_with_the_f16_projector(device, tmp_path):
    from core.inference.llama_cpp import _mmproj_picker

    pick = _mmproj_picker("org/collection-GGUF", "/c/snap/MiMo-VL-7B-F16.gguf")
    assert pick(F16_WEIGHT_LISTING) == "mmproj-MiMo-VL-7B-BF16.gguf"
    assert _local_pick(tmp_path, F16_WEIGHT_LISTING) == "mmproj-MiMo-VL-7B-BF16.gguf"


def test_exact_checkpoint_projector_beats_a_longer_name(device, tmp_path):
    from core.inference.llama_cpp import _mmproj_picker

    device("bf16")
    listing = ["Model-7B-Q4_K_M.gguf", "mmproj-Model-7B-F16.gguf", "mmproj-Model-7B-v2-BF16.gguf"]
    assert _mmproj_picker("org/collection-GGUF", "/c/snap/Model-7B-Q4_K_M.gguf")(listing) == (
        "mmproj-Model-7B-F16.gguf"
    )
    assert _local_pick(tmp_path, listing) == "mmproj-Model-7B-F16.gguf"


def test_local_folder_named_after_mimo_vl_with_a_generic_weight(device, tmp_path):
    folder = tmp_path / "MiMo-VL-7B"
    for n in ("model-Q4_K_M.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"):
        _touch(folder / n)
    assert Path(detect_mmproj_file(str(folder / "model-Q4_K_M.gguf"))).name == "mmproj-BF16.gguf"


@pytest.mark.parametrize(
    "name, precision",
    [
        ("mmproj-F16.gguf", "f16"),
        ("MiMo-VL-7B-F16-mmproj.gguf", "f16"),
        ("MiMo-VL-7B-RL-2508.mmproj-f16.gguf", "f16"),
        ("mmproj-MiMo-VL-7B-RL_BF16.gguf", "bf16"),
        ("Qwen3.5-9B-BF16-mmproj.gguf", "bf16"),
        ("MiMo-VL-7B-RL-2508.mmproj-Q8_0.gguf", None),
    ],
)
def test_projector_precision_tags(name, precision):
    assert model_config.mmproj_precision(name) == precision
