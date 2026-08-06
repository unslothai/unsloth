# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-class n_batch / n_ubatch load fields (llama-server --batch-size / --ubatch-size).

Compact sibling of test_parallel_slots_per_load.py: pydantic bounds, VRAM-budget
precedence, reload dedupe, shadow stripping and the stored-override mapping.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import httpx  # noqa: F401

from core.inference.llama_cpp import (
    GgufLoadIntent,
    LlamaCppBackend,
    _extra_args_n_ubatch,
)
from core.inference.llama_server_args import BATCH_MAX, BATCH_MIN, strip_shadowing_flags
from models.inference import LoadRequest, ValidateModelRequest
from utils.openai_auto_switch_settings import (
    model_override_load_kwargs,
    normalize_model_override,
)


@pytest.mark.parametrize("field", ["n_batch", "n_ubatch"])
def test_load_request_bounds(field):
    assert getattr(LoadRequest(model_path = "owner/repo"), field) is None
    assert getattr(LoadRequest(model_path = "owner/repo", **{field: BATCH_MAX}), field) == BATCH_MAX
    for bad in (BATCH_MIN - 1, BATCH_MAX + 1):
        with pytest.raises(ValueError):
            LoadRequest(model_path = "owner/repo", **{field: bad})
    # /validate sizes like /load, so it carries the same field and bounds
    assert getattr(ValidateModelRequest(model_path = "owner/repo", **{field: 256}), field) == 256


def test_effective_ubatch_prefers_first_class_over_env():
    env = {"LLAMA_ARG_BATCH": "128", "LLAMA_ARG_UBATCH": "128"}
    assert _extra_args_n_ubatch(None, env = env, n_batch = 4096, n_ubatch = 1024) == 1024


def test_effective_ubatch_lets_extras_override_first_class():
    # extras are appended after the emitted flags, so they last-wins at launch
    assert _extra_args_n_ubatch(["-ub", "256"], env = {}, n_ubatch = 1024) == 256


def test_effective_ubatch_caps_at_batch_and_ctx():
    assert _extra_args_n_ubatch(None, env = {}, n_batch = 512, n_ubatch = 2048) == 512
    assert _extra_args_n_ubatch(None, env = {}, n_ubatch = 4096, n_ctx = 1024) == 1024
    assert _extra_args_n_ubatch(None, env = {}) is None


def _loaded_backend() -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = object()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._requested_spec_mode = "auto"
    return backend


def _intent(**kwargs) -> GgufLoadIntent:
    return GgufLoadIntent(
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        speculative_type = "auto",
        **kwargs,
    )


def test_dedupe_matches_same_batch_sizes():
    backend = _loaded_backend()
    backend._requested_n_batch = 4096
    matches = backend._runtime_matches_intent(_intent(n_batch = 4096), None)
    assert matches is True


def test_dedupe_reloads_on_batch_change():
    backend = _loaded_backend()
    assert backend._runtime_matches_intent(_intent(n_batch = 4096), None) is False
    backend._requested_n_ubatch = 512
    assert backend._runtime_matches_intent(_intent(n_ubatch = 1024), None) is False


def test_dedupe_ignores_batch_for_diffusion():
    backend = _loaded_backend()
    backend._is_diffusion = True
    backend._diffusion_requested_ngl = None
    backend._gpu_layers = -1
    assert backend._runtime_matches_intent(_intent(n_batch = 4096), None) is True


def test_strip_shadowing_flags_batch_toggles():
    args = ["-b", "4096", "--ubatch-size=256", "--top-k", "20"]
    assert strip_shadowing_flags(args, strip_batch = True) == ["--ubatch-size=256", "--top-k", "20"]
    assert strip_shadowing_flags(args, strip_ubatch = True) == ["-b", "4096", "--top-k", "20"]
    assert strip_shadowing_flags(args) == args


def test_override_store_round_trip():
    entry = normalize_model_override({"n_batch": 4096, "n_ubatch": 1024})
    assert entry == {"n_batch": 4096, "n_ubatch": 1024}
    # out of range or boolean values drop silently, like the other knobs
    assert normalize_model_override({"n_batch": 0, "n_ubatch": True}) == {}
    kwargs = model_override_load_kwargs(entry, is_gguf = True)
    assert kwargs["n_batch"] == 4096 and kwargs["n_ubatch"] == 1024
    assert "n_batch" not in model_override_load_kwargs(entry, is_gguf = False)


def test_fast_path_intent_strips_inherited_batch_flags_when_field_set():
    # the already-loaded dedupe must see the same override the slow path launches
    from routes.inference import _active_gguf_intent

    backend = _loaded_backend()
    backend._extra_args = ["-b", "512", "--top-k", "20"]
    backend._extra_args_source = ("owner/repo", "Q4_K_M")

    kwargs = dict(
        model_identifier = "owner/repo",
        chat_template_override = None,
        n_parallel = 1,
        native_grant_backed = False,
    )
    overriding = _active_gguf_intent(
        LoadRequest(model_path = "owner/repo", n_batch = 4096), backend, **kwargs
    )
    assert overriding.extra_args == ("--top-k", "20")
    assert overriding.extra_args_inherited is False

    inheriting = _active_gguf_intent(LoadRequest(model_path = "owner/repo"), backend, **kwargs)
    assert inheriting.extra_args == ("-b", "512", "--top-k", "20")
    assert inheriting.extra_args_inherited is True


def test_remote_gguf_guard_counts_explicit_micro_batch():
    # a not-yet-downloaded gguf has no readable dims, but an explicit micro-batch
    # override still has to count its kq-mask growth against active training
    from types import SimpleNamespace
    from unittest.mock import patch

    from routes import inference as route

    config = SimpleNamespace(
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_hf_repo = "owner/repo",
        gguf_variant = "Q4_K_M",
    )
    remote_variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
    with (
        patch(
            "utils.models.model_config.list_gguf_variants",
            return_value = ([remote_variant], False),
        ),
        patch.object(route, "_remote_gguf_companion_bytes", return_value = 0),
    ):
        base = route._estimate_gguf_required_gb(config, max_seq_length = 32768)
        big = route._estimate_gguf_required_gb(
            config, max_seq_length = 32768, n_batch = 65536, n_ubatch = 65536
        )
    assert base == pytest.approx(1.0)
    # ctx-capped ubatch (32768) x ctx x 2 x 1.5 mask safety ~= 3 GiB on top
    assert big > base + 2.0


def test_override_strips_shadowing_batch_flags():
    kwargs = model_override_load_kwargs(
        {"n_batch": 4096, "llama_extra_args": ["-b", "512", "--top-k", "20"]},
        is_gguf = True,
    )
    assert kwargs["llama_extra_args"] == ["--top-k", "20"]
    # a flag with no first-class field behind it still passes through
    kwargs = model_override_load_kwargs(
        {"llama_extra_args": ["-ub", "256"]},
        is_gguf = True,
    )
    assert kwargs["llama_extra_args"] == ["-ub", "256"]
