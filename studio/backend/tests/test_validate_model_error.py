# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""routes/inference.py::validate_model surfaces actionable RuntimeError/ValueError
messages (e.g. "llama-server binary not found - run setup.sh") instead of a blank
"Invalid model", while keeping unexpected exceptions generic so internals never
leak to the client.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

import routes.inference as inf  # noqa: E402
from models.inference import ValidateModelRequest  # noqa: E402


def _provoke(
    monkeypatch,
    exc: BaseException,
    *,
    native: bool = False,
) -> HTTPException:
    """Drive validate_model so from_identifier raises ``exc``; return the
    HTTPException it converts that into."""
    monkeypatch.setattr(
        inf,
        "_resolve_model_identifier_for_request",
        lambda request, operation: ("org/repo", "org/repo", native),
    )

    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(inf.ModelConfig, "from_identifier", staticmethod(_raise))

    req = ValidateModelRequest(model_path = "org/repo")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.validate_model(req, current_subject = "tester"))
    return excinfo.value


def test_runtime_error_surfaces_actionable_message(monkeypatch):
    err = RuntimeError(
        "llama-server binary not found - cannot load GGUF models. "
        "Run setup.sh to build it, or set LLAMA_SERVER_PATH."
    )
    http = _provoke(monkeypatch, err)
    assert http.status_code == 400
    assert "llama-server binary not found" in http.detail
    assert http.detail != "Invalid model"


def test_value_error_not_supported_is_wrapped(monkeypatch):
    http = _provoke(monkeypatch, ValueError("architecture FooBar is not supported"))
    assert http.status_code == 400
    assert "not supported yet" in http.detail.lower()
    # Original cause is preserved for context.
    assert "FooBar" in http.detail


def test_unexpected_exception_stays_generic(monkeypatch):
    # A non-user-facing exception type must NOT have its message surfaced.
    http = _provoke(monkeypatch, KeyError("secret-internal-detail"))
    assert http.status_code == 400
    assert http.detail == "Invalid model"
    assert "secret-internal-detail" not in http.detail


def test_empty_runtime_error_falls_back_to_generic(monkeypatch):
    # A RuntimeError with no message should not produce an empty 400 detail.
    http = _provoke(monkeypatch, RuntimeError(""))
    assert http.status_code == 400
    assert http.detail == "Invalid model"


def _drive_validate(monkeypatch, *, is_gguf: bool):
    """Run validate_model with both security helpers forced True; return the response."""
    from types import SimpleNamespace

    import utils.models.model_config as mc

    monkeypatch.setattr(
        inf,
        "_resolve_model_identifier_for_request",
        lambda request, operation: ("org/mixed-repo", "org/mixed-repo", False),
    )
    config = SimpleNamespace(
        identifier = "org/mixed-repo",
        display_name = "org/mixed-repo",
        is_gguf = is_gguf,
        is_lora = False,
        is_vision = False,
        gguf_file = None,
    )
    monkeypatch.setattr(inf.ModelConfig, "from_identifier", staticmethod(lambda **_kw: config))
    # No LoRA base to resolve; keep it offline.
    monkeypatch.setattr(mc, "get_base_model_from_lora_identifier", lambda *_a, **_k: None)
    # Both gates WOULD flag this repo (mixed repo with auto_map + an unsafe pickle).
    monkeypatch.setattr(inf, "_requires_trust_remote_code_for_model", lambda *_a, **_k: True)
    monkeypatch.setattr(inf, "_requires_security_review_for_model", lambda *_a, **_k: True)

    req = ValidateModelRequest(model_path = "org/mixed-repo")
    return asyncio.run(inf.validate_model(req, current_subject = "tester"))


def test_selected_gguf_variant_skips_trc_and_security_review(monkeypatch):
    # GGUF loads via llama.cpp: auto_map and root pickles are inert, so neither gate fires.
    resp = _drive_validate(monkeypatch, is_gguf = True)
    assert resp.is_gguf is True
    assert resp.requires_trust_remote_code is False
    assert resp.requires_security_review is False


def test_non_gguf_load_still_runs_trc_and_security_review(monkeypatch):
    # Control: a Transformers (non-GGUF) load must still honor both gates.
    resp = _drive_validate(monkeypatch, is_gguf = False)
    assert resp.is_gguf is False
    assert resp.requires_trust_remote_code is True
    assert resp.requires_security_review is True


def test_resolve_loaded_trc_prefers_stored_value():
    # A value stored at load time wins, so a status refresh does not re-derive it.
    assert (
        inf._resolve_loaded_trust_remote_code("org/m", {"requires_trust_remote_code": True}, {})
        is True
    )
    assert (
        inf._resolve_loaded_trust_remote_code(
            "org/m", {"requires_trust_remote_code": False}, {"trust_remote_code": True}
        )
        is False
    )


def test_resolve_loaded_trc_uses_runtime_and_yaml():
    # No stored value: the trust_remote_code the load used, then the YAML default.
    assert (
        inf._resolve_loaded_trust_remote_code("org/m", {}, {}, trust_remote_code_used = True) is True
    )
    assert inf._resolve_loaded_trust_remote_code("org/m", {}, {"trust_remote_code": True}) is True


def test_resolve_loaded_trc_falls_back_to_raw_auto_map(monkeypatch):
    # No stored value or runtime/YAML signal: fall back to the raw auto_map check.
    monkeypatch.setattr(inf, "_requires_trust_remote_code_for_model", lambda *_a, **_k: True)
    assert inf._resolve_loaded_trust_remote_code("org/custom", {}, {}) is True
    monkeypatch.setattr(inf, "_requires_trust_remote_code_for_model", lambda *_a, **_k: False)
    assert inf._resolve_loaded_trust_remote_code("org/plain", {}, {}) is False


def _drive_validate_lora(monkeypatch, *, adapter_needs_trc, base_needs_trc):
    """Run validate_model for a LoRA adapter whose base resolves, with per-target
    trust_remote_code answers; return the response."""
    from types import SimpleNamespace

    import utils.models.model_config as mc

    adapter, base = "org/lora-adapter", "org/base-model"
    monkeypatch.setattr(
        inf,
        "_resolve_model_identifier_for_request",
        lambda request, operation: (adapter, adapter, False),
    )
    config = SimpleNamespace(
        identifier = adapter,
        display_name = adapter,
        is_gguf = False,
        is_lora = True,
        is_vision = False,
        gguf_file = None,
    )
    monkeypatch.setattr(inf.ModelConfig, "from_identifier", staticmethod(lambda **_kw: config))
    monkeypatch.setattr(mc, "get_base_model_from_lora_identifier", lambda *_a, **_k: base)
    trc = {adapter: adapter_needs_trc, base: base_needs_trc}
    monkeypatch.setattr(
        inf,
        "_requires_trust_remote_code_for_model",
        lambda target, *_a, **_k: trc.get(target, False),
    )
    monkeypatch.setattr(inf, "_requires_security_review_for_model", lambda *_a, **_k: False)
    req = ValidateModelRequest(model_path = adapter)
    return asyncio.run(inf.validate_model(req, current_subject = "tester"))


def test_validate_lora_flags_trc_from_adapter_only(monkeypatch):
    # Adapter ships auto_map, base does not: the requirement follows either repo.
    resp = _drive_validate_lora(monkeypatch, adapter_needs_trc = True, base_needs_trc = False)
    assert resp.requires_trust_remote_code is True


def test_validate_lora_flags_trc_from_base_only(monkeypatch):
    # The classic case: the base ships custom code, the adapter does not.
    resp = _drive_validate_lora(monkeypatch, adapter_needs_trc = False, base_needs_trc = True)
    assert resp.requires_trust_remote_code is True


def test_validate_lora_clean_when_neither_needs_trc(monkeypatch):
    resp = _drive_validate_lora(monkeypatch, adapter_needs_trc = False, base_needs_trc = False)
    assert resp.requires_trust_remote_code is False


@pytest.mark.parametrize("is_gguf", [True, False])
def test_validate_rejects_denied_llama_extra_args(monkeypatch, is_gguf):
    # /load validates extras unconditionally, before it resolves the model. If
    # /validate skipped the check whenever it classified the target as non-GGUF,
    # the preflight would pass and /load would 400 after unloading the active
    # model -- the failure this preflight exists to prevent.
    from types import SimpleNamespace

    import utils.models.model_config as mc

    monkeypatch.setattr(
        inf,
        "_resolve_model_identifier_for_request",
        lambda request, operation: ("org/gguf-repo", "org/gguf-repo", False),
    )
    config = SimpleNamespace(
        identifier = "org/gguf-repo",
        display_name = "org/gguf-repo",
        is_gguf = is_gguf,
        is_lora = False,
        is_vision = False,
        gguf_file = None,
    )
    monkeypatch.setattr(inf.ModelConfig, "from_identifier", staticmethod(lambda **_kw: config))
    monkeypatch.setattr(mc, "get_base_model_from_lora_identifier", lambda *_a, **_k: None)
    monkeypatch.setattr(inf, "_requires_trust_remote_code_for_model", lambda *_a, **_k: False)
    monkeypatch.setattr(inf, "_requires_security_review_for_model", lambda *_a, **_k: False)

    req = ValidateModelRequest(
        model_path = "org/gguf-repo",
        llama_extra_args = ["--port", "9999"],
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.validate_model(req, current_subject = "tester"))
    assert excinfo.value.status_code == 400
    assert "--port" in excinfo.value.detail


class _StopAfterGuard(Exception):
    """Sentinel so the test stops at the guard instead of building a response."""


def _drive_validate_guard(monkeypatch, *, request_extra_args, backend_extra_args):
    """Return the llama_extra_args /validate hands the training guard."""
    from types import SimpleNamespace

    import utils.models.model_config as mc

    monkeypatch.setattr(
        inf,
        "_resolve_model_identifier_for_request",
        lambda request, operation: ("org/gguf-repo", "org/gguf-repo", False),
    )
    config = SimpleNamespace(
        identifier = "org/gguf-repo",
        display_name = "org/gguf-repo",
        is_gguf = True,
        is_lora = False,
        is_vision = False,
        gguf_file = None,
        gguf_variant = "",
    )
    monkeypatch.setattr(inf.ModelConfig, "from_identifier", staticmethod(lambda **_kw: config))
    monkeypatch.setattr(mc, "get_base_model_from_lora_identifier", lambda *_a, **_k: None)
    monkeypatch.setattr(inf, "_effective_load_in_4bit", lambda *_a, **_k: False)
    monkeypatch.setattr(
        inf,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            extra_args = list(backend_extra_args),
            extra_args_source = ("org/gguf-repo", ""),
        ),
    )

    seen = {}

    def _capture(_config, **kwargs):
        seen["llama_extra_args"] = kwargs.get("llama_extra_args")
        raise _StopAfterGuard

    monkeypatch.setattr(inf, "_guard_chat_load_against_training", _capture)

    kwargs = {} if request_extra_args is None else {"llama_extra_args": request_extra_args}
    req = ValidateModelRequest(model_path = "org/gguf-repo", **kwargs)
    with pytest.raises(HTTPException):
        asyncio.run(inf.validate_model(req, current_subject = "tester"))
    return seen["llama_extra_args"]


def test_validate_guard_sizes_with_the_requests_own_extra_args(monkeypatch):
    # The guard's KV estimate reads -c/--ctx-size out of the extras. Dropping the
    # caller's own extras lets /validate pass an estimate smaller than the /load
    # that follows, so /load 409s after the UI already unloaded the live model.
    seen = _drive_validate_guard(
        monkeypatch,
        request_extra_args = ["-c", "65536"],
        backend_extra_args = [],
    )
    assert seen == ["-c", "65536"]


def test_validate_guard_still_inherits_when_request_omits_extra_args(monkeypatch):
    # Omitted means "inherit" on /load; /validate must size the same way.
    seen = _drive_validate_guard(
        monkeypatch,
        request_extra_args = None,
        backend_extra_args = ["-c", "65536"],
    )
    assert seen == ["-c", "65536"]


def test_validate_guard_honours_an_explicit_empty_extra_args(monkeypatch):
    # Explicit [] clears inherited extras on /load; the estimate must follow.
    seen = _drive_validate_guard(
        monkeypatch,
        request_extra_args = [],
        backend_extra_args = ["-c", "65536"],
    )
    assert seen == []


def _loaded_llama_backend(extra_args):
    """Minimal stand-in for a loaded llama-server, as /status reads it."""
    from types import SimpleNamespace
    return SimpleNamespace(
        is_loaded = True,
        model_identifier = "org/gguf-repo",
        is_vision = False,
        is_diffusion = False,
        hf_variant = "Q4_K_M",
        chat_template_override = None,
        supports_reasoning = False,
        reasoning_style = "enable_thinking",
        reasoning_effort_levels = [],
        reasoning_always_on = False,
        supports_preserve_thinking = False,
        supports_tools = False,
        chat_template = None,
        context_length = 4096,
        max_context_length = 4096,
        native_context_length = 4096,
        cache_type_kv = None,
        requested_spec_mode = None,
        spec_draft_n_max = None,
        tensor_parallel = False,
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        n_cpu_moe = 0,
        tensor_split = None,
        requested_n_ctx = 0,
        n_layers = None,
        n_moe_layers = 0,
        gpu_ids = None,
        requested_gpu_ids = None,
        extra_args = extra_args,
        spec_fallback_reason = None,
    )


@pytest.mark.parametrize("extra_args", [["--cpu-moe", "--no-mmap"], [], None])
def test_status_exposes_the_active_llama_extra_args(monkeypatch, extra_args):
    """/status must report the extras the running llama-server was launched with.

    They are the same list ``_resolve_inherited_extra_args`` treats as
    authoritative, so without them the UI shows an empty args field after a
    refresh while the server still runs them -- and a Reload then omits the
    field, which /load reads as "inherit", silently keeping args the user can
    neither see nor clear.
    """
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _loaded_llama_backend(extra_args))
    monkeypatch.setattr(inf, "load_inference_config", lambda *_a, **_k: None)
    monkeypatch.setattr(inf, "resolve_effective_chat_template_override", lambda **_k: None)

    resp = asyncio.run(inf.get_status(current_subject = "tester"))
    assert resp.is_gguf is True
    assert resp.llama_extra_args == extra_args


def test_load_response_reports_the_effective_llama_extra_args():
    """The GGUF /load response must echo what the server launched with.

    Manual GPU memory owns the offload group, so /load strips --cpu-moe /
    --gpu-layers / --fit from EXPLICIT extras before launching. Echoing the
    request instead would make the settings panel advertise a flag that was
    dropped, and leave dirty tracking baselined on a launch that never happened.
    """
    import inspect

    from core.inference.llama_server_args import strip_shadowing_flags
    from models.inference import LoadResponse

    # The strip the route applies in manual mode, with its own toggles.
    effective = strip_shadowing_flags(
        ["--cpu-moe", "--no-mmap"],
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = False,
        strip_tensor_split = False,
        strip_offload = True,
    )
    assert effective == ["--no-mmap"]

    # The response can carry it, and defaults to None for the non-GGUF paths.
    assert LoadResponse.model_fields["llama_extra_args"].default is None
    echoed = LoadResponse(
        status = "loaded",
        model = "org/gguf-repo",
        display_name = "org/gguf-repo",
        inference = {},
        llama_extra_args = effective,
    )
    assert echoed.llama_extra_args == effective

    # Both GGUF response sites (fresh load and the already-loaded dedupe) source
    # it from the backend's effective list, never from the request.
    src = inspect.getsource(inf._load_model_impl)
    assert src.count("llama_extra_args = llama_backend.extra_args,") == 2
    assert "llama_extra_args = request.llama_extra_args," not in src
