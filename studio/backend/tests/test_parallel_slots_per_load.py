# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the per-load parallel-slots knob.

The web UI threads a single optional ``n_parallel`` (llama-server
``--parallel``) from the run-settings form through LoadRequest; omitted, the
server-wide launch default (``run.py --parallel``) applies. These tests pin:

  * the pydantic request/response/status contract (default None, bounds shared
    with llama_server_args.PARALLEL_MIN/MAX and their run.py / CLI mirrors),
  * the backend ``requested_parallel_slots`` property and its lifecycle
    (committed from the pre-reduction pending kwargs, reset with the
    effective count),
  * the ``_already_in_target_state`` requested-vs-requested reload branch and
    its diffusion skip, and
  * the route wiring: one resolution point feeding the training guard, the
    load kwargs, the reload dedupe, and the /load, /validate and /status
    echoes.
"""

from __future__ import annotations

import inspect
import re
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Same external-dep stubs as the other llama_cpp unit tests so importing
# the backend doesn't drag in structlog / loggers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# Real httpx (installed backend dep): a hand-rolled stub would poison a
# combined pytest run (routes/inference references httpx attrs at def time).
import httpx  # noqa: F401

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_server_args import PARALLEL_MAX, PARALLEL_MIN
from core.inference.llama_cpp import LlamaCppBackend
from models.inference import (
    InferenceStatusResponse,
    LoadRequest,
    LoadResponse,
    ValidateModelRequest,
)


class _FakeProcess:
    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


# ── Pydantic contract ────────────────────────────────────────────────


def test_load_request_defaults_n_parallel_none():
    assert LoadRequest(model_path = "owner/repo").n_parallel is None


@pytest.mark.parametrize("value", [PARALLEL_MIN, 4, PARALLEL_MAX])
def test_load_request_accepts_in_range_n_parallel(value):
    assert LoadRequest(model_path = "owner/repo", n_parallel = value).n_parallel == value


@pytest.mark.parametrize("value", [0, -1, PARALLEL_MAX + 1])
def test_load_request_rejects_out_of_range_n_parallel(value):
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", n_parallel = value)


def test_load_request_round_trips_json_key():
    req = LoadRequest.model_validate({"model_path": "owner/repo", "n_parallel": 8})
    assert req.n_parallel == 8
    assert req.model_dump()["n_parallel"] == 8


def test_validate_request_n_parallel_contract():
    # /validate sizes the coexistence estimate like /load, so it carries the
    # same optional field with the same bounds.
    assert ValidateModelRequest(model_path = "owner/repo").n_parallel is None
    assert (
        ValidateModelRequest(model_path = "owner/repo", n_parallel = PARALLEL_MAX).n_parallel
        == PARALLEL_MAX
    )
    with pytest.raises(ValueError):
        ValidateModelRequest(model_path = "owner/repo", n_parallel = PARALLEL_MAX + 1)


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_response_models_emit_parallel_slot_fields(model_cls):
    kwargs = (
        dict(status = "loaded", model = "owner/repo", display_name = "repo", inference = {})
        if model_cls is LoadResponse
        else {}
    )
    empty = model_cls(**kwargs).model_dump()
    assert empty["requested_parallel_slots"] is None
    assert empty["parallel_slots"] is None
    dumped = model_cls(**kwargs, requested_parallel_slots = 8, parallel_slots = 4).model_dump()
    assert dumped["requested_parallel_slots"] == 8
    assert dumped["parallel_slots"] == 4


# ── Shared bounds and their deliberate mirrors ───────────────────────


def _mirrored_bounds(source_path: Path) -> tuple[int, int]:
    src = source_path.read_text(encoding = "utf-8")
    low = re.search(r"^_PARALLEL_MIN\s*=\s*(\d+)$", src, re.MULTILINE)
    high = re.search(r"^_PARALLEL_MAX\s*=\s*(\d+)$", src, re.MULTILINE)
    assert low and high, f"{source_path} must define _PARALLEL_MIN/_PARALLEL_MAX"
    return int(low.group(1)), int(high.group(1))


def test_run_py_mirror_matches_shared_bounds():
    assert _mirrored_bounds(Path(_BACKEND_DIR) / "run.py") == (PARALLEL_MIN, PARALLEL_MAX)


def test_cli_mirror_matches_shared_bounds():
    cli = Path(_BACKEND_DIR).parent.parent / "unsloth_cli" / "commands" / "studio.py"
    assert _mirrored_bounds(cli) == (PARALLEL_MIN, PARALLEL_MAX)


def test_frontend_mirror_matches_shared_bounds():
    # The web UI clamps the control with its own copy of the bounds; a bumped
    # PARALLEL_MAX that skips it would leave the UI silently capping lower.
    src = (
        Path(_BACKEND_DIR).parent
        / "frontend"
        / "src"
        / "features"
        / "model-picker"
        / "model-config"
        / "per-model-config.ts"
    ).read_text(encoding = "utf-8")
    low = re.search(r"^export const N_PARALLEL_MIN = (\d+);$", src, re.MULTILINE)
    high = re.search(r"^export const N_PARALLEL_MAX = (\d+);$", src, re.MULTILINE)
    assert low and high, "per-model-config.ts must export N_PARALLEL_MIN/MAX"
    assert (int(low.group(1)), int(high.group(1))) == (PARALLEL_MIN, PARALLEL_MAX)


def test_preset_model_reuses_shared_bounds():
    # ChatPresetLoadConfig is extra="forbid": hardcoded bounds drifting from
    # PARALLEL_MIN/MAX would 422 valid presets on every settings sync.
    from routes.chat_history import ChatPresetLoadConfig

    field = ChatPresetLoadConfig.model_fields["nParallel"]
    bounds = {type(m).__name__: getattr(m, "ge", getattr(m, "le", None)) for m in field.metadata}
    assert bounds.get("Ge") == PARALLEL_MIN
    assert bounds.get("Le") == PARALLEL_MAX


# ── requested_parallel_slots lifecycle ───────────────────────────────


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_kill_orphaned_servers", lambda self: 0)
    monkeypatch.setattr(llama_cpp_module.atexit, "register", lambda *_args, **_kwargs: None)
    return LlamaCppBackend()


def test_requested_parallel_slots_initial_value_is_one(backend):
    assert backend.requested_parallel_slots == 1


def test_requested_parallel_slots_reflects_field(backend):
    backend._requested_n_parallel = 8
    assert backend.requested_parallel_slots == 8


@pytest.mark.parametrize("value", [None, 0, -2, "not-an-int"])
def test_requested_parallel_slots_invalid_value_falls_back_to_one(backend, value):
    backend._requested_n_parallel = value
    assert backend.requested_parallel_slots == 1


def test_reset_effective_parallel_slots_also_resets_requested(backend):
    backend._requested_n_parallel = 8
    backend._commit_effective_parallel_slots(4)

    backend._reset_effective_parallel_slots()

    assert backend.requested_parallel_slots == 1
    assert backend.effective_parallel_slots == 1


def test_unload_resets_requested_parallel_slots(backend):
    backend._process = _FakeProcess()
    backend._requested_n_parallel = 8

    backend.unload_model()

    assert backend.requested_parallel_slots == 1


def test_load_model_commits_requested_from_pending_kwargs():
    # The local n_parallel may be rebound by the fit-time slot reduction before
    # the healthy commit; the requested value must come from the pre-reduction
    # pending snapshot, after _healthy flips True.
    src = inspect.getsource(LlamaCppBackend.load_model)
    commit = src.find(
        'self._requested_n_parallel = max(1, int(_pending_load_kwargs["n_parallel"]))'
    )
    healthy = src.find("self._healthy = True\n", 0, commit if commit != -1 else None)
    snapshot = src.find("self._last_load_kwargs = _pending_load_kwargs")
    assert commit != -1, "load_model must commit the requested slot count"
    assert healthy != -1 and healthy < commit < snapshot


# ── _already_in_target_state requested-vs-requested branch ───────────


def _loaded_backend() -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()  # is_loaded only checks "is not None"
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._gguf_path = None
    return backend


def _target_state(backend: LlamaCppBackend, n_parallel: int) -> bool:
    return backend._already_in_target_state(
        gguf_path = None,
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        n_parallel = n_parallel,
    )


def test_already_in_target_state_matches_same_slots():
    backend = _loaded_backend()
    backend._requested_n_parallel = 4
    assert _target_state(backend, 4) is True


def test_already_in_target_state_reloads_on_slots_change():
    backend = _loaded_backend()
    backend._requested_n_parallel = 4
    assert _target_state(backend, 8) is False


def test_already_in_target_state_compares_requested_not_effective():
    # The fitter may launch fewer slots than requested; an identical re-Apply
    # must still dedupe or it would reload (and re-reduce) forever.
    backend = _loaded_backend()
    backend._requested_n_parallel = 8
    backend._commit_effective_parallel_slots(4)
    assert _target_state(backend, 8) is True


def test_already_in_target_state_ignores_slots_for_diffusion():
    # The diffusion runner ignores --parallel, so a slots change must not
    # force a needless reload.
    backend = _loaded_backend()
    backend._is_diffusion = True
    backend._requested_n_parallel = 1
    assert _target_state(backend, 8) is True


# ── Route wiring (source contract, mirroring test_gpu_memory_mode) ───


def _route_source() -> str:
    return (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")


def _load_impl_source() -> str:
    """Body of _load_model_impl only, so positional assertions can't be
    satisfied by a later function further down the module."""
    src = _route_source()
    body = src[src.index("async def _load_model_impl") :]
    return body[: body.index("\n@router.")]


def test_route_resolves_slots_once_before_dedupe_guard_and_load():
    load_impl = _load_impl_source()
    resolve = load_impl.index("request.n_parallel")
    fallback = load_impl.index('getattr(fastapi_request.app.state, "llama_parallel_slots", 1)')
    dedupe = load_impl.index("requested_parallel_slots = _n_parallel")
    guard = load_impl.index("_guard_chat_load_against_training")
    # The GGUF launch kwargs, not the guard's own kwarg (which shares the spelling).
    load_kwargs = load_impl.index("_common_load_kwargs = dict(")
    assert resolve < dedupe, "resolution must precede the reload dedupe"
    assert fallback < dedupe
    assert resolve < guard < load_kwargs
    # The guard and the load kwargs consume the same resolved value, and
    # nothing re-reads app.state after the single resolution point.
    assert load_impl.count("n_parallel = _n_parallel") == 2
    assert "n_parallel = _n_parallel" in load_impl[load_kwargs : load_kwargs + 800]
    assert load_impl.count('getattr(fastapi_request.app.state, "llama_parallel_slots", 1)') == 1


def test_route_dedupe_compares_requested_slots_and_skips_diffusion():
    match_impl = _route_source()[_route_source().index("def _request_matches_loaded_settings") :]
    match_impl = match_impl[: match_impl.index("\ndef ")]
    assert "requested_parallel_slots is not None" in match_impl
    assert "not llama_backend.is_diffusion" in match_impl
    assert "llama_backend.requested_parallel_slots" in match_impl


def test_route_echoes_requested_and_effective_slots():
    route_src = _route_source()
    # Two /load returns (already_loaded + loaded) and the /status GGUF branch,
    # all through the shared echo helper.
    assert route_src.count("**_parallel_slot_echo(llama_backend)") == 3


def test_parallel_slot_echo_reports_none_for_diffusion():
    # The diffusion runner ignores --parallel and never commits a count, so
    # echoing its reset placeholder 1 would fabricate "invoked with 1 slot".
    from routes.inference import _parallel_slot_echo

    backend = _loaded_backend()
    backend._requested_n_parallel = 8
    backend._commit_effective_parallel_slots(4)
    assert _parallel_slot_echo(backend) == {"requested_parallel_slots": 8, "parallel_slots": 4}
    backend._is_diffusion = True
    assert _parallel_slot_echo(backend) == {
        "requested_parallel_slots": None,
        "parallel_slots": None,
    }


def test_validate_route_prefers_request_n_parallel():
    validate_impl = _route_source()[_route_source().index("async def validate_model") :]
    resolve = validate_impl.index("request.n_parallel")
    fallback = validate_impl.index('getattr(fastapi_request.app.state, "llama_parallel_slots", 1)')
    guard = validate_impl.index("_guard_chat_load_against_training")
    assert guard < resolve and guard < fallback, "the guard call resolves the slots inline"
