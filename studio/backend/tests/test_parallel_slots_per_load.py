# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the per-load parallel-slots knob.

An optional ``n_parallel`` (llama-server ``--parallel``) rides on LoadRequest;
omitted, the server-wide launch default (``run.py --parallel``) applies. These
tests pin the pydantic contract and the shared PARALLEL_MIN/MAX mirrors, the
``requested_parallel_slots`` lifecycle, the ``_already_in_target_state``
requested-vs-requested reload branch with its diffusion skip, and the route
wiring behind the /load, /validate and /status echoes.
"""

from __future__ import annotations

import inspect
import re
import struct
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Same external-dep stubs as the other llama_cpp unit tests.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# Real httpx: a stub would poison a combined run (routes/inference reads its
# attrs at def time).
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
    # /validate sizes like /load, so it carries the same field and bounds.
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
    # The UI clamps with its own copy; a bumped PARALLEL_MAX that skips it would
    # leave the UI silently capping lower.
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


def test_override_mirror_matches_shared_bounds():
    # Mirrored, not imported: llama_server_args owns the allow-list that module stays out of.
    from utils.openai_auto_switch_settings import PARALLEL_SLOTS_MAX, PARALLEL_SLOTS_MIN
    assert (PARALLEL_SLOTS_MIN, PARALLEL_SLOTS_MAX) == (PARALLEL_MIN, PARALLEL_MAX)


def test_preset_model_reuses_shared_bounds():
    # Bounds drifting from PARALLEL_MIN/MAX would 422 valid presets on every sync.
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
    # n_parallel may be reduced before the commit, so the requested value must
    # come from the pre-reduction pending snapshot.
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
    # An identical re-Apply must dedupe even after the fitter reduced the slots.
    backend = _loaded_backend()
    backend._requested_n_parallel = 8
    backend._commit_effective_parallel_slots(4)
    assert _target_state(backend, 8) is True


def test_already_in_target_state_ignores_slots_for_diffusion():
    # The diffusion runner ignores --parallel, so a slots change must not reload.
    backend = _loaded_backend()
    backend._is_diffusion = True
    backend._requested_n_parallel = 1
    assert _target_state(backend, 8) is True


# ── Route wiring (source contract, mirroring test_gpu_memory_mode) ───


def _route_source() -> str:
    return (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")


def _load_impl_source() -> str:
    """Body of _load_model_impl only, so positional assertions can't be
    satisfied by a later function in the module."""
    src = _route_source()
    body = src[src.index("async def _load_model_impl") :]
    return body[: body.index("\n@router.")]


def test_route_resolves_slots_once_before_dedupe_guard_and_load():
    load_impl = _load_impl_source()
    resolve = load_impl.index("request.n_parallel")
    fallback = load_impl.index('getattr(_app_state, "llama_parallel_slots", 1)')
    dedupe = load_impl.index("requested_parallel_slots = _n_parallel")
    guard = load_impl.index("_guard_chat_load_against_training")
    # The GGUF launch kwargs, not the guard's own kwarg (which shares the spelling).
    load_kwargs = load_impl.index("_common_load_kwargs = dict(")
    assert resolve < dedupe, "resolution must precede the reload dedupe"
    assert fallback < dedupe
    assert resolve < guard < load_kwargs
    # Guard and load kwargs share the resolved value; app.state is read once.
    assert load_impl.count("n_parallel = _n_parallel") == 2
    assert "n_parallel = _n_parallel" in load_impl[load_kwargs : load_kwargs + 800]
    assert load_impl.count('getattr(_app_state, "llama_parallel_slots", 1)') == 1
    # getattr, so a direct caller without an app cannot raise, and no re-read.
    assert "fastapi_request.app.state" not in load_impl


def test_route_dedupe_compares_requested_slots_and_skips_diffusion():
    match_impl = _route_source()[_route_source().index("def _request_matches_loaded_settings") :]
    match_impl = match_impl[: match_impl.index("\ndef ")]
    assert "requested_parallel_slots is not None" in match_impl
    assert "not llama_backend.is_diffusion" in match_impl
    assert "llama_backend.requested_parallel_slots" in match_impl


def test_route_echoes_requested_and_effective_slots():
    route_src = _route_source()
    # Both /load returns plus the /status GGUF branch, via the shared helper.
    assert route_src.count("**_parallel_slot_echo(llama_backend)") == 3


def test_parallel_slot_echo_reports_none_for_diffusion():
    # Diffusion never commits a count, so echoing the reset placeholder 1 would lie.
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
    fallback = validate_impl.index('"llama_parallel_slots",')
    guard = validate_impl.index("_guard_chat_load_against_training")
    assert guard < resolve and guard < fallback, "the guard call resolves the slots inline"


def _load_model_source() -> str:
    return inspect.getsource(LlamaCppBackend.load_model)


def test_slots_fall_back_to_one_without_kv_unified():
    # Without --kv-unified llama-server gives each slot -c/N, so an explicit
    # --parallel N shrinks every context window.
    src = _load_model_source()
    clamp = src.find("supports_kv_unified")
    assert clamp != -1, "load_model must check for --kv-unified before honouring the slots"
    block = src[clamp : clamp + 700]
    assert (
        "n_parallel > 1" in src[clamp - 300 : clamp]
    ), "only an explicit multi-slot load is clamped"
    assert "n_parallel = 1" in block


def test_clamp_sits_between_the_echo_and_the_fit():
    # The echo reports the ask and the fit uses what launches, so the clamp
    # belongs between the two.
    src = _load_model_source()
    pending = src.index("_pending_load_kwargs")
    clamp = src.index("supports_kv_unified")
    estimate = src.index("_estimate")
    commit = src.index("_commit_effective_parallel_slots")
    assert pending < clamp, "the requested count is captured before the clamp"
    assert clamp < estimate, "the fit must be estimated from the effective slot count"
    assert clamp < commit, "the committed effective count is the clamped one"


# ── Training-guard sizing ────────────────────────────────────────────


def _write_swa_gguf(path: Path) -> str:
    """Smallest DiffusionGemma-shaped header the KV estimator can size: the
    canvas marker routing it to the diffusion runner, plus the sliding-window
    dims that make llama.cpp's SWA cache slot-scaled."""

    def _kv_str(key: str, value: str) -> bytes:
        kb, vb = key.encode(), value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        )

    def _kv_u32(key: str, value: int) -> bytes:
        kb = key.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)

    arch = "diffusion-gemma"
    kvs = [
        _kv_str("general.architecture", arch),
        _kv_u32("diffusion.canvas_length", 256),
        _kv_u32(f"{arch}.context_length", 32768),
        _kv_u32(f"{arch}.block_count", 30),
        _kv_u32(f"{arch}.attention.head_count", 16),
        _kv_u32(f"{arch}.attention.head_count_kv", 8),
        _kv_u32(f"{arch}.attention.key_length", 512),
        _kv_u32(f"{arch}.attention.value_length", 512),
        _kv_u32(f"{arch}.attention.sliding_window", 1024),
        _kv_u32(f"{arch}.attention.key_length_swa", 256),
        _kv_u32(f"{arch}.attention.value_length_swa", 256),
    ]
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kvs)) + b"".join(kvs))
    return str(path)


def _guard_required_gb(
    monkeypatch,
    gguf_path: str,
    *,
    n_parallel: int,
    diffusion,
    caps = None,
    llama_extra_args = None,
) -> float:
    """Run the training guard over a local GGUF and return the size it budgeted."""
    import routes.inference as inf

    seen = {}

    core_training = _types.ModuleType("core.training")
    core_training.get_training_backend = lambda: _types.SimpleNamespace(
        is_training_active = lambda: True
    )

    def _can_load(**kwargs):
        seen.update(kwargs)
        return True, {"mode": "single_device"}

    training_vram = _types.ModuleType("routes.training_vram")
    training_vram.can_load_chat_during_training = _can_load
    monkeypatch.setitem(sys.modules, "core.training", core_training)
    monkeypatch.setitem(sys.modules, "routes.training_vram", training_vram)

    monkeypatch.setattr(inf, "_classify_diffusion_gguf", lambda _config: diffusion)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a, **k: False))
    monkeypatch.setattr(LlamaCppBackend, "_effective_gpu_count", staticmethod(lambda *a, **k: 1))
    monkeypatch.setattr(LlamaCppBackend, "_diffusion_gpu_arg", staticmethod(lambda *a, **k: "0"))
    # Pin the --kv-unified probe so the estimate cannot depend on a locally
    # installed llama-server. Default "no binary found" leaves the count alone.
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(caps or {})),
    )

    inf._guard_chat_load_against_training(
        _types.SimpleNamespace(is_gguf = True, gguf_file = gguf_path, identifier = "local/model"),
        model_identifier = "local/model",
        hf_token = None,
        load_in_4bit = False,
        max_seq_length = 8192,
        requested_gpu_ids = None,
        n_parallel = n_parallel,
        gpu_memory_mode = "auto",
        llama_extra_args = llama_extra_args,
    )
    return seen["required_override_gb"]


def test_training_guard_sizes_a_diffusion_gguf_at_one_slot(monkeypatch, tmp_path):
    # Diffusion ignores --parallel, so slots must not inflate the estimate and 409
    # a load that would have fitted beside training.
    gguf = _write_swa_gguf(tmp_path / "diffusion.gguf")
    one = _guard_required_gb(monkeypatch, gguf, n_parallel = 1, diffusion = True)
    many = _guard_required_gb(monkeypatch, gguf, n_parallel = 8, diffusion = True)
    assert one == many


def test_training_guard_still_sizes_slots_for_an_ordinary_gguf(monkeypatch, tmp_path):
    # llama-server does allocate per-slot SWA cells, so the reduction above must
    # be scoped to diffusion and not flatten every GGUF to one slot.
    gguf = _write_swa_gguf(tmp_path / "chat.gguf")
    one = _guard_required_gb(monkeypatch, gguf, n_parallel = 1, diffusion = False)
    many = _guard_required_gb(monkeypatch, gguf, n_parallel = 8, diffusion = False)
    assert many > one


def test_training_guard_sizes_one_slot_when_the_binary_has_no_kv_unified(monkeypatch, tmp_path):
    # load_model clamps a multi-slot request to 1 on such a build, where each slot
    # carries its own SWA stream, so sizing the asked count would 409 a load that fits.
    gguf = _write_swa_gguf(tmp_path / "chat.gguf")
    old = {"found": True, "supports_kv_unified": False}
    one = _guard_required_gb(monkeypatch, gguf, n_parallel = 1, diffusion = False, caps = old)
    many = _guard_required_gb(monkeypatch, gguf, n_parallel = 8, diffusion = False, caps = old)
    assert one == many


def test_training_guard_sizes_every_slot_when_kv_unified_exists(monkeypatch, tmp_path):
    # The clamp is scoped to binaries that cannot serve the slots; a capable one
    # really does allocate the SWA window per slot.
    gguf = _write_swa_gguf(tmp_path / "chat.gguf")
    new = {"found": True, "supports_kv_unified": True}
    one = _guard_required_gb(monkeypatch, gguf, n_parallel = 1, diffusion = False, caps = new)
    many = _guard_required_gb(monkeypatch, gguf, n_parallel = 8, diffusion = False, caps = new)
    assert many > one


def test_training_guard_keeps_the_asked_slots_for_an_explicit_mtp_load(monkeypatch, tmp_path):
    # load_model does clamp MTP to a single slot, but this estimate counts only the
    # drafter file and the main KV: not the draft KV, the duplicated target context MLA
    # keeps, or the draft compute reserve. Sizing for one slot would drop the slot KV
    # without adding those back, and a guard that under-sizes evicts the training run it
    # protects, so it keeps the larger number.
    gguf = _write_swa_gguf(tmp_path / "chat.gguf")
    mtp = ["--spec-type", "draft-mtp"]
    new = {"found": True, "supports_kv_unified": True}
    one = _guard_required_gb(
        monkeypatch, gguf, n_parallel = 1, diffusion = False, caps = new, llama_extra_args = mtp
    )
    many = _guard_required_gb(
        monkeypatch, gguf, n_parallel = 8, diffusion = False, caps = new, llama_extra_args = mtp
    )
    assert many > one


def test_training_guard_keeps_slots_when_the_launch_scrubs_the_mtp_env(monkeypatch, tmp_path):
    # An inherited LLAMA_ARG_SPEC_TYPE=draft-mtp really would launch MTP, since llama.cpp
    # appends spec types rather than replacing them. But the extras do not own
    # --spec-type here, so the launch scrubs the env and the server runs the slots asked
    # for; flattening the budget to one would under-size and let a load past the guard.
    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "draft-mtp")
    gguf = _write_swa_gguf(tmp_path / "chat.gguf")
    new = {"found": True, "supports_kv_unified": True}
    one = _guard_required_gb(monkeypatch, gguf, n_parallel = 1, diffusion = False, caps = new)
    many = _guard_required_gb(monkeypatch, gguf, n_parallel = 8, diffusion = False, caps = new)
    assert many > one


def test_training_guard_keeps_slots_for_an_unclassified_gguf(monkeypatch, tmp_path):
    # None = inconclusive header, so keep the larger estimate rather than
    # under-size against training.
    gguf = _write_swa_gguf(tmp_path / "unknown.gguf")
    one = _guard_required_gb(monkeypatch, gguf, n_parallel = 1, diffusion = None)
    many = _guard_required_gb(monkeypatch, gguf, n_parallel = 8, diffusion = None)
    assert many > one
