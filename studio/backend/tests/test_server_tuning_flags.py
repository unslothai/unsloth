# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The four first-class llama-server tuning fields.

load_mode (--load-mode), spec_draft_cache_type (--spec-draft-type-k/-v),
ctx_checkpoints (--ctx-checkpoints) and cache_ram (--cache-ram): pydantic bounds,
the Model Memory precedence the Run settings panel promises, shadow stripping,
reload dedupe and the stored-override mapping.

Sibling of test_batch_sizes_per_load.py, which covers the same shape for the
batch pair.
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
    _normalized_load_mode,
)
from core.inference import llama_server_args as lsa
from core.inference.llama_server_args import (
    CACHE_RAM_MAX_MIB,
    CTX_CHECKPOINTS_MAX,
    apply_load_mode_policy,
    parse_ctx_checkpoints_override,
    resolve_ctx_checkpoints,
    strip_shadowing_flags,
)
from models.inference import LoadRequest
from utils.openai_auto_switch_settings import (
    model_override_load_kwargs,
    normalize_model_override,
)


# --------------------------------------------------------------------------- request


def test_load_request_defaults_are_unset():
    request = LoadRequest(model_path = "owner/repo")
    assert request.load_mode is None
    assert request.spec_draft_cache_type is None
    assert request.ctx_checkpoints is None
    assert request.cache_ram is None


@pytest.mark.parametrize("mode", ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"])
def test_load_request_accepts_every_documented_mode(mode):
    assert LoadRequest(model_path = "owner/repo", load_mode = mode).load_mode == mode


def test_load_request_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", load_mode = "mmap + mlock")


def test_ctx_checkpoints_and_cache_ram_bounds():
    assert LoadRequest(model_path = "owner/repo", ctx_checkpoints = 0).ctx_checkpoints == 0
    assert (
        LoadRequest(model_path = "owner/repo", ctx_checkpoints = CTX_CHECKPOINTS_MAX).ctx_checkpoints
        == CTX_CHECKPOINTS_MAX
    )
    # -1 is "no limit" and 0 disables the cache, so both are inside the range
    assert LoadRequest(model_path = "owner/repo", cache_ram = -1).cache_ram == -1
    assert LoadRequest(model_path = "owner/repo", cache_ram = 0).cache_ram == 0
    for field, bad in (
        ("ctx_checkpoints", -1),
        ("ctx_checkpoints", CTX_CHECKPOINTS_MAX + 1),
        ("cache_ram", -2),
        ("cache_ram", CACHE_RAM_MAX_MIB + 1),
    ):
        with pytest.raises(ValueError):
            LoadRequest(model_path = "owner/repo", **{field: bad})


@pytest.mark.parametrize("field", ["ctx_checkpoints", "cache_ram"])
def test_integer_fields_reject_json_booleans(field):
    # bool subclasses int, so lax pydantic would turn `true` into 1 and launch the
    # child with a number nobody typed. Same guard the batch pair carries.
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", **{field: True})
    assert getattr(LoadRequest(model_path = "owner/repo", **{field: "16"}), field) == 16


# ------------------------------------------------------------------- load mode policy


@pytest.fixture
def memory_settings(monkeypatch):
    """Stand in for utils.model_memory_settings, which needs the settings DB."""
    state = {"keep_resident": False, "no_ram_reserve": False}
    module = _types.ModuleType("utils.model_memory_settings")
    module.get_model_memory_settings = lambda: (
        state["keep_resident"],
        state["no_ram_reserve"],
    )
    monkeypatch.setitem(sys.modules, "utils.model_memory_settings", module)
    return state


def test_load_mode_is_emitted_when_no_setting_objects(memory_settings):
    managed, extras = apply_load_mode_policy(
        [], supports_load_mode = True, requested_load_mode = "mlock"
    )
    assert managed == ["--load-mode", "mlock"]
    assert extras == []


def test_auto_and_unknown_modes_emit_nothing(memory_settings):
    for mode in (None, "", "auto", "AUTO", "mmap + mlock"):
        assert apply_load_mode_policy(
            ["--top-k", "20"], supports_load_mode = True, requested_load_mode = mode
        ) == ([], ["--top-k", "20"])


def test_an_explicitly_typed_load_mode_still_wins(memory_settings):
    # The control emits BEFORE the extras, so a flag typed for THIS load is
    # appended after it and last-wins, which is what the panel's diagnostics
    # promise. Only the route strips, and only an INHERITED copy.
    managed, extras = apply_load_mode_policy(
        ["--load-mode", "dio", "--top-k", "20"],
        supports_load_mode = True,
        requested_load_mode = "mmap",
    )
    assert managed == ["--load-mode", "mmap"]
    assert extras == ["--load-mode", "dio", "--top-k", "20"]


def test_the_route_strips_an_inherited_load_mode(memory_settings):
    # A trailing alias resets the whole mode in llama.cpp, so an INHERITED copy
    # would silently undo the pick. The route drops it when the field is set, the
    # same rule the batch pair follows; the policy itself strips nothing.
    inherited = ["--no-mmap", "--mlock", "--top-k", "20"]
    assert strip_shadowing_flags(
        inherited,
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = False,
        strip_load_mode = True,
        strip_load_mode_aliases = True,
    ) == ["--mlock", "--top-k", "20"]


def test_keep_resident_owns_the_mode(memory_settings):
    memory_settings["keep_resident"] = True
    assert apply_load_mode_policy([], supports_load_mode = True, requested_load_mode = "dio") == (
        [],
        [],
    )


def test_keep_resident_releases_the_mode_when_the_weights_are_not_host_resident(memory_settings):
    # The page-lock is skipped for a fully offloaded model, so nothing else is
    # claiming the mode and the pick applies.
    memory_settings["keep_resident"] = True
    managed, _ = apply_load_mode_policy(
        [],
        supports_load_mode = True,
        weights_in_host_memory = False,
        requested_load_mode = "dio",
    )
    assert managed == ["--load-mode", "dio"]


@pytest.mark.parametrize("mode", ["none", "mlock", "mmap+mlock"])
def test_no_ram_reserve_vetoes_the_reserving_modes(memory_settings, mode):
    memory_settings["no_ram_reserve"] = True
    assert apply_load_mode_policy([], supports_load_mode = True, requested_load_mode = mode) == ([], [])


@pytest.mark.parametrize("mode", ["mmap", "dio"])
def test_no_ram_reserve_leaves_the_non_reserving_modes(memory_settings, mode):
    # Neither holds a full host copy, so there is nothing for the setting to veto.
    memory_settings["no_ram_reserve"] = True
    managed, _ = apply_load_mode_policy([], supports_load_mode = True, requested_load_mode = mode)
    assert managed == ["--load-mode", mode]


def test_a_build_without_load_mode_falls_back_to_the_deprecated_spellings(memory_settings):
    assert apply_load_mode_policy([], supports_load_mode = False, requested_load_mode = "mmap+mlock")[
        0
    ] == ["--mlock"]
    assert apply_load_mode_policy([], supports_load_mode = False, requested_load_mode = "none")[0] == [
        "--no-mmap"
    ]
    # No pre-enum spelling for these two, so they are skipped rather than approximated
    for mode in ("mmap", "dio"):
        assert (
            apply_load_mode_policy([], supports_load_mode = False, requested_load_mode = mode)[0] == []
        )


def test_the_panel_and_the_policy_agree_on_which_modes_no_reserve_vetoes():
    # The Run settings note names the setting that wins, so the two sets have to
    # be the same one. RAM_RESERVING_LOAD_MODES in model-config-page.tsx.
    ui = (
        Path(_BACKEND_DIR).parent
        / "frontend"
        / "src"
        / "features"
        / "model-picker"
        / "components"
        / "model-config-page.tsx"
    ).read_text(encoding = "utf-8")
    listed = ui.split("const RAM_RESERVING_LOAD_MODES = new Set([", 1)[1].split("]")[0]
    assert {value.strip().strip('"') for value in listed.split(",") if value.strip()} == set(
        lsa._LOAD_MODE_MLOCK_VALUES | lsa._LOAD_MODE_RESERVING_VALUES
    )


# ---------------------------------------------------------------------- shadow strips


def test_strip_shadowing_flags_tuning_toggles():
    args = [
        "--ctx-checkpoints",
        "8",
        "--cache-ram=2048",
        "--spec-draft-type-k",
        "q8_0",
        "-ctvd",
        "q8_0",
        "--top-k",
        "20",
    ]
    assert strip_shadowing_flags(args, strip_ctx_checkpoints = True) == args[2:]
    assert "--cache-ram=2048" not in strip_shadowing_flags(args, strip_cache_ram = True)
    stripped = strip_shadowing_flags(args, strip_spec_draft_cache = True)
    assert stripped == ["--ctx-checkpoints", "8", "--cache-ram=2048", "--top-k", "20"]
    # nothing is stripped by default, so an inherited flag survives a load that
    # sets none of these fields
    assert strip_shadowing_flags(args) == args


def test_swa_checkpoints_is_the_same_setting():
    # upstream's older spelling of --ctx-checkpoints
    assert strip_shadowing_flags(["--swa-checkpoints", "4"], strip_ctx_checkpoints = True) == []


def test_the_effective_checkpoint_count_comes_from_the_extras():
    """A typed --ctx-checkpoints wins at launch, so it has to win in the sizing.

    The control emits its flag before the extras and llama.cpp is last-wins, so
    ctx_checkpoints=0 with "--ctx-checkpoints 256" in the extras allocates 256
    per-slot snapshots. Budgeting the field there under-reserves the fit.
    """
    assert parse_ctx_checkpoints_override(["--ctx-checkpoints", "256"]) == 256
    assert parse_ctx_checkpoints_override(["--swa-checkpoints=8"]) == 8
    # last-wins, like every other override parser here
    assert parse_ctx_checkpoints_override(["-ctxcp", "4", "--ctx-checkpoints", "16"]) == 16
    assert parse_ctx_checkpoints_override(["--top-k", "20"]) is None
    # malformed extras are refused at the boundary; sizing must not raise
    assert parse_ctx_checkpoints_override(["--ctx-checkpoints", "many"]) is None

    assert resolve_ctx_checkpoints(["--ctx-checkpoints", "256"], 0) == 256
    assert resolve_ctx_checkpoints(None, 8) == 8
    assert resolve_ctx_checkpoints([], None) == 0


def test_the_checkpoint_flag_falls_back_to_the_legacy_spelling():
    """A build carrying only --swa-checkpoints must still get the control's value.

    Upstream renamed --swa-checkpoints to --ctx-checkpoints and kept the old name
    as an alias, so a build older than the rename exposes only the old one.
    Probing the modern name alone dropped the Checkpoints pick there in silence.
    """
    import inspect

    from core.inference import llama_cpp

    probe = inspect.getsource(llama_cpp.LlamaCppBackend.probe_server_capabilities)
    # both spellings probed, most modern first, and WHICH one is recorded
    assert 'for _alias in ("--ctx-checkpoints", "--swa-checkpoints")' in probe
    assert "ctx_checkpoints_flag = _alias" in probe
    assert "supports_ctx_checkpoints = ctx_checkpoints_flag is not None" in probe
    # and the emission uses the recorded name, not a hard-coded one
    load = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    assert "cmd.extend([str(_ctxcp_flag), str(int(ctx_checkpoints))])" in load
    assert 'cmd.extend([str(server_caps["ctx_checkpoints_flag"]), "0"])' in load


def test_an_unsupported_draft_cache_dtype_is_dropped_not_launched():
    """llama-server exits on a dtype it cannot map, and by then the old model is gone."""
    import inspect

    from core.inference import llama_cpp

    assert "Q8_0".strip().lower() in llama_cpp._VALID_KV_CACHE_TYPES
    assert "fp16" not in llama_cpp._VALID_KV_CACHE_TYPES
    source = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    # normalized and allow-listed before emission, like the main cache dtype
    assert "_draft_cache_type not in _VALID_KV_CACHE_TYPES" in source
    assert "Ignoring unsupported draft KV cache type" in source


# ----------------------------------------------------------------------------- dedupe


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


def test_dedupe_matches_the_same_tuning():
    backend = _loaded_backend()
    backend._requested_load_mode = "dio"
    backend._requested_ctx_checkpoints = 8
    backend._requested_cache_ram = 2048
    assert (
        backend._runtime_matches_intent(
            _intent(load_mode = "dio", ctx_checkpoints = 8, cache_ram = 2048), None
        )
        is True
    )


def test_dedupe_reads_auto_and_unset_as_the_same_load():
    # Both launch the same command, so picking Auto must not reload a server
    # already running it.
    backend = _loaded_backend()
    assert backend._runtime_matches_intent(_intent(load_mode = "auto"), None) is True
    assert _normalized_load_mode("AUTO ") is None


@pytest.mark.parametrize(
    "changed",
    [
        {"load_mode": "mlock"},
        {"ctx_checkpoints": 16},
        {"cache_ram": 0},
    ],
)
def test_dedupe_reloads_on_a_tuning_change(changed):
    backend = _loaded_backend()
    backend._requested_load_mode = "dio"
    backend._requested_ctx_checkpoints = 8
    backend._requested_cache_ram = 2048
    intent = _intent(**{"load_mode": "dio", "ctx_checkpoints": 8, "cache_ram": 2048, **changed})
    assert backend._runtime_matches_intent(intent, None) is False


def test_dedupe_reloads_when_the_draft_cache_is_cleared():
    # Clearing the control back to the f16 default has to relaunch, or the server
    # keeps the quantized draft cache the panel no longer shows. Both sides hold
    # what was REQUESTED, so a load that asked for nothing still matches one that
    # asked for nothing.
    backend = _loaded_backend()
    backend._requested_spec_draft_cache_type = "q8_0"
    assert backend._runtime_matches_intent(_intent(), None) is False
    assert backend._runtime_matches_intent(_intent(spec_draft_cache_type = "q8_0"), None) is True
    assert backend._runtime_matches_intent(_intent(spec_draft_cache_type = "q4_0"), None) is False
    backend._requested_spec_draft_cache_type = None
    assert backend._runtime_matches_intent(_intent(), None) is True


def test_dedupe_ignores_the_tuning_for_diffusion():
    # The diffusion runner builds its own command and passes none of these.
    backend = _loaded_backend()
    backend._is_diffusion = True
    backend._diffusion_requested_ngl = None
    backend._gpu_layers = -1
    assert backend._runtime_matches_intent(_intent(load_mode = "mlock"), None) is True


def test_the_coexistence_estimate_charges_the_requested_checkpoints():
    """The training guard must size the SWA checkpoints the load will ask for.

    Checkpoints are per-slot snapshots whose size scales with the slot's context
    (ggml-org/llama.cpp#21690 is an OOM caused by exactly this), so an estimate
    that assumes zero can admit a load beside training that then runs out of VRAM.
    """
    import inspect

    from routes import inference as inference_routes

    # threaded end to end: the guard reads the field, the estimator forwards it,
    # and the KV sizing charges it
    assert (
        "ctx_checkpoints"
        in inspect.signature(inference_routes._estimate_gguf_required_gb).parameters
    )
    assert "ctx_checkpoints" in inspect.signature(inference_routes._estimate_gguf_kv_gb).parameters
    assert "ctx_checkpoints" in inspect.signature(inference_routes._gguf_runtime_bytes).parameters
    source = inspect.getsource(inference_routes._guard_chat_load_against_training)
    assert 'ctx_checkpoints = getattr(request, "ctx_checkpoints", None)' in source
    # _estimate_gguf_kv_gb is the guard's summing wrapper; the arithmetic lives in
    # _gguf_runtime_bytes, which the memory-estimate endpoint reads itemized. Both
    # links are asserted, so dropping the field in either place still fails here.
    assert "ctx_checkpoints = ctx_checkpoints" in inspect.getsource(
        inference_routes._estimate_gguf_kv_gb
    )
    kv_source = inspect.getsource(inference_routes._gguf_runtime_bytes)
    # priced on what the launch runs, so a typed --ctx-checkpoints wins here too
    assert "resolve_ctx_checkpoints(llama_extra_args, ctx_checkpoints)" in kv_source


# ------------------------------------------------------------------- override storage


def test_override_store_round_trip():
    entry = normalize_model_override(
        {
            "load_mode": "MMAP+MLOCK",
            "ctx_checkpoints": 0,
            "cache_ram": -1,
            "speculative_type": "dspark",
            "spec_draft_cache_type": "Q8_0",
        }
    )
    assert entry["load_mode"] == "mmap+mlock"
    # 0 and -1 are values, not "unset", so both are stored
    assert entry["ctx_checkpoints"] == 0
    assert entry["cache_ram"] == -1
    assert entry["spec_draft_cache_type"] == "q8_0"


def test_override_store_drops_values_the_loader_would_refuse():
    entry = normalize_model_override(
        {
            "load_mode": "swap",
            "ctx_checkpoints": True,
            "cache_ram": -2,
        }
    )
    assert "load_mode" not in entry
    assert "ctx_checkpoints" not in entry
    assert "cache_ram" not in entry


def test_override_store_drops_a_draft_dtype_without_a_separate_drafter():
    # ngram loads no draft model, so there is no draft context for the dtype to
    # apply to; storing it would show an edit the loader ignores.
    entry = normalize_model_override({"speculative_type": "ngram", "spec_draft_cache_type": "q8_0"})
    assert "spec_draft_cache_type" not in entry


def test_override_kwargs_are_gguf_only():
    override = {
        "load_mode": "dio",
        "spec_draft_cache_type": "q8_0",
        "ctx_checkpoints": 8,
        "cache_ram": 2048,
    }
    kwargs = model_override_load_kwargs(override, is_gguf = True)
    for key, value in override.items():
        assert kwargs[key] == value
    # the flags are llama-server's, so a transformers load carries none of them
    assert not set(override) & set(model_override_load_kwargs(override, is_gguf = False))
