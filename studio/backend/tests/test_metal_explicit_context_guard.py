# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A hand-set context above what unified memory holds must be refused, not launched.

The Metal branch of load_model already works out the largest context that fits, but only
Auto was moved to it: an explicit request was passed through verbatim, on the theory that
"--fit on" is a backstop. It is one, but not a trustworthy one here. llama.cpp will reduce
an explicit context (fit_params_min_ctx defaults to 4096; only "-c 0" disables it), but it
decides from ggml-metal's free-memory report, off the device's recommendedMaxWorkingSetSize,
which knows nothing of Unsloth's own resident gigabyte or two, other running apps, or the
iogpu wired limit actually being blown. When that estimate is optimistic the request stands
and the launch over-commits wired memory, which Jetsam cannot reclaim, so the machine
panics instead of the load failing. An M1 Max 32 GB hit exactly that on
Qwen3.8-27B-UD-Q4_K_XL, twice, as soon as the context was set by hand.

So the ceiling the branch computes now gates the explicit request too, and the refusal
names it. Two things it deliberately does not do: refuse against the 4096 fallback used
when KV cannot be sized (a guess, and refusing on it would block contexts that load fine
today), and refuse a manual load with a fixed layer count, which is the user taking the
memory budget over, as the other two Metal guards already treat it.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

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

if "jwt" not in sys.modules:
    try:
        import jwt  # noqa: F401
    except Exception:
        _jwt_stub = _types.ModuleType("jwt")
        _jwt_stub.decode = lambda *a, **k: {}
        _jwt_stub.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
        _jwt_stub.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
        sys.modules["jwt"] = _jwt_stub

from core.inference.llama_cpp import (  # noqa: E402
    _FIT_MIN_CTX,
    GgufLoadIntent,
    LlamaCppBackend,
)

_message = LlamaCppBackend._metal_context_overcommit_message
_ENV = LlamaCppBackend.METAL_CTX_OVERCOMMIT_ENV
_REAL_POPEN = subprocess.Popen

# What the stubbed fit reports as the largest context that fits, and the GGUF's native
# length. Anything between them is a context the user can type today and the machine
# cannot hold.
CEILING = 8192
NATIVE = 262144


@pytest.fixture(autouse = True)
def _no_opt_out(monkeypatch):
    """A real environment read, so a host that has it set must not turn every refusal
    test silently green."""
    monkeypatch.delenv(_ENV, raising = False)


def _write_gguf(path: Path) -> Path:
    """The smallest header load_model will parse."""

    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _ctx_values(cmd) -> list[str]:
    """Every context value in argv, in order. llama.cpp takes the last."""
    values = []
    for i, token in enumerate(cmd):
        if token in ("-c", "--ctx-size"):
            values.append(cmd[i + 1] if i + 1 < len(cmd) else None)
        elif token.startswith("-c=") or token.startswith("--ctx-size="):
            values.append(token.split("=", 1)[1])
    return values


def _launch(
    tmp_path,
    monkeypatch,
    *,
    n_ctx,
    metal = True,
    can_estimate_kv = True,
    gpu_memory_mode = "auto",
    gpu_layers = -1,
    extra_args = None,
    paravirtual = False,
    cache_type_kv = None,
    backend = None,
    real_fit = False,
    budget_bytes = 9 * 1024**3,
    weights_bytes = 1024,
    kv_per_token = 1024,
    native = NATIVE,
    mmproj_bytes = 0,
):
    """Drive the real load_model with no GPU enumerated (the Metal condition).

    The KV estimate is a flat 1 KiB per token and the compute buffer is zeroed, so the
    footprint check the branch runs before trusting its own ceiling passes on the tiny
    stub GGUF and the ceiling under test is the one the fit returns.

    Returns the launch capture ({"cmd": argv}, empty when nothing launched). Pass
    ``backend`` to drive a second load through the same instance, the only way to observe
    what a refusal does to state a previous load left behind.

    ``real_fit`` leaves _fit_context_to_vram unstubbed so the branch runs against the
    helper's actual return contract -- its 4096 floor, and its habit of handing the
    request straight back. ``budget_bytes`` / ``weights_bytes`` / ``kv_per_token`` /
    ``native`` then place the model against the budget, and only matter with ``real_fit``.
    """
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: budget_bytes if metal else 0),
    )
    if paravirtual:
        import core.inference.llama_cpp as _llama_cpp
        monkeypatch.setattr(_llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend = backend if backend is not None else LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: can_estimate_kv
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * kv_per_token
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    if not real_fit:
        backend._fit_context_to_vram = lambda target, *a, **k: min(int(target), CEILING)
    backend._get_gguf_size_bytes = lambda _path: weights_bytes
    backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
    backend._resolve_launch_mmproj_path = (
        (lambda **kwargs: str(_write_gguf(tmp_path / "mmproj-F16.gguf")))
        if mmproj_bytes
        else (lambda **kwargs: None)
    )
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    # This harness does not model host RAM, and None is the documented way to say so: both
    # _apu_ram_shortfall_message and _host_offload_shortfall_message treat unknown
    # available memory as "never refuse". Without it the sibling host-RAM guard fires on
    # the paravirtual path (the one placement here that reports child_has_no_gpu and so
    # gets past that guard's empty-pool early return) and prices the model against the
    # REAL machine, so the virtualised-device tests passed on a 16 GB runner and failed on
    # a 7 GB one. Host-memory dependent, not OS dependent.
    backend._available_system_memory_mib = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend._context_length = native

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                # One below pid_max: validly shaped but names no process, so the
                # lifetime registry's identity check drops it. Not inert decoration
                # -- load_model adopts whatever pid it is given and teardown signals
                # that process group, and killpg(1) is kill(-1), everything the user owns.
                "pid": 4194303,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(_write_gguf(tmp_path / "model.gguf")),
                model_identifier = "test",
                n_ctx = n_ctx,
                gpu_memory_mode = gpu_memory_mode,
                gpu_layers = gpu_layers,
                extra_args = extra_args,
                cache_type_kv = cache_type_kv,
                is_vision = bool(mmproj_bytes),
            )
        )
    captured["backend"] = backend
    return captured


class TestTheRefusalItself:
    """The message, in isolation from where it is raised."""

    def test_a_context_above_the_ceiling_is_refused(self):
        msg = _message(32768, CEILING)
        assert msg is not None
        # Both numbers, so the user can act on it without a second round trip.
        assert "32,768" in msg and "8,192" in msg

    def test_it_names_the_opt_out(self):
        assert _ENV in _message(32768, CEILING)

    def test_it_does_not_blame_system_ram(self):
        """The PC advice. There is no system RAM to spill to on unified memory, and
        saying so is what made the old warning read as survivable."""
        assert "system RAM" not in _message(32768, CEILING)

    @pytest.mark.parametrize("requested", [1, CEILING - 1, CEILING])
    def test_a_context_that_fits_is_allowed(self, requested):
        assert _message(requested, CEILING) is None

    @pytest.mark.parametrize("requested,ceiling", [(0, CEILING), (32768, 0), (-1, CEILING)])
    def test_an_unusable_pair_abstains(self, requested, ceiling):
        """No request, or no ceiling to measure against, is not a refusal."""
        assert _message(requested, ceiling) is None

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", " 1 "])
    def test_the_opt_out_abstains(self, monkeypatch, value):
        monkeypatch.setenv(_ENV, value)
        assert _message(32768, CEILING) is None

    @pytest.mark.parametrize("value", ["0", "no", "", "maybe"])
    def test_anything_else_still_refuses(self, monkeypatch, value):
        monkeypatch.setenv(_ENV, value)
        assert _message(32768, CEILING) is not None

    @pytest.mark.parametrize("cache_type", [None, "", "f16", "fp16"])
    def test_the_kv_hint_is_offered_on_an_unquantized_cache(self, cache_type):
        assert "q8_0" in _message(32768, CEILING, cache_type)

    @pytest.mark.parametrize("cache_type", ["q8_0", "q4_0"])
    def test_it_is_not_offered_once_the_cache_is_already_quantized(self, cache_type):
        """Advice the user has already taken reads as the refusal not having noticed."""
        assert "q8_0" not in _message(32768, CEILING, cache_type)


class TestWhatLoadModelDoes:
    def test_a_context_above_the_ceiling_never_reaches_llama_server(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768)

    def test_the_refusal_names_the_ceiling(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "8,192"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768)

    def test_a_context_that_fits_still_launches(self, tmp_path, monkeypatch):
        captured = _launch(tmp_path, monkeypatch, n_ctx = 4096)
        assert _ctx_values(captured["cmd"])[-1] == "4096"

    def test_the_ceiling_itself_is_allowed(self, tmp_path, monkeypatch):
        """Off-by-one on the boundary would refuse the number the message tells the
        user to type."""
        captured = _launch(tmp_path, monkeypatch, n_ctx = CEILING)
        assert _ctx_values(captured["cmd"])[-1] == str(CEILING)

    def test_auto_is_untouched(self, tmp_path, monkeypatch):
        """The path that already worked: shrink to the ceiling, never refuse."""
        captured = _launch(tmp_path, monkeypatch, n_ctx = 0)
        assert _ctx_values(captured["cmd"])[-1] == str(CEILING)

    def test_the_opt_out_loads_it_anyway(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        captured = _launch(tmp_path, monkeypatch, n_ctx = 32768)
        assert _ctx_values(captured["cmd"])[-1] == "32768"

    def test_a_fixed_manual_layer_count_is_the_callers_budget(self, tmp_path, monkeypatch):
        """Same exemption the floor and the zero-context drop already make."""
        captured = _launch(
            tmp_path, monkeypatch, n_ctx = 32768, gpu_memory_mode = "manual", gpu_layers = 20
        )
        assert _ctx_values(captured["cmd"])[-1] == "32768"

    def test_an_unsizeable_kv_cache_does_not_refuse(self, tmp_path, monkeypatch):
        """The branch falls back to a flat 4096 there. It is a guess, and refusing
        against it would block contexts that load fine today."""
        captured = _launch(tmp_path, monkeypatch, n_ctx = 32768, can_estimate_kv = False)
        assert _ctx_values(captured["cmd"])[-1] == "32768"

    def test_off_metal_nothing_is_refused(self, tmp_path, monkeypatch):
        """Linux and Windows spill to system RAM and report an error; not this guard's
        problem, and the budget reads 0 there so the branch is never entered."""
        captured = _launch(tmp_path, monkeypatch, n_ctx = 32768, metal = False)
        assert _ctx_values(captured["cmd"])[-1] == "32768"


def test_the_refusal_is_raised_outside_the_placement_handler():
    """Structural, because the failure it guards against is silent.

    The `except Exception` around GPU selection swallows any raise inside it and restores
    the original request, which is exactly the over-commit being refused. So the branch
    records the message and load_model raises it after that handler. Raising in place
    would leave every test above passing on a guard that does nothing.
    """
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    assigned = src.find("_metal_ctx_refusal = self._metal_context_overcommit_message(")
    handler = src.find("using --fit on")
    raised = src.find("raise RuntimeError(_metal_ctx_refusal)")
    assert assigned != -1 and handler != -1 and raised != -1
    assert assigned < handler < raised


class TestAVirtualisedMetalDevice:
    """A Mac VM runs GGUF entirely on CPU, so this budget is the wrong yardstick.

    The paravirtual pin rewrites every placement to manual/0 and launches behind
    --device none, because offloaded layers on a virtualised Metal device produce corrupt
    output. Nothing is allocated on the GPU, so refusing against a GPU working-set budget
    would break loads that work today on a Mac VM (and on the macOS GitHub Actions
    runners, which report exactly this device), and the message would describe hardware
    the launch never touches. Host RAM is the real limit, and
    _host_offload_shortfall_message already prices it.

    Caught by the pre-merge OS x GPU simulation, not by review: the exemption reads
    _paravirtual_cpu_forced, set from the hardware, while the neighbouring
    _caller_owns_budget is read off the REQUEST and stays False for the Auto load the
    pin rewrote.
    """

    def test_it_is_not_refused(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 32768, paravirtual = True)["cmd"]
        assert _ctx_values(cmd)[-1] == "32768"

    def test_a_physical_mac_in_the_same_shape_is_still_refused(self, tmp_path, monkeypatch):
        """Pins that the exemption is the virtualised device, not the CPU placement
        it happens to produce."""
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, paravirtual = False)

    def test_auto_is_still_capped_there(self, tmp_path, monkeypatch):
        """The exemption is from the refusal only. Auto still shrinks to the ceiling,
        which is what keeps a virtualised Mac off its native context."""
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, paravirtual = True)["cmd"]
        assert _ctx_values(cmd)[-1] == str(CEILING)


class TestTheMessageSurvivesTheRoute:
    """load_model raises; the route rewrites the text twice before the user reads it.

    The broad handler in _load_model_impl redacts native paths and then runs
    _maybe_unsupported_message over the result, exactly as for the existing APU and
    host-offload refusals. Both rewrites have to leave this message alone or the user is
    told something false about a fixable mistake.
    """

    def _message(self, tmp_path, monkeypatch) -> str:
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 32768)
        return str(excinfo.value)

    def test_it_is_not_relabelled_as_an_unsupported_model(self, tmp_path, monkeypatch):
        """_maybe_unsupported_message rewrites any error carrying one of these into "This
        model is not supported yet. Try a different model.", sending the user off to
        change models over a context they can simply lower.

        Read out of the route source rather than imported: the phrase list is the
        contract, and importing routes.inference would drag FastAPI in for four strings.
        """
        import ast
        import re

        # encoding is not optional: routes/inference.py carries non-ASCII (the DeepSeek
        # tool-call tokens), and read_text() defaults to cp1252 on Windows.
        route_src = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
            encoding = "utf-8"
        )
        hints = ast.literal_eval(
            re.search(r"_NOT_SUPPORTED_HINTS = (\(.*?\))", route_src, re.S).group(1)
        )
        # The list is only a contract if it is the real one.
        assert "is not supported" in hints
        message = self._message(tmp_path, monkeypatch).lower()
        assert [h for h in hints if h.lower() in message] == []

    def test_it_carries_nothing_for_the_path_redactor_to_eat(self, tmp_path, monkeypatch):
        """redact_native_paths replaces any leased path with <native_path>. A message
        with no path in it cannot be cut in half by that."""
        message = self._message(tmp_path, monkeypatch)
        assert "/" not in message.replace("q8_0", "")

    def test_it_is_a_single_line_of_plain_text(self, tmp_path, monkeypatch):
        """The route prefixes it ("Failed to load model: ...") and the UI renders the
        detail as one string."""
        message = self._message(tmp_path, monkeypatch)
        assert "\n" not in message


class TestWhatARefusedReloadCosts:
    """A refused reload ends with no model loaded, and that is the existing contract.

    load_model kills the resident server in Phase 1, long before the placement block that
    computes the ceiling, so every refusal raised from that block already behaves this way
    (the APU RAM shortfall, the unpinnable Vulkan ordinal). Refusing earlier would mean
    re-deriving the fit outside the one place that owns it, the drift
    _apu_ram_shortfall_message explicitly avoids.

    So this is pinned rather than fixed, and still the better end state: before this guard
    the same click took the whole machine down. The recovery path is what has to work, and
    the next test covers it.
    """

    def test_the_refused_reload_leaves_nothing_running(self, tmp_path, monkeypatch):
        # is_active, not is_loaded: this asks whether a child process exists, and
        # health is a separate signal the stubbed launch does not model.
        backend = _launch(tmp_path, monkeypatch, n_ctx = 4096)["backend"]
        assert backend.is_active
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, backend = backend)
        assert not backend.is_active

    def test_a_smaller_retry_after_a_refusal_succeeds(self, tmp_path, monkeypatch):
        """Nothing about the refusal is sticky: no half-written request state, and no
        dedupe that would read the retry as already loaded."""
        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, backend = backend)
        assert not backend.is_active
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 4096, backend = backend)["cmd"]
        assert _ctx_values(cmd)[-1] == "4096"


class TestTheContextCanArriveByAnotherDoor:
    """requested_ctx folds in a -c from extra args, so every spelling is covered.

    Worth pinning: reading intent.n_ctx directly would leave the guard one text box away
    from being bypassed, and the pass-through spelling is the one a user reaches for
    after being refused.
    """

    @pytest.mark.parametrize(
        "extra",
        [
            ("-c", "32768"),
            ("--ctx-size", "32768"),
            ("--ctx-size=32768",),
        ],
    )
    def test_a_pass_through_context_is_refused_too(self, tmp_path, monkeypatch, extra):
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 0, extra_args = list(extra))

    def test_a_pass_through_context_under_the_ceiling_still_launches(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, extra_args = ["-c", "4096"])["cmd"]
        assert _ctx_values(cmd)[-1] == "4096"

    def test_a_zero_pass_through_is_floored_not_refused(self, tmp_path, monkeypatch):
        """ "-c 0" is read as non-explicit and handled by the existing floor (#5118),
        so it must not turn into a refusal."""
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, extra_args = ["-c", "0"])["cmd"]
        assert _ctx_values(cmd) and _ctx_values(cmd)[-1] != "0"


# 1 MiB of KV per token, so a handful of thousand tokens is worth gigabytes and the
# fit's own 4096 floor can be pushed past the budget on a stub model.
_FAT_KV = 1024 * 1024
_BUDGET = 9 * 1024**3
# load_model folds a flat compute-buffer reserve into the weights before the fit sees
# them, so a 9 GiB budget leaves well under 9 GiB for weights + KV. Sized so the weights
# fit with room for a few hundred tokens and nothing like 4096.
_TIGHT_WEIGHTS = 3300 * 1024**2
_TIGHT_CEILING = 768


def _named_ceiling(message: str) -> int:
    """The ceiling the refusal quotes back, so a test can assert about it directly."""
    return int(message.split("The largest that fits is ")[1].split(" ")[0].replace(",", ""))


class TestWhenEvenTheFitsOwnMinimumDoesNotFit:
    """The fit floors at ``min_ctx`` (4096), so a 4096 coming back means either "4096
    fits" or "nothing fits, here is the floor". Reading the second as "the weights alone
    are over budget" skipped the refusal on exactly the machine that needs it: llama.cpp
    will not reduce below 4096 either, so "--fit on" has nothing left to give and the
    launch over-commits wired memory.
    """

    def test_the_premise_the_fit_hands_back_its_own_floor(self):
        """Not a behaviour assertion -- a guard on the return contract the branch reads.
        998 MiB of weights against a 1000 MiB budget leaves room for 2048 tokens at 1 KiB
        each, yet asking with the default floor still answers 4096."""
        backend = LlamaCppBackend()
        backend._can_estimate_kv = lambda: True
        backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * 1024

        def fit(min_ctx):
            return backend._fit_context_to_vram(
                NATIVE,
                1000,
                998 * 1024**2,
                None,
                min_ctx = min_ctx,
                budget_frac = 1.0,
                pooled = True,
                total_mib = None,
                compute_ctx_bytes_fn = lambda _ctx: 0,
            )

        assert fit(4096) == 4096  # the floor, not a measurement
        assert fit(256) == 2048  # what actually fits

    def _tight(self, tmp_path, monkeypatch, **kw):
        return _launch(
            tmp_path,
            monkeypatch,
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = _TIGHT_WEIGHTS,
            kv_per_token = _FAT_KV,
            **kw,
        )

    def test_an_explicit_context_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "unified"):
            self._tight(tmp_path, monkeypatch, n_ctx = 4096)

    def test_the_refusal_names_what_actually_fits(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = f"{_TIGHT_CEILING:,}"):
            self._tight(tmp_path, monkeypatch, n_ctx = 8192)

    def test_auto_starts_at_what_fits_not_at_the_floor(self, tmp_path, monkeypatch):
        """The same number the refusal names, or the UI advertises as its maximum a
        context that is itself the over-commit."""
        cmd = self._tight(tmp_path, monkeypatch, n_ctx = 0)["cmd"]
        assert _ctx_values(cmd)[-1] == str(_TIGHT_CEILING)

    def test_a_context_that_does_fit_still_launches(self, tmp_path, monkeypatch):
        cmd = self._tight(tmp_path, monkeypatch, n_ctx = 512)["cmd"]
        assert _ctx_values(cmd)[-1] == "512"

    def test_weights_over_budget_is_still_never_refused(self, tmp_path, monkeypatch):
        """The exemption the guard shipped with: nothing was measured there, so refusing
        would block loads that work today."""
        cmd = _launch(
            tmp_path,
            monkeypatch,
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = 10 * 1024**3,
            kv_per_token = _FAT_KV,
            n_ctx = 32768,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "32768"


class TestAContextAboveTheModelsNativeLength:
    """The fit is sized through the native length, so its ceiling can never exceed it and
    every request past it read as an over-commit whatever the machine had spare. Nothing
    clamps a request to native on the way in (the Extra Arguments box takes a raw
    --ctx-size and its placeholder suggests --rope-scaling yarn), and llama.cpp builds the
    context at the full -c, capping only the per-slot value afterwards, so the request is
    what actually gets allocated.
    """

    _NATIVE = 32768
    _ASKED = 131072

    def _above(self, tmp_path, monkeypatch, **kw):
        return _launch(
            tmp_path,
            monkeypatch,
            real_fit = True,
            budget_bytes = _BUDGET,
            native = self._NATIVE,
            **kw,
        )

    def test_it_launches_when_unified_memory_holds_it(self, tmp_path, monkeypatch):
        # 1 KiB per token: 131,072 tokens is 128 MiB against a 9 GiB budget.
        cmd = self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = 1024)["cmd"]
        assert _ctx_values(cmd)[-1] == str(self._ASKED)

    def test_the_load_does_not_arrive_carrying_a_warning_against_itself(
        self, tmp_path, monkeypatch
    ):
        """max_available_ctx is published as max_context_length, and both amber warnings
        fire when the loaded context exceeds it. Left at native, a load this branch
        measured and allowed reaches the user as "context length exceeds what fits in
        unified memory", naming a number smaller than the one running.
        """
        out = self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = 1024)
        loaded = int(_ctx_values(out["cmd"])[-1])
        published = out["backend"].max_context_length
        assert published == loaded

    def test_the_published_bound_never_runs_ahead_of_the_request(self, tmp_path, monkeypatch):
        """The fit is bounded by the request, so the bound may rise to the context that
        loaded and no further. A bound past it would invite a context nothing priced."""
        out = self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = 1024)
        assert out["backend"].max_context_length <= self._ASKED

    def test_a_refused_request_does_not_raise_the_published_bound(self, tmp_path, monkeypatch):
        """Only an accepted ceiling is published. A refusal measured nothing it can
        stand behind at the request, so the bound stays where the cap left it."""
        with pytest.raises(RuntimeError, match = "unified"):
            self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = _FAT_KV)

    def test_the_pass_through_spelling_launches_too(self, tmp_path, monkeypatch):
        """The spelling a RoPE-scaled request actually arrives in."""
        cmd = self._above(
            tmp_path,
            monkeypatch,
            n_ctx = 0,
            kv_per_token = 1024,
            extra_args = ["--rope-scaling", "yarn", "--ctx-size", str(self._ASKED)],
        )["cmd"]
        assert _ctx_values(cmd)[-1] == str(self._ASKED)

    def test_it_is_still_refused_when_the_memory_is_not_there(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "unified"):
            self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = _FAT_KV)

    def test_the_refusal_names_the_measured_ceiling_not_the_native_length(
        self, tmp_path, monkeypatch
    ):
        """A refusal that names the native length reports the wrong limit: memory holds
        sixteen times it here, so "lower the context to 4,096" throws away a context that
        would have loaded."""
        # 64 KiB per token against ~4 GiB of headroom: tens of thousands of tokens fit,
        # far past the 4096 this GGUF was trained at.
        with pytest.raises(RuntimeError) as excinfo:
            _launch(
                tmp_path,
                monkeypatch,
                real_fit = True,
                budget_bytes = _BUDGET,
                native = 4096,
                kv_per_token = 64 * 1024,
                n_ctx = self._ASKED,
            )
        message = str(excinfo.value)
        assert "4,096" not in message
        assert _named_ceiling(message) > 4096


class TestAnAboveNativeRequestOnAShortNativeModel:
    """Native below 4096, so the extension probe's own floor is above what fits.

    The probe re-prices the request through the fit to find a ceiling the native-sized cap
    could never reach. Its floor is 4096, a floor and not a measurement, and on a model
    trained at 2048 an above-native request can have room for something between the two.
    The floored result does not fit, the footprint check discards it, and the refusal
    falls back to naming the native-sized cap -- on a machine that launches the
    intermediate context when asked for it directly.

    Budget 9216 MiB against 5916 MiB of weights leaves 3300 MiB, so at 1 MiB per token
    the real ceiling is 3072 and 4096 misses by ~800 MiB.
    """

    _NATIVE = 2048
    _FITS = 3072
    _ASKED = 8192

    def _short(self, tmp_path, monkeypatch, **kw):
        return _launch(
            tmp_path,
            monkeypatch,
            real_fit = True,
            budget_bytes = 9216 * 1024**2,
            weights_bytes = 796 * 1024**2,
            kv_per_token = _FAT_KV,
            native = self._NATIVE,
            **kw,
        )

    def test_the_intermediate_context_launches(self, tmp_path, monkeypatch):
        """The other half of the contradiction, and what makes the number in the
        refusal checkable: this same load is one the guard already allows."""
        cmd = self._short(tmp_path, monkeypatch, n_ctx = self._FITS)["cmd"]
        assert _ctx_values(cmd)[-1] == str(self._FITS)

    def test_the_refusal_names_it_rather_than_the_native_length(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError) as excinfo:
            self._short(tmp_path, monkeypatch, n_ctx = self._ASKED)
        message = str(excinfo.value)
        assert _named_ceiling(message) == self._FITS
        # Naming 2,048 here sends the user to less than the machine holds.
        assert f"{self._NATIVE:,}" not in message

    def test_the_re_probe_only_ever_raises_the_ceiling(self, tmp_path, monkeypatch):
        """It runs on every above-native request, including ones where the floored
        probe already fits, so it must not talk a working ceiling back down."""
        with pytest.raises(RuntimeError) as excinfo:
            _launch(
                tmp_path,
                monkeypatch,
                real_fit = True,
                budget_bytes = _BUDGET,
                weights_bytes = _TIGHT_WEIGHTS,
                kv_per_token = _FAT_KV,
                native = 262144,
                n_ctx = 1048576,
            )
        assert _named_ceiling(str(excinfo.value)) >= _TIGHT_CEILING


class TestWhenNothingFitsAtAll:
    """Weights fit, and even the smallest context the search prices does not.

    The narrowest of the three states the over-budget arm has to tell apart, and the one
    with no number to lower to. It is a measurement, not an absence of one: the fit
    shrank, which is what says the weights themselves fit, and then the floor it shrank to
    did not fit either. Leaving it unmeasured let every explicit context through on a host
    where all of them over-commit, the crash this guard exists to stop.

    Told apart from weights-alone-over-budget by whether the re-priced answer is smaller:
    that arm returns the request untouched for any min_ctx, so it cannot shrink.
    """

    # Weights heavy enough that the budget cannot afford 256 tokens on top of them at
    # 1 MiB each, but light enough that the fit can shrink at all, the signal that
    # separates this state from weights-alone-over-budget. Measured window for this
    # harness: ~3850 to ~4050 MiB (3300 leaves room for 768 tokens, 4100 tips over).
    NOTHING_FITS = dict(
        real_fit = True,
        budget_bytes = _BUDGET,
        weights_bytes = 3950 * 1024**2,
        kv_per_token = _FAT_KV,
    )

    def test_an_explicit_context_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "No context fits"):
            _launch(tmp_path, monkeypatch, n_ctx = 8192, **self.NOTHING_FITS)

    def test_even_a_tiny_explicit_context_is_refused(self, tmp_path, monkeypatch):
        """There is no floor to fall back to: 512 over-commits the same as 32768."""
        with pytest.raises(RuntimeError, match = "No context fits"):
            _launch(tmp_path, monkeypatch, n_ctx = 512, **self.NOTHING_FITS)

    def test_the_refusal_names_no_ceiling(self, tmp_path, monkeypatch):
        """Naming one would be inventing a number the fit never vouched for, and the
        user would lower to it and hit the same wall."""
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 8192, **self.NOTHING_FITS)
        message = str(excinfo.value)
        assert "The largest that fits" not in message
        assert "smaller or more quantized GGUF" in message

    def test_it_still_names_the_opt_out(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = _ENV):
            _launch(tmp_path, monkeypatch, n_ctx = 8192, **self.NOTHING_FITS)

    def test_the_opt_out_loads_it_anyway(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 8192, **self.NOTHING_FITS)["cmd"]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_auto_is_untouched(self, tmp_path, monkeypatch):
        """Auto launches at this arm's floor on this host, and the guard still does not
        move it.

        That floor was a hardcoded 4096 and is now _FIT_MIN_CTX, which is the larger
        claim this docstring used to decline to make -- made deliberately elsewhere, so
        that Metal stops publishing half the context a discrete GPU does for the same
        model. What this test owns is unchanged: the explicit-context guard leaves Auto
        alone. Spelled against the constant so the next floor move does not land here.
        """
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, **self.NOTHING_FITS)["cmd"]
        assert _ctx_values(cmd)[-1] == str(_FIT_MIN_CTX)

    def test_a_fixed_manual_layer_count_is_still_exempt(self, tmp_path, monkeypatch):
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 8192,
            gpu_memory_mode = "manual",
            gpu_layers = 20,
            **self.NOTHING_FITS,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_a_virtualised_device_is_still_exempt(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 8192, paravirtual = True, **self.NOTHING_FITS)[
            "cmd"
        ]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_weights_over_budget_is_still_not_refused(self, tmp_path, monkeypatch):
        """The neighbouring state, and the discriminator between them. Here the fit
        cannot shrink, so nothing was measured and the host-RAM guard owns the failure."""
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 8192,
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = 4100 * 1024**2,
            kv_per_token = _FAT_KV,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_a_host_with_room_for_a_small_context_names_it(self, tmp_path, monkeypatch):
        """The third state, so all three arms are pinned against the real helper: the
        floor re-price finds something, and that something is what gets named."""
        with pytest.raises(RuntimeError, match = "The largest that fits"):
            _launch(
                tmp_path,
                monkeypatch,
                n_ctx = 32768,
                real_fit = True,
                budget_bytes = _BUDGET,
                weights_bytes = _TIGHT_WEIGHTS,
                kv_per_token = _FAT_KV,
            )


class TestAModelWhoseNativeLengthIsAtTheFloor:
    """The weights-only state has to be read off the budget, not off two fits agreeing.

    Both probes are bounded by the same target, so on a model whose native length is at or
    under the search's 256 alignment step they return the same number for a reason
    unrelated to the weights. Inferring "the fit priced nothing" from that agreement left
    both verdicts unset and let every explicit context through on a host where none of
    them fit. Reachable at native == 256 exactly, and whenever the GGUF carries no context
    length so the request itself becomes the target.
    """

    TIGHT = dict(
        real_fit = True,
        budget_bytes = _BUDGET,
        weights_bytes = 3950 * 1024**2,
        kv_per_token = _FAT_KV,
    )

    @pytest.mark.parametrize("native", [128, 256, 512, 4096])
    def test_an_explicit_context_is_refused_at_every_native_length(
        self, tmp_path, monkeypatch, native
    ):
        with pytest.raises(RuntimeError, match = "unified memory"):
            _launch(tmp_path, monkeypatch, n_ctx = 8192, native = native, **self.TIGHT)

    def test_weights_over_budget_is_still_not_refused_at_the_floor(self, tmp_path, monkeypatch):
        """The state the old comparison was trying to detect still has to pass through,
        and now it is detected by asking the budget rather than by the two fits tying."""
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 8192,
            native = 256,
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = 12 * 1024**3,
            kv_per_token = _FAT_KV,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_a_roomy_host_still_launches_at_a_tiny_native_length(self, tmp_path, monkeypatch):
        """Nothing about a small native length should refuse on its own."""
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 256,
            native = 256,
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = 100 * 1024**2,
            kv_per_token = _FAT_KV,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "256"


class TestACpuPinnedProjectorOnUnifiedMemory:
    """--no-mmproj-offload moves the projector off a discrete card. On unified memory
    there is nowhere to move it to: "host RAM" and "VRAM" are one pool, so its bytes
    still sit in the budget this guard measures.

    Dropping them overstates the context that fits and walks straight past the refusal
    into an OOM, which is the one outcome the guard exists to prevent. The APU shortfall
    guard already weighs a pinned projector for exactly this reason.

    Sized so the projector alone decides it: budget 8192 MiB against 1024 of weights and
    ~5120 of fixed overhead, with KV at 32 KiB per token. At 32768 the KV is 1024 MiB, so
    without the projector 7168 fits and with its 1536 the footprint is 8704 and does not.
    A KV rate any smaller and the pin is lost in the slack, which is how the first two
    versions of this test passed against the bug.
    """

    _COMMON = dict(real_fit = True, weights_bytes = 1024**3, kv_per_token = 32 * 1024)

    def test_the_pinned_projector_still_counts_against_the_budget(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(
                tmp_path,
                monkeypatch,
                n_ctx = 32768,
                budget_bytes = 8 * 1024**3,
                mmproj_bytes = int(1.5 * 1024**3),
                extra_args = ["--no-mmproj-offload"],
                **self._COMMON,
            )

    def test_the_same_load_without_the_projector_is_allowed(self, tmp_path, monkeypatch):
        """The control, and the whole point: 32768 fits on this machine once the
        projector is not in the pool, so the refusal above is about those bytes and not
        about a budget too small for anything."""
        captured = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 32768,
            budget_bytes = 8 * 1024**3,
            **self._COMMON,
        )
        assert _ctx_values(captured["cmd"])[-1] == "32768"

    def test_the_pinned_projector_is_charged_once_and_not_twice(self, tmp_path, monkeypatch):
        """The other side of the same coin. The shared-pool charge now lives in the
        common fit total, so an Apple-specific one on top of it prices the encoder
        twice and refuses loads that do fit.

        Sized so only the second charge decides it: 1024 of weights, ~5120 of fixed
        overhead and 1280 of KV at 40960 tokens leave 768 MiB of the 8192 budget, and
        a 512 MiB projector fits in that once but not twice.
        """
        captured = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 40960,
            budget_bytes = 8 * 1024**3,
            mmproj_bytes = 512 * 1024**2,
            extra_args = ["--no-mmproj-offload"],
            **self._COMMON,
        )
        assert _ctx_values(captured["cmd"])[-1] == "40960"
