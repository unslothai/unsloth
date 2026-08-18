# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A hand-set context above what unified memory holds must be refused, not launched.

The Metal branch of load_model already works out the largest context that fits, but
only Auto was ever moved to it: an explicit request was passed through verbatim, on the
theory that "--fit on" is a backstop. It is one, but not a trustworthy one here.
llama.cpp will reduce an explicit context (fit_params_min_ctx defaults to 4096; only
"-c 0" disables the reduction), but it decides from ggml-metal's free-memory report,
which comes off the device's recommendedMaxWorkingSetSize and knows nothing about
Studio's own resident gigabyte or two, other running apps, or the iogpu wired limit
that is the figure actually being blown. When that estimate is optimistic the request
stands and the launch over-commits wired memory, which Jetsam cannot reclaim, so the
machine panics instead of the load failing. An M1 Max 32 GB hit exactly that on
Qwen3.8-27B-UD-Q4_K_XL, twice, as soon as the context was set by hand.

So the ceiling the branch computes now gates the explicit request too, and the refusal
names it. Two things it deliberately does not do: refuse against the 4096 fallback the
branch uses when KV cannot be sized (a guess, not a measurement, and refusing on it
would block contexts that load fine today), and refuse a manual load with a fixed layer
count, which is the user taking the memory budget over, as the other two Metal guards
already treat it.
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

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_message = LlamaCppBackend._metal_context_overcommit_message
_ENV = LlamaCppBackend.METAL_CTX_OVERCOMMIT_ENV
_REAL_POPEN = subprocess.Popen

# What the stubbed fit reports as the largest context that fits, and the native length
# of the GGUF under test. Anything between them is a context the user can type into the
# box today and the machine cannot hold.
CEILING = 8192
NATIVE = 262144


@pytest.fixture(autouse = True)
def _no_opt_out(monkeypatch):
    """The opt-out is a real environment read, so a host that has it set must not
    silently turn every refusal test green."""
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
):
    """Drive the real load_model with no GPU enumerated (the Metal condition).

    The KV estimate is a flat 1 KiB per token and the compute buffer is zeroed, so the
    footprint check the branch runs before trusting its own ceiling passes on the tiny
    stub GGUF and the ceiling under test is the one the fit returns.

    Returns the launch capture ({"cmd": argv}, empty when nothing launched). Pass
    ``backend`` to drive a second load through the same instance, which is the only way
    to observe what a refusal does to state a previous load left behind.
    """
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: 9 * 1024**3 if metal else 0),
    )
    if paravirtual:
        import core.inference.llama_cpp as _llama_cpp
        monkeypatch.setattr(_llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend = backend if backend is not None else LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: can_estimate_kv
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * 1024
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._fit_context_to_vram = lambda native, *a, **k: min(int(native), CEILING)
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend._context_length = NATIVE

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
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

    The `except Exception` around GPU selection turns any raise inside it into
    "GPU selection failed", then restores the original request, which is exactly the
    over-commit being refused. So the branch records the message and load_model raises
    it after that handler. Raising in place would leave every test above passing on a
    guard that does nothing.
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
    --device none, because offloaded layers on a virtualised Metal device produce
    corrupt output. Nothing is allocated on the GPU, so refusing against a GPU
    working-set budget would break loads that work today on a Mac VM (and on the
    macOS GitHub Actions runners, which report exactly this device), and the message
    would describe hardware the launch never touches. Host RAM is the real limit
    there, and _host_offload_shortfall_message already prices that.

    Caught by the pre-merge OS x GPU simulation, not by review: the exemption reads
    _paravirtual_cpu_forced, which is set from the hardware, while the neighbouring
    _caller_owns_budget is read off the REQUEST and so stays False for the Auto load
    the pin rewrote.
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

    It is caught by the broad handler in _load_model_impl, which redacts native paths
    and then runs _maybe_unsupported_message over the result, exactly as the existing
    APU and host-offload refusals are. Both rewrites have to leave this message alone
    or the user is told something false about a fixable mistake.
    """

    def _message(self, tmp_path, monkeypatch) -> str:
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 32768)
        return str(excinfo.value)

    def test_it_is_not_relabelled_as_an_unsupported_model(self, tmp_path, monkeypatch):
        """_maybe_unsupported_message rewrites any error carrying one of these into
        "This model is not supported yet. Try a different model.", which would send
        the user off to change models over a context they can simply lower.

        Read out of the route source rather than imported: the phrase list is the
        contract, and importing routes.inference would drag FastAPI into a test that
        only needs four strings.
        """
        import ast
        import re

        route_src = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text()
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

    load_model kills the resident server in its Phase 1, long before the placement
    block that computes the ceiling. Every refusal raised from that block behaves this
    way already: the APU RAM shortfall, the unpinnable Vulkan ordinal. Refusing earlier
    would mean re-deriving the fit outside the one place that owns it, which is the
    drift _apu_ram_shortfall_message explicitly avoids.

    So this is pinned rather than fixed, and it is still the better end state: before
    this guard the same click took the whole machine down. The recovery path is what
    has to work, and the next test covers it.
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

    Worth pinning: if the guard read intent.n_ctx directly it would sit one text box
    away from being bypassed, and the pass-through spelling is the one a user reaches
    for after being refused.
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
