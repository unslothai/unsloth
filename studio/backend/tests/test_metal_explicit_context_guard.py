# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit context above the Metal ceiling is refused, except a manual layer count or unsized KV"""

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

CEILING = 8192
NATIVE = 262144


@pytest.fixture(autouse = True)
def _no_opt_out(monkeypatch):
    """Unset for real: a host with it set would turn every refusal test silently green."""
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
    wired_bytes = 0,
    ckpt_bytes = 0,
    refuse_partial_gpu_fit = False,
    launches = None,
    wired_exhausted = False,
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
    ``wired_bytes`` is the GPU wired headroom, 0 when unreadable; ``wired_exhausted`` reads
    it as known and used up. ``ckpt_bytes`` prices each
    SWA context checkpoint and advertises the flag, so ``--ctx-checkpoints`` is charged.
    """
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: budget_bytes if metal else 0),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_wired_ceiling_bytes",
        staticmethod(lambda: wired_bytes if metal else 0),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_wired_headroom_bytes",
        staticmethod(lambda: (0 if wired_exhausted else wired_bytes or None) if metal else None),
    )
    if paravirtual:
        import core.inference.llama_cpp as _llama_cpp
        monkeypatch.setattr(_llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend = backend if backend is not None else LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: can_estimate_kv
    backend._estimate_kv_cache_bytes = (
        lambda ctx, *a, ctx_checkpoints = 0, **k: int(ctx) * kv_per_token
        + ctx_checkpoints * ckpt_bytes
    )
    if ckpt_bytes:
        _probe = LlamaCppBackend.probe_server_capabilities.__func__
        monkeypatch.setattr(
            LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(
                lambda cls, binary = None: {
                    **_probe(cls, binary),
                    "ctx_checkpoints_flag": "--ctx-checkpoints",
                }
            ),
        )
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
    # None means host RAM unknown, so host-RAM guards never refuse in this harness.
    backend._available_system_memory_mib = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend._context_length = native

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        if launches is not None:
            launches.append(list(cmd))
        return type(
            "Process",
            (),
            {
                # Unused pid: teardown signals the group, and killpg(1) would hit everything.
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
                refuse_partial_gpu_fit = refuse_partial_gpu_fit,
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
        """Off-by-one at the boundary would refuse the number the message tells users to type."""
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
        """Falls back to a flat 4096 guess; refusing against it would block loads that work."""
        captured = _launch(tmp_path, monkeypatch, n_ctx = 32768, can_estimate_kv = False)
        assert _ctx_values(captured["cmd"])[-1] == "32768"

    def test_off_metal_nothing_is_refused(self, tmp_path, monkeypatch):
        """Linux and Windows spill to system RAM and report an error; not this guard's
        problem, and the budget reads 0 there so the branch is never entered."""
        captured = _launch(tmp_path, monkeypatch, n_ctx = 32768, metal = False)
        assert _ctx_values(captured["cmd"])[-1] == "32768"


def test_the_refusal_is_raised_outside_the_placement_handler():
    """Raise the refusal outside the broad placement except, or that handler restores the request."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    assigned = src.find("_metal_ctx_refusal = self._metal_context_overcommit_message(")
    handler = src.find("using --fit on")
    raised = src.find("raise RuntimeError(_metal_ctx_refusal)")
    assert assigned != -1 and handler != -1 and raised != -1
    assert assigned < handler < raised


class TestAVirtualisedMetalDevice:
    """A paravirtual Metal VM runs on CPU, so a GPU working-set budget is the wrong yardstick."""

    def test_it_is_not_refused(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 32768, paravirtual = True)["cmd"]
        assert _ctx_values(cmd)[-1] == "32768"

    def test_a_physical_mac_in_the_same_shape_is_still_refused(self, tmp_path, monkeypatch):
        """Pins the exemption to the virtualised device, not the CPU placement it produces."""
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, paravirtual = False)

    def test_auto_is_still_capped_there(self, tmp_path, monkeypatch):
        """Only the refusal is exempt; Auto still shrinks to the ceiling, not native context."""
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, paravirtual = True)["cmd"]
        assert _ctx_values(cmd)[-1] == str(CEILING)


class TestTheMessageSurvivesTheRoute:
    """Both route rewrites must leave this message alone, or the user is told something false."""

    def _message(self, tmp_path, monkeypatch) -> str:
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 32768)
        return str(excinfo.value)

    def test_it_is_not_relabelled_as_an_unsupported_model(self, tmp_path, monkeypatch):
        """Context errors must not be relabelled unsupported; the user can lower the context."""
        import ast
        import re

        # encoding required: the file has non-ASCII and Windows defaults to cp1252.
        route_src = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
            encoding = "utf-8"
        )
        hints = ast.literal_eval(
            re.search(r"_NOT_SUPPORTED_HINTS = (\(.*?\))", route_src, re.S).group(1)
        )
        assert "is not supported" in hints
        message = self._message(tmp_path, monkeypatch).lower()
        assert [h for h in hints if h.lower() in message] == []

    def test_it_carries_nothing_for_the_path_redactor_to_eat(self, tmp_path, monkeypatch):
        """redact_native_paths replaces leased paths, so the message must contain no path."""
        message = self._message(tmp_path, monkeypatch)
        assert "/" not in message.replace("q8_0", "")

    def test_it_is_a_single_line_of_plain_text(self, tmp_path, monkeypatch):
        """The route prefixes it ("Failed to load model: ...") and the UI renders the
        detail as one string."""
        message = self._message(tmp_path, monkeypatch)
        assert "\n" not in message


class TestWhatARefusedReloadCosts:
    """A refused reload leaves nothing running; Phase 1 kills the resident server before the ceiling."""

    def test_the_refused_reload_leaves_nothing_running(self, tmp_path, monkeypatch):
        backend = _launch(tmp_path, monkeypatch, n_ctx = 4096)["backend"]
        assert backend.is_active
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, backend = backend)
        assert not backend.is_active

    def test_a_smaller_retry_after_a_refusal_succeeds(self, tmp_path, monkeypatch):
        """A refusal leaves no half-written state, and no dedupe marks the retry as loaded."""
        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, backend = backend)
        assert not backend.is_active
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 4096, backend = backend)["cmd"]
        assert _ctx_values(cmd)[-1] == "4096"


class TestTheContextCanArriveByAnotherDoor:
    """The guard must read requested_ctx, not intent.n_ctx, or a -c in extra args bypasses it."""

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
        """The -c 0 pass-through is read as non-explicit and floored, not refused (#5118)."""
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, extra_args = ["-c", "0"])["cmd"]
        assert _ctx_values(cmd) and _ctx_values(cmd)[-1] != "0"


_FAT_KV = 1024 * 1024
_BUDGET = 9 * 1024**3
# load_model folds a compute reserve into weights, so less than 9 GiB remains for weights+KV.
_TIGHT_WEIGHTS = 3300 * 1024**2
_TIGHT_CEILING = 768


def _named_ceiling(message: str) -> int:
    """The ceiling the refusal quotes back, so a test can assert about it directly."""
    return int(message.split("The largest that fits is ")[1].split(" ")[0].replace(",", ""))


class TestWhenEvenTheFitsOwnMinimumDoesNotFit:
    """The fit floors at min_ctx, so a 4096 answer means it fits or nothing does; pinned here."""

    def test_the_premise_the_fit_hands_back_its_own_floor(self):
        """A guard on the return contract: 998 MiB of weights in a 1000 MiB budget leaves room for
        2048 tokens at 1 KiB each, yet the default floor still answers 4096."""
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
        assert fit(256) == 2048

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
        """Auto starts at what fits, the same number the refusal names, not an over-commit."""
        cmd = self._tight(tmp_path, monkeypatch, n_ctx = 0)["cmd"]
        assert _ctx_values(cmd)[-1] == str(_TIGHT_CEILING)

    def test_a_context_that_does_fit_still_launches(self, tmp_path, monkeypatch):
        cmd = self._tight(tmp_path, monkeypatch, n_ctx = 512)["cmd"]
        assert _ctx_values(cmd)[-1] == "512"

    def test_weights_over_budget_is_still_never_refused(self, tmp_path, monkeypatch):
        """The old exemption had no measurement, so refusing there would block loads that work."""
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
    """The fit caps at native length, so an above-native request is priced at what llama.cpp allocates."""

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
        cmd = self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = 1024)["cmd"]
        assert _ctx_values(cmd)[-1] == str(self._ASKED)

    def test_the_load_does_not_arrive_carrying_a_warning_against_itself(
        self, tmp_path, monkeypatch
    ):
        """The published max_context_length must equal the loaded one, or a warning fires."""
        out = self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = 1024)
        loaded = int(_ctx_values(out["cmd"])[-1])
        published = out["backend"].max_context_length
        assert published == loaded

    def test_the_published_bound_never_runs_ahead_of_the_request(self, tmp_path, monkeypatch):
        """The fit is bounded by the request, so the bound may rise to the context that
        loaded and no further. A bound past it would invite a context nothing priced."""
        out = self._above(tmp_path, monkeypatch, n_ctx = self._ASKED, kv_per_token = 1024)
        assert out["backend"].max_context_length <= self._ASKED

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
    """On a model trained at 2048, the probe's 4096 floor overstates what fits; the true ceiling is 3072."""

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
        """This same load is allowed, which makes the number in the refusal checkable."""
        cmd = self._short(tmp_path, monkeypatch, n_ctx = self._FITS)["cmd"]
        assert _ctx_values(cmd)[-1] == str(self._FITS)

    def test_the_refusal_names_it_rather_than_the_native_length(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError) as excinfo:
            self._short(tmp_path, monkeypatch, n_ctx = self._ASKED)
        message = str(excinfo.value)
        assert _named_ceiling(message) == self._FITS
        assert f"{self._NATIVE:,}" not in message

    def test_the_re_probe_only_ever_raises_the_ceiling(self, tmp_path, monkeypatch):
        """It runs on every above-native request, so it must never lower a working ceiling."""
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
    """Weights fit but no context does: the fit shrank, but its floor misses, so this state is measured."""

    # Measured window ~3850-4050 MiB: fit can shrink but not afford 256 tokens.
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
        """Naming one invents a number the fit never vouched for; users hit the same wall."""
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
        """Auto is left alone by the guard; the expected floor is _FIT_MIN_CTX, not a literal 4096."""
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
    """Read the weights-only state off the budget; two agreeing fits can coincide for unrelated reasons."""

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
    """Unified memory has no host RAM to move a pinned projector to, so its bytes still count."""

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
        """The control: 32768 fits once the projector is out of the pool, so the refusal is about
        those bytes, not a budget too small for anything."""
        captured = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 32768,
            budget_bytes = 8 * 1024**3,
            **self._COMMON,
        )
        assert _ctx_values(captured["cmd"])[-1] == "32768"

    def test_the_pinned_projector_is_charged_once_and_not_twice(self, tmp_path, monkeypatch):
        """A pinned projector is charged once; a second charge would refuse valid loads."""
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


_WIRED = 16 * 1024**3


class TestTheGpuWiredLimitDecidesTheRefusal:
    """#9942: a 38 GiB load refused against ~32 GB free ran fine with the opt-out."""

    _ROOMY = dict(real_fit = True, budget_bytes = _BUDGET, weights_bytes = 1024**3, kv_per_token = _FAT_KV)

    def test_a_context_over_free_memory_but_under_the_limit_loads(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 8192, wired_bytes = _WIRED, **self._ROOMY)["cmd"]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_it_loads_with_a_warning(self, tmp_path, monkeypatch):
        backend = _launch(tmp_path, monkeypatch, n_ctx = 8192, wired_bytes = _WIRED, **self._ROOMY)[
            "backend"
        ]
        assert "8,192" in backend.last_load_warning
        assert "loading anyway" in backend.last_load_warning

    def test_the_warning_rounds_need_up_and_free_down(self):
        message = LlamaCppBackend._metal_context_pressure_message(8192, 9 * 1024 + 1, 9 * 1024 - 1)
        assert "about 10 GB" in message and "the 8 GB" in message

    def test_a_context_under_free_memory_carries_no_warning(self, tmp_path, monkeypatch):
        backend = _launch(tmp_path, monkeypatch, n_ctx = 2048, wired_bytes = _WIRED, **self._ROOMY)[
            "backend"
        ]
        assert backend.last_load_warning is None

    def test_an_unreadable_limit_keeps_the_free_memory_refusal(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 8192, **self._ROOMY)

    def test_a_context_past_the_limit_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 32768, wired_bytes = _WIRED, **self._ROOMY)

    def test_the_refusal_names_what_the_limit_holds(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 32768, wired_bytes = _WIRED, **self._ROOMY)
        named = _named_ceiling(str(excinfo.value))
        assert named > 8192
        cmd = _launch(tmp_path, monkeypatch, n_ctx = named, wired_bytes = _WIRED, **self._ROOMY)["cmd"]
        assert _ctx_values(cmd)[-1] == str(named)
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = named + 1024, wired_bytes = _WIRED, **self._ROOMY)

    def test_the_refusal_can_name_a_context_under_the_fit_floor(self, tmp_path, monkeypatch):
        tight = dict(self._ROOMY, budget_bytes = 8 * 1024**3)
        with pytest.raises(RuntimeError) as free_only:
            _launch(tmp_path, monkeypatch, n_ctx = 8192, **tight)
        tight["wired_bytes"] = _BUDGET
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 8192, **tight)
        named = _named_ceiling(str(excinfo.value))
        assert _named_ceiling(str(free_only.value)) < named < 8192
        cmd = _launch(tmp_path, monkeypatch, n_ctx = named, **tight)["cmd"]
        assert _ctx_values(cmd)[-1] == str(named)

    def test_the_limit_can_refuse_the_context_auto_picks(self, tmp_path, monkeypatch):
        squeezed = dict(self._ROOMY, wired_bytes = 8 * 1024**3)
        auto_cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, **squeezed)["cmd"]
        auto_ctx = int(_ctx_values(auto_cmd)[-1])
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = auto_ctx, **squeezed)
        assert _named_ceiling(str(excinfo.value)) < auto_ctx
        assert "leave it on Auto" not in str(excinfo.value)

    def test_auto_still_sizes_from_free_memory(self, tmp_path, monkeypatch):
        with_limit = _launch(tmp_path, monkeypatch, n_ctx = 0, wired_bytes = _WIRED, **self._ROOMY)
        without = _launch(tmp_path, monkeypatch, n_ctx = 0, **self._ROOMY)
        assert _ctx_values(with_limit["cmd"])[-1] == _ctx_values(without["cmd"])[-1]

    def test_weights_over_free_memory_are_priced_against_the_limit(self, tmp_path, monkeypatch):
        heavy = dict(
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = 10 * 1024**3,
            kv_per_token = _FAT_KV,
            wired_bytes = 32 * 1024**3,
        )
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = 65536, **heavy)
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 8192, **heavy)["cmd"]
        assert _ctx_values(cmd)[-1] == "8192"

    def test_weights_alone_past_the_limit_are_not_refused(self, tmp_path, monkeypatch):
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 32768,
            wired_bytes = _WIRED,
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = 20 * 1024**3,
            kv_per_token = _FAT_KV,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "32768"

    def test_weights_past_the_limit_keep_the_free_memory_refusal(self, tmp_path, monkeypatch):
        """The headroom holds the GGUF but not the fitted weights, compute reserve included."""
        squeezed = dict(
            real_fit = True,
            budget_bytes = _BUDGET,
            weights_bytes = _TIGHT_WEIGHTS,
            kv_per_token = _FAT_KV,
            wired_bytes = _TIGHT_WEIGHTS + 1024**3,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 32768, **squeezed)
        assert _named_ceiling(str(excinfo.value)) == _TIGHT_CEILING
        assert "leave it on Auto" in str(excinfo.value)
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 512, **squeezed)["cmd"]
        assert _ctx_values(cmd)[-1] == "512"

    def test_swa_checkpoints_count_against_the_limit(self, tmp_path, monkeypatch):
        ckpt = dict(
            real_fit = True,
            budget_bytes = 20 * 1024**3,
            weights_bytes = 1024**3,
            kv_per_token = 256 * 1024,
            ckpt_bytes = 1024**3,
            extra_args = ["--ctx-checkpoints", "8"],
            wired_bytes = 24 * 1024**3,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 56000, **ckpt)
        named = _named_ceiling(str(excinfo.value))
        cmd = _launch(tmp_path, monkeypatch, n_ctx = named, **ckpt)["cmd"]
        assert _ctx_values(cmd)[-1] == str(named)
        with pytest.raises(RuntimeError, match = "unified"):
            _launch(tmp_path, monkeypatch, n_ctx = named + 1024, **ckpt)

    def test_swa_checkpoints_count_at_the_fit_floor(self, tmp_path, monkeypatch):
        swa = dict(
            real_fit = True,
            budget_bytes = 20 * 1024**3,
            weights_bytes = 1024**3,
            kv_per_token = 256 * 1024,
            extra_args = ["--ctx-checkpoints", "1"],
            wired_bytes = 24 * 1024**3,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _launch(tmp_path, monkeypatch, n_ctx = 16384, ckpt_bytes = 17 * 1024**3, **swa)
        assert _named_ceiling(str(excinfo.value)) < 8192
        with pytest.raises(RuntimeError, match = "No context fits"):
            _launch(tmp_path, monkeypatch, n_ctx = 1024, ckpt_bytes = 18 * 1024**3, **swa)

    def test_nothing_fitting_free_memory_loads_under_the_limit(self, tmp_path, monkeypatch):
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 4096,
            wired_bytes = _WIRED,
            **TestWhenNothingFitsAtAll.NOTHING_FITS,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "4096"

    def test_nothing_fitting_the_limit_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match = "No context fits"):
            _launch(
                tmp_path,
                monkeypatch,
                n_ctx = 4096,
                wired_bytes = _BUDGET,
                **TestWhenNothingFitsAtAll.NOTHING_FITS,
            )

    def test_a_fixed_manual_layer_count_is_still_exempt(self, tmp_path, monkeypatch):
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 32768,
            wired_bytes = _WIRED,
            gpu_memory_mode = "manual",
            gpu_layers = 20,
            **self._ROOMY,
        )["cmd"]
        assert _ctx_values(cmd)[-1] == "32768"


_GIB = 1024**3
_MIB = 1024**2


def _backend_with_embeddings(
    monkeypatch,
    *,
    embd,
    tensors = 12 * _GIB,
    measured = "mapped as designed",
    settings = (False, False),
    layout = None,
):
    """Default probe maps every loadable tensor to Metal except the input embeddings."""
    from core.inference.llama_server_args import MEMORY_ENV_VARS
    from core.inference.offload_layout import ModelLayout
    import utils.model_memory_settings as _mem_settings

    monkeypatch.setattr(_mem_settings, "get_model_memory_settings", lambda: settings)
    placement_vars = (
        "LLAMA_ARG_OVERRIDE_TENSOR",
        "LLAMA_ARG_CPU_MOE",
        "LLAMA_ARG_N_CPU_MOE",
        "LLAMA_ARG_N_CPU_FFN",
        "LLAMA_ARG_N_GPU_LAYERS",
        "LLAMA_ARG_DEVICE",
    )
    for name in (*MEMORY_ENV_VARS, *placement_vars):
        monkeypatch.delenv(name, raising = False)
    if layout is None:
        layout = ModelLayout(complete = True, token_embd_bytes = embd, tensor_bytes = tensors)
    if measured == "mapped as designed":
        metal = layout.tensor_bytes - embd - layout.excluded_block_bytes
        measured = (metal // _MIB, embd // _MIB, 2)
    backend = LlamaCppBackend()
    backend._tensor_spill_layout = lambda _path, **_kw: layout
    backend._metal_measured_model_mib = lambda _binary, _path: measured
    return backend


class TestInputEmbeddingsLeftInTheFileMapping:
    """Exercise Metal fitting when input embeddings remain pageable on the CPU."""

    _COMMON = dict(real_fit = True, budget_bytes = 16 * 1024**3, weights_bytes = 12 * 1024**3)

    def _published(
        self,
        tmp_path,
        monkeypatch,
        *,
        n_ctx,
        kv_per_token = 32 * 1024,
        extra_args = None,
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        **kw,
    ):
        return self._launched(
            tmp_path,
            monkeypatch,
            n_ctx = n_ctx,
            kv_per_token = kv_per_token,
            extra_args = extra_args,
            gpu_memory_mode = gpu_memory_mode,
            gpu_layers = gpu_layers,
            **kw,
        )[:2]

    def _launched(
        self,
        tmp_path,
        monkeypatch,
        *,
        n_ctx,
        kv_per_token = 32 * 1024,
        extra_args = None,
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        budget_bytes = 16 * _GIB,
        **kw,
    ):
        """``(published ceiling, launched context, argv)``."""
        kw.setdefault("embd", 6 * 1024**3)
        captured = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = n_ctx,
            kv_per_token = kv_per_token,
            extra_args = extra_args,
            gpu_memory_mode = gpu_memory_mode,
            gpu_layers = gpu_layers,
            backend = _backend_with_embeddings(monkeypatch, **kw),
            **{**self._COMMON, "budget_bytes": budget_bytes},
        )
        cmd = captured["cmd"]
        return captured["backend"]._max_context_length, _ctx_values(cmd)[-1], cmd

    @staticmethod
    def _fit_tokens(cmd):
        """The placement tokens this launch emitted, in order."""
        out = []
        for i, token in enumerate(cmd):
            if token in ("-ngl", "--fit") and i + 1 < len(cmd):
                out.extend([token, cmd[i + 1]])
        return out

    @pytest.mark.parametrize("n_ctx", [0, 32768])
    def test_a_discounted_ceiling_launches_pinned_to_the_measured_placement(
        self, tmp_path, monkeypatch, n_ctx
    ):
        """A discounted launch must preserve the probe's full-offload placement."""
        from core.inference.llama_server_args import fit_is_effectively_on

        published, _, cmd = self._launched(tmp_path, monkeypatch, n_ctx = n_ctx)
        assert published > _FIT_MIN_CTX
        assert self._fit_tokens(cmd) == ["-ngl", "-1", "--fit", "off"]
        assert not fit_is_effectively_on(cmd, {})

    def test_a_load_with_nothing_taken_out_keeps_the_fitter(self, tmp_path, monkeypatch):
        _, _, cmd = self._launched(tmp_path, monkeypatch, n_ctx = 32768, embd = 0)
        assert self._fit_tokens(cmd) == ["--fit", "on"]

    def test_a_floor_the_discount_did_not_lift_keeps_the_fitter(self, tmp_path, monkeypatch):
        published, _, cmd = self._launched(tmp_path, monkeypatch, n_ctx = 0, embd = _GIB)
        assert published == _FIT_MIN_CTX
        assert self._fit_tokens(cmd) == ["--fit", "on"]

    @pytest.mark.parametrize("extra_args", [["--fit", "on"], ["--fit=on"], ["-fit", "1"]])
    def test_an_extra_that_turns_the_fitter_back_on_keeps_the_charge(
        self, tmp_path, monkeypatch, extra_args
    ):
        """A later pass-through flag can override Unsloth's placement pin."""
        from core.inference.llama_server_args import fit_is_effectively_on

        published, _, cmd = self._launched(
            tmp_path, monkeypatch, n_ctx = 32768, extra_args = extra_args
        )
        assert published == _FIT_MIN_CTX
        assert fit_is_effectively_on(cmd, {})
        assert "-ngl" not in cmd

    def test_a_pass_through_fit_off_still_matches_the_pin(self, tmp_path, monkeypatch):
        from core.inference.llama_server_args import fit_is_effectively_on

        published, _, cmd = self._launched(
            tmp_path, monkeypatch, n_ctx = 32768, extra_args = ["--fit", "off"]
        )
        assert published > 32768
        assert self._fit_tokens(cmd)[:4] == ["-ngl", "-1", "--fit", "off"]
        assert not fit_is_effectively_on(cmd, {})

    def test_a_placement_it_did_not_measure_keeps_the_fitter(self, tmp_path, monkeypatch):
        _, _, cmd = self._launched(
            tmp_path, monkeypatch, n_ctx = 32768, extra_args = ["--load-mode", "none"]
        )
        assert self._fit_tokens(cmd) == ["--fit", "on"]

    def test_charged_whole_the_same_load_publishes_the_floor(self, tmp_path, monkeypatch):
        published, _ = self._published(tmp_path, monkeypatch, n_ctx = 32768, embd = 0)
        assert published == _FIT_MIN_CTX

    def test_an_explicit_context_publishes_a_measured_ceiling(self, tmp_path, monkeypatch):
        published, launched = self._published(tmp_path, monkeypatch, n_ctx = 32768)
        assert launched == "32768"
        assert published > 32768

    def test_the_ceiling_follows_the_kv_cache_size(self, tmp_path, monkeypatch):
        wide, _ = self._published(tmp_path, monkeypatch, n_ctx = 0, kv_per_token = 32 * 1024)
        narrow, _ = self._published(tmp_path, monkeypatch, n_ctx = 0, kv_per_token = 64 * 1024)
        assert _FIT_MIN_CTX < narrow < wide

    def test_a_short_measured_ceiling_is_refused_past_rather_than_floored(
        self, tmp_path, monkeypatch
    ):
        """A measured ceiling below the usual floor still refuses larger contexts."""
        with pytest.raises(RuntimeError, match = "unified") as refused:
            self._published(tmp_path, monkeypatch, n_ctx = 32768, kv_per_token = _FAT_KV)
        assert 0 < _named_ceiling(str(refused.value)) < _FIT_MIN_CTX

    def test_a_build_that_maps_the_embeddings_keeps_the_charge(self, tmp_path, monkeypatch):
        published, _ = self._published(
            tmp_path, monkeypatch, n_ctx = 32768, measured = (12 * 1024, 6 * 1024, 2)
        )
        assert published == _FIT_MIN_CTX

    def test_a_draft_that_loads_the_targets_own_mtp_blocks_keeps_the_charge(
        self, tmp_path, monkeypatch
    ):
        from core.inference.offload_layout import ModelLayout

        layout = ModelLayout(
            complete = True,
            token_embd_bytes = 6 * _GIB,
            tensor_bytes = 12 * _GIB,
            has_excluded_blocks = True,
            excluded_block_bytes = _GIB,
        )
        published, _ = self._published(
            tmp_path,
            monkeypatch,
            n_ctx = 32768,
            extra_args = ["--spec-type", "draft-mtp"],
            layout = layout,
            measured = (6 * 1024, 6 * 1024, 2),
        )
        assert published == _FIT_MIN_CTX

    def test_no_measurement_keeps_the_charge(self, tmp_path, monkeypatch):
        published, _ = self._published(tmp_path, monkeypatch, n_ctx = 32768, measured = None)
        assert published == _FIT_MIN_CTX

    def test_a_fixed_manual_layer_count_keeps_the_charge(self, tmp_path, monkeypatch):
        published, _ = self._published(
            tmp_path, monkeypatch, n_ctx = 32768, gpu_memory_mode = "manual", gpu_layers = 20
        )
        assert published == _FIT_MIN_CTX

    def test_the_fit_and_the_launch_price_the_same_settings_read(self, tmp_path, monkeypatch):
        """A settings change during loading must not split fit and launch decisions."""
        import utils.model_memory_settings as _mem_settings

        backend = _backend_with_embeddings(monkeypatch, embd = 6 * 1024**3)
        reads = []

        def settings():
            reads.append(None)
            return (False, False) if len(reads) == 1 else (True, False)

        monkeypatch.setattr(_mem_settings, "get_model_memory_settings", settings)
        captured = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 32768,
            kv_per_token = 32 * 1024,
            backend = backend,
            **self._COMMON,
        )
        locked = any("mlock" in str(token) for token in captured["cmd"])
        discounted = captured["backend"]._max_context_length > _FIT_MIN_CTX
        assert locked != discounted

    @pytest.mark.parametrize(
        "extra_args",
        [
            ["--no-mmap"],
            ["--mlock"],
            ["--load-mode", "none"],
            ["--load-mode", "dio"],
            ["--override-tensor", "per_layer_token_embd=CPU"],
            ["--cpu-moe"],
            ["-ncffn", "4"],
            ["--fit-target", "2048"],
        ],
    )
    def test_a_loader_or_placement_it_did_not_measure_keeps_the_charge(
        self, tmp_path, monkeypatch, extra_args
    ):
        published, _ = self._published(tmp_path, monkeypatch, n_ctx = 32768, extra_args = extra_args)
        assert published == _FIT_MIN_CTX


class TestWhichLoadsLeaveTheEmbeddingsInTheMapping:
    """Cover loader and placement inputs that control the discount."""

    BYTES = 3 * _GIB
    DISCOUNT = BYTES - 2 * _MIB

    def _bytes(
        self,
        monkeypatch,
        *,
        extra_args = None,
        env = None,
        load_mode = None,
        layers_fixed = False,
        mtp_may_engage = False,
        **kw,
    ):
        kw.setdefault("embd", self.BYTES)
        backend = _backend_with_embeddings(monkeypatch, **kw)
        return backend._metal_demand_paged_embedding_bytes(
            "/models/model.gguf",
            extra_args,
            binary = "/fake/llama-server",
            requested_load_mode = load_mode,
            supports_load_mode = True,
            settings = kw.get("settings", (False, False)),
            layers_fixed = layers_fixed,
            mtp_may_engage = mtp_may_engage,
            env = env or {},
        )

    @pytest.mark.parametrize("load_mode", [None, "auto", "mmap"])
    def test_a_mapped_load_discounts_the_unmapped_embeddings(self, monkeypatch, load_mode):
        assert self._bytes(monkeypatch, load_mode = load_mode) == self.DISCOUNT

    def test_weights_llama_cpp_moved_to_the_cpu_stay_charged(self, monkeypatch):
        """CPU fallback weights remain charged."""
        tensors = 12 * _GIB
        measured = (7 * 1024, 5 * 1024, 3)
        unmapped = self._bytes(monkeypatch, tensors = tensors, measured = measured)
        charged = tensors - unmapped
        assert charged >= 9 * _GIB
        assert unmapped == 3 * _GIB - 3 * _MIB

    def test_embeddings_mapped_into_metal_are_not_taken_out(self, monkeypatch):
        """A collapsed Metal span can include the embeddings."""
        tensors = 12 * _GIB
        measured = (10 * 1024, 5 * 1024, 3)
        assert self._bytes(monkeypatch, tensors = tensors, measured = measured) == 0

    def test_never_more_than_the_embeddings_comes_out(self, monkeypatch):
        """Unaccounted bytes cannot increase the discount past the embeddings."""
        measured = (4 * 1024, 3 * 1024, 2)
        assert self._bytes(monkeypatch, tensors = 12 * _GIB, measured = measured) == self.BYTES

    @staticmethod
    def _layout_with_nextn(has_nextn):
        from core.inference.offload_layout import ModelLayout
        return ModelLayout(
            complete = True,
            token_embd_bytes = 3 * _GIB,
            tensor_bytes = 12 * _GIB,
            has_excluded_blocks = has_nextn,
            excluded_block_bytes = 2 * _GIB if has_nextn else 0,
        )

    def test_its_own_mtp_blocks_loading_drops_it(self, monkeypatch):
        """The probe cannot model target-embedded MTP blocks."""
        layout = self._layout_with_nextn(True)
        assert self._bytes(monkeypatch, layout = layout, mtp_may_engage = True) == 0

    @pytest.mark.parametrize(("has_nextn", "mtp_may_engage"), [(True, False), (False, True)])
    def test_mtp_blocks_that_stay_skipped_keep_it(self, monkeypatch, has_nextn, mtp_may_engage):
        layout = self._layout_with_nextn(has_nextn)
        unmapped = self._bytes(monkeypatch, layout = layout, mtp_may_engage = mtp_may_engage)
        assert unmapped == self.DISCOUNT

    def test_skipped_mtp_bytes_are_not_read_as_unmapped_embeddings(self, monkeypatch):
        """TENSOR_SKIP keeps the trailing blocks out of every buffer AND every table row."""
        layout = self._layout_with_nextn(True)
        measured = (10 * 1024, 3 * 1024, 2)
        assert self._bytes(monkeypatch, layout = layout, measured = measured) == 0

    @pytest.mark.parametrize("load_mode", ["none", "mlock", "mmap+mlock", "dio"])
    def test_a_holding_per_model_mode_drops_it(self, monkeypatch, load_mode):
        assert self._bytes(monkeypatch, load_mode = load_mode) == 0

    @pytest.mark.parametrize(
        "extra_args",
        [
            ["--lora", "/a.gguf"],
            ["--lora-scaled", "/a.gguf:0.5"],
            ["--control-vector", "/v.gguf"],
            ["--control-vector-scaled", "/v.gguf:0.5"],
            ["--lora=/a.gguf"],
        ],
    )
    def test_a_pass_through_adapter_drops_it(self, monkeypatch, extra_args):
        """The probe loads the base GGUF alone, and no Apple term charges the adapter."""
        assert self._bytes(monkeypatch, extra_args = extra_args) == 0

    def test_the_adapter_gate_is_the_flag_set_the_other_consumers_price(self, monkeypatch):
        from core.inference.llama_cpp import _SIDECAR_ADAPTER_FLAGS
        for flag in _SIDECAR_ADAPTER_FLAGS:
            assert self._bytes(monkeypatch, extra_args = [flag, "/a.gguf:0.5"]) == 0

    def test_keep_model_in_gpu_memory_drops_it(self, monkeypatch):
        assert self._bytes(monkeypatch, settings = (True, False)) == 0

    def test_no_ram_reserve_vetoes_a_holding_mode_back_to_the_mapping(self, monkeypatch):
        assert self._bytes(monkeypatch, load_mode = "none", settings = (False, True)) == self.DISCOUNT

    @pytest.mark.parametrize(
        "extra_args",
        [["--no-mmap"], ["--mlock"], ["--load-mode=none"], ["-lm", "mmap+mlock"], ["--direct-io"]],
    )
    def test_a_holding_pass_through_flag_drops_it(self, monkeypatch, extra_args):
        assert self._bytes(monkeypatch, extra_args = extra_args) == 0

    def test_a_pass_through_mmap_keeps_it(self, monkeypatch):
        assert self._bytes(monkeypatch, extra_args = ["--load-mode", "mmap"]) == self.DISCOUNT

    @pytest.mark.parametrize("extra_args", [["--fit", "on"], ["--fit=on"], ["-fit", "true"]])
    def test_a_pass_through_fit_on_drops_it(self, monkeypatch, extra_args):
        assert self._bytes(monkeypatch, extra_args = extra_args) == 0

    @pytest.mark.parametrize("extra_args", [["--fit", "off"], ["--fit=off"]])
    def test_a_pass_through_fit_off_keeps_it(self, monkeypatch, extra_args):
        assert self._bytes(monkeypatch, extra_args = extra_args) == self.DISCOUNT

    def test_an_inherited_fit_on_keeps_it(self, monkeypatch):
        assert self._bytes(monkeypatch, env = {"LLAMA_ARG_FIT": "on"}) == self.DISCOUNT

    @pytest.mark.parametrize(
        "env",
        [{"LLAMA_ARG_MLOCK": "1"}, {"LLAMA_ARG_NO_MMAP": "0"}, {"LLAMA_ARG_LOAD_MODE": "none"}],
    )
    def test_a_holding_inherited_variable_drops_it(self, monkeypatch, env):
        assert self._bytes(monkeypatch, env = env) == 0

    @pytest.mark.parametrize(
        "extra_args",
        [
            ["-ot", "token_embd=MTL0"],
            ["--override-tensor", "per_layer_token_embd=CPU"],
            ["--override-tensor=exps=CPU"],
            ["--cpu-moe"],
            ["--n-cpu-moe", "4"],
            ["-ngl", "20"],
            ["--device", "none"],
            ["-ncffn", "4"],
            ["--n-cpu-ffn=4"],
            ["--fit-target", "2048"],
            ["-fitt", "2048"],
            ["--fit-ctx", "16384"],
            ["-fitc=16384"],
        ],
    )
    def test_a_pass_through_placement_drops_it(self, monkeypatch, extra_args):
        assert self._bytes(monkeypatch, extra_args = extra_args) == 0

    @pytest.mark.parametrize(
        "env",
        [
            {"LLAMA_ARG_OVERRIDE_TENSOR": "exps=CPU"},
            {"LLAMA_ARG_CPU_MOE": "1"},
            {"LLAMA_ARG_N_GPU_LAYERS": "20"},
            {"LLAMA_ARG_DEVICE": "none"},
            {"LLAMA_ARG_N_CPU_FFN": "4"},
            {"LLAMA_ARG_FIT_TARGET": "2048"},
            {"LLAMA_ARG_FIT_CTX": "16384"},
        ],
    )
    def test_an_inherited_placement_drops_it(self, monkeypatch, env):
        assert self._bytes(monkeypatch, env = env) == 0

    def test_a_request_with_a_fixed_layer_count_drops_it(self, monkeypatch):
        assert self._bytes(monkeypatch, layers_fixed = True) == 0

    def test_an_unreadable_layout_abstains(self, monkeypatch):
        from core.inference.offload_layout import ModelLayout
        assert self._bytes(monkeypatch, layout = ModelLayout()) == 0

    def test_no_measurement_abstains_and_says_so(self, monkeypatch):
        """Reads the module logger directly: the structlog stub makes caplog flaky under xdist."""
        import core.inference.llama_cpp as llama_cpp

        said = []
        monkeypatch.setattr(llama_cpp.logger, "info", lambda msg, *a, **kw: said.append(str(msg)))
        assert self._bytes(monkeypatch, measured = None) == 0
        assert any("could not measure" in line for line in said)


_E4B_TABLE = """0.00.408.825 I common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
0.00.429.748 I common_memory_breakdown_print: |   - MTL0 (Apple M1)    |  5461 = 5460 + (3642 =  3025 +      28 +     589) +       -3642 |
0.00.429.748 I common_memory_breakdown_print: |   - Host               |                 2352 =  2288 +       0 +      64                |
0.00.444.070 I llama_fit_params: printing fitted CLI arguments to stdout...
-c 512 -ngl 999
"""
_E4B_NCFFN_TABLE = """I common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
I common_memory_breakdown_print: |   - MTL0 (Apple M1)    |  5461 = 5460 + (2196 =  1625 +      28 +     543) +       -2196 |
I common_memory_breakdown_print: |   - Host               |                 2430 =  2288 +       0 +     142                |
I common_memory_breakdown_print: |   - CPU_REPACK         |                 1400 =  1400 +       0 +       0                |
"""


class TestTheMetalMemoryProbe:
    """Parse and invoke llama-fit-params conservatively."""

    def test_it_reads_the_metal_and_host_model_columns(self):
        from core.inference.llama_cpp import _parse_metal_memory_breakdown
        assert _parse_metal_memory_breakdown(_E4B_TABLE) == (3025, 2288, 2)

    def test_every_host_buffer_type_is_summed(self):
        from core.inference.llama_cpp import _parse_metal_memory_breakdown
        assert _parse_metal_memory_breakdown(_E4B_NCFFN_TABLE) == (1625, 2288 + 1400, 3)

    def test_only_the_first_table_is_read(self):
        from core.inference.llama_cpp import _parse_metal_memory_breakdown
        assert _parse_metal_memory_breakdown(_E4B_TABLE + _E4B_NCFFN_TABLE) == (3025, 2288, 2)

    @pytest.mark.parametrize(
        "output",
        [
            "",
            "error: invalid argument: -lv",
            _E4B_TABLE.replace("MTL0 (Apple M1)", "CUDA0 (RTX 4090)"),
            _E4B_TABLE.replace("2352 =  2288 +", "2352"),
            _E4B_TABLE.replace("3025 +", "x +"),
        ],
    )
    def test_anything_else_is_unknown(self, output):
        from core.inference.llama_cpp import _parse_metal_memory_breakdown
        assert _parse_metal_memory_breakdown(output) is None

    @staticmethod
    def _install(tmp_path, script):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "llama-server").write_text("")
        probe = bin_dir / "llama-fit-params"
        probe.write_text(script)
        probe.chmod(0o755)
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF")
        return str(bin_dir / "llama-server"), str(model)

    @pytest.mark.skipif(sys.platform == "win32", reason = "shell script stands in for the binary")
    def test_it_runs_the_probe_beside_the_binary_and_caches_the_answer(self, tmp_path, monkeypatch):
        calls = tmp_path / "calls"
        table = _E4B_TABLE.replace("\n", "\\n")
        script = f'#!/bin/sh\necho "$@" >> "{calls}"\nprintf "{table}"\n'
        binary, model = self._install(tmp_path, script)
        monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "10")
        backend = LlamaCppBackend()
        assert backend._metal_measured_model_mib(binary, model) == (3025, 2288, 2)
        assert backend._metal_measured_model_mib(binary, model) == (3025, 2288, 2)
        argv = calls.read_text().splitlines()
        assert argv == [f"-m {model} -ngl 999 -c 512 -lv 4"]

    @pytest.mark.skipif(sys.platform == "win32", reason = "shell script stands in for the binary")
    def test_a_failing_probe_is_unknown(self, tmp_path):
        binary, model = self._install(tmp_path, "#!/bin/sh\nexit 1\n")
        assert LlamaCppBackend()._metal_measured_model_mib(binary, model) is None

    def test_no_probe_is_unknown(self, tmp_path):
        (tmp_path / "llama-server").write_text("")
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF")
        backend = LlamaCppBackend()
        assert backend._metal_measured_model_mib(str(tmp_path / "llama-server"), str(model)) is None
        assert backend._metal_measured_model_mib(None, str(model)) is None


def test_every_forced_full_offload_arm_owes_the_fit_on_retry():
    """A forced -ngl -1 arm must set fully_gpu_offloaded, or a --fit retry after a crash never runs."""
    import ast
    import inspect
    import textwrap

    from core.inference.llama_cpp import LlamaCppBackend

    tree = ast.parse(textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model)))

    def pins_full_offload(stmt):
        """Matches only a DIRECT statement of the arm, so an enclosing if does not also count."""
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "extend":
            return False
        if not call.args or not isinstance(call.args[0], ast.List):
            return False
        values = [e.value for e in call.args[0].elts if isinstance(e, ast.Constant)]
        return values == ["-ngl", "-1", "--fit", "off"]

    def claims_full_offload(body):
        for stmt in body:
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Constant):
                continue
            if stmt.value.value is not True:
                continue
            if any(isinstance(t, ast.Name) and t.id == "fully_gpu_offloaded" for t in stmt.targets):
                return True
        return False

    arms = [
        branch
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        for branch in (node.body, node.orelse)
        if branch and any(pins_full_offload(stmt) for stmt in branch)
    ]
    assert len(arms) == 2, f"expected the two forced full-offload arms, found {len(arms)}"
    assert all(claims_full_offload(arm) for arm in arms)


class TestFitterKeepsTheWiredHeadroom:
    """Weights over the budget still launch, and llama.cpp's fitter places them against
    ggml-metal's free report: recommendedMaxWorkingSetSize minus this process's own
    allocations, blind to what WindowServer and other apps hold. On a 24 GB Mac that
    packs a 27B Q4 to within 1 GiB of the 16 GiB ceiling and the display starves
    (#9279), so the launch carries the measured gap as --fit-target."""

    GiB = 1024**3
    OVER_BUDGET = dict(real_fit = True, budget_bytes = 9 * 1024**3, weights_bytes = 16 * 1024**3)

    @pytest.fixture
    def fit_caps(self, monkeypatch):
        probe = LlamaCppBackend.probe_server_capabilities.__func__
        monkeypatch.setattr(
            LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(
                lambda cls, binary = None: {
                    **probe(cls, binary),
                    "supports_fit_target": True,
                    "supports_fit_ctx": True,
                }
            ),
        )

    @staticmethod
    def _floor(monkeypatch, mib):
        calls = []

        def floor():
            calls.append(1)
            return mib

        monkeypatch.setattr(LlamaCppBackend, "_metal_fit_target_floor_mib", staticmethod(floor))
        return calls

    @staticmethod
    def _fit_target(cmd):
        values = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--fit-target"]
        return values[-1] if values else None

    @pytest.mark.parametrize("n_ctx", [0, 32768])
    def test_an_over_budget_load_keeps_the_gap(self, tmp_path, monkeypatch, fit_caps, n_ctx):
        self._floor(monkeypatch, 2764.8)
        cmd = _launch(tmp_path, monkeypatch, n_ctx = n_ctx, **self.OVER_BUDGET)["cmd"]
        assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"
        assert self._fit_target(cmd) == "2764"

    def test_a_margin_below_llama_cpps_own_leaves_argv_alone(self, tmp_path, monkeypatch, fit_caps):
        self._floor(monkeypatch, 0.0)
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, **self.OVER_BUDGET)["cmd"]
        assert self._fit_target(cmd) is None

    def test_a_pass_through_fit_target_still_wins(self, tmp_path, monkeypatch, fit_caps):
        self._floor(monkeypatch, 2764.8)
        cmd = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 0,
            extra_args = ["--fit-target", "4096"],
            **self.OVER_BUDGET,
        )["cmd"]
        assert self._fit_target(cmd) == "4096"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gpu_memory_mode": "manual", "gpu_layers": 20},
            {"paravirtual": True},
        ],
        ids = ["caller_owns_layers", "paravirtual_cpu_pin"],
    )
    def test_loads_that_do_not_place_on_metal_are_untouched(
        self, tmp_path, monkeypatch, fit_caps, kwargs
    ):
        calls = self._floor(monkeypatch, 2764.8)
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, **self.OVER_BUDGET, **kwargs)["cmd"]
        assert not calls
        assert self._fit_target(cmd) is None

    def test_off_apple_silicon_the_floor_is_never_read(self, tmp_path, monkeypatch, fit_caps):
        calls = self._floor(monkeypatch, 2764.8)
        _launch(tmp_path, monkeypatch, n_ctx = 0, metal = False, **self.OVER_BUDGET)
        assert not calls

    def test_the_floor_is_not_a_budget_price(self, tmp_path, monkeypatch, fit_caps):
        """A VRAM slider change must not evict a Mac child over the machine's margin."""
        self._floor(monkeypatch, 2764.8)
        backend = _launch(tmp_path, monkeypatch, n_ctx = 0, **self.OVER_BUDGET)["backend"]
        assert getattr(backend, "_vram_fraction_launched", None) is None

    @staticmethod
    def _fake_mlx(
        monkeypatch,
        seen_bytes,
        available = True,
    ):
        mlx = _types.ModuleType("mlx")
        core = _types.ModuleType("mlx.core")
        core.metal = _types.SimpleNamespace(is_available = lambda: available)
        core.device_info = lambda: {"max_recommended_working_set_size": seen_bytes}
        mlx.core = core
        monkeypatch.setitem(sys.modules, "mlx", mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", core)
        # The working-set reader falls back to torch.mps; keep the host's torch out of it.
        monkeypatch.setitem(sys.modules, "torch", None)

    @pytest.mark.parametrize(
        "seen_gib,headroom_gib,expected_mib",
        [(16, 13.3, (16 - 13.3) * 1024), (16, 16, 0.0), (16, 20, 0.0)],
    )
    def test_the_floor_is_the_gap_between_what_ggml_sees_and_the_headroom(
        self, monkeypatch, seen_gib, headroom_gib, expected_mib
    ):
        self._fake_mlx(monkeypatch, int(seen_gib * self.GiB))
        monkeypatch.setattr(
            LlamaCppBackend,
            "_apple_metal_wired_ceiling_bytes",
            staticmethod(lambda: int(headroom_gib * self.GiB)),
        )
        assert LlamaCppBackend._metal_fit_target_floor_mib() == pytest.approx(expected_mib, abs = 1)

    @pytest.mark.parametrize(
        "headroom,seen,available", [(0, 16, True), (13, 0, True), (13, 16, False)]
    )
    def test_an_unreadable_side_keeps_llama_cpps_margin(
        self, monkeypatch, headroom, seen, available
    ):
        self._fake_mlx(monkeypatch, seen * self.GiB, available)
        monkeypatch.setattr(
            LlamaCppBackend,
            "_apple_metal_wired_ceiling_bytes",
            staticmethod(lambda: headroom * self.GiB),
        )
        assert LlamaCppBackend._metal_fit_target_floor_mib() == 0.0

    def test_an_old_server_without_the_flag_gets_none(self):
        assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(
            1, True, False, 0, 8192, {"supports_fit_target": False}, fit_target_floor_mib = 4096
        )


class TestAutoGivesUpContextBeforeLayers:
    """Auto launches at a fixed -c, so past the wired headroom the fitter can only move layers to
    CPU; Auto uses the measured headroom while it stays at or above the Auto floor."""

    GiB = 1024**3
    # 6 GiB + 64 KiB/token: 9 GiB fits 49,152; 7.5 GiB headroom 24,576; 6.4 GiB short of 8,192.
    FITS = dict(
        real_fit = True,
        budget_bytes = 9 * 1024**3,
        weights_bytes = 1024**3,
        kv_per_token = 64 * 1024,
        native = 131072,
    )

    def _ctx(
        self,
        tmp_path,
        monkeypatch,
        wired_gib,
        n_ctx = 0,
    ):
        cap = _launch(
            tmp_path, monkeypatch, n_ctx = n_ctx, wired_bytes = int(wired_gib * self.GiB), **self.FITS
        )
        return int(_ctx_values(cap["cmd"])[-1]), cap["backend"]

    def test_auto_launches_at_the_wired_ceiling(self, tmp_path, monkeypatch):
        budget_ctx, _ = self._ctx(tmp_path, monkeypatch, 0)
        ctx, backend = self._ctx(tmp_path, monkeypatch, 7.5)
        assert _FIT_MIN_CTX <= ctx <= 24576 < budget_ctx
        assert backend._max_context_length == ctx

    def test_the_ceiling_above_the_auto_default_still_trims_to_the_wired_limit(
        self, tmp_path, monkeypatch
    ):
        roomy = dict(self.FITS, budget_bytes = 32 * self.GiB)
        cap = _launch(tmp_path, monkeypatch, n_ctx = 0, wired_bytes = 12 * self.GiB, **roomy)
        assert _ctx_values(cap["cmd"])[-1] == "65536"
        assert 65536 < cap["backend"]._max_context_length < 131072

    def test_ample_headroom_keeps_the_budget_context(self, tmp_path, monkeypatch):
        assert self._ctx(tmp_path, monkeypatch, 20)[0] == self._ctx(tmp_path, monkeypatch, 0)[0]

    def test_a_ceiling_under_the_auto_floor_leaves_it_to_the_fitter(self, tmp_path, monkeypatch):
        """Under 8,192 the context is worth more than the layers the fitter would move."""
        assert self._ctx(tmp_path, monkeypatch, 6.4)[0] == self._ctx(tmp_path, monkeypatch, 0)[0]

    def test_a_hand_set_context_is_not_lowered(self, tmp_path, monkeypatch):
        """The explicit path refuses against the wired limit instead (#10804)."""
        with pytest.raises(RuntimeError, match = "24,576"):
            self._ctx(tmp_path, monkeypatch, 7.5, n_ctx = 32768)


class TestMetalAlongsideAdmission:
    """An alongside load must report a shared-memory shortfall before spawning."""

    @pytest.mark.parametrize("n_ctx", [0, 32768])
    @pytest.mark.parametrize("budget_gib,wired_gib", [(9, 8), (32, 8)])
    def test_weights_over_remaining_memory_never_spawn(
        self, tmp_path, monkeypatch, n_ctx, budget_gib, wired_gib
    ):
        from core.inference.llama_cpp import GpuMemoryShortError

        launches = []
        with pytest.raises(GpuMemoryShortError) as caught:
            _launch(
                tmp_path,
                monkeypatch,
                n_ctx = n_ctx,
                real_fit = True,
                weights_bytes = 16 * 1024**3,
                budget_bytes = budget_gib * 1024**3,
                wired_bytes = wired_gib * 1024**3,
                cache_type_kv = "q8_0",
                refuse_partial_gpu_fit = True,
                launches = launches,
            )
        assert launches == []
        assert caught.value.short_mib > 0
        assert not caught.value.capped
        # Metal and MLX residents share one pool, not discrete GPU ordinals.
        assert caught.value.gpu_indices is None and not caught.value.discrete

    def test_minimum_context_must_fit_too(self, tmp_path, monkeypatch):
        from core.inference.llama_cpp import GpuMemoryShortError
        with pytest.raises(GpuMemoryShortError):
            _launch(
                tmp_path,
                monkeypatch,
                n_ctx = 0,
                real_fit = True,
                weights_bytes = 1024**3,
                budget_bytes = 9 * 1024**3,
                kv_per_token = 16 * 1024**2,
                refuse_partial_gpu_fit = True,
            )

    def test_used_up_wired_headroom_refuses_a_small_model(self, tmp_path, monkeypatch):
        from core.inference.llama_cpp import GpuMemoryShortError

        launches = []
        with pytest.raises(GpuMemoryShortError):
            _launch(
                tmp_path,
                monkeypatch,
                n_ctx = 0,
                real_fit = True,
                weights_bytes = 1024**3,
                budget_bytes = 32 * 1024**3,
                wired_exhausted = True,
                refuse_partial_gpu_fit = True,
                launches = launches,
            )
        assert launches == []

    def test_auto_context_that_fits_still_launches(self, tmp_path, monkeypatch):
        captured = _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 0,
            real_fit = True,
            weights_bytes = 1024**3,
            budget_bytes = 9 * 1024**3,
            refuse_partial_gpu_fit = True,
        )
        assert captured["cmd"][0] == "/fake/llama-server"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"refuse_partial_gpu_fit": False},
            {"refuse_partial_gpu_fit": True, "gpu_memory_mode": "manual", "gpu_layers": 0},
            {"refuse_partial_gpu_fit": True, "paravirtual": True},
            {"refuse_partial_gpu_fit": True, "extra_args": ["--device", "none"]},
        ],
    )
    def test_existing_explicit_exemptions(self, tmp_path, monkeypatch, kwargs):
        assert _launch(
            tmp_path,
            monkeypatch,
            n_ctx = 0,
            real_fit = True,
            weights_bytes = 16 * 1024**3,
            budget_bytes = 9 * 1024**3,
            **kwargs,
        )["cmd"]

    @pytest.mark.parametrize(
        "media, gib",
        [((20480, 0), 22), ((1, 20480), 22), ((6144, 65536), 128), ((30720, 0), 22)],
    )
    def test_media_comes_off_the_metal_budget(self, tmp_path, monkeypatch, media, gib):
        import core.inference.llama_cpp as llama_cpp
        import utils.hardware.hardware as hardware

        memory = _types.SimpleNamespace(total = gib << 30, available = gib << 30)
        monkeypatch.setattr("psutil.virtual_memory", lambda: memory)
        monkeypatch.setattr(hardware, "_metal_recommended_working_set_bytes", lambda: gib << 30)
        monkeypatch.setattr(LlamaCppBackend, "_metal_media_mib", lambda self: media)
        args = dict(n_ctx = 0, real_fit = True, refuse_partial_gpu_fit = True, weights_bytes = 4 << 30)
        with pytest.raises(llama_cpp.GpuMemoryShortError):
            _launch(tmp_path, monkeypatch, budget_bytes = 32 << 30, wired_bytes = 64 << 30, **args)

    def test_an_mlx_chat_models_growth_comes_off_beside_media(self, monkeypatch):
        import core.inference.gpu_residency as residency
        import core.inference.llama_cpp as llama_cpp

        backend = LlamaCppBackend()
        monkeypatch.setattr(llama_cpp, "_serving_backends_snapshot", lambda: [backend])
        monkeypatch.setattr(residency, "media_held_mib", lambda: {0: 6144})
        monkeypatch.setattr(residency, "media_unallocated_mib", lambda: {0: 1024})
        monkeypatch.setattr(residency, "_mlx_chat_unallocated_mib", lambda: 2048)
        assert backend._metal_media_mib() == (6144, 3072)

    def test_retry_rechecks_memory_after_eviction(self, tmp_path, monkeypatch):
        from core.inference.llama_cpp import GpuMemoryShortError

        backend = LlamaCppBackend()
        args = dict(
            n_ctx = 0,
            real_fit = True,
            weights_bytes = 16 * 1024**3,
            refuse_partial_gpu_fit = True,
            backend = backend,
        )
        with pytest.raises(GpuMemoryShortError):
            _launch(tmp_path, monkeypatch, budget_bytes = 9 * 1024**3, **args)
        assert not backend.is_active
        assert _launch(tmp_path, monkeypatch, budget_bytes = 32 * 1024**3, **args)["cmd"]


class TestAutoOnAMacWithRoomToSpare:
    _ROOMY = dict(real_fit = True, budget_bytes = 64 * 1024**3, weights_bytes = 1024**3)

    def test_auto_does_not_reserve_the_whole_native_window(self, tmp_path, monkeypatch):
        captured = _launch(tmp_path, monkeypatch, n_ctx = 0, **self._ROOMY)
        assert _ctx_values(captured["cmd"])[-1] == "65536"
        assert captured["backend"]._max_context_length == NATIVE

    def test_an_explicit_context_above_the_auto_default_still_loads(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 131072, **self._ROOMY)["cmd"]
        assert _ctx_values(cmd)[-1] == "131072"

    def test_a_shorter_native_window_keeps_its_own_length(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, n_ctx = 0, native = 40960, **self._ROOMY)["cmd"]
        assert _ctx_values(cmd)[-1] == "40960"
