# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metal must never be sent "-c 0".

llama.cpp reads "-c 0" as fit_params_min_ctx = UINT32_MAX, pinning the model's
full native context and disabling --fit's reduction. No GPU is enumerated on
Apple Silicon, so the Apple cap in load_model is the only thing holding the
context down, and two paths reach the command builder with a zero context after
that cap has been skipped or discarded: a GGUF carrying no context length in its
metadata (the cap is guarded on effective_ctx > 0), and the broad
`except Exception` around GPU selection, which restores the original request.
"""

from __future__ import annotations

import inspect
import re
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
    _LLAMA_FIT_MIN_CTX,
    GgufLoadIntent,
    LlamaCppBackend,
)

_floor = LlamaCppBackend._metal_zero_ctx_floor
_drops = LlamaCppBackend._metal_drops_zero_ctx_override
_REAL_POPEN = subprocess.Popen
# argv and child env of the most recent _launch, for the tests that assert on the env.
_LAST_LAUNCH: dict = {}


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
    metal = True,
    ctx_metadata = None,
    extra_args = None,
    gpu_memory_mode = "auto",
    gpu_layers = -1,
    paravirtual = False,
):
    """Drive the real load_model with no GPU enumerated (the Metal condition)."""
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: 16 * 1024**3 if metal else 0),
    )
    if paravirtual:
        import core.inference.llama_cpp as _llama_cpp
        monkeypatch.setattr(_llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
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
    backend._context_length = ctx_metadata

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
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
                n_ctx = 0,
                gpu_memory_mode = gpu_memory_mode,
                gpu_layers = gpu_layers,
                extra_args = extra_args,
            )
        )
    _LAST_LAUNCH.clear()
    _LAST_LAUNCH.update(captured)
    return captured["cmd"], backend


def _launch_env(tmp_path, monkeypatch, *, env_ctx, **kwargs):
    """As _launch, returning what the child inherits as LLAMA_ARG_CTX_SIZE."""
    if env_ctx is None:
        monkeypatch.delenv("LLAMA_ARG_CTX_SIZE", raising = False)
    else:
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", env_ctx)
    cmd, _ = _launch(tmp_path, monkeypatch, **kwargs)
    return cmd, _LAST_LAUNCH["env"].get("LLAMA_ARG_CTX_SIZE")


@pytest.fixture
def on_metal(monkeypatch):
    """A resolvable Apple unified-memory budget."""
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 9 * 1024**3)
    )


@pytest.fixture
def off_metal(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )


class TestOnMetal:
    def test_a_zero_context_is_floored(self, on_metal):
        """The exception path: auto request restored to 0 after the cap ran."""
        assert _floor(0, False, False, 262144) == _FIT_MIN_CTX

    def test_a_model_shorter_than_the_floor_keeps_its_own_length(self, on_metal):
        assert _floor(0, False, False, 2048) == 2048

    def test_no_metadata_still_gets_a_floor(self, on_metal):
        """The cap is guarded on ctx > 0, so this GGUF was never capped."""
        assert _floor(0, False, False, None) == _FIT_MIN_CTX

    def test_a_cap_below_the_floor_wins(self, on_metal):
        """The exception path keeps max_available_ctx, whose KV answer can sit
        under the floor. Floating back up would re-create the over-commit."""
        assert _floor(0, False, False, 262144, 2048) == 2048

    def test_a_cap_above_the_floor_does_not_raise_it(self, on_metal):
        assert _floor(0, False, False, 262144, 131072) == _FIT_MIN_CTX

    def test_no_cap_yet_still_floors(self, on_metal):
        """The no-metadata path never ran the cap, so there is no ceiling to respect."""
        assert _floor(0, False, False, None, 0) == _FIT_MIN_CTX

    def test_a_positive_context_is_left_alone(self, on_metal):
        assert _floor(8192, False, False, 262144) == 0

    def test_auto_layers_is_left_alone(self, on_metal):
        """It omits -c entirely and lets --fit size it, which is correct."""
        assert _floor(0, True, False, 262144) == 0

    def test_manual_offload_is_left_alone(self, on_metal):
        """There the user owns memory management, context cap included."""
        assert _floor(0, False, True, 262144) == 0


def test_every_caller_of_the_floor_states_whether_the_fitter_runs():
    """``fitter_runs`` defaults to True, the permissive value, so a call that omits it
    gets the 8192 ceiling and the auto_fit exemption back.

    That default is deliberate: there is one production caller and 60-odd test call
    sites, and making it required would churn the latter to constrain the former. What
    it costs is that a SECOND production caller could reintroduce this bug by saying
    nothing, which is how the previous version of this arm went wrong -- a docstring
    claim about what callers pass, with nothing checking that they do. So the claim is
    checked here instead of asserted in prose.
    """
    source = Path(inspect.getfile(LlamaCppBackend)).read_text(encoding = "utf-8")
    calls = [m for m in re.finditer(r"self\._metal_zero_ctx_floor\(", source)]
    assert calls, "the floor is no longer called from the backend; this guard is stale"
    for match in calls:
        line = source[: match.start()].count("\n") + 1
        # Balance the parens rather than stopping at the first ")": the argument this
        # guard is looking for is itself a call, so a naive scan ends inside it.
        depth, end = 1, match.end()
        while end < len(source) and depth:
            depth += {"(": 1, ")": -1}.get(source[end], 0)
            end += 1
        args = source[match.end() : end - 1]
        assert "fitter_runs" in args, (
            f"llama_cpp.py:{line} calls _metal_zero_ctx_floor without fitter_runs, so it "
            "takes the default True and can hand a Mac 8192 tokens with the child fitter "
            "off, which nothing can then reduce. Pass fit_is_effectively_on(...) for this "
            "launch."
        )


class TestWithNoFitterToReduceIt:
    """_FIT_MIN_CTX is a ceiling the child's --fit is expected to come down from.

    "Being BELOW llama.cpp's own floor is what made that safe to raise" -- so with
    the fitter off, this arm has to hand over what the fitter would have reduced
    to instead, or a Mac with room for 4096 and not 8192 has no way down.
    """

    def test_the_floor_drops_to_llama_cpps_own(self, on_metal):
        assert _floor(0, False, False, 262144, fitter_runs = False) == _LLAMA_FIT_MIN_CTX
        assert _LLAMA_FIT_MIN_CTX < _FIT_MIN_CTX

    def test_a_running_fitter_still_gets_the_raised_floor(self, on_metal):
        """The control: nothing about the ordinary Metal Auto path moved."""
        assert _floor(0, False, False, 262144, fitter_runs = True) == _FIT_MIN_CTX
        assert _floor(0, False, False, 262144) == _FIT_MIN_CTX

    def test_the_auto_layers_exemption_is_withdrawn_with_the_fitter(self, on_metal):
        """The docstring's claim, taken here rather than trusted to the caller:
        auto_fit means "--fit sizes it", which is false once --fit is off."""
        assert _floor(0, True, False, 262144, fitter_runs = False) == _LLAMA_FIT_MIN_CTX
        assert _floor(0, True, False, 262144, fitter_runs = True) == 0

    def test_a_measured_ceiling_below_it_still_wins(self, on_metal):
        assert _floor(0, False, False, 262144, 2048, fitter_runs = False) == 2048

    def test_a_model_shorter_than_it_keeps_its_own_length(self, on_metal):
        assert _floor(0, False, False, 2048, fitter_runs = False) == 2048

    def test_a_caller_owned_budget_is_still_exempt(self, on_metal):
        assert _floor(0, False, True, 262144, fitter_runs = False) == 0

    def test_it_stays_inert_off_apple_silicon(self, off_metal):
        assert _floor(0, False, False, 262144, fitter_runs = False) == 0


class TestEverywhereElse:
    @pytest.mark.parametrize("owns", [False, True])
    @pytest.mark.parametrize("ctx", [0, 4096])
    def test_no_budget_means_no_change(self, off_metal, owns, ctx):
        """0 off Apple Silicon, so Linux and Windows never enter this."""
        assert _floor(ctx, False, owns, 262144) == 0


class TestAPassThroughZeroContext:
    """A user "-c 0" is the same over-commit, arriving by another door."""

    @pytest.mark.parametrize("override", [0])
    def test_it_is_dropped_on_metal(self, on_metal, override):
        assert _drops(override, False) is True

    @pytest.mark.parametrize("override", [None, 2048, 262144])
    def test_a_positive_or_absent_override_is_left_alone(self, on_metal, override):
        assert _drops(override, False) is False

    def test_auto_layers_drops_it_too(self, on_metal):
        """Auto-layers omits -c so --fit sizes it; a trailing zero disables that."""
        assert _drops(0, False) is True

    def test_manual_offload_is_left_alone(self, on_metal):
        assert _drops(0, True) is False

    def test_it_is_kept_everywhere_else(self, off_metal):
        assert _drops(0, False) is False


class TestTheEmittedCommand:
    """What llama-server actually receives, argv-level.

    Floor and drop are only correct together: extras are appended after Unsloth's
    own -c and llama.cpp is last-wins, so a surviving "-c 0" undoes the floor.
    """

    def test_a_zero_override_does_not_outlive_the_floor(self, tmp_path, monkeypatch):
        cmd, _ = _launch(tmp_path, monkeypatch, extra_args = ["-c", "0", "--top-k", "5"])
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]
        # Only the context is dropped; the rest of the user's extras survive.
        assert "--top-k" in cmd and "5" in cmd

    def test_the_long_spelling_is_dropped_too(self, tmp_path, monkeypatch):
        cmd, _ = _launch(tmp_path, monkeypatch, extra_args = ["--ctx-size=0"])
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]

    def test_a_capped_context_also_drops_the_zero_override(self, tmp_path, monkeypatch):
        """The Apple cap ran, so the floor stays inert -- the drop still has to fire."""
        cmd, _ = _launch(tmp_path, monkeypatch, ctx_metadata = 262144, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]

    def test_the_context_studio_computed_is_what_survives(self, tmp_path, monkeypatch):
        """Not a constant: the cap's own answer stands, here the model's 2048.

        load_model already treats "-c 0" as non-explicit, so the cap overrides it
        either way. The drop only stops the trailing copy from undoing that.
        """
        cmd, _ = _launch(tmp_path, monkeypatch, ctx_metadata = 2048, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == ["2048"]

    def test_a_positive_override_is_still_honored(self, tmp_path, monkeypatch):
        cmd, _ = _launch(tmp_path, monkeypatch, extra_args = ["-c", "8192"])
        assert _ctx_values(cmd)[-1] == "8192"

    def test_without_an_override_the_floor_stands(self, tmp_path, monkeypatch):
        cmd, _ = _launch(tmp_path, monkeypatch)
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]

    def test_off_metal_nothing_is_touched(self, tmp_path, monkeypatch):
        """Linux and Windows keep today's behaviour, zero override included."""
        cmd, _ = _launch(tmp_path, monkeypatch, metal = False, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == ["0", "0"]


class TestAutoLayers:
    """gpu_memory_mode "manual" with gpu_layers < 0: no -c, --fit sizes it.

    The context is decided entirely by --fit here, so a pass-through "-c 0" is
    worse in this mode than anywhere else: the fit never runs at all.
    """

    def _launch_auto_layers(self, tmp_path, monkeypatch, **kwargs):
        return _launch(tmp_path, monkeypatch, gpu_memory_mode = "manual", gpu_layers = -1, **kwargs)

    def test_no_context_is_passed_without_an_override(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_auto_layers(tmp_path, monkeypatch)
        assert _ctx_values(cmd) == []
        assert "--fit" in cmd

    def test_a_zero_override_does_not_reach_the_server(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_auto_layers(
            tmp_path, monkeypatch, extra_args = ["-c", "0", "--top-k", "5"]
        )
        assert _ctx_values(cmd) == []
        assert "--fit" in cmd
        assert "--top-k" in cmd and "5" in cmd

    def test_a_positive_override_is_still_honored(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_auto_layers(tmp_path, monkeypatch, extra_args = ["-c", "8192"])
        assert _ctx_values(cmd)[-1] == "8192"

    def test_off_metal_nothing_is_touched(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_auto_layers(
            tmp_path, monkeypatch, metal = False, extra_args = ["-c", "0"]
        )
        assert _ctx_values(cmd) == ["0"]

    def test_a_fixed_manual_layer_count_keeps_the_override(self, tmp_path, monkeypatch):
        """Not Auto-layers: the user owns the budget, so nothing is dropped."""
        cmd, _ = _launch(
            tmp_path,
            monkeypatch,
            gpu_memory_mode = "manual",
            gpu_layers = 20,
            extra_args = ["-c", "0"],
        )
        assert _ctx_values(cmd)[-1] == "0"


class TestAutoLayersWithTheFitterTurnedOff:
    """The Auto-layers exemption is only as good as the fitter it defers to.

    Extras land after Unsloth's own "--fit on" and win, so a pass-through
    "--fit off" leaves a command carrying no -c and no fitter, which is
    llama.cpp's native context and the over-commit this branch prevents.

    The floor that replaces it cannot be _FIT_MIN_CTX either. 8192 is defensible
    on a Mac only as a ceiling the child's own --fit can still reduce, down to
    llama.cpp's fit_params_min_ctx; with the fitter off there is no reduction
    path at all, so a Mac with room for 4096 and not 8192 fails at startup or at
    decode. The user typed --fit off, so the conservative context is what gives
    way, not their flag.
    """

    AUTO_LAYERS = {"gpu_memory_mode": "manual", "gpu_layers": -1}

    def test_the_floor_applies_once_fitting_is_off(self, tmp_path, monkeypatch):
        cmd, _ = _launch(tmp_path, monkeypatch, extra_args = ["--fit", "off"], **self.AUTO_LAYERS)
        assert _ctx_values(cmd) == [str(_LLAMA_FIT_MIN_CTX)]

    def test_the_users_fit_off_is_not_taken_back_to_keep_the_higher_floor(
        self, tmp_path, monkeypatch
    ):
        """The other shape this could have taken. Overriding an explicit --fit off
        would let the floor stay at 8192, and would also re-arm a fitter the user
        may have turned off because it aborts on their host."""
        cmd, _ = _launch(tmp_path, monkeypatch, extra_args = ["--fit", "off"], **self.AUTO_LAYERS)
        assert cmd[-2:] == ["--fit", "off"]

    def test_the_no_kv_cap_lands_on_the_same_floor(self, tmp_path, monkeypatch):
        """The cap path, not the zero-context one, and in DEFAULT memory mode.

        Manual with Auto layers resolves an Auto request to 0, so it never reaches
        the cap; the default mode expands it to the model's native length, which
        does.

        A GGUF that DOES carry a context length takes the Apple cap, which assigns
        a positive context of its own. _metal_zero_ctx_floor never sees it: its
        first condition returns 0 for any positive effective_ctx, so the no-fitter
        reduction has to be made where the cap is chosen. Without that, this arm
        hands a Mac 8192 with nothing able to bring it down.
        """
        cmd, _ = _launch(
            tmp_path,
            monkeypatch,
            ctx_metadata = 262144,
            extra_args = ["--fit", "off"],
        )
        assert _ctx_values(cmd) == [str(_LLAMA_FIT_MIN_CTX)]

    def test_the_no_kv_cap_keeps_the_raised_floor_while_a_fitter_runs(self, tmp_path, monkeypatch):
        """The control: same path, fitter left on, so the exemption stands and the
        command carries no -c at all, leaving the child to size the context. What
        must NOT happen is this arm quietly adopting llama.cpp's lower floor for a
        launch that still has a fitter to come down from."""
        cmd, _ = _launch(tmp_path, monkeypatch, ctx_metadata = 262144)
        assert _ctx_values(cmd) == [
            str(_FIT_MIN_CTX)
        ], f"the raised ceiling was given up on a launch that still has a fitter: {cmd}"

    def test_a_zero_override_alongside_it_is_still_dropped(self, tmp_path, monkeypatch):
        cmd, _ = _launch(
            tmp_path,
            monkeypatch,
            extra_args = ["--fit", "off", "-c", "0"],
            **self.AUTO_LAYERS,
        )
        assert _ctx_values(cmd) == [str(_LLAMA_FIT_MIN_CTX)]

    def test_an_inherited_fit_off_reaches_the_same_floor(self, tmp_path, monkeypatch):
        """llama.cpp applies LLAMA_ARG_FIT before argv, so the env twin turns the
        fitter off just as an extra does and withdraws the same exemption."""
        monkeypatch.setenv("LLAMA_ARG_FIT", "off")
        cmd, _ = _launch(tmp_path, monkeypatch, **self.AUTO_LAYERS)
        assert _ctx_values(cmd) == [str(_LLAMA_FIT_MIN_CTX)]

    def test_an_explicit_fit_on_keeps_the_exemption(self, tmp_path, monkeypatch):
        cmd, _ = _launch(tmp_path, monkeypatch, extra_args = ["--fit", "on"], **self.AUTO_LAYERS)
        assert _ctx_values(cmd) == []

    def test_off_metal_nothing_is_touched(self, tmp_path, monkeypatch):
        cmd, _ = _launch(
            tmp_path,
            monkeypatch,
            extra_args = ["--fit", "off"],
            metal = False,
            **self.AUTO_LAYERS,
        )
        assert _ctx_values(cmd) == []

    def test_a_caller_owned_budget_is_left_alone(self, tmp_path, monkeypatch):
        cmd, _ = _launch(
            tmp_path,
            monkeypatch,
            extra_args = ["--fit", "off"],
            gpu_memory_mode = "manual",
            gpu_layers = 20,
        )
        assert _ctx_values(cmd) == ["0"]


class TestAnInheritedContextEnvironment:
    """LLAMA_ARG_CTX_SIZE runs -c's own handler, and env parses before argv.

    So the command line wins wherever Unsloth emits one. Auto-layers emits none,
    on purpose, leaving an inherited 0 to cancel the --fit that sizes the mode.
    """

    AUTO_LAYERS = {"gpu_memory_mode": "manual", "gpu_layers": -1}

    def test_a_zero_is_dropped_where_no_context_is_emitted(self, tmp_path, monkeypatch):
        cmd, env_ctx = _launch_env(tmp_path, monkeypatch, env_ctx = "0", **self.AUTO_LAYERS)
        assert _ctx_values(cmd) == []
        assert env_ctx is None

    def test_a_positive_inherited_context_is_kept(self, tmp_path, monkeypatch):
        """Still the legitimate way to set a context for an Auto-layers launch."""
        cmd, env_ctx = _launch_env(tmp_path, monkeypatch, env_ctx = "8192", **self.AUTO_LAYERS)
        assert _ctx_values(cmd) == []
        assert env_ctx == "8192"

    def test_an_emitted_context_leaves_the_environment_alone(self, tmp_path, monkeypatch):
        """Automatic mode passes -c, and argv is parsed after the environment."""
        cmd, env_ctx = _launch_env(tmp_path, monkeypatch, env_ctx = "0")
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]
        assert env_ctx == "0"

    def test_a_caller_owned_budget_is_left_alone(self, tmp_path, monkeypatch):
        cmd, env_ctx = _launch_env(
            tmp_path, monkeypatch, env_ctx = "0", gpu_memory_mode = "manual", gpu_layers = 20
        )
        assert env_ctx == "0"

    def test_off_metal_nothing_is_touched(self, tmp_path, monkeypatch):
        cmd, env_ctx = _launch_env(
            tmp_path, monkeypatch, env_ctx = "0", metal = False, **self.AUTO_LAYERS
        )
        assert env_ctx == "0"

    @pytest.mark.parametrize("value", ["", "  ", "abc", "-1"])
    def test_only_a_zero_counts(self, tmp_path, monkeypatch, value):
        """Anything else is llama.cpp's to interpret, or reject."""
        cmd, env_ctx = _launch_env(tmp_path, monkeypatch, env_ctx = value, **self.AUTO_LAYERS)
        assert env_ctx == value


class TestAVirtualisedMetalDevice:
    """The paravirtual pin rewrites every placement to manual/0 before these guards.

    Auto is the default, so reading the mode off the rewritten placement made the
    common case on a virtualised Mac look caller-owned and emit "-c 0" anyway.
    """

    def _launch_pv(self, tmp_path, monkeypatch, **kwargs):
        return _launch(tmp_path, monkeypatch, paravirtual = True, **kwargs)

    def test_an_auto_request_still_gets_the_floor(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_pv(tmp_path, monkeypatch)
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]

    def test_an_auto_request_still_drops_a_zero_override(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_pv(tmp_path, monkeypatch, extra_args = ["-c", "0", "--top-k", "5"])
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]
        assert "--top-k" in cmd and "5" in cmd

    def test_auto_layers_is_treated_the_same(self, tmp_path, monkeypatch):
        """The pin took the layer freedom --fit needed, so the floor applies."""
        cmd, _ = self._launch_pv(tmp_path, monkeypatch, gpu_memory_mode = "manual", gpu_layers = -1)
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]

    def test_a_fixed_manual_layer_count_is_still_the_callers(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_pv(
            tmp_path,
            monkeypatch,
            gpu_memory_mode = "manual",
            gpu_layers = 20,
            extra_args = ["-c", "0"],
        )
        assert _ctx_values(cmd)[-1] == "0"

    def test_off_metal_nothing_is_touched(self, tmp_path, monkeypatch):
        cmd, _ = self._launch_pv(tmp_path, monkeypatch, metal = False, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == ["0", "0"]


def test_the_emission_guard_is_still_in_place():
    """Pins the existing contract the floor sits in front of."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    zero = src.find('cmd.extend(["-c", "0"])')
    assert zero != -1
    guard = src.rfind("elif not auto_fit:", 0, zero)
    assert guard != -1 and zero - guard < 120
    assert "_metal_zero_ctx_floor(" in src


class TestTheAdvertisedCeilingMatchesWhatWeLaunch:
    """max_context_length must not outlive the cap that never ran.

    On the exception path max_available_ctx still holds the native length its
    initialiser put there, which nothing has said fits. Publishing it makes the
    UI call it the largest context that fits, advertising the over-commit as safe.
    """

    NATIVE = 262144

    def test_the_native_length_is_not_advertised_after_the_floor(self, on_metal):
        floor = _floor(0, False, False, self.NATIVE, self.NATIVE)
        assert floor == _FIT_MIN_CTX
        # What load_model now publishes: the floor itself, not max(ceiling, floor).
        assert floor < self.NATIVE

    def test_a_real_ceiling_below_the_floor_still_wins(self, on_metal):
        """The cap did run and its KV answer is smaller, so keep the smaller."""
        assert _floor(0, False, False, self.NATIVE, 3000) == 3000

    def test_the_published_ceiling_is_never_above_what_we_launch(self, on_metal):
        for max_avail in (None, 3000, 4096, self.NATIVE):
            floor = _floor(0, False, False, self.NATIVE, max_avail)
            assert floor <= (max_avail or _FIT_MIN_CTX)
            assert floor <= _FIT_MIN_CTX


class TestTheStripDoesNotRewriteWhatWasRequested:
    """The zero-context strip is a launch decision, not a record of the ask.

    _requested_extra_args is the comparator a later Apply is matched against, so
    storing the stripped list there reloaded the model on every Apply.
    """

    def test_the_strip_removes_only_the_context_pair(self):
        from core.inference.llama_cpp import strip_context_only
        user = ["--threads", "8", "-c", "0", "--mlock"]
        assert strip_context_only(list(user)) == ["--threads", "8", "--mlock"]

    def test_a_suppressed_drafter_does_not_snapshot_the_strip(self, tmp_path, monkeypatch):
        """The paravirtual drafter drop takes its own copy, and that copy wins
        below. Taken after the zero-context strip, it would put the rewrite back
        into the comparator and cause the reload this class exists to prevent.
        """
        import core.inference.llama_cpp as _llama_cpp

        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"\x00" * 16)
        # No draft-layer flag, so the drafter cannot be pinned and is dropped instead.
        monkeypatch.setattr(_llama_cpp, "_paravirtual_draft_ngl_flag", lambda caps: None)
        requested = ["-md", str(draft), "-c", "0", "--top-k", "5"]
        cmd, backend = _launch(tmp_path, monkeypatch, extra_args = list(requested), paravirtual = True)
        assert _ctx_values(cmd) == [str(_FIT_MIN_CTX)]
        assert backend._requested_extra_args == requested

    @pytest.mark.parametrize("mode,layers", [("auto", -1), ("manual", -1)])
    def test_the_comparator_still_holds_what_was_asked_for(
        self, tmp_path, monkeypatch, mode, layers
    ):
        """A launch that stripped -c 0 must still compare equal to its own request."""
        requested = ["-c", "0", "--top-k", "5"]
        cmd, backend = _launch(
            tmp_path,
            monkeypatch,
            extra_args = list(requested),
            gpu_memory_mode = mode,
            gpu_layers = layers,
        )
        assert "0" not in _ctx_values(cmd)
        assert backend._requested_extra_args == requested
