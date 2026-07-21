# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the llama-server mmproj text-only fallback.

A GGUF vision model is launched with ``--mmproj <projector>``. When the
installed llama.cpp prebuilt is older than the model's projector format,
llama-server aborts at startup with ``clip.cpp:NNNN: Unknown projector
type`` (exit -6). load_model now retries once WITHOUT ``--mmproj`` so the
base model still loads text-only, warns the user to update llama.cpp, and
marks the session non-vision. These tests pin the two decision helpers:
``_is_projector_incompatibility`` (when to retry) and ``_strip_mmproj_args``
(how the retry argv is built). Unrelated failures must NOT trigger a retry.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Match the stubbing pattern in sibling tests so the module imports in a
# lightweight env without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger(
    "structlog"
)
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_detect = LlamaCppBackend._is_projector_incompatibility
_strip = LlamaCppBackend._strip_mmproj_args
_signal_crash = LlamaCppBackend._is_signal_crash
_flash_off = LlamaCppBackend._with_flash_attn_off
_nonproj = LlamaCppBackend._output_has_nonprojector_diagnostic

# Real abort captured loading gemma-4 on a 3-day-old prebuilt (build b9496).
_GEMMA4_OLD_LLAMACPP_OUT = (
    "srv    load_model: loading model 'gemma-4-E2B-it-UD-Q4_K_XL.gguf'\n"
    "/build_work/src/llama.cpp-b9496/tools/mtmd/clip.cpp:4391: "
    "Unknown projector type\n"
    "libggml-base.so.0(ggml_abort+0x152)\n"
    "libmtmd.so.0(clip_n_mmproj_embd)\n"
)
# Unrelated failures that must keep their own handling (no projector retry).
_OOM_OUT = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on "
    "device 0: cudaMalloc failed: out of memory"
)
_BAD_ARCH_OUT = (
    "llama_model_load: error loading model: unknown model architecture: 'qwen_image'"
)
_PORT_OUT = "srv start: failed to bind: address already in use"
_MISSING_OUT = "error: failed to open GGUF file: no such file or directory"
# A healthy startup log that merely mentions the projector must not match.
_HEALTHY_VISION_OUT = (
    "Using mmproj for vision: /cache/mmproj-F16.gguf\n"
    "clip_model_loader: loaded meta data with 20 key-value pairs\n"
    "srv  update_slots: all slots are idle"
)


class TestProjectorIncompatibilityDetector:
    def test_gemma4_on_old_llamacpp_triggers_retry(self):
        # Headline case: a 3-day-old llama.cpp aborts on Gemma-4's projector.
        assert _detect(_GEMMA4_OLD_LLAMACPP_OUT) is True

    @pytest.mark.parametrize(
        "out",
        [
            "clip.cpp:4391: Unknown projector type",
            "error: unsupported projector type for this model",
            "llama_mmproj: unsupported mmproj file version",
            "clip.cpp: projector type 'gemma4' is not supported",
        ],
    )
    def test_projector_format_errors_match(self, out):
        assert _detect(out) is True

    def test_case_insensitive(self):
        assert _detect("UNKNOWN PROJECTOR TYPE") is True

    @pytest.mark.parametrize(
        "out",
        [
            _OOM_OUT,
            _BAD_ARCH_OUT,
            _PORT_OUT,
            _MISSING_OUT,
            _HEALTHY_VISION_OUT,
            "",
            # bare multimodal words without a failure term must not match
            "loading clip model",
            "mmproj file resolved from cache",
        ],
    )
    def test_unrelated_failures_do_not_retry(self, out):
        assert _detect(out) is False


class TestSignalCrashDetector:
    """_is_signal_crash flags a hard fault (SIGSEGV/SIGABRT/SIGILL/SIGFPE/SIGBUS
    or a Windows 0xC0000000+ fault); not a clean exit, hung (None), or an
    external kill (SIGKILL/SIGTERM/SIGINT) that an OOM/unload would cause."""

    @pytest.mark.parametrize("rc", [-11, -6, -4, -7, -8, 0xC0000005, 0xC000001D])
    def test_program_faults_are_hard_crashes(self, rc):
        assert _signal_crash(rc) is True

    @pytest.mark.parametrize("rc", [0, 1, 2, 137, None, -9, -15, -2])
    def test_clean_hung_or_external_kill_is_not(self, rc):
        assert _signal_crash(rc) is False


# A realistic vision launch argv (mirrors the live "Starting llama-server"
# command), projector pair at the end.
_VISION_CMD = [
    "/home/u/.unsloth/llama.cpp/build/bin/llama-server",
    "-m",
    "/cache/gemma-4-E2B-it-UD-Q4_K_XL.gguf",
    "--port",
    "55473",
    "-c",
    "131072",
    "--parallel",
    "1",
    "--flash-attn",
    "on",
    "--no-context-shift",
    "-ngl",
    "-1",
    "--threads",
    "-1",
    "--jinja",
    "--spec-default",
    "--mmproj",
    "/cache/mmproj-F16.gguf",
]


class TestStripMmprojArgs:
    def test_removes_mmproj_pair(self):
        stripped = _strip(_VISION_CMD)
        assert "--mmproj" not in stripped
        assert "/cache/mmproj-F16.gguf" not in stripped

    def test_preserves_every_text_flag(self):
        stripped = _strip(_VISION_CMD)
        for flag in (
            "-m",
            "/cache/gemma-4-E2B-it-UD-Q4_K_XL.gguf",
            "--port",
            "55473",
            "-c",
            "131072",
            "-ngl",
            "-1",
            "--jinja",
            "--spec-default",
            "--flash-attn",
            "on",
        ):
            assert flag in stripped
        # Exactly the two projector tokens are dropped.
        assert len(stripped) == len(_VISION_CMD) - 2

    def test_strips_mmproj_in_the_middle(self):
        cmd = ["llama-server", "--mmproj", "/p/mm.gguf", "-c", "4096", "--jinja"]
        assert _strip(cmd) == ["llama-server", "-c", "4096", "--jinja"]

    def test_noop_when_no_mmproj(self):
        cmd = ["llama-server", "-m", "/p/model.gguf", "-c", "4096", "--jinja"]
        assert _strip(cmd) == cmd

    def test_returns_new_list(self):
        cmd = ["llama-server", "--mmproj", "/p/mm.gguf"]
        out = _strip(cmd)
        assert out is not cmd
        assert cmd[-1] == "/p/mm.gguf"  # input untouched


class TestFlashAttnOff:
    """_with_flash_attn_off is the least-destructive recovery rung: flip
    '--flash-attn on' to 'off' (keeps vision + MTP), or None when there is
    nothing to disable."""

    def test_flips_on_to_off_keeping_vision_and_mtp(self):
        out = _flash_off(_VISION_CMD)
        assert out is not None
        # FA disabled, every other capability (mmproj, MTP, ctx) preserved.
        i = out.index("--flash-attn")
        assert out[i + 1] == "off"
        assert "--mmproj" in out and "--spec-default" in out
        assert len(out) == len(_VISION_CMD)

    def test_none_when_already_off(self):
        assert _flash_off(["llama-server", "--flash-attn", "off", "-c", "4096"]) is None

    def test_none_when_no_flash_attn(self):
        assert _flash_off(["llama-server", "-m", "/m.gguf", "-c", "4096"]) is None

    def test_returns_new_list_input_untouched(self):
        cmd = ["llama-server", "--flash-attn", "on"]
        out = _flash_off(cmd)
        assert out == ["llama-server", "--flash-attn", "off"]
        assert cmd[-1] == "on"  # input not mutated

    def test_flips_equals_form(self):
        out = _flash_off(["llama-server", "--flash-attn=on", "-c", "4096"])
        assert out == ["llama-server", "--flash-attn=off", "-c", "4096"]

    def test_flips_fa_alias_and_auto(self):
        assert _flash_off(["llama-server", "-fa", "auto"]) == [
            "llama-server",
            "-fa",
            "off",
        ]
        assert _flash_off(["llama-server", "-fa=on"]) == ["llama-server", "-fa=off"]

    def test_flips_every_occurrence_last_wins(self):
        # extra_args can re-enable FA after Unsloth's flag; llama.cpp is last-wins,
        # so one leftover 'on' would re-crash the retry. Every enable must flip.
        cmd = [
            "llama-server",
            "--flash-attn",
            "on",
            "--mmproj",
            "/p",
            "--flash-attn",
            "on",
        ]
        out = _flash_off(cmd)
        assert out is not None
        assert "on" not in out
        assert out.count("off") == 2

    def test_none_when_equals_off(self):
        assert _flash_off(["llama-server", "--flash-attn=off"]) is None

    def test_none_when_user_off_wins_last(self):
        # User appended 'off' after Unsloth's 'on'; effective (last-wins) is off,
        # so there is nothing to retry.
        assert (
            _flash_off(["llama-server", "--flash-attn", "on", "--flash-attn", "off"])
            is None
        )

    def test_neutralizes_trailing_bare_flag(self):
        # A bare --flash-attn reads as on under last-wins; it must be neutralized
        # too, else the retry re-enables FA and re-crashes.
        out = _flash_off(["llama-server", "--flash-attn", "on", "--flash-attn"])
        assert out == ["llama-server", "--flash-attn", "off", "--flash-attn=off"]
        assert "on" not in out

    def test_bare_flag_only(self):
        assert _flash_off(["llama-server", "--flash-attn"]) == [
            "llama-server",
            "--flash-attn=off",
        ]
        assert _flash_off(["llama-server", "-fa"]) == ["llama-server", "-fa=off"]


class TestNonProjectorDiagnostic:
    """_output_has_nonprojector_diagnostic gates the signal-only text-only retry:
    a hard crash that already names OOM / a bad arch / a TP limit must surface
    that error, not be silently downgraded to a non-vision session."""

    @pytest.mark.parametrize(
        "out",
        [
            _OOM_OUT,
            _BAD_ARCH_OUT,
            "ggml_backend_cuda_buffer_type_alloc: failed to allocate buffer",
            "split_mode_tensor not implemented for this architecture",
        ],
    )
    def test_known_nonprojector_causes_match(self, out):
        assert _nonproj(out) is True

    @pytest.mark.parametrize(
        "out",
        [
            "",  # bare crash: no marker -> still eligible for the text-only retry
            _GEMMA4_OLD_LLAMACPP_OUT,  # a real projector abort must NOT be suppressed
            _HEALTHY_VISION_OUT,
            _PORT_OUT,
        ],
    )
    def test_bare_or_projector_output_does_not_match(self, out):
        assert _nonproj(out) is False


class TestRetryContract:
    """The two helpers compose into the load_model retry decision."""

    def test_gemma4_failure_yields_valid_text_only_command(self):
        # Old-llama.cpp projector abort -> retry, and the retry argv is a
        # valid text-only launch (model + context kept, projector gone).
        assert _detect(_GEMMA4_OLD_LLAMACPP_OUT) is True
        retry_cmd = _strip(_VISION_CMD)
        assert "--mmproj" not in retry_cmd
        assert "-m" in retry_cmd and "--jinja" in retry_cmd

    def test_oom_does_not_retry_text_only(self):
        # An OOM with --mmproj present must NOT be treated as a projector
        # problem: load_model errors out instead of dropping vision.
        assert _detect(_OOM_OUT) is False

    def test_bare_segfault_with_mmproj_yields_text_only_retry(self):
        # Field report: a -11 SIGSEGV on --mmproj has no projector line; the
        # signal path fires only when no other diagnostic explains the crash.
        out = ""  # a SIGSEGV produced no projector-format line
        assert _detect(out) is False
        should_retry = _detect(out) or (_signal_crash(-11) and not _nonproj(out))
        assert should_retry is True
        retry_cmd = _strip(_VISION_CMD)
        assert "--mmproj" not in retry_cmd and "-m" in retry_cmd

    def test_signal_crash_with_oom_output_keeps_the_real_error(self):
        # A hard fault that already printed an OOM must surface it, not silently
        # drop --mmproj and tell the user to update llama.cpp.
        assert _signal_crash(-6) is True
        should_retry = _detect(_OOM_OUT) or (
            _signal_crash(-6) and not _nonproj(_OOM_OUT)
        )
        assert should_retry is False

    def test_signal_crash_with_bad_arch_does_not_drop_vision(self):
        should_retry = _detect(_BAD_ARCH_OUT) or (
            _signal_crash(-6) and not _nonproj(_BAD_ARCH_OUT)
        )
        assert should_retry is False

    def test_clean_nonzero_exit_with_mmproj_does_not_retry(self):
        # Clean non-zero exit (bad path, port bind) is not a hard crash; stay message-based.
        assert (_detect(_MISSING_OUT) or _signal_crash(1)) is False

    def test_signal_crash_tries_flash_attn_off_before_dropping_vision(self):
        # Ladder order: a hard fault retries FA-off FIRST (keeps vision + MTP);
        # only if THAT is None/fails do we fall back to stripping --mmproj.
        assert _signal_crash(-11) is True
        fa_retry = _flash_off(_VISION_CMD)
        assert fa_retry is not None
        assert "--mmproj" in fa_retry  # vision preserved at this rung
        # Last resort still available and strictly more destructive.
        text_only = _strip(fa_retry)
        assert "--mmproj" not in text_only

    def test_external_kill_skips_flash_attn_retry(self):
        # SIGKILL (-9, OOM killer) is not a program fault: no FA-off retry.
        assert _signal_crash(-9) is False
