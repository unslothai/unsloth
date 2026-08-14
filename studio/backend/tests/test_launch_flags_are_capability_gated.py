# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every launch flag is gated on the binary advertising it.

The probe checks 17 optional flags with _is_real before emitting them, but
--flash-attn, --no-context-shift and --jinja were emitted unconditionally. That
is fine for the pinned prebuilt, which has all three. It is not fine for a stale
or user-supplied LLAMA_SERVER_PATH, which Studio explicitly supports: an unknown
argument makes llama-server exit immediately rather than start degraded, and the
user sees a generic startup failure.

These three differ from the other 17 in one important way: they are part of
today's command on every launch, so their gates FAIL OPEN. An unreadable or
unparseable --help must keep emitting them; only a build whose help positively
lacks one may drop it.
"""

from __future__ import annotations

import inspect
import subprocess
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
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference import llama_cpp as llama_cpp_module  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

GATED = ("supports_no_context_shift", "supports_jinja", "flash_attn_takes_value")

# Help text as llama.cpp actually prints it: the declaration starts at column 0
# and its description is padded to column 40 (common_arg::to_string).
NEW_HELP = (
    "usage: llama-server [options]\n"
    "\n"
    "-m,    --model FNAME                    model path\n"
    "-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or\n"
    "                                        'auto', default: 'auto')\n"
    "--context-shift, --no-context-shift     whether to use context shift on infinite\n"
    "                                        text generation (default: disabled)\n"
    "--jinja, --no-jinja                     whether to use jinja template engine for\n"
    "                                        chat (default: disabled)\n"
)
# The pre-enum vintage: -fa is a bare boolean and there is no --jinja.
OLD_HELP = (
    "usage: llama-server [options]\n"
    "\n"
    "-m,    --model FNAME                    model path\n"
    "-fa,   --flash-attn                     enable Flash Attention (default: disabled)\n"
    "--no-context-shift                      disables context shift on infinite text\n"
    "                                        generation\n"
)
# What a wrapper prints before it gives up: real flags, but only some of them.
PARTIAL_HELP = (
    "usage: llama-server [options]\n\n-m,    --model FNAME                    model path\n"
)


def probe(
    tmp_path,
    monkeypatch,
    help_text,
    returncode = 0,
    stream = "stdout",
):
    """Run the real probe against a stubbed ``llama-server --help``.

    ``subprocess`` is stubbed rather than a script dropped on disk so the test
    means the same thing on Windows, where a shebang file is not executable.
    """
    binary = tmp_path / "llama-server"
    binary.write_text("")
    completed = subprocess.CompletedProcess(
        args = [str(binary), "--help"],
        returncode = returncode,
        stdout = help_text if stream == "stdout" else "",
        stderr = help_text if stream == "stderr" else "",
    )
    monkeypatch.setattr(llama_cpp_module.subprocess, "run", lambda *a, **k: completed)
    return LlamaCppBackend.probe_server_capabilities(str(binary))


class TestTheGatesFailOpen:
    """A failed or unreadable probe must not silently drop a flag."""

    @pytest.mark.parametrize(
        "label,help_text,returncode",
        [
            ("nothing at all", "", 1),
            ("a banner but no flags", "Segmentation fault\n", 2),
            ("a partial listing", PARTIAL_HELP, 1),
            ("a full listing", NEW_HELP, 1),
            ("an old-shaped partial listing", PARTIAL_HELP, 3),
        ],
    )
    @pytest.mark.parametrize("key", GATED)
    def test_a_nonzero_help_keeps_every_flag(
        self, tmp_path, monkeypatch, label, help_text, returncode, key
    ):
        """A partial listing is still a FAILED probe.

        The wrapper case that motivates this: it prints a few parseable options
        and then exits nonzero. ``blocks`` is non-empty, but nothing in it is
        authoritative, and reading it as "flag absent" would drop
        ``--no-context-shift`` (context silently rotates again) or ``--jinja``
        (no template rendering) on a launch the probe never understood.
        """
        assert probe(tmp_path, monkeypatch, help_text, returncode)[key] is True

    @pytest.mark.parametrize("key", GATED)
    def test_an_empty_help_does_not_read_as_absent(self, tmp_path, monkeypatch, key):
        """Exit 0 with no output means the probe told us nothing, not "missing"."""
        assert probe(tmp_path, monkeypatch, "", 0)[key] is True

    @pytest.mark.parametrize("key", GATED)
    def test_an_unprobeable_binary_keeps_every_flag(self, tmp_path, key):
        assert LlamaCppBackend.probe_server_capabilities(str(tmp_path / "absent"))[key] is True

    @pytest.mark.parametrize(
        "key,default",
        [
            ("supports_no_context_shift", True),
            ("supports_jinja", True),
            ("flash_attn_takes_value", True),
        ],
    )
    def test_the_emission_site_reads_the_key_with_a_true_default(self, key, default):
        src = inspect.getsource(LlamaCppBackend.load_model)
        assert f'_caps.get("{key}", {default})' in src


class TestASuccessfulProbeIsAuthoritative:
    """Exit 0 with a parseable listing is the one case that may drop a flag."""

    def test_a_current_binary_keeps_all_three(self, tmp_path, monkeypatch):
        caps = probe(tmp_path, monkeypatch, NEW_HELP)
        assert caps["supports_no_context_shift"] is True
        assert caps["supports_jinja"] is True
        assert caps["flash_attn_takes_value"] is True

    def test_an_older_binary_drops_what_it_lacks(self, tmp_path, monkeypatch):
        caps = probe(tmp_path, monkeypatch, OLD_HELP)
        assert caps["supports_no_context_shift"] is True
        assert caps["supports_jinja"] is False
        assert caps["flash_attn_takes_value"] is False

    def test_help_on_stderr_reads_the_same(self, tmp_path, monkeypatch):
        caps = probe(tmp_path, monkeypatch, NEW_HELP, stream = "stderr")
        assert caps["supports_jinja"] is True
        assert caps["flash_attn_takes_value"] is True

    def test_windows_line_endings_read_the_same(self, tmp_path, monkeypatch):
        caps = probe(tmp_path, monkeypatch, OLD_HELP.replace("\n", "\r\n"))
        assert caps["supports_no_context_shift"] is True
        assert caps["supports_jinja"] is False
        assert caps["flash_attn_takes_value"] is False


class TestFlashAttentionValueForm:
    """The value form is the part that actually breaks an older binary."""

    VALUE = "-fa, --flash-attn [on|off|auto]   set flash attention"
    BOOLEAN = "-fa, --flash-attn                 enable flash attention"

    def test_an_enum_means_the_value_is_accepted(self):
        assert LlamaCppBackend._flash_attn_takes_value(self.VALUE) is True

    def test_a_bare_boolean_declaration_drops_the_value(self):
        assert LlamaCppBackend._flash_attn_takes_value(self.BOOLEAN) is False

    @pytest.mark.parametrize("help_text", ["", "-m, --model FNAME", "unrelated output"])
    def test_it_fails_open_when_the_flag_is_not_mentioned(self, help_text):
        assert LlamaCppBackend._flash_attn_takes_value(help_text) is True

    def test_the_flag_itself_is_always_emitted(self):
        """Only the VALUE is conditional. Dropping -fa entirely would be a
        silent performance regression rather than a startup failure, which is
        the worse trade."""
        src = inspect.getsource(LlamaCppBackend.load_model)
        assert 'cmd.append("--flash-attn")' in src
        idx = src.find('cmd.append("--flash-attn")')
        assert "if _caps" not in src[max(0, idx - 200) : idx].split("\n")[-1]

    def test_the_real_master_declaration_reads_as_the_value_form(self):
        assert LlamaCppBackend._flash_attn_takes_value(NEW_HELP) is True

    def test_the_real_pre_enum_declaration_reads_as_boolean(self):
        assert LlamaCppBackend._flash_attn_takes_value(OLD_HELP) is False


class TestTheCrashRecoveryMatchesTheEmittedForm:
    """``_with_flash_attn_off`` has to speak the same dialect the launch does.

    llama.cpp looks every argv token up verbatim (only ``_`` becomes ``-``); it
    never splits on ``=``. So ``--flash-attn=off`` is "invalid argument" on every
    build, and the bare form only exists on builds whose flag takes no value at
    all. The retry therefore drops a bare flag instead of giving it a value.
    """

    @pytest.mark.parametrize("flag", ["--flash-attn", "-fa"])
    def test_a_bare_flag_is_dropped(self, flag):
        assert LlamaCppBackend._with_flash_attn_off(
            ["llama-server", "-m", "m.gguf", flag, "--no-context-shift"]
        ) == ["llama-server", "-m", "m.gguf", "--no-context-shift"]

    @pytest.mark.parametrize("value", ["on", "auto", "1", "true", "enabled"])
    def test_a_valued_flag_is_flipped_in_place(self, value):
        assert LlamaCppBackend._with_flash_attn_off(
            ["llama-server", "--flash-attn", value, "--jinja"]
        ) == ["llama-server", "--flash-attn", "off", "--jinja"]

    @pytest.mark.parametrize(
        "cmd",
        [
            ["llama-server", "-m", "m.gguf", "--flash-attn"],
            ["llama-server", "-m", "m.gguf", "-fa"],
            ["llama-server", "--flash-attn", "on"],
            ["llama-server", "-fa", "--flash-attn", "auto"],
        ],
    )
    def test_the_retry_never_carries_an_equals_form(self, cmd):
        retry = LlamaCppBackend._with_flash_attn_off(cmd)
        assert retry is not None
        assert not [t for t in retry if t.startswith("-") and "=" in t]

    @pytest.mark.parametrize(
        "cmd",
        [
            ["llama-server", "--flash-attn", "off"],
            ["llama-server", "-fa", "off"],
            ["llama-server", "-m", "m.gguf"],
        ],
    )
    def test_nothing_to_retry_stays_none(self, cmd):
        assert LlamaCppBackend._with_flash_attn_off(cmd) is None

    def test_the_env_goes_with_the_dropped_flag(self):
        """Dropping a bare flag is not enough on its own: llama.cpp applies
        LLAMA_ARG_FLASH_ATTN before argv, and argv can no longer say "off"."""
        env = {"LLAMA_ARG_FLASH_ATTN": "1", "LLAMA_ARG_CTX_SIZE": "4096"}
        assert LlamaCppBackend._drop_env_flash_attn(env) is True
        assert env == {"LLAMA_ARG_CTX_SIZE": "4096"}

    def test_dropping_an_absent_env_is_a_no_op(self):
        env = {"LLAMA_ARG_CTX_SIZE": "4096"}
        assert LlamaCppBackend._drop_env_flash_attn(env) is False
        assert env == {"LLAMA_ARG_CTX_SIZE": "4096"}
