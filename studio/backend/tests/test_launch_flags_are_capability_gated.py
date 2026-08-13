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

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

GATED = ("supports_no_context_shift", "supports_jinja", "flash_attn_takes_value")


class TestTheGatesFailOpen:
    """A failed or unreadable probe must not silently drop a flag."""

    @pytest.mark.parametrize("key", GATED)
    def test_the_unprobeable_binary_dict_defaults_true(self, key):
        src = inspect.getsource(LlamaCppBackend.probe_server_capabilities)
        assert f'"{key}": True' in src

    @pytest.mark.parametrize("key", GATED)
    def test_the_pre_probe_initialiser_defaults_true(self, key):
        src = inspect.getsource(LlamaCppBackend.probe_server_capabilities)
        var = key if key != "flash_attn_takes_value" else "flash_attn_takes_value"
        assert f"{var} = True" in src

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

    def test_an_empty_help_does_not_read_as_absent(self):
        """`blocks` empty means the probe told us nothing, not "flag missing"."""
        src = inspect.getsource(LlamaCppBackend.probe_server_capabilities)
        assert "if blocks:" in src
        gated_block = src[src.find("if blocks:") : src.find("if blocks:") + 400]
        for key in ("--no-context-shift", "--jinja"):
            assert key in gated_block


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
