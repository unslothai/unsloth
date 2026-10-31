# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launch flags are gated on --help; the three always-emitted ones fail open on unreadable help."""

from __future__ import annotations

import ast
import inspect
import logging
import subprocess
import sys
import textwrap
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
from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _flash_attn_enabled_from_args,
)

GATED = (
    "supports_no_context_shift",
    "supports_jinja",
    "supports_flash_attn",
    "flash_attn_takes_value",
)

# llama.cpp help format: declaration at column 0, description padded to column 40
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
OLD_HELP = (
    "usage: llama-server [options]\n"
    "\n"
    "-m,    --model FNAME                    model path\n"
    "-fa,   --flash-attn                     enable Flash Attention (default: disabled)\n"
    "--no-context-shift                      disables context shift on infinite text\n"
    "                                        generation\n"
)
NO_FLASH_ATTN_HELP = (
    "usage: llama-server [options]\n"
    "\n"
    "-m,    --model FNAME                    model path\n"
    "-c,    --ctx-size N                     size of the prompt context\n"
    "--no-context-shift                      disables context shift on infinite text\n"
    "                                        generation\n"
)
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
    """Stubs subprocess instead of writing a script, since a shebang file is not executable on Windows."""
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
        """A partial listing from a failed probe is not evidence of absence, so no flag may be dropped."""
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
            ("supports_flash_attn", True),
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
        assert caps["supports_flash_attn"] is True
        assert caps["flash_attn_takes_value"] is False

    def test_a_build_without_flash_attention_drops_the_flag_itself(self, tmp_path, monkeypatch):
        """Builds older than b2775 have no flash attention flag, and emitting -fa exits as invalid
        argument."""
        caps = probe(tmp_path, monkeypatch, NO_FLASH_ATTN_HELP)
        assert caps["supports_flash_attn"] is False
        assert caps["supports_no_context_shift"] is True

    def test_the_missing_flag_still_fails_open_on_a_failed_probe(self, tmp_path, monkeypatch):
        caps = probe(tmp_path, monkeypatch, NO_FLASH_ATTN_HELP, returncode = 1)
        assert caps["supports_flash_attn"] is True

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

    def test_the_flag_and_its_value_are_gated_separately(self):
        """Two independent answers, so a build that has the flag but not the
        enum keeps flash attention, and one that has neither drops both."""
        src = inspect.getsource(LlamaCppBackend.load_model)
        assert '_caps.get("supports_flash_attn", True)' in src
        assert 'cmd.append("--flash-attn")' in src
        assert '_caps.get("flash_attn_takes_value", True)' in src

    def test_the_real_master_declaration_reads_as_the_value_form(self):
        assert LlamaCppBackend._flash_attn_takes_value(NEW_HELP) is True

    def test_the_real_pre_enum_declaration_reads_as_boolean(self):
        assert LlamaCppBackend._flash_attn_takes_value(OLD_HELP) is False


class TestTheCrashRecoveryMatchesTheEmittedForm:
    """llama.cpp matches argv tokens verbatim and never splits on =, so the retry drops a bare flag."""

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


def _flash_attn_env_scrub(*, known_off: bool) -> dict:
    """Run load_model's real inherited-flash-attention env scrub, and report the env."""
    source = textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model))
    blocks = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and "_drop_env_flash_attn"
        in {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}
        and "_flash_attn_known_off"
        in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
    ]
    assert len(blocks) == 1, f"expected one scrub block, found {len(blocks)}"
    scope = {
        "self": LlamaCppBackend,
        "logger": logging.getLogger(__name__),
        "_flash_attn_known_off": known_off,
        "env": {"LLAMA_ARG_FLASH_ATTN": "1", "LLAMA_ARG_CTX_SIZE": "4096"},
    }
    exec(ast.unparse(ast.Module(body = blocks, type_ignores = [])), scope)
    return scope["env"]


class TestAFlaglessBuildIgnoresTheFlashAttentionEnv:
    """Without the flag llama.cpp ignores LLAMA_ARG_FLASH_ATTN; recording it on under-sizes the V cache."""

    CMD = ["llama-server", "-m", "m.gguf", "--no-context-shift", "-c", "8192"]

    def test_the_inherited_value_is_dropped(self):
        env = _flash_attn_env_scrub(known_off = True)
        assert "LLAMA_ARG_FLASH_ATTN" not in env
        assert env == {"LLAMA_ARG_CTX_SIZE": "4096"}

    def test_the_recorded_state_then_matches_the_launch(self):
        env = _flash_attn_env_scrub(known_off = True)
        assert _flash_attn_enabled_from_args(self.CMD, default = False, env = env) is False

    @pytest.mark.parametrize("value", ["1", "on", "auto", "true"])
    def test_every_enabling_spelling_would_otherwise_win(self, value):
        """The unscrubbed env overrides the default on all of llama.cpp's truthy forms."""
        env = {"LLAMA_ARG_FLASH_ATTN": value}
        assert _flash_attn_enabled_from_args(self.CMD, default = False, env = env) is True

    def test_a_build_that_has_the_flag_keeps_the_inherited_value(self):
        """The scrub is scoped to the flagless build: everywhere else a deliberate
        LLAMA_ARG_FLASH_ATTN must reach llama-server untouched."""
        env = _flash_attn_env_scrub(known_off = False)
        assert env["LLAMA_ARG_FLASH_ATTN"] == "1"


class _SelfShim:
    """Stand-in self for the fixup block; unset attributes fall through to the class; methods bind to it."""

    def __init__(
        self,
        kv_lora_rank = None,
        architecture = None,
        mtp_draft_path = None,
    ):
        self._kv_lora_rank = kv_lora_rank
        self._architecture = architecture
        self._mtp_draft_path = mtp_draft_path

    def __getattr__(self, name):
        attr = getattr(LlamaCppBackend, name)
        # A staticmethod reached through the class is a plain function too, but it
        # takes no self, so only genuine instance methods get bound.
        declared = inspect.getattr_static(LlamaCppBackend, name, None)
        if inspect.isfunction(attr) and not isinstance(declared, staticmethod):
            return _types.MethodType(attr, self)
        return attr


def _flagless_v_cache_fixup(
    *,
    known_off: bool,
    cmd: list,
    env: dict,
    mla: bool = False,
) -> tuple:
    """Run load_model's real flagless-build V-cache fixup over cmd and env."""
    source = textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model))
    blocks = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}
        & {"_reset_quantized_v_cache", "_drop_env_quantized_v_cache"}
        and "_flash_attn_known_off"
        in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
    ]
    assert len(blocks) == 2, f"expected two V-cache fixup blocks, found {len(blocks)}"
    scope = {
        "self": _SelfShim(kv_lora_rank = 512 if mla else None),
        "logger": logging.getLogger(__name__),
        "_flash_attn_known_off": known_off,
        "cmd": list(cmd),
        "env": dict(env),
    }
    exec(ast.unparse(ast.Module(body = blocks, type_ignores = [])), scope)
    return scope["cmd"], scope["env"]


class TestAFlaglessBuildCannotRunAQuantizedVCache:
    """A quantized V cache also needs flash attention, so dropping the flag must reset V too."""

    CMD = [
        "llama-server",
        "-m",
        "m.gguf",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]

    def test_the_v_cache_is_reset_and_the_k_cache_is_not(self):
        cmd, _ = _flagless_v_cache_fixup(known_off = True, cmd = self.CMD, env = {})
        assert cmd[cmd.index("--cache-type-v") + 1] == "f16"
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"

    def test_the_recovery_rung_would_not_have_caught_it(self):
        assert LlamaCppBackend._with_flash_attn_off(self.CMD) is None

    def test_the_inherited_quantized_v_env_goes_too(self):
        _, env = _flagless_v_cache_fixup(
            known_off = True,
            cmd = self.CMD,
            env = {"LLAMA_ARG_CACHE_TYPE_V": "q8_0", "LLAMA_ARG_CACHE_TYPE_K": "q8_0"},
        )
        assert "LLAMA_ARG_CACHE_TYPE_V" not in env
        assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"

    def test_a_build_that_has_the_flag_keeps_its_quantized_v_cache(self):
        cmd, env = _flagless_v_cache_fixup(
            known_off = False,
            cmd = self.CMD,
            env = {"LLAMA_ARG_CACHE_TYPE_V": "q8_0"},
        )
        assert cmd == self.CMD
        assert env == {"LLAMA_ARG_CACHE_TYPE_V": "q8_0"}

    @pytest.mark.parametrize("value", ["f16", "bf16", "f32"])
    def test_an_unquantized_v_cache_is_left_alone(self, value):
        cmd = [c if c != "q8_0" else value for c in self.CMD]
        assert _flagless_v_cache_fixup(known_off = True, cmd = cmd, env = {})[0] == cmd

    def test_the_reset_lands_before_the_launch_is_logged(self):
        """Otherwise "Starting llama-server: ..." names a V cache type the child
        never runs, and the log is the only record of what was launched."""
        src = inspect.getsource(LlamaCppBackend.load_model)
        assert src.index("_reset_quantized_v_cache") < src.index("Starting llama-server")


class TestTheFlaglessFixupKeepsMlaKAndVEqual:
    """MLA rejects unequal K and V cache types, so the launch-site V reset must bring K down too."""

    CMD = [
        "llama-server",
        "-m",
        "ds.gguf",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]

    def test_mla_brings_k_down_with_v(self):
        cmd, _ = _flagless_v_cache_fixup(known_off = True, cmd = self.CMD, env = {}, mla = True)
        assert cmd[cmd.index("--cache-type-k") + 1] == "f16"
        assert cmd[cmd.index("--cache-type-v") + 1] == "f16"

    def test_a_non_mla_model_still_keeps_its_quantized_k(self):
        cmd, _ = _flagless_v_cache_fixup(known_off = True, cmd = self.CMD, env = {}, mla = False)
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
        assert cmd[cmd.index("--cache-type-v") + 1] == "f16"


TRACE_HELP = NEW_HELP + (
    "-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold. Messages with a higher verbosity will be\n"
    "                                        ignored. Values:\n"
    "                                         - 0: generic output\n"
    "                                         - 1: error\n"
    "                                         - 2: warning\n"
    "                                         - 3: info\n"
    "                                         - 4: trace (more info)\n"
    "                                         - 5: debug\n"
    "                                        (default: 3)\n"
)
PRE_TRACE_HELP = NEW_HELP + (
    "-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold. Messages with a higher verbosity will be\n"
    "                                        ignored.\n"
)


class TestTraceVerbosityRestoresTheLoadLog:
    """Load lines the offload report and classifiers read are trace-level on current builds."""

    def test_a_trace_build_is_detected(self, tmp_path, monkeypatch):
        assert probe(tmp_path, monkeypatch, TRACE_HELP)["supports_trace_verbosity"] is True

    @pytest.mark.parametrize(
        "help_text,returncode",
        [(PRE_TRACE_HELP, 0), (NEW_HELP, 0), (TRACE_HELP, 1), ("", 0)],
    )
    def test_anything_else_fails_closed(self, tmp_path, monkeypatch, help_text, returncode):
        caps = probe(tmp_path, monkeypatch, help_text, returncode)
        assert caps["supports_trace_verbosity"] is False

    def test_an_unprobeable_binary_fails_closed(self, tmp_path):
        caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "absent"))
        assert caps["supports_trace_verbosity"] is False

    def test_a_trace_build_gets_level_4(self):
        args = llama_cpp_module._trace_verbosity_args({"supports_trace_verbosity": True}, [], {})
        assert args == ["--verbosity", "4"]

    @pytest.mark.parametrize(
        "extra_args",
        [["-lv", "2"], ["--verbosity=5"], ["--log-verbosity", "1"], ["-v"], ["--log-disable"]],
    )
    def test_the_users_own_logging_choice_wins(self, extra_args):
        caps = {"supports_trace_verbosity": True}
        assert llama_cpp_module._trace_verbosity_args(caps, extra_args, {}) == []

    def test_the_env_form_wins_too(self):
        caps = {"supports_trace_verbosity": True}
        env = {"LLAMA_ARG_LOG_VERBOSITY": "3"}
        assert llama_cpp_module._trace_verbosity_args(caps, None, env) == []

    def test_other_builds_get_nothing(self):
        assert llama_cpp_module._trace_verbosity_args({}, None, {}) == []

    def test_the_launch_emits_it(self):
        src = inspect.getsource(LlamaCppBackend.load_model)
        assert "cmd.extend(_trace_verbosity_args(_caps, extra_args, os.environ))" in src


class TestTheStdoutBufferIsBounded:
    """Level 4 adds about 30 lines a request; the buffer lives as long as the server."""

    def _drain(self, lines):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._stdout_lines = []
        backend._process = _types.SimpleNamespace(stdout = iter(f"{l}\n" for l in lines))
        backend._drain_stdout()
        return backend._stdout_lines

    def test_the_startup_head_and_the_tail_survive(self):
        startup = [f"load {i}" for i in range(200)] + ["main: server is listening on 127.0.0.1"]
        requests = [f"request {i}" for i in range(60000)]
        kept = self._drain(startup + requests)
        assert len(kept) <= llama_cpp_module._STDOUT_TRIM_AT
        assert kept[: len(startup)] == startup
        assert kept[-1] == "request 59999"
        assert len(kept) >= len(startup) + llama_cpp_module._STDOUT_TAIL_KEEP

    def test_a_short_log_is_untouched(self):
        lines = [f"line {i}" for i in range(500)]
        assert self._drain(lines) == lines


def test_a_flood_before_readiness_keeps_the_startup_head():
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._stdout_lines = []
    lines = ["load_tensors: offloaded 13/37 layers to GPU"] + [f"trace {i}" for i in range(30000)]
    backend._process = _types.SimpleNamespace(stdout = iter(f"{l}\n" for l in lines))
    backend._drain_stdout()
    assert backend._stdout_lines[0] == lines[0]
    assert len(backend._stdout_lines) <= llama_cpp_module._STDOUT_TRIM_AT


def test_the_offload_report_covers_metal_and_env_pinned_layers():
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "(_detected_gpus or _metal_capable_host())" in src
    assert "_offload_counts[0] <= 0" in src
    start = src.index("self._offload_overridden = ")
    assignment = src[start : src.index("\n                if ", start)]
    assert "self._strip_device_extra_args(extra_args)" in assignment
    assert "if _gpu_ids_own_device_flags" in assignment
    assert "_env_fixes_gpu_layers(env)" in assignment


def test_the_active_log_file_stops_growing_past_its_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp_module, "_LLAMA_LOG_FULL_BYTES", 4096)
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._stdout_lines = []
    log = tmp_path / "llama.log"
    backend._llama_log_fh = open(log, "w", encoding = "utf-8")
    lines = [f"0.00.{i:06d} I slot launch: id 0 | task {i} | padding padding" for i in range(2000)]
    lines += ["0.09.000000 W srv  a warning after the cap", "\trepeat_last_n = 64", "error: boom"]
    backend._process = _types.SimpleNamespace(stdout = iter(f"{l}\n" for l in lines))
    backend._drain_stdout()
    backend._llama_log_fh.close()
    text = log.read_text(encoding = "utf-8")
    assert len(text) < 4096 + 512
    assert "keeping warnings and errors only" in text
    assert "a warning after the cap" in text and "error: boom" in text
    assert "repeat_last_n" not in text


def test_metal_device_rows_count_as_gpu():
    rows = [
        "device_info:",
        "  - MTL0    : Apple M3 Max (98304 MiB, 98303 MiB free)",
        "  - CPU     : Apple M3 Max",
    ]
    assert llama_cpp_module.llama_saw_gpu_device(rows) is True
    assert llama_cpp_module.llama_saw_gpu_device([rows[0], rows[2]]) is False
