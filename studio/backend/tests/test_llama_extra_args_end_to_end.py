# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the child process actually receives when the editor sends extra args.

The unit tests either side of this one pin the validator and the row's copy. This
one pins the thing both exist to produce: the argv llama-server is launched with.
It reuses the placement suite's harness, which runs the real ``load_model`` and
captures the command at the Popen boundary instead of spawning anything.

The bar the whole feature is measured against is the first test here: with nothing
in the box, the command must be byte-identical to the one Unsloth emitted before.
"""

from __future__ import annotations

import pytest

# The harness (module stubs, a fake GGUF, a fake GPU probe, the captured Popen)
# already exists; importing it keeps one copy of that setup. By path, because the
# tests directory is not a package and rootdir-relative imports do not resolve.
import importlib.util as _importlib_util
from pathlib import Path as _Path

_PLACEMENT_PATH = _Path(__file__).resolve().parent / "test_llama_cpp_placement.py"
_spec = _importlib_util.spec_from_file_location("_placement_harness", _PLACEMENT_PATH)
_placement = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_placement)
_backend = _placement._backend
_launch = _placement._launch


def _cmd(tmp_path, **load_kwargs):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    return _launch(backend, gguf, **load_kwargs)["cmd"]


def _stable(cmd: list[str]) -> list[str]:
    """The command with the two values that differ per launch masked.

    The port is picked from whatever is free and the GGUF lives under a per-test
    tmp dir, so a literal comparison could never hold. Everything else is the
    part this feature must not move.
    """
    masked = list(cmd)
    for index, token in enumerate(masked):
        if index and masked[index - 1] == "--port":
            masked[index] = "<port>"
        elif index and masked[index - 1] in {"-m", "--model"}:
            masked[index] = "<model>"
    return masked


def test_no_extra_args_leaves_the_command_unchanged(tmp_path):
    # The acceptance bar: someone who never opens the box must see exactly what
    # they saw before. None (inherit) and [] (explicitly none) both mean that here.
    baseline = _stable(_cmd(tmp_path))
    assert _stable(_cmd(tmp_path, extra_args = None)) == baseline
    assert _stable(_cmd(tmp_path, extra_args = [])) == baseline
    # And the masking is not hiding the whole command.
    assert "--flash-attn" in baseline and "<port>" in baseline


def test_an_extra_arg_reaches_the_child(tmp_path):
    cmd = _cmd(tmp_path, extra_args = ["--top-k", "20"])

    assert "--top-k" in cmd
    assert cmd[cmd.index("--top-k") + 1] == "20"


def test_extra_args_come_after_unsloths_own(tmp_path):
    # This is what makes the row's "passed after the settings above" true, and it
    # is llama.cpp's last-wins parsing that turns position into precedence.
    cmd = _cmd(tmp_path, extra_args = ["--top-k", "20"])
    managed = [i for i, token in enumerate(cmd) if token in {"--model", "-m", "--port"}]

    assert managed, "expected Unsloth's own flags in the command"
    assert cmd.index("--top-k") > max(managed)


def test_a_shadowing_value_wins_by_arriving_last(tmp_path):
    # --ctx-size is Context Length's flag. The backend deliberately allows the
    # shadow and reconciles its own sizing, so the user's value has to be the one
    # llama.cpp reads: last.
    cmd = _cmd(tmp_path, extra_args = ["--ctx-size", "4096"])
    positions = [i for i, token in enumerate(cmd) if token in {"--ctx-size", "-c"}]

    assert positions, "expected a context flag in the command"
    assert cmd[max(positions) + 1] == "4096"


def test_a_multi_token_value_stays_one_argv_entry(tmp_path):
    # The tokeniser's whole purpose: a template with spaces must not become two
    # arguments, which is what would happen if the string were split by the shell
    # or by the backend.
    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    cmd = _cmd(tmp_path, extra_args = ["--chat-template", template])

    assert cmd[cmd.index("--chat-template") + 1] == template


@pytest.mark.parametrize(
    "extra_args,family",
    [
        pytest.param(["--cache-type-k", "q4_0"], {"-ctk", "--cache-type-k"}, id = "cache-k"),
        pytest.param(["-ctk", "q4_0"], {"-ctk", "--cache-type-k"}, id = "cache-k-short"),
        pytest.param(["--cache-type-v", "q4_0"], {"-ctv", "--cache-type-v"}, id = "cache-v"),
        pytest.param(["-ctv", "q4_0"], {"-ctv", "--cache-type-v"}, id = "cache-v-short"),
        pytest.param(["--flash-attn", "off"], {"-fa", "--flash-attn"}, id = "flash-off"),
        pytest.param(["-fa", "off"], {"-fa", "--flash-attn"}, id = "flash-short"),
        pytest.param(
            ["--context-shift"],
            {"--context-shift", "--no-context-shift"},
            id = "context-shift",
        ),
        pytest.param(
            ["--no-context-shift"],
            {"--context-shift", "--no-context-shift"},
            id = "no-context-shift",
        ),
        pytest.param(["--jinja"], {"--jinja", "--no-jinja"}, id = "jinja"),
        pytest.param(["--no-jinja"], {"--jinja", "--no-jinja"}, id = "no-jinja"),
    ],
)
def test_an_explicit_semantic_override_suppresses_the_automatic_atom(tmp_path, extra_args, family):
    cmd = _cmd(tmp_path, cache_type_kv = "q8_0", extra_args = extra_args)

    assert cmd[-len(extra_args) :] == extra_args
    assert family.isdisjoint(cmd[: -len(extra_args)])


@pytest.mark.parametrize("extra_arg", ["--metrics", "--no-metrics"])
def test_a_supported_metrics_override_suppresses_the_automatic_flag(tmp_path, extra_arg):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    caps = backend.probe_server_capabilities()
    caps["supports_metrics"] = True
    backend.probe_server_capabilities = lambda _binary = None: caps

    cmd = _launch(backend, gguf, extra_args = [extra_arg])["cmd"]

    assert cmd[-1] == extra_arg
    assert not {"--metrics", "--no-metrics"}.intersection(cmd[:-1])


@pytest.mark.parametrize(
    "extra_args,other_axis",
    [
        pytest.param(["-ctk", "q4_0"], "--cache-type-v", id = "k-keeps-v"),
        pytest.param(["-ctv", "q4_0"], "--cache-type-k", id = "v-keeps-k"),
    ],
)
def test_cache_axis_overrides_are_independent(tmp_path, extra_args, other_axis):
    cmd = _cmd(tmp_path, cache_type_kv = "q8_0", extra_args = extra_args)

    assert other_axis in cmd[: -len(extra_args)]
    assert cmd[-len(extra_args) :] == extra_args


def test_a_supported_flash_override_is_not_treated_as_a_missing_flag(tmp_path):
    extra_args = ["-fa", "on"]
    cmd = _cmd(tmp_path, cache_type_kv = "q8_0", extra_args = extra_args)

    cache_v = cmd.index("--cache-type-v")
    assert cmd[cache_v + 1] == "q8_0"
    assert cmd[-len(extra_args) :] == extra_args


def test_unknown_and_repeated_extra_args_remain_verbatim_and_last(tmp_path):
    extra_args = [
        "--future-flag",
        "alpha",
        "--metrics",
        "--future-flag",
        "omega",
        "--no-metrics",
    ]
    cmd = _cmd(tmp_path, extra_args = extra_args)

    assert cmd[-len(extra_args) :] == extra_args
    assert not {"--metrics", "--no-metrics"}.intersection(cmd[: -len(extra_args)])


@pytest.mark.parametrize(
    "denied",
    [
        ["--agent"],
        ["--tools-runtime", "docker:x"],
        ["--mcp-servers-json", "{}"],
        ["--parallel", "8"],
        ["--api-key", "secret"],
    ],
)
def test_a_denied_flag_never_reaches_the_child(tmp_path, denied):
    # The load has to refuse rather than launch and hope: llama-server would
    # happily honour any of these.
    with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
        from core.inference.llama_server_args import validate_extra_args
        validate_extra_args(denied)


def test_the_denied_env_twin_does_not_survive_the_launch(tmp_path, monkeypatch):
    # Denying the token is only half of it: llama.cpp reads LLAMA_ARG_* before argv,
    # so an inherited value would reach the child with nothing in the command to
    # show for it.
    monkeypatch.setenv("LLAMA_ARG_AGENT", "1")
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "all")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    env = _launch(backend, gguf)["env"]

    assert "LLAMA_ARG_AGENT" not in env
    assert "LLAMA_ARG_TOOLS" not in env
