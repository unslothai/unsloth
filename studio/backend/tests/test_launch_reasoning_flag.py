# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch-time thinking default goes out as --reasoning where the build has it.

llama-server prints "Setting 'enable_thinking' via --chat-template-kwargs is
deprecated. Use --reasoning on / --reasoning off instead." on every Studio launch
of a reasoning model (#7526). The warning is per KEY: only enable_thinking has a
flag of its own, and --chat-template-kwargs is still the only way to set an
arbitrary template variable, so only that key moves and every other key stays
where it was.

The gate is the flag catalogue from `llama-server --help`, and it is closed on
anything short of a positive answer, because a build that predates --reasoning
exits with "error: invalid argument: --reasoning" rather than starting without it.
The help text below is copied verbatim out of two real binaries: the bundled
prebuilt b10909, which has the flag, and the b6277 release tag, which has
--reasoning-format, --reasoning-budget and nothing else, and which really does
refuse to start when handed --reasoning.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _build_launch_reasoning_args,
)
from core.inference.llama_server_args import strip_shadowing_flags  # noqa: E402

# --- help text, verbatim from the two binaries ----------------------------------------------

# b10909 (the bundled prebuilt). Note the three name-alike flags around it: a
# substring match on "reasoning" would report the flag on a build that lacks it.
MODERN_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
--reasoning-format FORMAT               controls whether thought tags are allowed and/or extracted from the
                                        response, and in which format they're returned; one of:
                                        - none: leaves thoughts unparsed in `message.content`
-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ('on', 'off', or 'auto', default:
                                        'auto' (detect from template))
                                        (env: LLAMA_ARG_REASONING)
--reasoning-effort LEVEL                reasoning effort level given to the chat template: 'default' to keep
                                        the template default,
--reasoning-budget N                    token budget for thinking: -1 for unrestricted, 0 for immediate end,
                                        N>0 for token budget (default: -1)
                                        (env: LLAMA_ARG_THINK_BUDGET)
--reasoning-preserve, --no-reasoning-preserve
                                        preserve reasoning trace in the full history, not just the last
                                        assistant message (default: enabled)
--chat-template-kwargs STRING           sets additional params for the json template parser, must be a valid
                                        JSON object
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: disabled)
"""

# b6277. Same three name-alikes, no --reasoning.
OLD_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
--reasoning-format FORMAT               controls whether thought tags are allowed and/or extracted from the
                                        response, and in which format they're returned; one of:
                                        - none: leaves thoughts unparsed in `message.content`
--reasoning-budget N                    controls the amount of thinking allowed; currently only one of: -1 for
                                        unrestricted thinking budget, or 0 to disable thinking (default: -1)
                                        (env: LLAMA_ARG_THINK_BUDGET)
--chat-template-kwargs STRING           sets additional params for the json template parser
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: disabled)
"""

# The shape a future llama.cpp removal would take: the flag is still listed, with the
# description the rename machinery gives a dead name.
REMOVAL_STUB_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
-rea,  --reasoning [on|off|auto]        (argument has been removed)
--chat-template-kwargs STRING           sets additional params for the json template parser
"""

MODERN_CAPS = {"supports_reasoning_flag": True}
OLD_CAPS = {"supports_reasoning_flag": False}


def _probe(
    tmp_path,
    monkeypatch,
    help_text,
    returncode = 0,
    stream = "stdout",
):
    """The real probe against a stubbed ``llama-server --help``."""
    binary = tmp_path / "llama-server"
    binary.write_text("")
    completed = subprocess.CompletedProcess(
        args = [str(binary), "--help"],
        returncode = returncode,
        stdout = help_text if stream == "stdout" else "",
        stderr = help_text if stream == "stderr" else "",
    )
    monkeypatch.setattr(llama_cpp.subprocess, "run", lambda *a, **k: completed)
    LlamaCppBackend._capability_cache.clear()
    return LlamaCppBackend.probe_server_capabilities(str(binary))


class TestTheProbeReadsTheFlagAndNotItsNamesakes:
    def test_the_bundled_prebuilt_advertises_the_flag(self, tmp_path, monkeypatch):
        assert _probe(tmp_path, monkeypatch, MODERN_HELP)["supports_reasoning_flag"] is True

    def test_a_build_with_only_the_namesakes_does_not(self, tmp_path, monkeypatch):
        """--reasoning-format / -budget / -preserve all contain the string."""
        assert _probe(tmp_path, monkeypatch, OLD_HELP)["supports_reasoning_flag"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"help_text": "", "returncode": 0},
            {"help_text": "", "returncode": 1},
            {"help_text": "usage: llama-server [options]\n"},
            {"help_text": REMOVAL_STUB_HELP},
        ],
        ids = ["empty-help", "failed-probe", "usage-only", "removal-stub"],
    )
    def test_the_gate_is_closed_on_anything_but_a_positive_answer(
        self, tmp_path, monkeypatch, kwargs
    ):
        """Unlike --jinja, this gate must fail CLOSED: the flag aborts an old build."""
        assert _probe(tmp_path, monkeypatch, **kwargs)["supports_reasoning_flag"] is False

    def test_a_parsed_catalogue_counts_even_on_a_nonzero_exit(self, tmp_path, monkeypatch):
        """Same policy as the sibling flags, rather than a third one.

        A build that prints its whole help and then exits nonzero has still named
        its flags, and --reasoning-budget is read off that catalogue today. The
        capabilities carry the doubt separately, in the *_probe_inconclusive keys.
        """
        caps = _probe(tmp_path, monkeypatch, MODERN_HELP, returncode = 1)
        assert caps["supports_reasoning_flag"] == caps["supports_reasoning_budget"] is True
        assert caps["reasoning_budget_probe_inconclusive"] is True

    def test_a_missing_binary_reports_it_unsupported(self, tmp_path):
        caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "nope"))
        assert caps["supports_reasoning_flag"] is False

    def test_the_flag_is_read_from_stderr_too(self, tmp_path, monkeypatch):
        caps = _probe(tmp_path, monkeypatch, MODERN_HELP, stream = "stderr")
        assert caps["supports_reasoning_flag"] is True


class TestOnlyTheDeprecatedKeyMoves:
    """The kwargs channel keeps carrying every key that has no flag of its own."""

    def test_thinking_on_becomes_the_flag_and_nothing_else_is_emitted(self):
        assert _build_launch_reasoning_args(MODERN_CAPS, {"enable_thinking": True}) == [
            "--reasoning",
            "on",
        ]

    def test_thinking_off_becomes_the_flag(self):
        assert _build_launch_reasoning_args(MODERN_CAPS, {"enable_thinking": False}) == [
            "--reasoning",
            "off",
        ]

    @pytest.mark.parametrize("preserve", [True, False])
    def test_preserve_thinking_stays_in_the_kwargs(self, preserve):
        """It is a template variable with no flag: #9096's default has to survive."""
        args = _build_launch_reasoning_args(
            MODERN_CAPS, {"enable_thinking": True, "preserve_thinking": preserve}
        )
        assert args[:2] == ["--reasoning", "on"]
        assert args[2] == "--chat-template-kwargs"
        assert json.loads(args[3]) == {"preserve_thinking": preserve}

    def test_a_reasoning_effort_ladder_is_untouched(self):
        """gpt-oss style never sets enable_thinking, so nothing is deprecated there."""
        kwargs = {"reasoning_effort": "high"}
        assert _build_launch_reasoning_args(MODERN_CAPS, kwargs) == [
            "--chat-template-kwargs",
            json.dumps(kwargs),
        ]

    def test_an_unrelated_future_key_is_untouched(self):
        kwargs = {"enable_thinking": False, "some_new_template_variable": "x"}
        args = _build_launch_reasoning_args(MODERN_CAPS, kwargs)
        assert args[:2] == ["--reasoning", "off"]
        assert json.loads(args[3]) == {"some_new_template_variable": "x"}

    def test_the_caller_s_dict_is_not_mutated(self):
        kwargs = {"enable_thinking": True}
        _build_launch_reasoning_args(MODERN_CAPS, kwargs)
        assert kwargs == {"enable_thinking": True}

    @pytest.mark.parametrize(
        "caps",
        [OLD_CAPS, {}, {"supports_reasoning_flag": None}],
        ids = ["old-build", "nothing-known", "unset"],
    )
    def test_without_the_flag_the_argv_is_what_main_emits(self, caps):
        kwargs = {"enable_thinking": True, "preserve_thinking": False}
        assert _build_launch_reasoning_args(caps, kwargs) == [
            "--chat-template-kwargs",
            json.dumps(kwargs),
        ]


# --- the real launch, per host ---------------------------------------------------------------

# enable_thinking style, with preserve_thinking so both channels are exercised at once.
THINKING_TEMPLATE = (
    "{% if enable_thinking %}<think>\n{% endif %}"
    "{% if preserve_thinking %}{{ messages[0].reasoning_content }}{% endif %}"
    "{{ messages[0].content }}"
)
# gpt-oss style: a ladder, no on/off gate, nothing deprecated.
EFFORT_TEMPLATE = "{{ reasoning_effort }}{% if reasoning_effort == 'high' %}<think>{% endif %}"

OSES = {
    "linux": ("linux", False),
    "windows": ("win32", False),
    "wsl": ("linux", True),
    "macos": ("darwin", False),
}
VENDORS = ("nvidia", "amd", "cpu")


def _fake_torch(*, rocm: bool, device_count: int) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+rocm6.2" if rocm else "2.11.0+cu130"
    version = types.ModuleType("torch.version")
    version.hip = "6.2.0" if rocm else None
    version.cuda = None if rocm else "13.0"
    torch.version = version
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: device_count > 0,
        device_count = lambda: device_count,
    )
    return torch


def _argv_for(tmp_path, monkeypatch, *, os_label, vendor, modern, template):
    """The argv the real ``load_model`` builds on one host, for one vintage."""
    platform, is_wsl = OSES[os_label]
    monkeypatch.setattr(llama_cpp.sys, "platform", platform, raising = False)
    monkeypatch.setattr(llama_cpp, "_is_wsl", lambda: is_wsl, raising = False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(rocm = vendor == "amd", device_count = 0 if vendor == "cpu" else 1),
    )
    memory = [] if vendor == "cpu" else [(0, 24_000, 24_000)]
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(MODERN_CAPS if modern else OLD_CAPS)),
    )
    captured = _launch(
        backend,
        gguf,
        n_ctx = 4096,
        chat_template_override = template,
    )
    return captured["cmd"]


def _reasoning_slice(cmd: list[str]) -> list[str]:
    """Just the reasoning arguments, in order, out of the whole command line."""
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] in {"--reasoning", "--chat-template-kwargs"}:
            out.extend(cmd[i : i + 2])
            i += 2
            continue
        i += 1
    return out


_HOSTS = [(o, v) for o in OSES for v in VENDORS]


class TestTheRealLaunchOnEveryHost:
    """The reasoning arguments are invariant under OS and device.

    They are appended into the same command the placement code then adds -ngl,
    --device and the split mode to, so this drives the real ``load_model`` on each
    host rather than the helper alone.
    """

    @pytest.mark.parametrize("os_label,vendor", _HOSTS)
    def test_a_modern_build_gets_the_flag_and_keeps_the_preserve_kwarg(
        self, tmp_path, monkeypatch, os_label, vendor
    ):
        cmd = _argv_for(
            tmp_path,
            monkeypatch,
            os_label = os_label,
            vendor = vendor,
            modern = True,
            template = THINKING_TEMPLATE,
        )
        assert _reasoning_slice(cmd) == [
            "--reasoning",
            "on",
            "--chat-template-kwargs",
            '{"preserve_thinking": false}',
        ]
        assert cmd.count("--reasoning") == 1

    @pytest.mark.parametrize("os_label,vendor", _HOSTS)
    def test_an_old_build_never_sees_the_flag(self, tmp_path, monkeypatch, os_label, vendor):
        cmd = _argv_for(
            tmp_path,
            monkeypatch,
            os_label = os_label,
            vendor = vendor,
            modern = False,
            template = THINKING_TEMPLATE,
        )
        assert "--reasoning" not in cmd
        assert _reasoning_slice(cmd) == [
            "--chat-template-kwargs",
            '{"enable_thinking": true, "preserve_thinking": false}',
        ]

    @pytest.mark.parametrize("modern", [True, False], ids = ["modern", "old"])
    def test_an_effort_ladder_model_is_unchanged_on_both_vintages(
        self, tmp_path, monkeypatch, modern
    ):
        cmd = _argv_for(
            tmp_path,
            monkeypatch,
            os_label = "linux",
            vendor = "nvidia",
            modern = modern,
            template = EFFORT_TEMPLATE,
        )
        assert "--reasoning" not in cmd
        assert _reasoning_slice(cmd)[0] == "--chat-template-kwargs"
        assert "reasoning_effort" in _reasoning_slice(cmd)[1]


class TestInheritedExtrasTreatBothSpellingsAlike:
    """Applying a chat template override recomputes the reasoning default.

    An inherited copy of the launch flag is appended after ours and would
    last-wins-override it, which is why --chat-template-kwargs has always been in
    the template group. Now that --reasoning carries the same intent, the outcome
    must not depend on which spelling the user typed.
    """

    @pytest.mark.parametrize(
        "args",
        [
            ["--reasoning", "off"],
            ["--reasoning=off"],
            ["-rea", "off"],
            ["--chat-template-kwargs", '{"enable_thinking": false}'],
        ],
    )
    def test_a_template_override_strips_the_inherited_default(self, args):
        assert strip_shadowing_flags([*args, "--threads", "8"], strip_template = True) == [
            "--threads",
            "8",
        ]

    @pytest.mark.parametrize(
        "args",
        [["--reasoning", "off"], ["--chat-template-kwargs", '{"enable_thinking": false}']],
    )
    def test_without_a_template_override_both_survive(self, args):
        assert strip_shadowing_flags([*args], strip_template = False) == args
