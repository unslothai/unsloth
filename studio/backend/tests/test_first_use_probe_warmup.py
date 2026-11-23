# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-use costs moved off the request path: the llama-server build probe and the date probe."""

from __future__ import annotations

import subprocess
import sys
import threading
import types
from datetime import date
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import main  # noqa: E402
import utils.current_date_prompt_settings as cdps  # noqa: E402
import utils.llama_cpp_freshness as freshness  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


class _FakeApp:
    class _State:
        pass

    def __init__(self) -> None:
        self.state = _FakeApp._State()
        self.state.llama_cpp_capabilities = None
        self.state.llama_cpp_freshness = None


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"v1")
    version_calls: list[str] = []
    real_run = subprocess.run

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and list(cmd[1:]) == ["--version"]:
            version_calls.append(cmd[0])
            return types.SimpleNamespace(
                stdout = "version: 1.0.0-dev (build 12631, commit 59f66204f)\n",
                stderr = "",
                returncode = 0,
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(LlamaCppBackend, "_build_number_cache", {})
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: str(binary))
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda _b: {"found": True, "supports_mtp": True}),
    )
    monkeypatch.setattr(freshness, "check_prebuilt_freshness", lambda _b, **_kw: {"stale": False})
    monkeypatch.delenv("UNSLOTH_DISABLE_UPDATE_CHECK", raising = False)
    return binary, version_calls


class TestStartupBuildProbe:
    def test_startup_probe_fills_the_cache_the_first_load_reads(self, fake_server):
        binary, version_calls = fake_server
        main._run_llama_cpp_startup_probes(_FakeApp())
        assert len(version_calls) == 1
        assert LlamaCppBackend._build_number_cache, "startup probe left the build cache empty"

        assert LlamaCppBackend.reserves_micro_batch_outputs(str(binary)) is False
        assert LlamaCppBackend.resumes_thoughts(str(binary)) is True
        assert LlamaCppBackend.reserves_micro_batch_outputs() is False
        assert LlamaCppBackend.probe_build_number(str(binary)) == 12631
        assert len(version_calls) == 1, "the first load spawned --version again"

    def test_a_replaced_binary_is_probed_again(self, fake_server):
        binary, version_calls = fake_server
        main._run_llama_cpp_startup_probes(_FakeApp())
        binary.write_bytes(b"updated build")
        assert LlamaCppBackend.probe_build_number(str(binary)) == 12631
        assert len(version_calls) == 2

    def test_no_binary_spawns_nothing(self, fake_server, monkeypatch):
        _, version_calls = fake_server
        monkeypatch.setattr(
            LlamaCppBackend,
            "probe_server_capabilities",
            staticmethod(lambda _b: {"found": False}),
        )
        main._run_llama_cpp_startup_probes(_FakeApp())
        assert version_calls == []

    def test_a_failing_build_probe_keeps_the_freshness_check(self, fake_server, monkeypatch):
        def _boom(cls, binary = None):
            raise RuntimeError("probe failed")

        monkeypatch.setattr(LlamaCppBackend, "probe_build_number", classmethod(_boom))
        app = _FakeApp()
        main._run_llama_cpp_startup_probes(app)
        assert app.state.llama_cpp_freshness == {"stale": False}

    def test_disabled_probes_skip_the_build_probe(self, fake_server, monkeypatch):
        _, version_calls = fake_server
        monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
        before = set(threading.enumerate())
        main._start_llama_cpp_probes_if_enabled(_FakeApp())
        started = [
            t
            for t in threading.enumerate()
            if t not in before and t.name == "llama-cpp-startup-probe"
        ]
        assert started == []
        assert version_calls == []
        assert LlamaCppBackend._build_number_cache == {}


_TODAY = date(2026, 11, 22)
# Shapes of real defaults: Qwen2.5's default system, Llama 3's dated header, GLM's fixed prefix.
_TEMPLATES = {
    "qwen25_default": (
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
        "{%- else %}{{- '<|im_start|>system\\nYou are Qwen, a helpful assistant.<|im_end|>\\n' }}"
        "{%- endif %}{%- for m in messages %}{%- if m.role != 'system' %}"
        "{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif %}"
        "{%- endfor %}{%- if tools %}{{ tools | tojson }}{%- endif %}"
    ),
    "llama3_dated": (
        "{{ bos_token }}{%- set date_string = strftime_now('%d %b %Y') %}"
        "{%- if messages[0]['role'] == 'system' %}{%- set sys = messages[0]['content'] %}"
        "{%- set messages = messages[1:] %}{%- else %}{%- set sys = '' %}{%- endif %}"
        "<|start_header_id|>system<|end_header_id|>\n\nToday Date: {{ date_string }}\n\n"
        "{{ sys }}<|eot_id|>{%- for m in messages %}<|start_header_id|>{{ m.role }}"
        "<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{%- endfor %}"
    ),
    "fixed_prefix": (
        "[gMASK]<sop>{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}{% endfor %}"
    ),
    "parts_only": "{% for m in messages %}{{ m.content[0].text }}\n{% endfor %}",
    "raises_on_system": (
        "{% for m in messages %}{% if m.role == 'system' %}"
        "{{ raise_exception('no system role') }}{% endif %}{{ m.content }}{% endfor %}"
    ),
    "carries_token": (
        "{% if messages[0].role != 'system' %}{{ eos_token }}default{% endif %}"
        "{% for m in messages %}{{ m.content }}{% endfor %}"
    ),
    "generation_block": (
        "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}"
        "{% if loop.last %}{% break %}{% endif %}{% endfor %}"
    ),
    "control_kwarg": (
        "{% if enable_thinking is defined and not enable_thinking %}/no_think{% endif %}"
        "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
    ),
    "broken_syntax": "{% for m in messages %}{{ m.content }",
    "undefined_call": "{{ missing_helper() }}",
}


class _CompileEveryRender:
    """The previous renderer: a fresh environment and compile for every render."""

    def __init__(self, chat_template, today):
        self._chat_template = chat_template
        self._today = today

    def __call__(
        self,
        messages,
        tools = None,
        controls = None,
    ):
        return cdps._compile_probe(self._chat_template, self._today).render(
            messages = messages,
            tools = tools,
            documents = None,
            add_generation_prompt = False,
            **cdps._PROBE_SPECIAL_TOKENS,
            **(controls or {}),
        )


def _probe(template, tools, controls):
    return cdps.template_system_turn.__wrapped__(template, _TODAY, tools, controls)


class TestDateProbeCompilesOnce:
    @pytest.mark.parametrize("name", sorted(_TEMPLATES))
    @pytest.mark.parametrize("tools", [False, True])
    @pytest.mark.parametrize("controls", [(), (("enable_thinking", False),)])
    def test_result_matches_compiling_on_every_render(self, monkeypatch, name, tools, controls):
        template = _TEMPLATES[name]
        head = _probe(template, tools, controls)
        monkeypatch.setattr(cdps, "_ProbeRenderer", _CompileEveryRender)
        assert _probe(template, tools, controls) == head

    def test_known_results(self):
        assert _probe(_TEMPLATES["qwen25_default"], False, ()) == (
            True,
            "You are Qwen, a helpful assistant.",
        )
        assert _probe(_TEMPLATES["raises_on_system"], False, ()) == (False, None)
        assert _probe(_TEMPLATES["broken_syntax"], False, ()) == (True, "")
        assert _probe(_TEMPLATES["carries_token"], False, ()) == (True, None)

    @pytest.mark.parametrize("name", sorted(_TEMPLATES))
    def test_one_compile_per_probe(self, monkeypatch, name):
        compiles: list[str] = []
        real = cdps._compile_probe

        def _counting(chat_template, today):
            compiles.append(chat_template)
            return real(chat_template, today)

        monkeypatch.setattr(cdps, "_compile_probe", _counting)
        _probe(_TEMPLATES[name], True, ())
        assert len(compiles) == 1

    def test_a_compile_error_is_raised_on_every_render(self):
        from jinja2.exceptions import TemplateSyntaxError
        render = cdps._ProbeRenderer(_TEMPLATES["broken_syntax"], _TODAY)
        for _ in range(3):
            with pytest.raises(TemplateSyntaxError):
                render([])
