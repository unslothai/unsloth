# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server's GPU sampler (--backend-sampling) on the launches where it pays off.

Without it every decode step copies each slot's full-vocabulary logits to the host and
runs the sampler chain on the CPU. Measured on a Colab L4 with 4 slots (-c 8192
--kv-unified, Studio's own sampler chains): Qwen3.5-0.8B 20.3 -> 7.0 ms per token at
4 streams, Qwen3-0.6B 7.3 -> 6.1 ms, flood throughput up to 2.2x, for ~20-25 ms more TTFT
per request. With identical logits and seed it draws the same tokens as the CPU chain. It is per request and steps aside on
its own for grammar / tools, a reasoning budget and pre-sampling logprobs.

The gate: the binary must advertise the flag (an unknown argument stops an older
llama-server from starting), a GPU must be there and hold the output layer, the backend
must be one it was measured on (not Vulkan, not macOS), and no speculative decoding may
run (penalties cannot run on the GPU when a step verifies several tokens per slot). The
user's own flag and UNSLOTH_LLAMA_BACKEND_SAMPLING=0 win.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch, _write_gguf  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _backend_sampling_flags,
    _cpu_sampling_for_min_p,
    _launch_backend_sampling_min_p,
    _launch_speculates,
)

# As llama.cpp #269 (18b4ec5f1) prints it: common/arg.cpp:2347.
NEW_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
-bs,   --backend-sampling               enable backend sampling (experimental) (default: disabled)
                                        (env: LLAMA_ARG_BACKEND_SAMPLING)
--spec-draft-backend-sampling, --no-spec-draft-backend-sampling
                                        offload draft sampling to the backend (default: enabled)
--kv-unified, -kvu                      use single unified KV buffer shared across all sequences
"""
# A build from before backend sampling; the draft-only namesake must not count.
OLD_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
--spec-draft-backend-sampling, --no-spec-draft-backend-sampling
                                        offload draft sampling to the backend (default: enabled)
--kv-unified, -kvu                      use single unified KV buffer shared across all sequences
"""
REMOVED_HELP = """usage: llama-server [options]

-bs,   --backend-sampling               (argument has been removed)
"""

ON = {"supports_backend_sampling": True}
GPU = {"gpus_detected": True, "vulkan": False, "platform": "linux", "env": {}, "spec_env": {}}


def _probe(
    tmp_path,
    monkeypatch,
    help_text,
    returncode = 0,
):
    binary = tmp_path / "llama-server"
    binary.write_text("")
    completed = subprocess.CompletedProcess(
        args = [str(binary), "--help"], returncode = returncode, stdout = help_text, stderr = ""
    )
    monkeypatch.setattr(llama_cpp.subprocess, "run", lambda *a, **k: completed)
    LlamaCppBackend._capability_cache.clear()
    return LlamaCppBackend.probe_server_capabilities(str(binary))


class TestTheProbe:
    def test_a_build_with_the_flag_advertises_it(self, tmp_path, monkeypatch):
        assert _probe(tmp_path, monkeypatch, NEW_HELP)["supports_backend_sampling"] is True

    @pytest.mark.parametrize(
        "help_text,returncode",
        [(OLD_HELP, 0), (REMOVED_HELP, 0), ("", 0), ("", 1)],
        ids = ["old-build", "removal-stub", "empty-help", "failed-probe"],
    )
    def test_the_gate_is_closed_on_anything_but_a_positive_answer(
        self, tmp_path, monkeypatch, help_text, returncode
    ):
        caps = _probe(tmp_path, monkeypatch, help_text, returncode)
        assert caps["supports_backend_sampling"] is False

    def test_a_missing_binary_reports_it_unsupported(self, tmp_path):
        LlamaCppBackend._capability_cache.clear()
        caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "nope"))
        assert caps["supports_backend_sampling"] is False


class TestTheGate:
    BASE = ["/x/llama-server", "-m", "m.gguf", "--parallel", "4", "-ngl", "-1"]

    def test_a_plain_gpu_launch_gets_it(self):
        assert _backend_sampling_flags(ON, self.BASE, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize(
        "caps", [{}, {"supports_backend_sampling": False}], ids = ["unknown", "old"]
    )
    def test_an_older_build_gets_nothing(self, caps):
        assert _backend_sampling_flags(caps, self.BASE, **GPU) == []

    @pytest.mark.parametrize(
        "kw",
        [{"gpus_detected": False}, {"vulkan": True}, {"cuda_build": False}, {"platform": "darwin"}],
        ids = ["no-gpu", "vulkan", "rocm-or-other", "macos"],
    )
    def test_unmeasured_or_cpu_backends_get_nothing(self, kw):
        assert _backend_sampling_flags(ON, self.BASE, **{**GPU, **kw}) == []

    @pytest.mark.parametrize(
        "tail",
        [["-ngl", "0"], ["--gpu-layers", "0"], ["--n-gpu-layers=0"], ["--device", "none"]],
        ids = ["ngl0", "gpu-layers0", "inline", "device-none"],
    )
    def test_a_cpu_placed_model_gets_nothing(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    def test_a_later_gpu_layer_count_wins(self):
        argv = self.BASE + ["-ngl", "0", "-ngl", "99"]
        assert _backend_sampling_flags(ON, argv, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize(
        "tail",
        [
            ["--spec-default"],
            ["--spec-type", "ngram-mod"],
            ["--spec-type=draft-mtp"],
            ["-md", "d.gguf"],
        ],
        ids = ["spec-default", "ngram-mod", "mtp-inline", "draft-model"],
    )
    def test_speculative_launches_get_nothing(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    def test_spec_type_none_is_not_speculation(self):
        argv = self.BASE + ["--spec-default", "--spec-type", "none"]
        assert _backend_sampling_flags(ON, argv, **GPU) == ["--backend-sampling"]

    def test_a_kept_spec_env_counts(self):
        kw = {**GPU, "spec_env": {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}}
        assert _backend_sampling_flags(ON, self.BASE, **kw) == []

    @pytest.mark.parametrize("flag", ["--backend-sampling", "-bs"])
    def test_the_users_own_flag_is_not_doubled(self, flag):
        assert _backend_sampling_flags(ON, self.BASE + [flag], **GPU) == []

    def test_the_env_twin_is_the_users_too(self):
        kw = {**GPU, "env": {"LLAMA_ARG_BACKEND_SAMPLING": "1"}}
        assert _backend_sampling_flags(ON, self.BASE, **kw) == []

    @pytest.mark.parametrize(
        "tail, env",
        [(["--min-p", "1"], {}), (["--min-p=1.0"], {}), ([], {"LLAMA_ARG_MIN_P": "1"})],
        ids = ["flag", "inline", "env"],
    )
    def test_a_default_min_p_of_one_gets_nothing(self, tail, env):
        # Every request without its own min_p would sample on a GPU min_p that drops every token.
        assert _backend_sampling_flags(ON, self.BASE + tail, **{**GPU, "env": env}) == []

    def test_a_default_min_p_below_one_still_gets_it(self):
        argv = self.BASE + ["--min-p", "0.99"]
        assert _backend_sampling_flags(ON, argv, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    def test_the_opt_out(self, monkeypatch, value):
        monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", value)
        assert _backend_sampling_flags(ON, self.BASE, **GPU) == []


@pytest.mark.parametrize(
    "argv,env,expected",
    [
        ([], {}, False),
        (["--spec-default"], {}, True),
        (["--spec-type", "none"], {}, False),
        (["--spec-default", "--spec-type", "none"], {}, False),
        (["--spec-type", "none", "--spec-default"], {}, True),
        ([], {"LLAMA_ARG_SPEC_TYPE": "none"}, False),
        ([], {"LLAMA_ARG_MODEL_DRAFT": "d.gguf"}, True),
        (["--spec-type", "none"], {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}, False),
    ],
)
def test_launch_speculates(argv, env, expected):
    assert _launch_speculates(argv, env) is expected


# ---- the real load path ----

_MEMORY = [(0, 40 * 1024**3, 48 * 1024**3)]
_CAPS = {"found": True, "supports_kv_unified": True, "supports_backend_sampling": True}


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", raising = False)
    monkeypatch.delenv("LLAMA_ARG_BACKEND_SAMPLING", raising = False)
    backend, _ = _backend(tmp_path, vulkan = False, memory = _MEMORY)
    gguf = _write_gguf(tmp_path / "Qwen3-0.6B-Q8_0.gguf")

    def go(
        caps = _CAPS,
        ggml_backends = frozenset({"cpu", "cuda"}),
        **kw,
    ):
        monkeypatch.setattr(
            type(backend), "probe_server_capabilities", classmethod(lambda cls, binary = None: caps)
        )
        monkeypatch.setattr(
            type(backend),
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: ggml_backends),
        )
        kw.setdefault("n_parallel", 4)
        kw.setdefault("speculative_type", "off")
        return _launch(backend, gguf, **kw)["cmd"]

    go.backend = backend
    return go


def test_a_multi_slot_gpu_load_launches_with_it(launch):
    cmd = launch()
    assert cmd.count("--backend-sampling") == 1


def test_a_single_slot_load_launches_with_it_too(launch):
    assert "--backend-sampling" in launch(n_parallel = 1)


def test_a_rocm_build_keeps_the_cpu_sampler(launch):
    assert "--backend-sampling" not in launch(ggml_backends = frozenset({"cpu", "hip"}))


def test_an_older_binary_launches_as_before(launch):
    assert "--backend-sampling" not in launch(caps = {**_CAPS, "supports_backend_sampling": False})


def test_auto_speculation_keeps_the_cpu_sampler(launch):
    cmd = launch(speculative_type = "auto")
    assert "--spec-default" in cmd or "--spec-type" in cmd
    assert "--backend-sampling" not in cmd


def test_a_cpu_placed_load_keeps_the_cpu_sampler(launch):
    assert "--backend-sampling" not in launch(gpu_memory_mode = "manual", gpu_layers = 0)


def test_the_users_flag_is_passed_once(launch):
    cmd = launch(extra_args = ["--backend-sampling"])
    assert cmd.count("--backend-sampling") == 1


def test_the_opt_out_launches_as_before(launch, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", "0")
    assert "--backend-sampling" not in launch()


# ---- min_p at 1 on the GPU sampler ----
# llama.cpp's GPU min_p keeps logit - (max + log p) > 0, so p = 1 drops even the top token and the
# draw is NaN; those requests have to go back to the CPU chain, and only on a server with it on.

_ON = ["/fake/llama-server", "--backend-sampling"]


@pytest.mark.parametrize(
    "argv, env, expected",
    [
        (["/fake/llama-server"], {}, None),
        (["/fake/llama-server", "--spec-draft-backend-sampling"], {}, None),
        (_ON, {}, 0.0),
        (["/fake/llama-server", "-bs"], {}, 0.0),
        (["/fake/llama-server"], {"LLAMA_ARG_BACKEND_SAMPLING": "1"}, 0.0),
        (["/fake/llama-server"], {"LLAMA_ARG_BACKEND_SAMPLING": "0"}, None),
        ([*_ON, "--min-p", "1"], {}, 1.0),
        ([*_ON, "--min-p=0.5"], {"LLAMA_ARG_MIN_P": "1"}, 0.5),
        (_ON, {"LLAMA_ARG_MIN_P": "1"}, 1.0),
    ],
)
def test_launch_backend_sampling_min_p(argv, env, expected):
    assert _launch_backend_sampling_min_p(argv, env) == expected


@pytest.mark.parametrize("min_p", [1, 1.0, "1", 1.5, 0.9999])
def test_min_p_at_one_goes_back_to_the_cpu(min_p):
    payload = {"messages": [], "min_p": min_p}
    assert _cpu_sampling_for_min_p(payload, 0.0) == {**payload, "backend_sampling": False}
    assert "backend_sampling" not in payload


@pytest.mark.parametrize(
    "payload", [{"min_p": 0.99}, {"min_p": 0.999}, {"min_p": 0}, {}, {"min_p": None}]
)
def test_other_min_p_values_leave_the_payload_alone(payload):
    assert _cpu_sampling_for_min_p(payload, 0.0) is payload


def test_an_omitted_min_p_takes_the_launch_default():
    assert _cpu_sampling_for_min_p({}, 1.0) == {"backend_sampling": False}
    assert _cpu_sampling_for_min_p({"min_p": 0.1}, 1.0) == {"min_p": 0.1}


def test_a_server_without_the_flag_never_sees_the_field():
    payload = {"min_p": 1.0}
    assert _cpu_sampling_for_min_p(payload, None) is payload


def test_a_clients_own_opt_in_is_pinned_off_too():
    assert _cpu_sampling_for_min_p({"min_p": 1, "backend_sampling": True}, None) == {
        "min_p": 1,
        "backend_sampling": False,
    }


def test_a_load_with_the_flag_guards_its_requests(launch):
    assert "--backend-sampling" in launch()
    assert launch.backend.guard_gpu_min_p({"min_p": 1.0}) == {
        "min_p": 1.0,
        "backend_sampling": False,
    }


def test_a_load_without_the_flag_leaves_its_requests_alone(launch):
    assert "--backend-sampling" not in launch(caps = {**_CAPS, "supports_backend_sampling": False})
    assert launch.backend.guard_gpu_min_p({"min_p": 1.0}) == {"min_p": 1.0}


def _guarded_backend(launch_min_p = 0.0):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._backend_sampling_min_p = launch_min_p
    return backend


def test_every_stream_is_guarded(monkeypatch):
    # _open_stream is the one transport of generate_chat_completion and both tool loops.
    import contextlib

    sent = []

    @contextlib.contextmanager
    def fake_stream(
        client,
        url,
        payload,
        cancel_event = None,
        **kwargs,
    ):
        sent.append(payload)
        yield type("Response", (), {"status_code": 200})()

    monkeypatch.setattr(LlamaCppBackend, "_stream_with_retry", staticmethod(fake_stream))
    monkeypatch.setattr(LlamaCppBackend, "_auth_headers", {}, raising = False)
    for launch_min_p, min_p, expected in [
        (0.0, 1.0, {"backend_sampling": False}),
        (0.0, 0.99, {}),
        (None, 1.0, {}),
    ]:
        backend = _guarded_backend(launch_min_p)
        with backend._open_stream("http://llama.test/v1/chat/completions", {"min_p": min_p}, None):
            pass
        assert sent.pop() == {"min_p": min_p, **expected}


def test_gguf_tts_is_guarded(monkeypatch):
    sent = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, json):
            sent.append(json)
            raise httpx.ConnectError("stop here")

    import httpx

    monkeypatch.setattr(llama_cpp.httpx, "Client", Client)
    monkeypatch.setattr(LlamaCppBackend, "_auth_headers", {}, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "base_url", "http://llama.test", raising = False)
    for launch_min_p, min_p, expected in [(0.0, 1.0, False), (0.0, 0.99, None), (None, 1.0, None)]:
        with pytest.raises(httpx.ConnectError):
            _guarded_backend(launch_min_p).generate_audio_response("hi", "snac", min_p = min_p)
        assert sent.pop().get("backend_sampling", None) is expected


def _route_backend(launch_min_p = 0.0):
    return type(
        "RouteBackend",
        (),
        {
            "is_loaded": True,
            "is_vision": False,
            "base_url": "http://llama.test",
            "context_length": 4096,
            "model_identifier": "gguf",
            "markup_profile": None,
            "guard_gpu_min_p": _guarded_backend(launch_min_p).guard_gpu_min_p,
            "_request_reasoning_kwargs": lambda self, *a: None,
        },
    )()


def test_every_chat_passthrough_body_is_guarded():
    from models.inference import ChatCompletionRequest, ChatMessage
    from routes.inference import _build_openai_passthrough_body, _build_passthrough_payload

    args = ([{"role": "user", "content": "hi"}], None, 0.7, 0.9, 20, 16, False)
    assert (
        _build_passthrough_payload(*args, min_p = 1.0, llama_backend = _route_backend())[
            "backend_sampling"
        ]
        is False
    )
    for backend, min_p in [
        (_route_backend(), 0.99),
        (_route_backend(), None),
        (_route_backend(None), 1.0),
    ]:
        assert "backend_sampling" not in _build_passthrough_payload(
            *args, min_p = min_p, llama_backend = backend
        )
    # OpenAI /v1/chat/completions and the streaming /v1/responses passthrough build through it.
    request = ChatCompletionRequest(
        model = "default", messages = [ChatMessage(role = "user", content = "hi")], min_p = 1.0
    )
    body = _build_openai_passthrough_body(request, llama_backend = _route_backend())
    assert body["backend_sampling"] is False


def test_every_passthrough_build_hands_over_the_backend():
    # Both /v1/messages passthroughs and the OpenAI body: a call without the backend is unguarded.
    import ast
    import inspect

    import routes.inference as inf_mod

    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(inf_mod)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_build_passthrough_payload"
    ]
    assert len(calls) == 3
    for call in calls:
        assert "llama_backend" in {kw.arg for kw in call.keywords}


@pytest.mark.parametrize(
    "launch_min_p, raw, expected",
    [
        (0.0, {"min_p": 1}, False),
        (0.0, {"min_p": 0.99}, None),
        (0.0, {}, None),
        (None, {"min_p": 1}, None),
        (None, {"min_p": 1, "backend_sampling": True}, False),
    ],
)
def test_the_raw_completions_proxy_is_guarded(monkeypatch, launch_min_p, raw, expected):
    import asyncio
    from types import SimpleNamespace

    import httpx

    import routes.inference as inf_mod
    from core.inference.api_monitor import ApiMonitor

    sent = []

    class Request:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/completions")
        method = "POST"
        headers = {}

        async def json(self):
            return {"prompt": "hi", "stream": False, **raw}

        async def is_disconnected(self):
            return False

    class Client:
        async def aclose(self):
            return None

        async def post(self, _url, *, json, **_kwargs):
            sent.append(dict(json))
            return httpx.Response(200, json = {"id": "cmpl-test", "choices": [{"text": "ok"}]})

    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: Client())
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: _route_backend(launch_min_p))
    asyncio.run(inf_mod.openai_completions(Request(), current_subject = "test"))
    assert sent[0].get("backend_sampling", None) is expected
