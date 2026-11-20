# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD RyzenAI NPU fallback: its OpenAI server, runtime install and NpuBackend routing. No NPU needed."""

from __future__ import annotations

import errno
import hashlib
import json
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from core.inference import npu_backend as nb
from core.inference import ryzenai_runtime as rr
from core.inference import ryzenai_server as rs

LLAMA = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"


class FakeModel:
    def __init__(
        self,
        pieces = ("Hel", "lo", " there", "!"),
        prompt_tokens = 10,
    ):
        self.pieces, self.prompt_tokens = list(pieces), prompt_tokens
        self.prompts, self.closed, self.search = [], 0, None

    def count_tokens(self, text):
        return self.prompt_tokens

    def generate(self, prompt, max_new_tokens, **search):
        self.prompts.append(prompt)
        self.search = search
        try:
            yield from self.pieces[:max_new_tokens]
        finally:
            self.closed += 1


@pytest.fixture
def served():
    servers = []

    def start(model = None, context_length = 64):
        model = model or FakeModel()
        state = rs.State(
            model, "ryzenai-m", context_length, LLAMA, {"bos_token": "", "eos_token": ""}
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(state, "k"))
        threading.Thread(target = server.serve_forever, daemon = True).start()
        servers.append(server)
        client = httpx.Client(
            base_url = f"http://127.0.0.1:{server.server_port}", headers = {"Authorization": "Bearer k"}
        )
        return client, model

    yield start
    for server in servers:
        server.shutdown()


def _chat(client, **body):
    return client.post(
        "/v1/chat/completions", json = {"messages": [{"role": "user", "content": "hi"}], **body}
    )


def test_requests_need_the_key(served):
    client, _ = served()
    assert client.get("/health").json()["context_length"] == 64
    assert httpx.get(f"{client.base_url}/health").status_code == 401
    assert (
        httpx.get(f"{client.base_url}/health", headers = {"Authorization": "Bearer nope"}).status_code
        == 401
    )


def test_a_completion_reports_text_usage_and_why_it_stopped(served):
    client, model = served()
    body = _chat(client).json()
    assert body["choices"][0]["message"]["content"] == "Hello there!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    assert model.prompts == ["<user>hi<assistant>"]
    assert _chat(client, max_tokens = 2).json()["choices"][0]["finish_reason"] == "length"


def test_a_stream_is_openai_sse_with_usage_and_done(served):
    client, _ = served()
    lines = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hello there!"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 4


def test_a_stop_string_cuts_the_answer(served):
    client, model = served()
    body = _chat(client, stop = [" th"]).json()
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert model.closed == 1


def test_a_prompt_longer_than_the_compiled_context_is_a_400(served):
    client, _ = served(FakeModel(prompt_tokens = 64))
    response = _chat(client)
    assert response.status_code == 400 and "holds 64" in response.json()["error"]["message"]


@pytest.mark.parametrize("payload", [b"not json", b'{"messages": "x"}'])
def test_a_malformed_body_is_a_400(served, payload):
    client, _ = served()
    response = client.post(
        "/v1/chat/completions", content = payload, headers = {"Content-Type": "application/json"}
    )
    assert response.status_code == 400


def test_sampling_follows_temperature(served):
    client, model = served()
    _chat(client, temperature = 0)
    assert model.search == {"do_sample": False}
    _chat(client, temperature = 0.7, top_p = 0.9, top_k = 40)
    assert model.search == {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 40}


def test_a_dropped_stream_frees_the_generator(served):
    client, model = served(FakeModel(pieces = ["x"] * 2000), context_length = 4096)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json = {"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        next(r.iter_lines())
    for _ in range(100):
        if model.closed:
            break
        threading.Event().wait(0.05)
    assert model.closed == 1
    assert _chat(client, max_tokens = 3).status_code == 200


def test_chat_templates_from_either_file_and_text_parts(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [{"name": "default", "template": LLAMA}],
                "bos_token": {"content": "<s>"},
                "eos_token": "</s>",
            }
        )
    )
    template, tokens = rs.load_chat_template(tmp_path)
    assert template == LLAMA and tokens == {"bos_token": "<s>", "eos_token": "</s>"}
    (tmp_path / "chat_template.jinja").write_text("{{ bos_token }}{{ messages[0].content }}")
    template, tokens = rs.load_chat_template(tmp_path)
    parts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url"},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    assert rs.render_prompt(template, tokens, parts) == "<s>ab"
    (tmp_path / "chat_template.jinja").unlink()
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match = "no chat template"):
        rs.load_chat_template(tmp_path)


def test_a_template_that_refuses_the_conversation_is_refused():
    state_template = "{{ raise_exception('System role not supported') }}"
    with pytest.raises(Exception, match = "System role"):
        rs.render_prompt(state_template, {}, [{"role": "system", "content": "x"}])


# Runtime install and catalog.


def _pins(
    tmp_path,
    zip_bytes: bytes,
    models = None,
) -> Path:
    path = tmp_path / "pins.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime": {
                    "version": "1.7.0",
                    "repo": "lemonade-sdk/ryzenai-server",
                    "asset": {
                        "name": "ryzenai-server.zip",
                        "sha256": hashlib.sha256(zip_bytes).hexdigest(),
                    },
                },
                "context_length": 4096,
                "models": models
                or [
                    {"repo": "amd/Tiny-onnx-ryzenai-npu", "size_gb": 1.5, "labels": ["chat"]},
                    {
                        "repo": "amd/R1-onnx-ryzenai-npu",
                        "size_gb": 2.0,
                        "labels": ["chat", "reasoning"],
                    },
                ],
            }
        )
    )
    return path


def _zip() -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in (
            "onnxruntime.dll",
            "onnxruntime_providers_ryzenai.dll",
            "onnxruntime-genai.dll",
            "ryzen_mm.dll",
            "AMD_LICENSE",
        ):
            zf.writestr(f"ryzenai-server/{name}", b"MZ" + name.encode())
        zf.writestr("ryzenai-server/ryzenai-server.exe", b"MZexe")
    return buf.getvalue()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    blob = _zip()
    import install_lemonade_prebuilt as installer

    downloads = []

    def fake_download(url, destination, *, expected_sha256, progress, cancel):
        downloads.append(url)
        if hashlib.sha256(blob).hexdigest() != expected_sha256:
            raise RuntimeError("sha256 mismatch")
        Path(destination).write_bytes(blob)

    monkeypatch.setattr(installer, "_download", fake_download)
    runtime = rr.RyzenAiRuntime(tmp_path / "ryzenai", pins_path = _pins(tmp_path, blob))
    runtime.downloads = downloads
    return runtime


def test_install_keeps_the_dlls_never_the_exe_and_is_reused(runtime):
    assert runtime.installed() is None
    target = runtime.install()
    assert sorted(p.name for p in target.iterdir()) == sorted(
        [
            "AMD_LICENSE",
            "UNSLOTH_RYZENAI_INFO.json",
            "onnxruntime-genai.dll",
            "onnxruntime.dll",
            "onnxruntime_providers_ryzenai.dll",
            "ryzen_mm.dll",
        ]
    )
    assert "ryzenai-server/releases/download/v1.7.0/ryzenai-server.zip" in runtime.downloads[0]
    assert runtime.install() == target and len(runtime.downloads) == 1


def test_a_pin_change_or_a_missing_dll_reads_as_not_installed(runtime, tmp_path):
    target = runtime.install()
    (target / "onnxruntime-genai.dll").unlink()
    assert runtime.installed() is None
    runtime.install()
    runtime.pins_path = _pins(tmp_path, b"another zip")
    assert runtime.installed() is None
    with pytest.raises(RuntimeError, match = "sha256"):
        runtime.install()


def test_the_catalog_names_ids_without_slashes(runtime):
    rows = runtime.catalog()
    assert [r.id for r in rows] == ["ryzenai-Tiny-onnx-ryzenai-npu", "ryzenai-R1-onnx-ryzenai-npu"]
    assert rows[1].labels == ("chat", "reasoning") and not rows[0].downloaded
    assert nb.looks_like_npu_display_name("Tiny-onnx-ryzenai-npu")
    with pytest.raises(rr.RyzenAiError):
        runtime.model("ryzenai-nope")


def test_download_reports_progress_marks_complete_and_honours_cancel(runtime, monkeypatch):
    import huggingface_hub

    class Sibling:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class Info:
        sha = "abc123"
        siblings = [Sibling("genai_config.json", 10), Sibling("fusion.onnx.data", 90)]

    fetched = []
    monkeypatch.setattr(
        huggingface_hub.HfApi, "model_info", lambda self, repo, files_metadata: Info()
    )

    def fake_fetch(repo, name, revision, local_dir):
        fetched.append((repo, name, revision))
        Path(local_dir).mkdir(parents = True, exist_ok = True)
        (Path(local_dir) / name).write_text("x")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_fetch)
    events = list(runtime.download("ryzenai-Tiny-onnx-ryzenai-npu"))
    assert [e.get("percent") for e in events] == [0, 10, 99, 100] and events[-1][
        "event"
    ] == "complete"
    assert fetched[0] == ("amd/Tiny-onnx-ryzenai-npu", "genai_config.json", "abc123")
    assert runtime.model("ryzenai-Tiny-onnx-ryzenai-npu").downloaded
    cancel = threading.Event()
    gen = runtime.download("ryzenai-Tiny-onnx-ryzenai-npu", cancel)
    next(gen)
    cancel.set()
    with pytest.raises(rr.RyzenAiDownloadCancelled):
        next(gen)
    assert not runtime.model("ryzenai-Tiny-onnx-ryzenai-npu").downloaded
    runtime.delete("ryzenai-Tiny-onnx-ryzenai-npu")
    assert not runtime.model_dir("ryzenai-Tiny-onnx-ryzenai-npu").exists()


@pytest.mark.parametrize(
    "stderr, code, expect",
    [
        (
            "OSError: [WinError 4551] An Application Control policy has blocked this file",
            1,
            "Application Control",
        ),
        ("", 0xC0E90002, "Smart App Control"),
        ("OSError: [WinError 126] The specified module could not be found", 1, "did not load"),
    ],
)
def test_validate_names_a_code_integrity_block(runtime, monkeypatch, stderr, code, expect):
    runtime.install()

    class Done:
        returncode, stdout = code, ""

    Done.stderr = stderr
    monkeypatch.setattr(rr.subprocess, "run", lambda *a, **k: Done())
    assert expect in runtime.validate()


def test_a_refused_python_spawn_names_the_policy(monkeypatch, tmp_path):
    exc = OSError(errno.EACCES, "blocked")
    exc.winerror = 4551

    def refuse(*a, **k):
        raise exc

    monkeypatch.setattr(rr.subprocess, "Popen", refuse)
    server = rr.RyzenAiServer(tmp_path, tmp_path, "ryzenai-m", 4096)
    with pytest.raises(rr.RyzenAiError, match = "code integrity"):
        server.start(timeout = 5)
    assert not server.is_alive()


def test_a_worker_that_exits_reports_its_last_line(monkeypatch, tmp_path):
    import sys as _sys

    script = "import sys; print('loading'); print('OgaError: model is broken'); sys.exit(3)"
    real = rr.subprocess.Popen
    monkeypatch.setattr(
        rr.subprocess,
        "Popen",
        lambda cmd, **k: real([_sys.executable, "-c", script], **{**k, "cwd": None}),
    )
    server = rr.RyzenAiServer(tmp_path, tmp_path, "ryzenai-m", 4096)
    with pytest.raises(rr.RyzenAiError, match = "OgaError: model is broken"):
        server.start(timeout = 30)


# NpuBackend routing.


class FakeServer:
    started = []

    def __init__(self, dll_dir, model_dir, model_id, context_length):
        self.model_id, self.context_length = model_id, context_length
        self.port, self.api_key, self.alive = None, None, False

    def start(
        self,
        timeout = 900.0,
        cancel = None,
    ):
        self.port, self.api_key, self.alive = 4321, "secret", True
        FakeServer.started.append(self)

    def is_alive(self):
        return self.alive

    def stop(self):
        self.alive, self.port = False, None

    close = stop


@pytest.fixture
def backend(tmp_path, monkeypatch, runtime):
    monkeypatch.setattr(
        nb, "detect_amd_npu", lambda: {"present": True, "supported": True, "family": "XDNA2"}
    )
    monkeypatch.setattr(rr, "supported_platform", lambda: True)
    monkeypatch.setattr(rr, "RyzenAiServer", FakeServer)
    pins_path = runtime.pins_path
    monkeypatch.setattr(
        nb.LemonadeNpuBackend,
        "_ryzen",
        property(lambda self: rr.RyzenAiRuntime(self.root / "ryzenai", pins_path)),
    )
    monkeypatch.setattr(rr.RyzenAiRuntime, "validate", lambda self, timeout = 120.0: None)
    npu = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    npu.flm_calls = []

    def flm_refused():
        npu.flm_calls.append(1)
        raise nb.NpuError(
            "Windows blocked part of the local model runtime: Smart App Control blocked this program."
        )

    monkeypatch.setattr(npu, "_enable_flm", flm_refused)
    monkeypatch.setattr(npu, "_flm_rows", lambda: pytest.fail("lemond must not start for RyzenAI"))
    FakeServer.started = []
    yield npu
    npu.shutdown()


def _downloaded(npu, model_id = "ryzenai-Tiny-onnx-ryzenai-npu"):
    folder = npu._ryzen.model_dir(model_id)
    folder.mkdir(parents = True)
    (folder / ".unsloth-complete").write_text("{}")


def test_enable_falls_back_to_ryzenai_when_fastflowlm_is_refused(backend):
    status = backend.enable()
    assert status["ready"] and status["state"] == "ready" and status["error"] is None
    assert status["versions"]["ryzenai"] == "1.7.0"
    assert "Smart App Control" in status["validation"]["fastflowlm"]
    fresh = nb.LemonadeNpuBackend(root = backend.root)  # a restart stays enabled
    assert fresh.status()["ready"] and fresh.can_run()


def test_the_catalog_lists_ryzenai_models_without_starting_lemond(backend):
    backend.enable()
    _downloaded(backend)
    rows = {m.id: m for m in backend.catalog()}
    assert set(rows) == {"ryzenai-Tiny-onnx-ryzenai-npu", "ryzenai-R1-onnx-ryzenai-npu"}
    tiny = rows["ryzenai-Tiny-onnx-ryzenai-npu"]
    assert (
        tiny.downloaded
        and tiny.max_context_length == 4096
        and tiny.model_path == "lemonade:ryzenai-Tiny-onnx-ryzenai-npu"
    )
    assert rows["ryzenai-R1-onnx-ryzenai-npu"].reasoning


def test_load_chat_upstream_and_unload(backend):
    backend.enable()
    _downloaded(backend)
    model = backend.load("ryzenai-Tiny-onnx-ryzenai-npu", context_length = 99999)
    up = backend.upstream()
    assert (up.base_url, up.api_key, up.context_length) == (
        "http://127.0.0.1:4321/v1",
        "secret",
        4096,
    )
    assert (
        up.public_model == model.model_path and backend.status()["loaded_model"] == model.model_path
    )
    assert (
        backend.load("ryzenai-Tiny-onnx-ryzenai-npu", context_length = 2048)
        and backend.upstream().context_length == 2048
    )
    assert not FakeServer.started[0].alive  # the previous child was stopped first
    assert backend.unload() == model.model_path
    assert backend.resident() is None and not FakeServer.started[1].alive
    with pytest.raises(nb.NpuError, match = "not downloaded"):
        backend.load("ryzenai-R1-onnx-ryzenai-npu")


def test_a_dead_worker_is_not_resident(backend):
    backend.enable()
    _downloaded(backend)
    backend.load("ryzenai-Tiny-onnx-ryzenai-npu")
    FakeServer.started[-1].alive = False
    assert backend.resident() is None
    with pytest.raises(nb.NpuError, match = "No NPU model is loaded"):
        backend.upstream()


def test_delete_refuses_the_loaded_model_and_clears_the_rest(backend):
    backend.enable()
    _downloaded(backend)
    _downloaded(backend, "ryzenai-R1-onnx-ryzenai-npu")
    backend.load("ryzenai-Tiny-onnx-ryzenai-npu")
    with pytest.raises(nb.NpuError, match = "Unload"):
        backend.delete("ryzenai-Tiny-onnx-ryzenai-npu")
    backend.delete("ryzenai-R1-onnx-ryzenai-npu")
    assert not backend._ryzen.model_dir("ryzenai-R1-onnx-ryzenai-npu").exists()
    backend.unload()
    assert backend.delete_downloaded() == (1, [])


def test_enable_reports_both_failures(backend, monkeypatch):
    monkeypatch.setattr(
        rr.RyzenAiRuntime, "validate", lambda self, timeout = 120.0: "The NPU driver is too old."
    )
    with pytest.raises(nb.NpuError) as info:
        backend.enable()
    assert "Smart App Control" in str(
        info.value
    ) and "AMD RyzenAI also failed: The NPU driver is too old." in str(info.value)
    assert backend.status()["state"] == "failed" and not backend.status()["ready"]


def test_no_fallback_off_windows(backend, monkeypatch):
    monkeypatch.setattr(rr, "supported_platform", lambda: False)
    with pytest.raises(nb.NpuError, match = "Smart App Control"):
        backend.enable()
    assert backend._ryzen.installed() is None  # nothing was downloaded


def test_cancel_load_stops_the_starting_worker(backend, monkeypatch):
    backend.enable()
    _downloaded(backend)
    entered, release = threading.Event(), threading.Event()

    def slow_start(
        self,
        timeout = 900.0,
        cancel = None,
    ):
        self.port, self.api_key, self.alive = 4321, "secret", True
        entered.set()
        release.wait(5)
        if cancel is not None and cancel.is_set():
            raise rr.RyzenAiError("Loading ryzenai-Tiny-onnx-ryzenai-npu was cancelled.")

    monkeypatch.setattr(FakeServer, "start", slow_start)
    errors = []
    worker = threading.Thread(
        target = lambda: errors.append(_catch(backend.load, "ryzenai-Tiny-onnx-ryzenai-npu"))
    )
    worker.start()
    assert entered.wait(5)
    assert backend.cancel_load("ryzenai-Tiny-onnx-ryzenai-npu")
    release.set()
    worker.join(5)
    assert isinstance(errors[0], nb.NpuLoadCancelled) and backend.resident() is None


def _catch(fn, *args):
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


def test_fastflowlm_refused_after_it_was_enabled_offers_the_fallback(backend, monkeypatch):
    backend._write_marker(lemond = "lemond.exe")
    refused = OSError(errno.EACCES, "An Application Control policy has blocked this file")
    refused.winerror = 4551

    def flm_rows():
        raise refused

    monkeypatch.setattr(backend, "_flm_rows", flm_rows)
    with pytest.raises(nb.NpuError, match = "Enable the NPU again to use AMD RyzenAI models"):
        backend.catalog()
    assert backend.status()["state"] == "failed" and not backend.status()["ready"]
    backend.enable()
    assert {m.id for m in backend.catalog()} == {
        "ryzenai-Tiny-onnx-ryzenai-npu",
        "ryzenai-R1-onnx-ryzenai-npu",
    }


def test_an_ordinary_fastflowlm_failure_is_not_turned_into_a_fallback(backend, monkeypatch):
    backend._write_marker(lemond = "lemond.exe")

    def flm_rows():
        raise nb.NpuError("Listing NPU models failed: timeout")

    monkeypatch.setattr(backend, "_flm_rows", flm_rows)
    with pytest.raises(nb.NpuError, match = "^Listing NPU models failed: timeout$"):
        backend.catalog()
    assert backend.status()["state"] != "failed"


def _split(prompt, pieces):
    splitter, thought, text = rs.ThinkSplitter(prompt), "", ""
    for piece in pieces:
        r, c = splitter.feed(piece)
        thought, text = thought + r, text + c
    r, c = splitter.flush()
    return thought + r, text + c


@pytest.mark.parametrize(
    "prompt, pieces, expect",
    [
        (
            "<|Assistant|><think>\n",
            ["Let me", " think</th", "ink>\n\nAnswer: 4"],
            ("Let me think", "Answer: 4"),
        ),
        ("plain", ["<thi", "nk>hidden</think>", "shown"], ("hidden", "shown")),
        ("plain", ["a ", "<", " b"], ("", "a < b")),
        ("plain", ["Sure. ", "<think>stays</think>"], ("", "Sure. <think>stays</think>")),
        ("x<think>", ["never closed"], ("never closed", "")),
        ("plain", ["one", " two"], ("", "one two")),
    ],
)
def test_think_blocks_become_reasoning_content(prompt, pieces, expect):
    assert _split(prompt, pieces) == expect


def test_a_reasoning_answer_is_split_in_the_response():
    model = FakeModel(pieces = ["Hmm", "</think>", "\n42"])
    state = rs.State(model, "m", 64, "{{ messages[0].content }}<think>\n", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), rs.make_handler(state, "k"))
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        auth = {"Authorization": "Bearer k"}
        msg = httpx.post(
            url, headers = auth, json = {"messages": [{"role": "user", "content": "q"}]}
        ).json()["choices"][0]["message"]
        assert msg == {"role": "assistant", "content": "42", "reasoning_content": "Hmm"}
        with httpx.stream(
            "POST",
            url,
            headers = auth,
            json = {"messages": [{"role": "user", "content": "q"}], "stream": True},
        ) as r:
            deltas = [
                json.loads(l[6:])["choices"][0]["delta"]
                for l in r.iter_lines()
                if l.startswith("data: {")
            ]
        assert {"reasoning_content": "Hmm"} in deltas and {"content": "42"} in deltas
    finally:
        server.shutdown()
