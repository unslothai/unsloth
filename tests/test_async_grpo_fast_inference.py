# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""In-process vLLM server for TRL AsyncGRPOTrainer with fast_inference=True, against a fake engine (CPU)."""

import importlib.util
import json
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_async_grpo_vllm.py"
_spec = importlib.util.spec_from_file_location("_async_grpo_vllm", _MODULE)
server_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server_module)


class _Sampling:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Lora:
    def __init__(self, path, lora_request_id):
        self.lora_path = path
        self.lora_int_id = lora_request_id


class _FakeEngine:
    def __init__(self):
        lora_config = types.SimpleNamespace(max_lora_rank = 64)
        model_config = types.SimpleNamespace(max_model_len = 2048, dtype = "torch.bfloat16")
        scheduler_config = types.SimpleNamespace(max_num_seqs = 64)
        self.vllm_config = types.SimpleNamespace(
            model_config = model_config, lora_config = lora_config, scheduler_config = scheduler_config
        )
        self.requests = {}
        self.added_loras = []
        self.removed_loras = []
        self.aborted = []
        self.sleeping = False
        self.hold = threading.Event()
        self.hold.set()

    def add_request(
        self,
        request_id,
        prompt,
        params,
        lora_request = None,
    ):
        self.requests[request_id] = (prompt["prompt_token_ids"], params, lora_request)

    def step(self):
        self.hold.wait()
        outputs = []
        for request_id, (prompt, params, lora) in list(self.requests.items()):
            del self.requests[request_id]
            tokens = [prompt[-1] + 1, prompt[-1] + 2]
            completion = types.SimpleNamespace(
                token_ids = tokens,
                logprobs = [
                    {t: types.SimpleNamespace(logprob = -0.5 * (i + 1))} for i, t in enumerate(tokens)
                ],
                finish_reason = "stop",
            )
            output = types.SimpleNamespace(
                request_id = request_id, finished = True, outputs = [completion]
            )
            output.params, output.lora = params, lora
            outputs.append(output)
        time.sleep(0.01)
        return outputs

    def abort_request(self, ids):
        self.aborted.extend(ids)
        for request_id in ids:
            self.requests.pop(request_id, None)

    def add_lora(self, lora):
        self.added_loras.append(lora.lora_int_id)

    def remove_lora(self, lora_id):
        self.removed_loras.append(lora_id)

    def is_sleeping(self):
        return self.sleeping


class _FakeLLM:
    def __init__(self):
        self.llm_engine = _FakeEngine()
        self.woken = 0

    def get_default_sampling_params(self):
        return types.SimpleNamespace(
            temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0, repetition_penalty = 1.0
        )

    def wake_up(self):
        self.woken += 1
        self.llm_engine.sleeping = False


def _model(llm):
    model = types.SimpleNamespace(
        vllm_engine = llm, config = types.SimpleNamespace(_name_or_path = "org/tiny")
    )
    model.load_lora = lambda path, lora_request_id = None: _Lora(path, lora_request_id)
    return model


@pytest.fixture
def served():
    llm = _FakeLLM()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    yield server, llm
    server.stop()


def _get(server, path):
    with urllib.request.urlopen(server.url + path, timeout = 10) as response:
        return response.status, json.loads(response.read())


def _post(
    server,
    path,
    payload = None,
    timeout = 10,
):
    request = urllib.request.Request(
        server.url + path,
        data = json.dumps(payload or {}).encode(),
        headers = {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout = timeout) as response:
        return response.status, json.loads(response.read())


def _complete(
    server,
    prompt,
    model = "org/tiny",
    **extra,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "repetition_penalty": 1.0,
        "n": 1,
        "return_token_ids": True,
        "logprobs": 0,
        "seed": 7,
        **extra,
    }
    return _post(server, "/v1/completions", payload)[1]


def test_introspection_endpoints_match_what_trl_reads(served):
    server, _ = served
    assert _get(server, "/health")[0] == 200
    assert _get(server, "/v1/models")[1]["data"][0]["max_model_len"] == 2048
    info = _get(server, "/server_info?config_format=json")[1]["vllm_config"]
    assert info["model_config"]["dtype"] == "torch.bfloat16"
    assert info["lora_config"]["max_lora_rank"] == 64
    assert info["lora_config"]["max_loras"] >= 10
    assert info["parallel_config"]["data_parallel_size"] == 1
    assert _get(server, "/get_world_size")[1]["world_size"] == 1


def test_completion_returns_token_ids_and_sampled_logprobs(served):
    server, llm = served
    choice = _complete(server, [5, 6, 7])["choices"][0]
    assert choice["token_ids"] == [8, 9]
    assert choice["logprobs"]["token_logprobs"] == [-0.5, -1.0]


def test_unset_fields_take_the_generation_config_defaults_and_set_fields_win(served):
    server, llm = served
    seen = {}
    original = llm.llm_engine.add_request

    def spy(
        request_id,
        prompt,
        params,
        lora_request = None,
    ):
        seen.update(params.kwargs)
        return original(request_id, prompt, params, lora_request)

    llm.llm_engine.add_request = spy
    _complete(server, [1])
    assert seen["top_k"] == 20 and seen["temperature"] == 1.0 and seen["seed"] == 7
    assert seen["logprobs"] == 0 and seen["n"] == 1 and seen["max_tokens"] == 8


def test_concurrent_requests_all_complete(served):
    server, _ = served
    results = [None] * 32

    def worker(i):
        results[i] = _complete(server, [i])["choices"][0]["token_ids"]

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == [[i + 1, i + 2] for i in range(32)]


def test_pause_holds_new_requests_until_resume(served):
    server, _ = served
    _post(server, "/pause")
    box = {}
    thread = threading.Thread(target = lambda: box.update(out = _complete(server, [3])))
    thread.start()
    thread.join(0.5)
    assert thread.is_alive() and "out" not in box
    _post(server, "/resume")
    thread.join(10)
    assert box["out"]["choices"][0]["token_ids"] == [4, 5]


def test_adapters_load_eagerly_serve_newest_and_unload_after_drain(served):
    server, llm = served
    engine = llm.llm_engine
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    first = engine.added_loras[-1]
    seen = []
    original = engine.add_request
    engine.add_request = lambda rid, p, params, lora_request = None: (
        seen.append(lora_request),
        original(rid, p, params, lora_request),
    )[1]
    _complete(server, [1], model = "trl-policy-v1")
    assert seen[-1].lora_int_id == first
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/v2"})
    second = engine.added_loras[-1]
    assert second != first
    # A request still naming v1 is served by the newest adapter.
    _complete(server, [1], model = "trl-policy-v1")
    assert seen[-1].lora_int_id == second

    _post(server, "/v1/unload_lora_adapter", {"lora_name": "trl-policy-v1"})
    assert first in engine.removed_loras
    _post(server, "/v1/unload_lora_adapter", {"lora_name": "missing"})


def test_unload_of_an_adapter_in_use_waits_for_its_requests(served):
    server, llm = served
    engine = llm.llm_engine
    _post(server, "/v1/load_lora_adapter", {"lora_name": "a-v1", "lora_path": "/tmp/v1"})
    lora_id = engine.added_loras[-1]
    engine.hold.clear()
    box = {}
    thread = threading.Thread(target = lambda: box.update(out = _complete(server, [9], model = "a-v1")))
    thread.start()
    deadline = time.time() + 5
    while not engine.requests and time.time() < deadline:
        time.sleep(0.01)
    # The engine thread is blocked in step(); queue the unload behind it, then release.
    unload = threading.Thread(
        target = lambda: _post(server, "/v1/unload_lora_adapter", {"lora_name": "a-v1"})
    )
    unload.start()
    time.sleep(0.1)
    assert lora_id not in engine.removed_loras
    engine.hold.set()
    thread.join(10)
    unload.join(10)
    assert box["out"]["choices"][0]["token_ids"] == [10, 11]
    assert lora_id in engine.removed_loras


def test_bad_prompts_and_engine_errors_are_http_errors(served):
    server, llm = served
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(server, "/v1/completions", {"model": "x", "prompt": "text", "max_tokens": 4})
    assert error.value.code == 400

    def boom(*args, **kwargs):
        raise ValueError("prompt too long")

    llm.llm_engine.add_request = boom
    with pytest.raises(urllib.error.HTTPError) as error:
        _complete(server, [1])
    assert error.value.code == 400 and b"prompt too long" in error.value.read()


def test_abort_all_fails_in_flight_requests_and_the_server_keeps_serving(served):
    server, llm = served
    llm.llm_engine.hold.clear()
    box = {}

    def run():
        try:
            box["out"] = _complete(server, [1])
        except urllib.error.HTTPError as error:
            box["code"] = error.code

    thread = threading.Thread(target = run)
    thread.start()
    deadline = time.time() + 5
    while not llm.llm_engine.requests and time.time() < deadline:
        time.sleep(0.01)
    # abort_all is a command: it runs once the blocked step returns.
    aborter = threading.Thread(target = server.abort_all)
    aborter.start()
    llm.llm_engine.hold.set()
    aborter.join(10)
    thread.join(10)
    assert "out" in box or box.get("code") == 400
    assert _complete(server, [4])["choices"][0]["token_ids"] == [5, 6]


def test_abort_all_also_fails_requests_still_queued(served):
    server, llm = served
    _post(server, "/pause")
    box = {}

    def run():
        try:
            box["out"] = _complete(server, [7])
        except urllib.error.HTTPError as error:
            box["code"] = error.code

    thread = threading.Thread(target = run)
    thread.start()
    deadline = time.time() + 5
    while not server.engine.pending and time.time() < deadline:
        time.sleep(0.01)
    server.abort_all()
    thread.join(10)
    assert box == {"code": 400}
    _post(server, "/resume")
    assert not llm.llm_engine.requests
    assert _complete(server, [4])["choices"][0]["token_ids"] == [5, 6]


def test_sleeping_engine_is_woken_once(monkeypatch):
    llm = _FakeLLM()
    llm.llm_engine.sleeping = True
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    try:
        assert llm.woken == 1
    finally:
        server.stop()


def test_stop_is_idempotent_and_frees_the_port():
    llm = _FakeLLM()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    url = server.url
    server.stop()
    server.stop()
    with pytest.raises(OSError):
        urllib.request.urlopen(url + "/health", timeout = 2)


def test_model_without_load_lora_is_refused():
    llm = _FakeLLM()
    model = types.SimpleNamespace(vllm_engine = llm, config = types.SimpleNamespace(_name_or_path = "x"))
    with pytest.raises(ValueError, match = "get_peft_model"):
        server_module.UnslothAsyncVLLMServer(model, sampling_params = _Sampling)


def test_trl_vllm_client_reads_this_server():
    pytest.importorskip("requests")
    try:
        from accelerate.state import PartialState
        from trl.experimental.async_grpo.vllm_client import VLLMClient
    except Exception:
        pytest.skip("TRL AsyncGRPO not installed")
    PartialState()  # TRL logs through accelerate's logger
    llm = _FakeLLM()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    try:
        client = VLLMClient(server.url, server_timeout = 10)
        client.wait_for_server_ready(poll_interval_s = 0.1)
        assert client.get_max_model_len() == 2048
        assert client.get_dtype() == "torch.bfloat16"
        assert client.get_world_size() == 1
        client.pause()
        client.load_lora_adapter("trl-policy-v1", "/tmp/v1")
        client.resume()
        client.unload_lora_adapter("trl-policy-v1")
        assert llm.llm_engine.added_loras and llm.llm_engine.removed_loras
    finally:
        server.stop()
