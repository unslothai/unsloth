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
import urllib.parse
import urllib.request
import weakref
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
        model_config = types.SimpleNamespace(
            max_model_len = 2048, dtype = "torch.bfloat16", max_logprobs = 0
        )
        scheduler_config = types.SimpleNamespace(max_num_seqs = 64)
        cache_config = types.SimpleNamespace(kv_cache_size_tokens = 16384)
        self.vllm_config = types.SimpleNamespace(
            model_config = model_config,
            lora_config = lora_config,
            scheduler_config = scheduler_config,
            cache_config = cache_config,
        )
        # The in-process V1 layout the server reads its load from: LLMEngine -> InprocClient -> EngineCore.
        scheduler = types.SimpleNamespace(
            kv_cache_manager = types.SimpleNamespace(usage = 0.25), running = [1, 2], waiting = [3]
        )
        self.engine_core = types.SimpleNamespace(
            engine_core = types.SimpleNamespace(scheduler = scheduler)
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
        # vLLM's own check (SamplingParams._validate_logprobs), made when a request is added.
        top = getattr(params, "kwargs", {}).get("logprobs") or 0
        if top > self.vllm_config.model_config.max_logprobs:
            raise ValueError(
                f"Requested sample logprobs of {top}, which is greater than max allowed"
            )
        self.requests[request_id] = (prompt["prompt_token_ids"], params, lora_request)

    def step(self):
        self.hold.wait()
        outputs = []
        for request_id, (prompt, params, lora) in list(self.requests.items()):
            del self.requests[request_id]
            tokens = [prompt[-1] + 1, prompt[-1] + 2]
            top = int(getattr(params, "kwargs", {}).get("logprobs") or 0)
            logprobs = []
            for i, t in enumerate(tokens):
                # Like vLLM: the sampled token first, then the top `logprobs` by rank (one may be -inf).
                entry = {t: types.SimpleNamespace(logprob = -0.5 * (i + 1))}
                for rank in range(top):
                    value = float("-inf") if rank == top - 1 else -1.0 - rank
                    entry[900 + rank] = types.SimpleNamespace(logprob = value)
                logprobs.append(entry)
            completion = types.SimpleNamespace(
                token_ids = tokens, logprobs = logprobs, finish_reason = "stop"
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
    # 256 is well past socketserver's default listen backlog of 5, where a burst this size resets
    # connections every run; 32 only tripped it now and then on a loaded runner.
    server, _ = served
    count = 256
    results = [None] * count
    errors = []

    def worker(i):
        try:
            results[i] = _complete(server, [i])["choices"][0]["token_ids"]
        except Exception as error:
            errors.append(f"{i}: {type(error).__name__}: {error}")

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors, errors[:5]
    assert results == [[i + 1, i + 2] for i in range(count)]


@pytest.fixture
def served_pausing(monkeypatch):
    # Kill switch: /pause stops the engine until /resume, as before.
    monkeypatch.setenv("UNSLOTH_ASYNC_GRPO_PAUSE_ON_SYNC", "1")
    llm = _FakeLLM()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    yield server, llm
    server.stop()


def test_pause_holds_new_requests_until_resume(served_pausing):
    server, _ = served_pausing
    _post(server, "/pause")
    box = {}
    thread = threading.Thread(target = lambda: box.update(out = _complete(server, [3])))
    thread.start()
    thread.join(0.5)
    assert thread.is_alive() and "out" not in box
    _post(server, "/resume")
    thread.join(10)
    assert box["out"]["choices"][0]["token_ids"] == [4, 5]


def _raw_post(
    url,
    payload,
    content_type = "application/json",
):
    request = urllib.request.Request(
        url, data = json.dumps(payload).encode(), headers = {"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(request, timeout = 10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_control_endpoints_need_the_url_secret(served):
    server, llm = served
    origin = "http://127.0.0.1:{}".format(urllib.parse.urlsplit(server.url).port)
    adapter = {"lora_name": "other", "lora_path": "/tmp/other"}
    for base in (origin, origin + "/wrong-token"):
        assert _raw_post(base + "/v1/load_lora_adapter", adapter) == 404
        assert _raw_post(base + "/pause", {}) == 404
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/v1/models", timeout = 10)
        assert error.value.code == 404
    assert llm.llm_engine.added_loras == [] and server.engine.paused is False
    assert _get(server, "/health") == (200, {"status": "ok"})


def test_text_plain_posts_are_refused(served):
    server, llm = served
    adapter = {"lora_name": "other", "lora_path": "/tmp/other"}
    assert _raw_post(server.url + "/v1/load_lora_adapter", adapter, "text/plain") == 415
    assert _raw_post(server.url + "/pause", {}, "text/plain;charset=UTF-8") == 415
    assert llm.llm_engine.added_loras == [] and server.engine.paused is False


def test_versioned_adapters_load_eagerly_serve_newest_and_unload_after_drain(monkeypatch):
    # UNSLOTH_ASYNC_GRPO_INPLACE_LORA=0: every sync loads a new adapter id, as before.
    monkeypatch.setenv("UNSLOTH_ASYNC_GRPO_INPLACE_LORA", "0")
    llm = _FakeLLM()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    try:
        _versioned_adapters(server, llm)
    finally:
        server.stop()


def _versioned_adapters(server, llm):
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


def test_a_second_server_on_one_engine_gets_new_lora_ids():
    llm = _FakeLLM()
    engine = llm.llm_engine
    model = _model(llm)
    first = server_module.UnslothAsyncVLLMServer(model, sampling_params = _Sampling)
    _post(first, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    first.stop()
    # The stopped server leaves nothing registered in the shared engine.
    assert engine.removed_loras == engine.added_loras
    second = server_module.UnslothAsyncVLLMServer(model, sampling_params = _Sampling)
    try:
        _post(
            second, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"}
        )
    finally:
        second.stop()
    assert len(set(engine.added_loras)) == 2


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


def test_abort_all_also_fails_requests_still_queued(served_pausing):
    server, llm = served_pausing
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
        try:
            from trl.generation.async_vllm_client import AsyncVLLMClient as VLLMClient
            timeout = {"timeout": 10}
        except ImportError:
            from trl.experimental.async_grpo.vllm_client import VLLMClient
            timeout = {"server_timeout": 10}
    except Exception:
        pytest.skip("TRL AsyncGRPO not installed")
    if not hasattr(VLLMClient, "load_lora_adapter"):
        # TRL 1.13's client syncs full weights only; fast_inference requires TRL >= 1.14 for this.
        pytest.skip("TRL < 1.14: VLLMClient has no LoRA adapter calls")
    PartialState()  # TRL logs through accelerate's logger
    llm = _FakeLLM()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    try:
        client = VLLMClient(server.url, **timeout)
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


def _by_token_id(top_logprobs):
    try:
        from trl.trainer.async_grpo_trainer import _by_token_id as parse
    except Exception:
        try:
            from trl.experimental.async_grpo.async_rollout_worker import _by_token_id as parse
        except Exception:

            def parse(top_logprobs):
                return [
                    {int(token.removeprefix("token_id:")): value for token, value in top.items()}
                    for top in top_logprobs
                ]

    return parse(top_logprobs)


def test_top_logprobs_are_keyed_by_token_id_as_trl_reads_them(served):
    server, llm = served
    choice = _complete(server, [5], logprobs = 3)["choices"][0]
    # The engine's max_logprobs=0 cap was raised to what was asked, as `vllm serve --max-logprobs` would allow.
    assert llm.llm_engine.vllm_config.model_config.max_logprobs == 3
    logprobs = choice["logprobs"]
    parsed = _by_token_id(logprobs["top_logprobs"])
    assert len(parsed) == len(choice["token_ids"]) == 2
    for token_id, sampled, top in zip(choice["token_ids"], logprobs["token_logprobs"], parsed):
        # The sampled token plus the top 3, -inf clamped to plain JSON like `vllm serve`.
        assert top[token_id] == sampled
        assert top[900] == -1.0 and top[901] == -2.0 and top[902] == -9999.0
        assert len(top) == 4
    # logprobs=0 (TRL <= 1.16) still gets the sampled token, as vLLM returns it.
    zero = _complete(server, [5], logprobs = 0)["choices"][0]["logprobs"]["top_logprobs"]
    assert [list(_by_token_id([t])[0]) for t in zero] == [[6], [7]]


def test_a_sampled_token_missing_from_the_logprobs_is_refused_not_substituted():
    # The first entry is another token's logprob: returning it would feed TRL a wrong behaviour logprob.
    entry = {
        900: types.SimpleNamespace(logprob = -0.25),
        8: types.SimpleNamespace(logprob = -3.0),
    }
    assert server_module._sampled_logprob([entry], 0, 8) == -3.0
    with pytest.raises(ValueError, match = "no logprob for sampled token 7 at position 0"):
        server_module._sampled_logprob([entry], 0, 7)
    completion = types.SimpleNamespace(token_ids = [7], logprobs = [entry], finish_reason = "length")
    output = types.SimpleNamespace(request_id = "r", outputs = [completion])
    with pytest.raises(ValueError, match = "sampled token 7"):
        server_module.UnslothAsyncVLLMServer.finish(output)


def test_metrics_report_kv_capacity_and_load_in_vllm_format(served):
    import re

    server, llm = served
    with urllib.request.urlopen(server.url + "/metrics", timeout = 10) as response:
        assert response.status == 200
        text = response.read().decode()
    # The exact patterns TRL's KV-cap control loop matches.
    kv = re.findall(r'^vllm:cache_config_info\{.*kv_cache_size_tokens="(\d+)"', text, re.M)
    assert kv == ["16384"]
    assert re.findall(r"^vllm:kv_cache_usage_perc\{.*\} (\S+)", text, re.M) == ["0.25"]
    assert re.findall(r"^vllm:num_requests_running\{.*\} (\S+)", text, re.M) == ["2.0"]
    assert re.findall(r"^vllm:num_requests_waiting\{.*\} (\S+)", text, re.M) == ["1.0"]
    # Without a reachable scheduler it still answers, from its own request counts.
    del llm.llm_engine.engine_core
    with urllib.request.urlopen(server.url + "/metrics", timeout = 10) as response:
        text = response.read().decode()
    assert "kv_cache_usage_perc" not in text and "num_requests_waiting" in text
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(server.url.rsplit("/", 1)[0] + "/wrong/metrics", timeout = 10)
    assert error.value.code == 404


class _Request:
    def __init__(self, num_preemptions = 0):
        self.num_preemptions = num_preemptions


class _PreemptingScheduler:
    """vLLM V1 Scheduler subset: `requests`, `Request.num_preemptions`, `_free_blocks`."""

    def __init__(self):
        self.kv_cache_manager = types.SimpleNamespace(usage = 0.5)
        self.running = []
        self.waiting = []
        self.requests = {}
        self.freed = 0

    def _free_blocks(self, request):
        self.freed += 1
        del self.requests[request.request_id]


def _metric(text, name):
    import re
    return [float(x) for x in re.findall(rf"^{name}\{{.*\}} (\S+)", text, re.M)]


def _scrape(server):
    with urllib.request.urlopen(server.url + "/metrics", timeout = 10) as response:
        assert response.status == 200
        return response.read().decode()


def test_metrics_export_a_monotonic_preemption_total_across_finished_requests():
    llm = _FakeLLM()
    scheduler = _PreemptingScheduler()
    llm.llm_engine.engine_core.engine_core.scheduler = scheduler
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    try:
        assert _metric(_scrape(server), "vllm:num_preemptions_total") == [0.0]
        a, b = _Request(2), _Request(1)
        a.request_id, b.request_id = "a", "b"
        scheduler.requests.update(a = a, b = b)
        assert _metric(_scrape(server), "vllm:num_preemptions_total") == [3.0]
        scheduler._free_blocks(a)
        assert scheduler.freed == 1 and "a" not in scheduler.requests
        assert _metric(_scrape(server), "vllm:num_preemptions_total") == [3.0]
        b.num_preemptions += 1
        scheduler._free_blocks(b)
        assert _metric(_scrape(server), "vllm:num_preemptions_total") == [4.0]
        second = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
        try:
            c = _Request(5)
            c.request_id = "c"
            scheduler.requests["c"] = c
            scheduler._free_blocks(c)
            assert _metric(_scrape(second), "vllm:num_preemptions_total") == [9.0]
        finally:
            second.stop()
    finally:
        server.stop()


def test_metrics_omit_preemptions_without_a_countable_scheduler(served):
    # No `_free_blocks`: omit rather than emit a non-monotonic total.
    server, _ = served
    text = _scrape(server)
    assert "num_preemptions_total" not in text
    assert _metric(text, "vllm:num_requests_running") == [2.0]


def test_metrics_export_the_engine_thread_counters(served):
    server, _ = served
    for prompt in ([5], [9, 10]):
        _complete(server, prompt)
    _post(server, "/pause")
    _post(server, "/resume")
    text = _scrape(server)
    assert _metric(text, "unsloth:async_grpo_requests_finished_total") == [2.0]
    assert _metric(text, "unsloth:async_grpo_generation_tokens_total") == [4.0]
    assert _metric(text, "unsloth:async_grpo_syncs_total") == [1.0]
    assert _metric(text, "unsloth:async_grpo_engine_steps_total")[0] >= 1.0
    assert _metric(text, "unsloth:async_grpo_engine_step_seconds_total")[0] > 0.0
    assert _metric(text, "unsloth:async_grpo_engine_max_batch")[0] >= 1.0


def test_installed_vllm_scheduler_has_what_the_preemption_count_wraps():
    pytest.importorskip("vllm")
    import inspect

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import Request
    except Exception:
        pytest.skip("vLLM without the V1 scheduler")
    assert "del self.requests[" in inspect.getsource(Scheduler._free_blocks)
    assert "num_preemptions += 1" in inspect.getsource(Scheduler)
    assert "self.num_preemptions = 0" in inspect.getsource(Request.__init__)


def test_trl_control_loop_reports_preemptions_from_this_server():
    import asyncio

    loop_class = _trl_rollout_loop()
    if not hasattr(loop_class, "_control_loop"):
        pytest.skip("TRL AsyncGRPO without the KV-cap control loop")
    import aiohttp

    llm = _FakeLLM()
    scheduler = _PreemptingScheduler()
    llm.llm_engine.engine_core.engine_core.scheduler = scheduler
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    loop = _rollout_loop(loop_class, server)
    loop._metrics_poll_s = 0.05
    loop._concurrency = 0
    loop._total_rollouts = 0
    pushed = []
    stop = []

    def push(values):
        pushed.append(values)
        request = _Request(len(pushed))
        request.request_id = str(len(pushed))
        scheduler.requests[request.request_id] = request
        scheduler._free_blocks(request)
        if len(pushed) >= 3:
            stop[0].set()

    loop._push_metrics = push

    async def run():
        stop.append(asyncio.Event())
        async with aiohttp.ClientSession() as session:
            loop.session = session
            await asyncio.wait_for(loop._control_loop(stop[0]), timeout = 30)

    try:
        asyncio.run(run())
    finally:
        server.stop()
    assert [p["rollout/vllm_preemption_count"] for p in pushed] == [0.0, 1.0, 2.0]


def _trl_rollout_loop():
    pytest.importorskip("aiohttp")
    try:
        from trl.trainer import async_grpo_trainer as trl_module
    except Exception:
        pytest.skip("TRL >= 1.18 AsyncGRPO not importable")
    loop_class = getattr(trl_module, "AsyncRolloutLoop", None)
    if loop_class is None or not hasattr(loop_class, "_init_kv_cap"):
        pytest.skip("TRL AsyncGRPO without the KV-cap rollout loop")
    return loop_class


def _rollout_loop(loop_class, server, **config):
    import collections

    loop = loop_class.__new__(loop_class)
    loop.config = types.SimpleNamespace(
        max_completion_length = 8,
        temperature = 1.0,
        top_p = 1.0,
        top_k = 0,
        min_p = None,
        vllm_server_timeout = 30.0,
        max_concurrent_rollouts = 512,
        **config,
    )
    loop.vllm_server_url = server.url
    loop.model_name = "org/tiny"
    loop.lora_name = None
    loop.sampling_replay = False
    loop.num_sampler_logprobs = 8
    loop._model_version_value = types.SimpleNamespace(value = 0)
    loop._counters = collections.defaultdict(int)
    loop._kv_cap = 512
    return loop


def test_trl_rollout_loop_samples_and_sizes_its_kv_cap_from_this_server(served):
    import asyncio

    loop_class = _trl_rollout_loop()
    import aiohttp
    from accelerate.state import PartialState

    PartialState()  # TRL logs through accelerate's logger

    server, _ = served
    loop = _rollout_loop(loop_class, server)

    async def run():
        async with aiohttp.ClientSession() as session:
            loop.session = session
            await loop._init_kv_cap()
            return await loop.generate_one_turn([5])

    turn = asyncio.run(run())
    # TRL's own client code: score centering (the TRL 1.17+ default) reads top_logprobs.
    assert turn.output_ids == [6, 7]
    assert [entry[t] for entry, t in zip(turn.output_sampler_logprobs, turn.output_ids)] == [
        -0.5,
        -1.0,
    ]
    assert turn.output_log_probs == [-0.5, -1.0]
    # 16384 KV tokens / max_model_len 2048, not the 512 fallback with a warning.
    assert loop._kv_cap == 8


def _recording(engine):
    seen = []
    original = engine.add_request

    def add_request(
        request_id,
        prompt,
        params,
        lora_request = None,
    ):
        seen.append((prompt, lora_request))
        return original(request_id, prompt, params, lora_request)

    engine.add_request = add_request
    return seen


def test_syncs_overwrite_one_adapter_slot_and_salt_the_prefix_cache(served):
    server, llm = served
    engine = llm.llm_engine
    seen = _recording(engine)
    _complete(server, [1])
    assert seen[-1] == ({"prompt_token_ids": [1]}, None)  # the base model: no adapter, no salt
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    (slot,) = engine.added_loras
    assert engine.removed_loras == []
    _complete(server, [1], model = "trl-policy-v1")
    first_salt = seen[-1][0]["cache_salt"]
    assert seen[-1][1].lora_int_id == slot and seen[-1][1].lora_path == "/tmp/v1"
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/v2"})
    # Same id, reloaded from the new path: dropped and added back in one engine-thread command.
    assert engine.added_loras == [slot, slot] and engine.removed_loras == [slot]
    _complete(server, [1], model = "trl-policy-v1")
    assert seen[-1][1].lora_int_id == slot and seen[-1][1].lora_path == "/tmp/v2"
    # vLLM keys cached blocks on the adapter name, which the slot keeps: a new version gets a new salt.
    assert seen[-1][0]["cache_salt"] != first_salt
    # TRL unloads stale versions; the slot stays while a newer name is served from it.
    _post(server, "/v1/unload_lora_adapter", {"lora_name": "trl-policy-v1"})
    assert engine.removed_loras == [slot]
    _complete(server, [1], model = "trl-policy-v2")
    assert seen[-1][1].lora_int_id == slot
    _post(server, "/v1/unload_lora_adapter", {"lora_name": "trl-policy-v2"})
    assert engine.removed_loras == [slot, slot]
    _complete(server, [1])
    assert seen[-1] == ({"prompt_token_ids": [1]}, None)
    # Loaded again after a full unload: a plain add, nothing left to drop.
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v3", "lora_path": "/tmp/v3"})
    assert engine.added_loras == [slot, slot, slot] and engine.removed_loras == [slot, slot]


class _MaxLorasEngine(_FakeEngine):
    """Admits a waiting request only while its adapter fits next to the running ones (max_loras=1), like vLLM's
    scheduler; a prompt ending in 999 keeps running until `release` is set."""

    def __init__(self):
        super().__init__()
        self.waiting = []
        self.running = {}
        self.release = threading.Event()

    def add_request(
        self,
        request_id,
        prompt,
        params,
        lora_request = None,
    ):
        self.waiting.append((request_id, prompt["prompt_token_ids"], lora_request))

    def step(self):
        held = {lora.lora_int_id for _, lora in self.running.values() if lora is not None}
        while self.waiting:
            request_id, prompt, lora = self.waiting[0]
            if lora is not None and len(held | {lora.lora_int_id}) > 1:
                break
            self.waiting.pop(0)
            self.running[request_id] = (prompt, lora)
            if lora is not None:
                held.add(lora.lora_int_id)
        outputs = []
        for request_id, (prompt, lora) in list(self.running.items()):
            if prompt[-1] == 999 and not self.release.is_set():
                continue
            del self.running[request_id]
            completion = types.SimpleNamespace(
                token_ids = [prompt[-1] + 1],
                logprobs = [{prompt[-1] + 1: types.SimpleNamespace(logprob = -0.5)}],
                finish_reason = "stop",
            )
            outputs.append(
                types.SimpleNamespace(request_id = request_id, finished = True, outputs = [completion])
            )
        time.sleep(0.005)
        return outputs

    def abort_request(self, ids):
        super().abort_request(ids)
        self.waiting = [w for w in self.waiting if w[0] not in ids]
        for request_id in ids:
            self.running.pop(request_id, None)


def test_a_sync_does_not_stall_new_requests_behind_the_previous_version():
    llm = _FakeLLM()
    llm.llm_engine = engine = _MaxLorasEngine()
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    box = {}
    straggler = threading.Thread(
        target = lambda: box.update(out = _complete(server, [999], model = "trl-policy-v1"))
    )
    try:
        _post(
            server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"}
        )
        straggler.start()
        deadline = time.time() + 5
        while not engine.running and time.time() < deadline:
            time.sleep(0.01)
        assert engine.running
        _post(
            server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/v2"}
        )
        # The next version's rollout is served while the previous version's straggler is still decoding.
        payload = {"model": "trl-policy-v2", "prompt": [5], "max_tokens": 4, "logprobs": 0}
        choice = _post(server, "/v1/completions", payload, timeout = 5)[1]["choices"][0]
        assert choice["token_ids"] == [6] and straggler.is_alive()
    finally:
        engine.release.set()
        if straggler.is_alive():
            straggler.join(10)
        server.stop()
    assert box["out"]["choices"][0]["token_ids"] == [1000]


def test_a_failed_slot_reload_puts_the_previous_version_back(served):
    server, llm = served
    engine = llm.llm_engine
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    added = []
    original = engine.add_lora

    def add_lora(lora):
        added.append(lora.lora_path)
        if lora.lora_path == "/tmp/bad":
            raise RuntimeError("bad adapter")
        return original(lora)

    engine.add_lora = add_lora
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(
            server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/bad"}
        )
    assert error.value.code == 400
    assert added == ["/tmp/bad", "/tmp/v1"]
    seen = _recording(engine)
    _complete(server, [1], model = "trl-policy-v2")
    assert seen[-1][1].lora_path == "/tmp/v1"


def test_unloading_the_newest_version_puts_the_previous_one_back_in_the_slot(served):
    server, llm = served
    engine = llm.llm_engine
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/v2"})
    seen = _recording(engine)
    _complete(server, [1], model = "trl-policy-v2")
    v2_salt = seen[-1][0]["cache_salt"]
    _post(server, "/v1/unload_lora_adapter", {"lora_name": "trl-policy-v2"})
    (slot,) = set(engine.added_loras)
    assert engine.added_loras == [slot] * 3 and engine.removed_loras == [slot] * 2
    _complete(server, [1], model = "trl-policy-v1")
    assert seen[-1][1].lora_path == "/tmp/v1" and seen[-1][0]["cache_salt"] != v2_salt


def test_decoding_goes_on_between_pause_and_resume(served):
    # TRL brackets the adapter save with /pause and /resume; the reload is an engine-thread command, so the engine
    # keeps decoding through the save on the adapter it already serves.
    server, llm = served
    engine = llm.llm_engine
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    seen = _recording(engine)
    _post(server, "/pause")
    assert server.engine.paused is False and server.engine.syncing is True
    choice = _post(
        server, "/v1/completions", {"prompt": [3], "max_tokens": 4, "logprobs": 0}, timeout = 5
    )
    assert choice[1]["choices"][0]["token_ids"] == [4, 5]
    assert seen[-1][1].lora_path == "/tmp/v1"
    v1_salt = seen[-1][0]["cache_salt"]
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/v2"})
    _post(server, "/resume")
    _complete(server, [3])
    assert seen[-1][1].lora_path == "/tmp/v2" and seen[-1][0]["cache_salt"] != v1_salt
    stats = server.engine.stats
    assert stats["syncs"] == 1 and stats["sync_steps"] >= 1 and server.engine.syncing is False


def test_kill_switch_stops_decoding_between_pause_and_resume(served_pausing):
    server, llm = served_pausing
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"})
    _post(server, "/pause")
    box = {}
    thread = threading.Thread(target = lambda: box.update(out = _complete(server, [3])))
    thread.start()
    thread.join(0.3)
    assert thread.is_alive() and server.engine.paused is True
    _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/tmp/v2"})
    _post(server, "/resume")
    thread.join(10)
    assert box["out"]["choices"][0]["token_ids"] == [4, 5]
    assert server.engine.stats["syncs"] == 1 and server.engine.stats["sync_steps"] == 0


@pytest.mark.parametrize("pause_on_sync", ["0", "1"])
def test_a_failed_load_during_a_sync_keeps_serving_the_previous_version(monkeypatch, pause_on_sync):
    monkeypatch.setenv("UNSLOTH_ASYNC_GRPO_PAUSE_ON_SYNC", pause_on_sync)
    llm = _FakeLLM()
    engine = llm.llm_engine
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    try:
        _post(
            server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"}
        )
        version = server.engine.slot_version
        original = engine.add_lora

        def add_lora(lora):
            if lora.lora_path == "/tmp/bad":
                raise RuntimeError("bad adapter")
            return original(lora)

        engine.add_lora = add_lora
        seen = _recording(engine)
        _post(server, "/pause")
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(
                server,
                "/v1/load_lora_adapter",
                {"lora_name": "trl-policy-v2", "lora_path": "/tmp/bad"},
            )
        assert error.value.code == 400
        # Rolled back: same slot version (same prefix-cache salt), the previous adapter back in the slot.
        assert server.engine.slot_version == version and server.engine.newest == "trl-policy-v1"
        if pause_on_sync == "1":
            _post(server, "/resume")
        # TRL raises before /resume; without the kill switch the engine still serves v1 meanwhile.
        _complete(server, [1])
        assert seen[-1][1].lora_path == "/tmp/v1"
    finally:
        server.stop()


def _trl_sync_weight_lora():
    import importlib
    for name in (
        "trl.trainer.async_grpo_trainer",
        "trl.experimental.async_grpo.async_grpo_trainer",
    ):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        trainer = getattr(module, "AsyncGRPOTrainer", None)
        if trainer is not None and hasattr(trainer, "_sync_weight_lora"):
            return module, trainer._sync_weight_lora
    pytest.skip("TRL AsyncGRPOTrainer with LoRA sync (>= 1.14) not installed")


@pytest.mark.parametrize("pause_on_sync", ["0", "1"])
def test_trl_lora_sync_decodes_through_the_save_on_the_previous_version(
    monkeypatch, tmp_path, pause_on_sync
):
    # TRL's own `_sync_weight_lora` against this server: a rollout dispatched while the adapter is being saved is
    # decoded during the save on the previous adapter and stamped with the previous version, because TRL bumps
    # `model_version` only after the load. The kill switch restores the stall.
    pytest.importorskip("requests")
    module, sync_weight_lora = _trl_sync_weight_lora()
    try:
        from accelerate.state import PartialState
        try:
            from trl.generation.async_vllm_client import AsyncVLLMClient as VLLMClient
            timeout = {"timeout": 10}
        except ImportError:
            from trl.experimental.async_grpo.vllm_client import VLLMClient
            timeout = {"server_timeout": 10}
    except Exception:
        pytest.skip("TRL AsyncGRPO client not installed")
    PartialState()
    monkeypatch.setenv("UNSLOTH_ASYNC_GRPO_PAUSE_ON_SYNC", pause_on_sync)
    llm = _FakeLLM()
    engine = llm.llm_engine
    server = server_module.UnslothAsyncVLLMServer(_model(llm), sampling_params = _Sampling)
    seen = _recording(engine)
    stamps = []

    class _Worker:
        def update_model_version(self, version):
            stamps.append(("bump", version))

    trainer = types.SimpleNamespace(
        model_version = 1,
        _lora_dir = str(tmp_path),
        _lora_name = "trl-policy",
        _adapter_name = "default",
        model = object(),
        accelerator = types.SimpleNamespace(
            is_main_process = True, wait_for_everyone = lambda: None, unwrap_model = lambda m: m
        ),
        args = types.SimpleNamespace(max_staleness = 4),
        vllm_client = VLLMClient(server.url, **timeout),
        rollout_worker = _Worker(),
    )
    rollout = {}

    def save_lora_adapter(model, accelerator, adapter_name, dest_dir):
        # A rollout dispatched mid-save: TRL stamps its group with the version current at dispatch.
        stamps.append(("dispatch", trainer.model_version))
        thread = threading.Thread(target = lambda: rollout.update(out = _complete(server, [3])))
        thread.start()
        thread.join(1.0)
        rollout["done_during_save"] = not thread.is_alive()
        rollout["thread"] = thread

    try:
        _post(
            server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/tmp/v1"}
        )
        monkeypatch.setattr(module, "save_lora_adapter", save_lora_adapter)
        sync_weight_lora(trainer, time.time())
        rollout["thread"].join(10)
        assert rollout["out"]["choices"][0]["token_ids"] == [4, 5]
        assert stamps == [("dispatch", 1), ("bump", 2)] and trainer.model_version == 2
        if pause_on_sync == "0":
            assert rollout["done_during_save"] is True
            # Decoded on v1 and stamped v1: the staleness TRL computes is the true one.
            assert [s[1].lora_path for s in seen] == ["/tmp/v1"]
            assert server.engine.stats["sync_steps"] >= 1
        else:
            assert rollout["done_during_save"] is False
            # Held until after the load: served by v2 under a v1 stamp.
            assert [s[1].lora_path for s in seen] == [str(tmp_path / "trl-policy-v2")]
            assert server.engine.stats["sync_steps"] == 0
        assert server.engine.stats["syncs"] == 1 and server.engine.syncing is False
    finally:
        server.stop()


# Weight sync through tensors: TRL's save is replaced by `stage_adapter`, its load by the staged snapshot.
class _TensorLora:
    def __init__(self, lora_request_id, tensors):
        self.lora_int_id = lora_request_id
        self.lora_path = ""
        self.lora_tensors = tensors


class _PeftModel:
    def __init__(
        self,
        llm,
        adapters = ("default",),
    ):
        self.vllm_engine = llm
        self.config = types.SimpleNamespace(_name_or_path = "org/tiny")
        self.peft_config = {name: object() for name in adapters}
        self.file_loads = []
        torch = pytest.importorskip("torch")
        self.weight = torch.arange(6, dtype = torch.float32).reshape(2, 3) / 7

    def load_lora(
        self,
        path,
        lora_request_id = None,
    ):
        self.file_loads.append(path)
        return _Lora(path, lora_request_id)


def _tensor_served(
    monkeypatch,
    adapters = ("default",),
    builder = None,
):
    torch = pytest.importorskip("torch")
    built = []

    def build(model, save_directory, load_tensors, lora_request_id):
        assert load_tensors
        built.append(save_directory)
        # Aliases of the live weights, as zoo's builder returns them.
        return _TensorLora(lora_request_id, {"layers.0.q_proj.lora_A.weight": model.weight})

    monkeypatch.setattr(server_module, "_tensor_request_builder", lambda: builder or build)
    llm = _FakeLLM()
    llm.llm_engine.vllm_config.lora_config.lora_dtype = torch.bfloat16
    model = _PeftModel(llm, adapters)
    server = server_module.UnslothAsyncVLLMServer(model, sampling_params = _Sampling)
    return server, llm, model, built


def test_a_staged_adapter_is_served_from_its_snapshot_without_files(monkeypatch):
    torch = pytest.importorskip("torch")
    server, llm, model, built = _tensor_served(monkeypatch)
    try:
        engine = llm.llm_engine
        seen = _recording(engine)
        saved = model.weight.clone()
        assert server.stage_adapter(model, "default", "/o/.vllm_lora/trl-policy-v1")
        assert built == ["/o/.vllm_lora/.unsloth-live-adapter"]
        model.weight.add_(1.0)  # the optimizer steps the live adapter after the save
        _post(
            server,
            "/v1/load_lora_adapter",
            {"lora_name": "trl-policy-v1", "lora_path": "/o/.vllm_lora/trl-policy-v1"},
        )
        assert model.file_loads == [], "the load read TRL's files although a snapshot was staged"
        assert server.engine.stats["tensor_syncs"] == 1
        _complete(server, [1], model = "trl-policy-v1")
        (tensor,) = seen[-1][1].lora_tensors.values()
        assert tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16
        assert torch.equal(
            tensor, saved.to(torch.bfloat16)
        ), "the served version moved with the live weights"
        first_salt = seen[-1][0]["cache_salt"]
        # Next sync: same slot, new values, new salt.
        assert server.stage_adapter(model, "default", "/o/.vllm_lora/trl-policy-v2")
        _post(
            server,
            "/v1/load_lora_adapter",
            {"lora_name": "trl-policy-v2", "lora_path": "/o/.vllm_lora/trl-policy-v2"},
        )
        _complete(server, [1], model = "trl-policy-v2")
        (tensor,) = seen[-1][1].lora_tensors.values()
        assert torch.equal(tensor, (saved + 1.0).to(torch.bfloat16))
        assert seen[-1][0]["cache_salt"] != first_salt
        slot = seen[-1][1].lora_int_id
        assert engine.added_loras == [slot, slot] and engine.removed_loras == [slot]
        # Unloading the newest restores the previous snapshot, not the live weights.
        _post(server, "/v1/unload_lora_adapter", {"lora_name": "trl-policy-v2"})
        _complete(server, [1], model = "trl-policy-v1")
        (tensor,) = seen[-1][1].lora_tensors.values()
        assert torch.equal(tensor, saved.to(torch.bfloat16))
    finally:
        server.stop()


def test_a_load_of_an_unstaged_path_reads_the_files(monkeypatch):
    server, llm, model, _ = _tensor_served(monkeypatch)
    try:
        assert server.stage_adapter(model, "default", "/o/v1")
        _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v2", "lora_path": "/o/v2"})
        assert model.file_loads == ["/o/v2"] and server.engine.stats["tensor_syncs"] == 0
        # The unused snapshot is dropped, so a later load of its path reads the files too.
        _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/o/v1"})
        assert model.file_loads == ["/o/v2", "/o/v1"]
    finally:
        server.stop()


def test_tensor_sync_kill_switch_and_unsupported_layouts_keep_trl_files(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ASYNC_GRPO_TENSOR_SYNC", "0")
    server, _, model, built = _tensor_served(monkeypatch)
    try:
        assert not server.stage_adapter(model, "default", "/o/v1")
    finally:
        server.stop()
    monkeypatch.delenv("UNSLOTH_ASYNC_GRPO_TENSOR_SYNC")
    server, _, model, built = _tensor_served(monkeypatch, adapters = ("default", "other"))
    try:
        assert not server.stage_adapter(
            model, "default", "/o/v1"
        )  # several adapters: TRL picks the files
        assert not server.stage_adapter(
            _PeftModel(model.vllm_engine), "default", "/o/v1"
        )  # another model
        assert built == []
    finally:
        server.stop()


def test_a_failed_snapshot_falls_back_to_files_for_the_rest_of_the_run(monkeypatch):
    def build(*args):
        raise RuntimeError("unservable")

    server, _, model, _ = _tensor_served(monkeypatch, builder = build)
    try:
        assert not server.stage_adapter(model, "default", "/o/v1")
        assert not server.engine.tensor_sync
        _post(server, "/v1/load_lora_adapter", {"lora_name": "trl-policy-v1", "lora_path": "/o/v1"})
        assert model.file_loads == ["/o/v1"]
    finally:
        server.stop()


def test_staged_versions_get_their_own_ids_without_the_shared_slot(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ASYNC_GRPO_INPLACE_LORA", "0")
    server, llm, model, _ = _tensor_served(monkeypatch)
    try:
        ids = []
        for version in (1, 2):
            path = f"/o/v{version}"
            assert server.stage_adapter(model, "default", path)
            _post(
                server,
                "/v1/load_lora_adapter",
                {"lora_name": f"trl-policy-v{version}", "lora_path": path},
            )
            ids.append(llm.llm_engine.added_loras[-1])
        assert (
            ids[0] != ids[1] and model.file_loads == [] and server.engine.stats["tensor_syncs"] == 2
        )
    finally:
        server.stop()


def test_host_snapshot_is_one_copy_per_dtype_and_keeps_order():
    torch = pytest.importorskip("torch")
    tensors = {
        "b": torch.ones(2, 2),
        "a": torch.arange(3, dtype = torch.float32),
        "c": torch.zeros(1, dtype = torch.float64),
    }
    out = server_module._host_snapshot(tensors)
    assert list(out) == ["b", "a", "c"]
    assert out["b"].untyped_storage().data_ptr() == out["a"].untyped_storage().data_ptr()
    assert out["c"].dtype == torch.float64 and torch.equal(out["a"], tensors["a"])
    tensors["a"].add_(5)
    assert torch.equal(out["a"], torch.arange(3, dtype = torch.float32))
    cast = server_module._host_snapshot(tensors, torch.bfloat16)
    assert all(t.dtype == torch.bfloat16 for t in cast.values())


def test_trl_lora_sync_hands_the_adapter_over_as_tensors(monkeypatch, tmp_path):
    # TRL's own `_sync_weight_lora` with rl.py's wrapper around TRL's `save_lora_adapter`: no adapter files are
    # written, and the load serves the snapshot taken at the save.
    torch = pytest.importorskip("torch")
    pytest.importorskip("requests")
    module, sync_weight_lora = _trl_sync_weight_lora()
    try:
        from accelerate.state import PartialState
        try:
            from trl.generation.async_vllm_client import AsyncVLLMClient as VLLMClient
            timeout = {"timeout": 10}
        except ImportError:
            from trl.experimental.async_grpo.vllm_client import VLLMClient
            timeout = {"server_timeout": 10}
    except Exception:
        pytest.skip("TRL AsyncGRPO client not installed")
    PartialState()
    spec = importlib.util.spec_from_file_location(
        "_async_grpo_patch_tests", Path(__file__).with_name("test_async_grpo_patch.py")
    )
    patch_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_tests)
    rl = patch_tests._load()
    server, llm, model, built = _tensor_served(monkeypatch)
    rl["_ASYNC_GRPO_TENSOR_SERVERS"][model] = weakref.ref(server)
    trl_saves = []
    monkeypatch.setattr(
        module, "save_lora_adapter", rl["_async_grpo_tensor_save"](lambda *a: trl_saves.append(a))
    )
    seen = _recording(llm.llm_engine)

    class _Worker:
        def update_model_version(self, version):
            pass

    trainer = types.SimpleNamespace(
        model_version = 1,
        _lora_dir = str(tmp_path),
        _lora_name = "trl-policy",
        _adapter_name = "default",
        model = model,
        accelerator = types.SimpleNamespace(
            is_main_process = True,
            num_processes = 1,
            wait_for_everyone = lambda: None,
            unwrap_model = lambda m: m,
        ),
        args = types.SimpleNamespace(max_staleness = 4),
        vllm_client = VLLMClient(server.url, **timeout),
        rollout_worker = _Worker(),
    )
    try:
        saved = model.weight.clone()
        sync_weight_lora(trainer, time.time())
        assert trainer.model_version == 2 and trl_saves == []
        assert not (tmp_path / "trl-policy-v2").exists(), "TRL's adapter files were written"
        assert server.engine.stats["tensor_syncs"] == 1 and model.file_loads == []
        model.weight.add_(1.0)
        _complete(server, [1], model = "trl-policy-v2")
        (tensor,) = seen[-1][1].lora_tensors.values()
        assert torch.equal(tensor, saved.to(torch.bfloat16))
    finally:
        server.stop()
