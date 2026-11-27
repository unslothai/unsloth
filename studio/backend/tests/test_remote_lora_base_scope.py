# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One inference or export load reads a repo's adapter_config.json once; only definitive answers are shared."""

import io
import json
import os
import subprocess
import sys
import threading
import urllib.error
from pathlib import Path

import pytest

import utils.transformers_version as tv


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(b""))


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(tv, "_hf_json_ahead", {})
    monkeypatch.setattr(tv, "_adapter_base_from_hf_cache", lambda m: "cached/" + m)
    state = {"answers": {}, "calls": []}

    def read(url, token):
        state["calls"].append((url, token))
        queue = state["answers"][url.split("/resolve/")[0].split("/", 3)[-1]]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, int):
            raise _http_error(url, answer)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(tv, "_read_hf_json", read)
    return state


@pytest.mark.parametrize(
    "answer, expected",
    [
        (404, None),
        ({"base_model_name_or_path": "org/base"}, "org/base"),
        ({"peft_type": "LORA"}, None),
        ({"base_model_name_or_path": ""}, None),
    ],
)
def test_definitive_answer_is_read_once_per_scope(hub, answer, expected):
    hub["answers"]["org/model"] = [answer]
    with tv.remote_lora_base_scope():
        first = tv._remote_lora_base("org/model", "tok")
        second = tv._remote_lora_base("org/model", "tok")
    assert first == second == expected
    assert len(hub["calls"]) == 1


@pytest.mark.parametrize(
    "failure",
    [500, 401, 403, urllib.error.URLError("down"), [1, 2], TimeoutError("slow")],
)
def test_failure_is_never_reused(hub, failure):
    hub["answers"]["org/model"] = [failure, 404]
    with tv.remote_lora_base_scope():
        assert tv._remote_lora_base("org/model", None) == "cached/org/model"
        assert tv._remote_lora_base("org/model", None) is None
        assert tv._remote_lora_base("org/model", None) is None
    assert len(hub["calls"]) == 2


def test_nothing_is_reused_outside_or_after_a_scope(hub):
    hub["answers"]["org/model"] = [404]
    tv._remote_lora_base("org/model", None)
    tv._remote_lora_base("org/model", None)
    assert len(hub["calls"]) == 2
    with tv.remote_lora_base_scope():
        tv._remote_lora_base("org/model", None)
    hub["answers"]["org/model"] = [{"base_model_name_or_path": "org/base"}]
    assert tv._remote_lora_base("org/model", None) == "org/base"
    assert len(hub["calls"]) == 4


def test_answers_are_keyed_by_token_and_repo(hub):
    hub["answers"]["org/model"] = [404]
    hub["answers"]["org/other"] = [404]
    with tv.remote_lora_base_scope():
        for token in (None, "a", "b", False, None, "a"):
            tv._remote_lora_base("org/model", token)
        tv._remote_lora_base("org/other", None)
    assert len(hub["calls"]) == 5


def test_nested_scope_shares_and_outer_exit_clears(hub):
    hub["answers"]["org/model"] = [404]
    with tv.remote_lora_base_scope():
        tv._remote_lora_base("org/model", None)
        with tv.remote_lora_base_scope():
            tv._remote_lora_base("org/model", None)
        tv._remote_lora_base("org/model", None)
    assert len(hub["calls"]) == 1
    assert tv._remote_lora_base_answers.get() is None


def test_scope_does_not_leak_into_other_threads(hub):
    hub["answers"]["org/model"] = [404]
    with tv.remote_lora_base_scope():
        tv._remote_lora_base("org/model", None)
        worker = threading.Thread(target = tv._remote_lora_base, args = ("org/model", None))
        worker.start()
        worker.join()
    assert len(hub["calls"]) == 2


def test_offline_and_local_names_never_reach_the_scope(hub, monkeypatch):
    hub["answers"]["org/model"] = [404]
    with tv.remote_lora_base_scope():
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert tv._remote_lora_base("org/model", None) == "cached/org/model"
        monkeypatch.delenv("HF_HUB_OFFLINE")
        assert tv._remote_lora_base("org/model", None) is None
        assert tv._remote_lora_base("plainname", None) is None
    assert len(hub["calls"]) == 1


def test_activation_reuses_the_worker_lookup(hub, monkeypatch):
    hub["answers"]["org/model"] = [404]
    seen = []
    monkeypatch.setattr(tv, "get_transformers_tier", lambda m, t = None: seen.append(m) or "default")
    with tv.remote_lora_base_scope():
        base = tv._remote_lora_base("org/model", "tok") or tv._resolve_base_model("org/model")
        assert tv.get_transformers_activation_tier(base, "tok") == "default"
    assert seen == ["org/model"]
    assert len(hub["calls"]) == 1


_WORKER_DRIVER = """
import faulthandler, io, json, os, queue, sys, urllib.error

faulthandler.dump_traceback_later(120, exit = True)
import utils.transformers_version as tv
import utils.utils as uu

uu.hf_dns_dead = lambda *a, **k: False
tv.hf_endpoint_unreachable = lambda *a, **k: False
out, worker, model = sys.argv[1:4]
reads = []


def read(url, token):
    reads.append(url.rsplit("/", 1)[-1])
    raise urllib.error.HTTPError(url, 404, "x", {}, io.BytesIO(b""))


def done(*a, **k):
    with open(out, "w") as f:
        json.dump(reads, f)
    os._exit(0)


tv._read_hf_json = read
config = {"model_name": model, "checkpoint_path": model, "hf_token": None, "device_backend": "cuda"}
if worker == "inference":
    import core.inference.worker as w

    w._native_audio_security_targets_or_error = done
    w.run_inference_process(cmd_queue = queue.Queue(), resp_queue = queue.Queue(), cancel_event = None, config = config)
else:
    import core._torchao_stub as stub
    import core.export.worker as w
    import utils.ssm_runtime as ssm

    w._run_export_security_gates = lambda *a, **k: True
    ssm.provision_ssm_kernels = lambda *a, **k: None
    stub.install_torchao_windows_rocm_real_or_stub = done
    w.run_export_process(cmd_queue = queue.Queue(), resp_queue = queue.Queue(), config = config)
"""


@pytest.mark.parametrize("worker", ["inference", "export"])
def test_worker_load_reads_adapter_config_once(worker, tmp_path):
    if worker == "inference" and sys.platform == "darwin":
        pytest.skip("macOS inference takes the MLX command loop, not this activation path")
    backend = Path(__file__).resolve().parents[1]
    out = tmp_path / "reads.json"
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env["PYTHONPATH"] = str(backend)
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_DRIVER, str(out), worker, "org/model"],
        cwd = tmp_path,
        env = env,
        timeout = 240,
        capture_output = True,
        text = True,
    )
    assert out.exists(), proc.stderr[-4000:]
    reads = json.loads(out.read_text())
    assert reads.count("adapter_config.json") == 1, reads
