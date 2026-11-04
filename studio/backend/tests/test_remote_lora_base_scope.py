"""One load reads a repo's adapter_config.json once: the inference worker's own lookup and
its activation share the definitive answer, and nothing else is reused."""

import io
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


def test_inference_worker_wraps_lookup_and_activation_in_one_scope():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "worker.py").read_text(
        encoding = "utf-8"
    )
    scope = src.index("with remote_lora_base_scope():")
    lookup = src.index("_lora_base = _remote_lora_base(model_name, hf_token = _hf_token)")
    activate = src.index("_activate_transformers_version(_base, _hf_token)")
    gates = src.index("_native_audio_security_targets_or_error(model_name")
    assert scope < lookup < activate < gates
