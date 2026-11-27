# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention card verdicts and race floors persist across processes; sink decisions stay per load. CPU-only fakes."""

from __future__ import annotations

import contextlib

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_dispatch as D
from core.inference import diffusion_probe_cache as pc

PASS = {
    "ok": True,
    "impl": "cuda",
    "cos": 0.9998,
    "rel_l1": 0.01,
    "sharp_cos": 0.999,
    "min_tokens": 0,
    "ms": {4096: {"incumbent": 2.0, "int8": 1.0}, 1024: {"incumbent": 0.5, "int8": 0.3}},
}


@pytest.fixture
def machine(monkeypatch):
    fp = {"gpu_uuid": "GPU-1", "torch": "fake"}
    monkeypatch.setattr(pc, "fingerprint", lambda card: dict(fp, card = card))
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROBE_CACHE", raising = False)
    monkeypatch.setattr(I, "_PERSISTED", [None], raising = False)
    monkeypatch.setattr(I, "implementation", lambda index, hd: "cuda")
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    warmed = []
    monkeypatch.setattr(I, "_warm", lambda *a: warmed.append(a[:4]), raising = False)
    checks = []
    state = {"checks": checks, "warmed": warmed, "fp": fp, "result": dict(PASS)}

    def check(device, dtype, heads, hd, incumbent, timed, impl):
        checks.append((heads, hd, timed, impl))
        return dict(state["result"])

    monkeypatch.setattr(I, "_check", check)

    def new_process():
        for name in ("_VERDICTS", "_LONG_CHECKS", "_CUDA_FAILED"):
            monkeypatch.setattr(I, name, {})
        for name in ("_VERIFIED", "_LONG_VERIFIED"):
            monkeypatch.setattr(I, name, set())
        monkeypatch.setattr(I, "_PERSISTED", [None], raising = False)
        monkeypatch.setattr(D, "_LONG_VERDICTS", {})

    new_process()
    state["new_process"] = new_process
    return state


def _verdict(**kw):
    return I.verdict("cuda:0", torch.bfloat16, 8, 128, None, kw.pop("name", "native"), **kw)


def test_a_second_process_reuses_the_verdict_without_probing(machine):
    first = _verdict()
    assert first["ok"] and len(machine["checks"]) == 1 and not machine["warmed"]
    machine["new_process"]()
    second = _verdict()
    assert machine["checks"] == [(8, 128, True, "cuda")]  # no second check, no second race
    assert second["persisted"] and second["min_tokens"] == 0
    assert second["ms"] == PASS["ms"]
    assert (0, 128, "torch.bfloat16", "cuda") in I._VERIFIED
    assert machine["warmed"] == [(0, torch.bfloat16, 128, "cuda")]


def test_the_key_keeps_dtype_geometry_incumbent_and_timing_apart(machine):
    _verdict()
    machine["new_process"]()
    _verdict(timed = False)
    I.verdict("cuda:0", torch.float16, 8, 128, None, "native")
    I.verdict("cuda:0", torch.bfloat16, 8, 128, None, "dispatch:_native_cudnn")
    assert len(machine["checks"]) == 4


def test_a_new_toolchain_or_card_runs_the_check_again(machine):
    _verdict()
    machine["new_process"]()
    machine["fp"]["torch"] = "other"
    _verdict()
    assert len(machine["checks"]) == 2


def test_a_check_that_raised_is_never_persisted(machine):
    machine["result"] = dict(PASS, ok = False, failed = True, error = "RuntimeError('nvrtc')")
    _verdict()
    machine["new_process"]()
    machine["result"] = dict(
        PASS, ms = {4096: {"incumbent": 2.0, "int8": 1.0}}, error = "timing failed"
    )
    _verdict()
    machine["new_process"]()
    _verdict()
    assert len(machine["checks"]) == 3


def test_a_failed_accuracy_check_is_persisted(machine):
    machine["result"] = {
        "ok": False,
        "impl": "cuda",
        "cos": 0.5,
        "rel_l1": 0.4,
        "ms": {},
        "min_tokens": None,
    }
    _verdict()
    machine["new_process"]()
    assert not _verdict()["ok"]
    assert len(machine["checks"]) == 1 and not machine["warmed"]
    assert not I._VERIFIED


def test_kernels_that_no_longer_load_run_the_check_again(machine, monkeypatch):
    _verdict()
    machine["new_process"]()
    monkeypatch.setattr(
        I, "_warm", lambda *a: (_ for _ in ()).throw(RuntimeError("module load failed"))
    )
    assert _verdict()["ok"]
    assert len(machine["checks"]) == 2


def test_the_long_variant_check_comes_back_with_the_verdict(machine):
    machine["result"] = dict(PASS, long = {"ok": True, "keys": 8100, "cos": 0.9997})
    _verdict()
    machine["new_process"]()
    _verdict()
    assert (0, 128, "torch.bfloat16", "cuda") in I._LONG_VERIFIED
    assert I._LONG_CHECKS[(0, 128, "torch.bfloat16", "cuda")]["keys"] == 8100


def test_the_switch_turns_persistence_off(machine, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_CACHE", "0")
    _verdict()
    machine["new_process"]()
    _verdict()
    assert len(machine["checks"]) == 2


def test_long_and_mid_races_persist_but_an_out_of_memory_loss_does_not(machine, monkeypatch):
    timed = []

    def median_ms(fns, rounds = 5):
        timed.append(1)
        return {"incumbent": 10.0, "int8": 5.0}

    monkeypatch.setattr(I, "_median_ms", median_ms)
    monkeypatch.setattr(I, "probe_inputs", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    machine["new_process"]()
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    assert len(timed) == 1

    def oom(fns, rounds = 5):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(I, "_median_ms", oom)
    assert not D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 8192)
    machine["new_process"]()
    monkeypatch.setattr(I, "_median_ms", median_ms)
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 8192)
    assert len(timed) == 2


def test_a_transient_out_of_memory_race_is_retried_and_never_cached(machine, monkeypatch):
    # the family race (H3 at 16384 tokens): one OOM, then memory is back
    timed = []
    script = {"ooms": 1}

    def median_ms(fns, rounds = 5):
        if script["ooms"]:
            script["ooms"] -= 1
            raise torch.OutOfMemoryError("CUDA out of memory")
        timed.append(1)
        return {"incumbent": 10.0, "int8": 5.0}

    monkeypatch.setattr(I, "_median_ms", median_ms)
    monkeypatch.setattr(I, "probe_inputs", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # retried within the call: the second probe wins
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    assert len(timed) == 1
    # two OOMs in a row lose this call only: the same process races again on the next load
    script["ooms"] = 2
    assert not D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 8192)
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 8192)
    assert len(timed) == 2


class _Event:
    def synchronize(self):
        pass


def _plan(decision):
    return {
        "sinks": None,
        "calls": I.SINK_DECIDE_AT,
        "checks": [(_Event(), torch.tensor([1]))],
        "decision": decision,
        "shape": (1, 4096, 4096, 24, 128),
        "index": 0,
        "mode": I.MODE_RACE,
    }


def test_a_sink_decision_is_raced_again_for_every_load(machine, monkeypatch):
    """Sink race verdicts are per load: never reused across processes or after forget_plans."""
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(I, "_GRAPHED", [False])
    dropped = []
    monkeypatch.setattr(I, "_drop_routes", lambda plan: dropped.append(plan["shape"]))
    decision = (0, (("bn", 128),), 1, 24, 128, torch.bfloat16, None, 16, 16, I.MODE_RACE)
    races = []

    def race(plan, answer):
        def run():
            races.append(answer)
            plan["sink_race"] = {"incumbent": 1.0, "sinks": 1.2, "plain": 1.1}
            return answer

        return run

    I._decisions().clear()
    plan = _plan(decision)
    assert I._choose_path(plan, race(plan, False))[0] is False
    plan = _plan(decision)
    assert I._choose_path(plan, race(plan, False))[0] is False
    assert races == [False]
    I.forget_plans()
    machine["new_process"]()
    plan = _plan(decision)
    assert I._choose_path(plan, race(plan, True))[0] is True
    assert races == [False, True]
    I.forget_plans()
    plan = _plan(decision)
    assert I._choose_path(plan, race(plan, "native"))[0] == "native"
    assert races == [False, True, "native"] and len(dropped) == 1
    I.forget_plans()


def test_a_load_that_kept_the_sink_path_for_its_first_render_persists_nothing(machine, monkeypatch):
    """The first-render rule keeps the sink path where a used call clipped, yet the verdict stays per load: nothing
    reaches the probe cache, and the next load (and the next process) races again."""
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(I, "_GRAPHED", [False])
    monkeypatch.setattr(
        I, "_drop_routes", lambda plan: pytest.fail("the kept sink path moves no route")
    )
    writes = []
    store = I.persisted()
    real_put = store.put
    monkeypatch.setattr(store, "put", lambda *a: writes.append(a) or real_put(*a))
    decision = (0, (("bn", 128),), 1, 24, 128, torch.bfloat16, None, 16, 16, I.MODE_RACE)
    races = []

    def used_plan():
        plan = _plan(decision)
        plan["checks"] = [
            (_Event(), torch.tensor([1]), True)
        ]  # a used call clipped on the sink path
        return plan

    def race(plan, answer):
        def run():
            races.append(answer)
            plan["sink_race"] = {"incumbent": 1.0, "sinks": 1.2, "plain": 1.1}
            return answer

        return run

    I._decisions().clear()
    plan = used_plan()
    assert I._choose_path(plan, race(plan, False))[0] is True and races == [False]
    I.forget_plans()
    plan = used_plan()
    assert I._choose_path(plan, race(plan, False))[0] is True and races == [False, False]
    I.forget_plans()
    machine["new_process"]()
    plan = _plan(decision)
    assert I._choose_path(plan, race(plan, False))[0] is False and races == [False, False, False]
    assert writes == []
    I.forget_plans()


def test_an_out_of_memory_sink_race_is_not_persisted(machine, monkeypatch):
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(I, "_GRAPHED", [False])
    monkeypatch.setattr(I, "_drop_routes", lambda plan: None)
    decision = (0, (("bn", 64),), 1, 8, 128, torch.bfloat16, None, 4, 4, I.MODE_RACE)
    races = []

    def race():
        races.append(1)
        return "native"

    I._decisions().clear()
    I._choose_path(_plan(decision), race)
    I.forget_plans()
    machine["new_process"]()
    I._choose_path(_plan(decision), race)
    assert len(races) == 2
    I.forget_plans()


def test_namespace_reads_survive_a_torn_file_and_rewrite_it(machine):
    ns = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert ns.put("cuda:0", "k", 1)
    path = ns._path()
    path.write_text("{not json", encoding = "utf-8")
    fresh = pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",))
    assert fresh.get("cuda:0", "k") is None
    assert fresh.put("cuda:0", "k", 2)
    assert (
        pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",)).get("cuda:0", "k") == 2
    )


def test_namespace_sources_are_part_of_the_key(machine):
    assert pc.Namespace("int8_attention_test", ("diffusion_probe_cache.py",)).put("cuda:0", "k", 1)
    other = pc.Namespace(
        "int8_attention_test", ("diffusion_probe_cache.py", "diffusion_int8_attention.py")
    )
    assert other.get("cuda:0", "k") is None


def test_a_strided_cudnn_incumbent_has_its_own_verdict(machine):
    from core.inference import diffusion_qwenimage21_attn as Q21

    strided = D._race_name(Q21.StridedCudnn((0,)))
    plain = D._race_name("_native_cudnn")
    assert plain == "dispatch:_native_cudnn" and strided == "dispatch:_native_cudnn:strided"
    assert D._race_name(None) == "dispatch:native"
    I.verdict("cuda:0", torch.bfloat16, 8, 128, None, plain)
    machine["new_process"]()
    I.verdict("cuda:0", torch.bfloat16, 8, 128, None, strided)
    assert (
        len(machine["checks"]) == 2
    )  # the plain cuDNN verdict is not reused for the strided incumbent


def test_the_strided_incumbent_source_is_part_of_the_key():
    assert "diffusion_qwenimage21_attn.py" in I.PERSIST_SOURCES
    assert "diffusion_int8_attention_cuda.py" in I.PERSIST_SOURCES


def test_a_new_cudnn_diffusers_or_nvrtc_runs_the_check_again(machine, monkeypatch):
    versions = {"diffusers": "0.37.0", "cudnn": 91000, "nvrtc": [12, 8]}
    monkeypatch.setattr(I, "_library_versions", lambda: dict(versions))
    _verdict()
    machine["new_process"]()
    _verdict()
    assert len(machine["checks"]) == 1
    for name, value in (("cudnn", 91100), ("diffusers", "0.38.0"), ("nvrtc", [13, 0])):
        versions[name] = value
        machine["new_process"]()
        _verdict()
    assert len(machine["checks"]) == 4


def test_an_unreadable_key_component_is_never_persisted(machine, monkeypatch):
    monkeypatch.setattr(
        I, "_library_versions", lambda: (_ for _ in ()).throw(RuntimeError("no cuDNN"))
    )
    _verdict()
    machine["new_process"]()
    _verdict()
    assert len(machine["checks"]) == 2


def test_a_corrupted_entry_runs_the_check_again(machine):
    import json

    _verdict()
    path = I.persisted()._path()
    data = json.loads(path.read_text(encoding = "utf-8"))
    (entry,) = data.values()
    for key in entry["decisions"]:
        entry["decisions"][key] = "garbage"
    path.write_text(json.dumps(data), encoding = "utf-8")
    machine["new_process"]()
    assert _verdict()["ok"]
    assert len(machine["checks"]) == 2
    data = json.loads(path.read_text(encoding = "utf-8"))
    (entry,) = data.values()
    machine["new_process"]()
    assert _verdict()["ok"]
    assert len(machine["checks"]) == 2


def test_an_entry_under_another_fingerprint_is_not_reused(machine):
    import json

    _verdict()
    path = I.persisted()._path()
    data = json.loads(path.read_text(encoding = "utf-8"))
    (entry,) = data.values()
    entry["fingerprint"]["driver"] = "stale"
    path.write_text(json.dumps(data), encoding = "utf-8")
    machine["new_process"]()
    _verdict()
    assert len(machine["checks"]) == 2


def test_a_long_race_is_keyed_by_the_implementation_that_served_it(machine, monkeypatch):
    """Race floors are keyed by the serving implementation (CUDA vs its Triton fallback)."""
    timed = []
    serving = {"impl": "triton"}
    monkeypatch.setattr(I, "implementation", lambda index, hd: serving["impl"])

    def median_ms(fns, rounds = 5):
        timed.append(serving["impl"])
        return (
            {"incumbent": 10.0, "int8": 5.0}
            if serving["impl"] == "cuda"
            else {"incumbent": 5.0, "int8": 10.0}
        )

    monkeypatch.setattr(I, "_median_ms", median_ms)
    monkeypatch.setattr(I, "probe_inputs", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert not D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    machine["new_process"]()
    serving["impl"] = "cuda"
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    assert timed == ["triton", "cuda"]
    machine["new_process"]()
    assert D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    serving["impl"] = "triton"
    assert not D._race_wins("cuda:0", torch.bfloat16, 2, 64, None, "native", 16384)
    assert timed == ["triton", "cuda"]
