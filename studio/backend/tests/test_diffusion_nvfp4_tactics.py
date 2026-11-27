# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bucketed, persisted mm_fp4 tactics: a new prompt length never retunes, a warm disk never tunes."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from core.inference import diffusion_nvfp4_dispatch as dispatch
from core.inference import diffusion_nvfp4_linear as nl
from core.inference import diffusion_nvfp4_ops as ops
from core.inference import diffusion_nvfp4_tactics as tactics

torch = pytest.importorskip("torch")

_BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(slots = True, unsafe_hash = True)
class _Spec:
    input_idx: tuple
    dim_idx: tuple
    gen_tuning_buckets: Any
    map_to_tuning_buckets: Callable


@dataclasses.dataclass(kw_only = True, unsafe_hash = True)
class _Config:
    dynamic_tensor_specs: tuple
    constraint_specs: tuple = ()
    use_cuda_graph: bool = True


def _pow2_floor(x: int) -> int:
    return 1 << (max(int(x), 1).bit_length() - 1)


def _pow2_buckets(m: int) -> tuple:
    top, out = _pow2_floor(m), []
    while top >= 1:
        out.append(top)
        top //= 2
    return tuple(out)


# FlashInfer 0.6.6's mm_fp4 config shape: all powers of two up to M, keyed by the floor power of two.
_BASE_CONFIG = _Config(dynamic_tensor_specs = (_Spec((0,), (0,), _pow2_buckets, _pow2_floor),))

_GPUS = {
    0: ((10, 0), "NVIDIA B200", 148),
    1: ((10, 0), "NVIDIA B200", 148),
    2: ((12, 0), "NVIDIA RTX PRO 6000 Blackwell", 188),
}


class _Device:
    type = "cuda"

    def __init__(self, index: int):
        self.index = index

    def __repr__(self):  # pragma: no cover - debug aid
        return f"cuda:{self.index}"


class _T:
    """Just enough tensor for the plan: a shape and a device."""

    def __init__(self, shape, device):
        self.shape = tuple(shape)
        self.device = device


class _Runner:
    def __init__(self, count: int):
        self.count = count
        self.runs: list = []

    def get_valid_tactics(self, inputs, profile):
        return list(range(self.count))

    def __call__(
        self,
        inputs,
        tactic = -1,
    ):
        self.runs.append((inputs[0].shape[0], tactic))
        return inputs[6]


class _Fake:
    """Records every profiling request FlashInfer would see."""

    def __init__(
        self,
        *,
        version = "0.7.3",
        tactic_count = 32,
    ):
        self.version = version
        self.runner = _Runner(tactic_count)
        self.tunes: list = []
        self.public_tunes: list = []
        self.tuning = False
        self.delay = 0.0
        self.result: Callable = (
            lambda k, n, bucket: (k // 16 + n // 16 + bucket) % self.runner.count
        )

    def install(self, monkeypatch):
        fake = self

        @contextlib.contextmanager
        def autotune(flag = True, **_kw):
            old = fake.tuning
            fake.tuning = bool(flag)
            try:
                yield
            finally:
                fake.tuning = old

        class _AutoTuner:
            @staticmethod
            def get():
                return _AutoTuner

            @staticmethod
            def choose_one(op, runners, config, inputs):
                spec = config.dynamic_tensor_specs[0]
                m, half_k, n = inputs[0].shape[0], inputs[0].shape[1], inputs[6].shape[1]
                if not fake.tuning:
                    return runners[0], -1
                buckets = spec.gen_tuning_buckets
                buckets = tuple(buckets) if isinstance(buckets, tuple) else tuple(buckets(m))
                fake.tunes.append(
                    {
                        "op": op,
                        "m": m,
                        "k": half_k * 2,
                        "n": n,
                        "profiled": buckets,
                        "key_m": spec.map_to_tuning_buckets(m),
                    }
                )
                if fake.delay:
                    time.sleep(fake.delay)
                return runners[0], fake.result(half_k * 2, n, buckets[0])

        def mm_fp4(
            a,
            *args,
            out = None,
            **kwargs,
        ):
            if fake.tuning:
                fake.public_tunes.append(a.shape[0])
            return out

        root = types.ModuleType("flashinfer")
        root.__version__ = self.version
        root.__path__ = []
        root.autotune = autotune
        root.mm_fp4 = mm_fp4
        gemm = types.ModuleType("flashinfer.gemm")
        gemm.__path__ = []
        base = types.ModuleType("flashinfer.gemm.gemm_base")
        base.DEFAULT_WORKSPACE_SIZE = 1 << 20
        base._MM_FP4_TUNING_CONFIG_128x4 = _BASE_CONFIG
        base._get_cache_buf = lambda name, size, device: ("workspace", device.index)
        base.get_cutlass_fp4_gemm_module = lambda major, minor: types.SimpleNamespace(
            cutlass_fp4_gemm_runner = lambda: fake.runner
        )
        autotuner = types.ModuleType("flashinfer.autotuner")
        autotuner.AutoTuner = _AutoTuner
        utils = types.ModuleType("flashinfer.utils")
        utils.get_compute_capability = lambda device: _GPUS[device.index][0]
        for name, module in {
            "flashinfer": root,
            "flashinfer.gemm": gemm,
            "flashinfer.gemm.gemm_base": base,
            "flashinfer.autotuner": autotuner,
            "flashinfer.utils": utils,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv(tactics.NVFP4_TACTIC_CACHE_DIR_ENV, str(tmp_path / "tactics"))
    monkeypatch.delenv(tactics.NVFP4_TACTIC_CACHE_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(
            name = _GPUS[index][1], multi_processor_count = _GPUS[index][2]
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: _GPUS[index][0])
    monkeypatch.setattr(ops, "_is_capturing", lambda: False)
    monkeypatch.setattr(dispatch, "enabled", lambda device: True)
    _fresh_process()
    yield _Fake().install(monkeypatch)
    _fresh_process()


def _fresh_process() -> None:
    """Everything a restart forgets; only the cache directory survives."""
    dispatch.reset()
    tactics.forget_memory()
    nl.reset_tuned_shapes()
    dispatch._BUCKET_CONFIGS.clear()


def _gemm(
    m: int,
    k: int,
    n: int,
    device: int = 0,
):
    dev = _Device(device)
    return dispatch.gemm_plan(
        _T((m, k // 2), dev),
        _T((k // 2, n), dev),
        _T((m, k // 16), dev),
        _T((k // 16, n), dev),
        _T((1,), dev),
        _T((m, n), dev),
        n,
        "cutlass",
    )


# A FLUX-like DiT: joint img / txt streams, a fused single stream on the packed sequence, modulation.
_STREAM = ((3072, 9216), (3072, 3072), (3072, 12288), (12288, 3072))
_SINGLE = ((3072, 21504), (15360, 3072))
_MOD = ((3072, 18432),)


def _render(
    height: int,
    width: int,
    text_tokens: int,
    batch: int = 1,
    device: int = 0,
) -> set:
    img = (height // 16) * (width // 16)
    seen = set()
    for m, shapes in (
        (img * batch, _STREAM),
        (text_tokens * batch, _STREAM),
        ((img + text_tokens) * batch, _SINGLE),
        (batch, _MOD),
    ):
        for k, n in shapes:
            plan = _gemm(m, k, n, device)
            assert plan is not None
            seen.add((m, k, n))
    return seen


_RENDERS = [
    (h, w, text, batch)
    for (h, w) in ((512, 512), (768, 768), (1024, 1024), (1024, 768), (1344, 768))
    for text in (17, 77, 93, 154, 230, 301, 377, 512)
    for batch in (1, 2)  # 2 = CFG batched
]


def test_the_bucket_ladder_is_a_floor_with_bounded_spacing():
    last = 0
    for m in range(1, 70000):
        b = tactics.m_bucket(m)
        assert 1 <= b <= m
        assert b >= last  # monotone
        last = b
        assert m < 2 * b
        if m >= 512:
            assert m < 1.5 * b
    assert [
        tactics.m_bucket(m) for m in (1, 2, 3, 255, 256, 511, 512, 767, 768, 1023, 4608, 9216)
    ] == [
        1,
        2,
        2,
        128,
        256,
        256,
        512,
        512,
        768,
        768,
        4096,
        8192,
    ]
    # 1..128k tokens (up to long video) is 26 buckets in all.
    assert len({tactics.m_bucket(m) for m in range(1, 131073)}) == 26


def test_many_renders_with_many_lengths_tune_only_the_bucket_set(fake):
    exact: set = set()
    for h, w, text, batch in _RENDERS:
        exact |= _render(h, w, text, batch)
    expected = {(k, n, tactics.m_bucket(m)) for (m, k, n) in exact}
    got = [(t["k"], t["n"], t["profiled"][0]) for t in fake.tunes]

    assert sorted(got) == sorted(expected)  # one tune per (shape, bucket), never a repeat
    assert len(got) == len(set(got))
    assert all(len(t["profiled"]) == 1 for t in fake.tunes)  # one M profiled per tune, not a ladder
    assert all(t["profiled"][0] == t["key_m"] == tactics.m_bucket(t["m"]) for t in fake.tunes)
    assert all(t["op"] == dispatch.BUCKET_OP for t in fake.tunes)
    assert fake.public_tunes == []
    # The old path tuned once per exact length; this workload has far more lengths than buckets.
    assert len(exact) > 3 * len(expected)


def test_a_new_prompt_length_in_a_tuned_bucket_is_never_a_new_tune(fake):
    _render(1024, 1024, 77)
    before = len(fake.tunes)
    for text in range(78, 512):
        # img + text stays in [4096, 6144) and the text stream in its bucket's range.
        if tactics.m_bucket(text) == tactics.m_bucket(77):
            _render(1024, 1024, text)
    assert len(fake.tunes) == before


def test_every_length_in_a_bucket_runs_the_buckets_tactic(fake):
    plans = {id(_gemm(m, 3072, 3072)) for m in range(4096, 6144, 37)}
    assert len(plans) == 1
    assert len(fake.tunes) == 1
    tactic = _gemm(5000, 3072, 3072)[1]
    assert tactic == fake.result(3072, 3072, 4096)


def test_a_warm_disk_cache_means_zero_tunes_after_a_restart(fake, tmp_path):
    for h, w, text, batch in _RENDERS[:12]:
        _render(h, w, text, batch)
    tuned = {
        (t["k"], t["n"], t["profiled"][0]): fake.result(t["k"], t["n"], t["profiled"][0])
        for t in fake.tunes
    }
    assert tuned

    _fresh_process()
    fake.tunes.clear()
    for h, w, text, batch in _RENDERS[:12]:
        _render(h, w, text, batch)
    assert fake.tunes == []
    for (k, n, bucket), tactic in tuned.items():
        assert _gemm(bucket, k, n)[1] == tactic
    assert tactics.describe()["disk_entries_loaded"] == len(tuned)
    files = list((tmp_path / "tactics").glob("mm_fp4_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding = "utf-8"))
    assert data["meta"]["device"] == "NVIDIA B200" and data["meta"]["tactic_count"] == 32
    assert len(data["entries"]) == len(tuned)


@pytest.mark.parametrize(
    "change",
    ["flashinfer_version", "arch", "device_name", "tactic_table"],
)
def test_the_disk_cache_invalidates_on_any_identity_change(fake, monkeypatch, change):
    _gemm(4608, 3072, 3072)
    assert len(fake.tunes) == 1
    _fresh_process()
    if change == "flashinfer_version":
        sys.modules["flashinfer"].__version__ = "0.6.6"
    elif change == "arch":
        monkeypatch.setitem(_GPUS, 0, ((10, 3), "NVIDIA B200", 148))
    elif change == "device_name":
        monkeypatch.setitem(_GPUS, 0, ((10, 0), "NVIDIA B300", 148))
    else:
        fake.runner.count = 40
    _gemm(4608, 3072, 3072)
    assert len(fake.tunes) == 2


def test_a_new_weight_shape_is_its_own_tune(fake):
    _gemm(4608, 3072, 3072)
    _gemm(4608, 3072, 12288)
    _gemm(4608, 12288, 3072)
    _gemm(4700, 3072, 3072)
    assert [(t["k"], t["n"]) for t in fake.tunes] == [(3072, 3072), (3072, 12288), (12288, 3072)]


def test_a_corrupt_or_foreign_or_out_of_range_file_costs_a_retune_never_a_raise(fake, tmp_path):
    _gemm(4608, 3072, 3072)
    (path,) = (tmp_path / "tactics").glob("mm_fp4_*.json")

    path.write_text("{not json", encoding = "utf-8")
    _fresh_process()
    _gemm(4608, 3072, 3072)
    assert len(fake.tunes) == 2

    data = json.loads(path.read_text(encoding = "utf-8"))
    data["entries"] = {key: 999 for key in data["entries"]}
    path.write_text(json.dumps(data), encoding = "utf-8")
    _fresh_process()
    _gemm(4608, 3072, 3072)
    assert len(fake.tunes) == 3

    data = json.loads(path.read_text(encoding = "utf-8"))
    data["meta"]["device"] = "another card"
    path.write_text(json.dumps(data), encoding = "utf-8")
    _fresh_process()
    _gemm(4608, 3072, 3072)
    assert len(fake.tunes) == 4


def test_concurrent_callers_on_one_bucket_tune_it_exactly_once(fake):
    fake.delay = 0.05
    barrier = threading.Barrier(8)
    plans: list = []
    errors: list = []

    def _worker(i):
        try:
            barrier.wait()
            plans.append(_gemm(4096 + 97 * i, 3072, 3072, device = i % 2))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target = _worker, args = (i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # Devices 0 and 1 are the same card model, so they share the one tuned entry.
    assert len(fake.tunes) == 1
    assert len({plan[1] for plan in plans}) == 1


def test_concurrent_processes_merge_their_entries(tmp_path):
    root = tmp_path / "shared"
    script = (
        "import sys\n"
        "from core.inference import diffusion_nvfp4_tactics as t\n"
        "env = t.environment(flashinfer_version='0.7.3', capability=(10, 0), device_name='B200',"
        " sm_count=148, cuda='13.0', tactic_count=32)\n"
        "w = int(sys.argv[1])\n"
        "for i in range(20):\n"
        "    t.record(env, 3072, 3072 + 32 * i, 1 << w, (w + i) % 32)\n"
    )
    env = dict(os.environ, **{tactics.NVFP4_TACTIC_CACHE_DIR_ENV: str(root)})
    env.pop(tactics.NVFP4_TACTIC_CACHE_ENV, None)
    procs = [
        subprocess.Popen([sys.executable, "-c", script, str(w)], cwd = str(_BACKEND_DIR), env = env)
        for w in range(6)
    ]
    assert [p.wait(timeout = 120) for p in procs] == [0] * 6
    (path,) = root.glob("mm_fp4_*.json")
    entries = json.loads(path.read_text(encoding = "utf-8"))["entries"]
    assert len(entries) == 6 * 20
    assert not list(root.glob(".*.tmp"))


def test_different_cards_in_one_process_keep_separate_tables(fake):
    _gemm(4608, 3072, 3072, device = 0)
    _gemm(4608, 3072, 3072, device = 1)  # same model as 0: shared
    _gemm(4608, 3072, 3072, device = 2)  # sm_120: its own tune
    assert len(fake.tunes) == 2


def test_a_cold_bucket_under_capture_neither_builds_nor_tunes(fake, monkeypatch):
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)
    assert _gemm(4608, 3072, 3072) is None
    assert fake.tunes == []


def test_a_bucket_tuned_on_disk_serves_a_capture_without_any_tune(fake, monkeypatch):
    _gemm(4608, 3072, 3072)  # tunes + persists
    _fresh_process()
    _gemm(1024, 3072, 3072)  # another bucket: builds the runner, tunes 1024
    assert len(fake.tunes) == 2
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)
    plan = _gemm(5000, 3072, 3072)  # 4096 bucket: on disk, plan from cached parts
    assert plan is not None and plan[1] == fake.result(3072, 3072, 4096)
    assert len(fake.tunes) == 2


def test_memory_mode_never_writes(fake, monkeypatch, tmp_path):
    monkeypatch.setenv(tactics.NVFP4_TACTIC_CACHE_ENV, "memory")
    _gemm(4608, 3072, 3072)
    _gemm(4700, 3072, 3072)
    assert len(fake.tunes) == 1
    assert not (tmp_path / "tactics").exists()


def test_the_kill_switch_restores_the_exact_length_plans(fake, monkeypatch):
    monkeypatch.setenv(tactics.NVFP4_TACTIC_CACHE_ENV, "0")
    _gemm(4608, 3072, 3072)
    _gemm(4700, 3072, 3072)
    assert fake.tunes == []  # legacy plans only look up FlashInfer's own cache
    assert (4608, 1536, 3072, "cutlass", 0) in dispatch._GEMM_PLAN
    assert (4700, 1536, 3072, "cutlass", 0) in dispatch._GEMM_PLAN


def test_a_failed_tune_is_never_persisted(fake, tmp_path):
    fake.result = lambda k, n, bucket: -1  # every tactic failed (or an OOM) inside FlashInfer
    plan = _gemm(4608, 3072, 3072)
    assert plan[1] == -1
    assert not list((tmp_path / "tactics").glob("mm_fp4_*.json"))
    _gemm(4700, 3072, 3072)
    assert len(fake.tunes) == 1  # not retried in this process either


def test_a_tune_that_raises_falls_back_once_and_is_not_persisted(fake, tmp_path):
    def _boom(*_a, **_kw):
        raise RuntimeError("profiling failed")

    fake.result = _boom
    assert _gemm(4608, 3072, 3072)[1] == -1
    assert _gemm(4700, 3072, 3072)[1] == -1
    assert len(fake.tunes) == 1
    assert not list((tmp_path / "tactics").glob("mm_fp4_*.json"))


def test_verify_keeps_the_autotuner_lookup_and_caches_nothing(fake):
    dev = _Device(0)
    plan = dispatch.gemm_plan(
        _T((256, 256), dev), None, None, None, None, _T((256, 128), dev), 128, "cutlass", force = True
    )
    assert plan is not None and plan[1] == -1
    assert fake.tunes == [] and dispatch._GEMM_PLAN == {}


def test_the_bucket_config_profiles_one_m_and_keys_every_m_to_it():
    config = dispatch.bucket_config(_BASE_CONFIG, 4096)
    assert dispatch.bucket_config(_BASE_CONFIG, 4096) is config  # stable for FlashInfer's LRU
    spec = config.dynamic_tensor_specs[0]
    assert spec.gen_tuning_buckets == (4096,)
    assert {spec.map_to_tuning_buckets(m) for m in (1, 4096, 5000, 6143)} == {4096}
    assert spec.input_idx == (0,) and spec.dim_idx == (0,)
    assert (
        config.use_cuda_graph is True and config.constraint_specs == _BASE_CONFIG.constraint_specs
    )
    assert _BASE_CONFIG.dynamic_tensor_specs[0].gen_tuning_buckets is _pow2_buckets  # untouched


def _mm_through_the_op(monkeypatch, m: int, k: int, n: int):
    monkeypatch.setattr(ops, "_device_guard", lambda t: contextlib.nullcontext())
    monkeypatch.setattr(ops, "_fire_barrier", lambda device: None)
    monkeypatch.setattr(ops, "_device_index", lambda device: 0)
    monkeypatch.setattr(dispatch, "transposed", lambda t: t)
    xq = torch.zeros(m, k // 2, dtype = torch.uint8)
    wq = torch.zeros(n, k // 2, dtype = torch.uint8)
    return ops._mm_impl(xq, wq, None, wq, None, n, "cutlass")


def test_the_op_takes_the_bucketed_plan_and_never_the_public_tune(fake, monkeypatch):
    monkeypatch.setattr(
        tactics,
        "device_environment",
        lambda device, count, backend = "cutlass": tactics.environment(
            flashinfer_version = "0.7.3",
            capability = (10, 0),
            device_name = "NVIDIA B200",
            sm_count = 148,
            cuda = "13.0",
            tactic_count = count,
        ),
    )
    monkeypatch.setattr(dispatch, "_runner", lambda device, index: (fake.runner, "ws", 32))
    for m in (4100, 4608, 5000, 6000):
        _mm_through_the_op(monkeypatch, m, 64, 64)
    assert fake.public_tunes == []
    assert len(fake.tunes) == 1
    assert [tactic for _, tactic in fake.runner.runs] == [fake.result(64, 64, 4096)] * 4


def test_the_public_path_claims_flashinfers_own_bucket_not_the_exact_length(fake, monkeypatch):
    monkeypatch.setattr(dispatch, "enabled", lambda device: False)
    for m in (4100, 4608, 5000, 8191, 8192, 9000):
        _mm_through_the_op(monkeypatch, m, 64, 64)
    # 0.6.6-shaped config: floor power of two, so 4100..8191 share one tune and 8192.. the next.
    assert fake.public_tunes == [4100, 8192]


def test_the_public_path_under_the_kill_switch_claims_every_length(fake, monkeypatch):
    monkeypatch.setenv(tactics.NVFP4_TACTIC_CACHE_ENV, "0")
    monkeypatch.setattr(dispatch, "enabled", lambda device: False)
    for m in (4100, 4608, 4608):
        _mm_through_the_op(monkeypatch, m, 64, 64)
    assert fake.public_tunes == [4100, 4608]


def test_describe_reports_the_tactic_cache(fake):
    _gemm(4608, 3072, 3072)
    record = dispatch.describe()["tactics"]
    assert record["mode"] == "auto" and record["tuned"] == 1 and record["entries"] == 1
