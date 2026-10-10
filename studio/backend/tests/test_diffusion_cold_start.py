# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cold-start levers for the diffusion backend: background compile of a dense denoiser, the persisted quant smoke-probe
table, the mapped pre-quant checkpoint read, and the deeper diffusers prewarm.

CPU only. The background compile is driven through a real ``torch.compile`` with a counting backend, so "no compile on
the render" is observed, not assumed.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_bg_compile as bg  # noqa: E402
from core.inference import diffusion_cuda_graph as cg  # noqa: E402
from core.inference import diffusion_probe_cache as probe_cache  # noqa: E402
from core.inference import diffusion_speed as speed  # noqa: E402


class _Counter:
    """A dynamo backend that counts compiles and runs the traced graph as is."""

    def __init__(self) -> None:
        self.compiles = 0

    def __call__(self, gm, example_inputs):
        self.compiles += 1
        return gm.forward


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(
        self,
        x,
        *,
        scale = 1.0,
        return_dict = True,
    ):
        return self.lin(x) * scale


@pytest.fixture(autouse = True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _whole_compiled(counter):
    net = _Net().eval()
    net.compile(backend = counter, fullgraph = True, dynamic = False)
    return net


_REAL_COMPILE = r"""
import torch
from core.inference import diffusion_bg_compile as bg


class Counter:
    def __init__(self):
        self.compiles = 0

    def __call__(self, gm, example_inputs):
        self.compiles += 1
        return gm.forward


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x, *, scale = 1.0, return_dict = True):
        return self.lin(x) * scale


counter = Counter()
net = Net().eval()
net.compile(backend = counter, fullgraph = True, dynamic = False)
job = bg.arm(net)
assert job is not None and job.pending()

x = torch.randn(2, 8)
with torch.no_grad(), bg.force_eager():
    eager_out = net(x, scale = 2.0, return_dict = False)
assert counter.compiles == 0, "a render under force_eager compiled the denoiser"
assert len(job.samples) == 1

assert job.kick() is True
job._thread.join(60)
assert job.state == "done", job.error
assert counter.compiles >= 1, "the background thread never compiled"
compiled_after_bg = counter.compiles

with torch.no_grad():
    out = net(x, scale = 2.0, return_dict = False)
assert counter.compiles == compiled_after_bg, "the first compiled render recompiled what the background built"
assert torch.equal(out, eager_out)
job.close()
assert not hasattr(net._compiled_call_impl, "_unsloth_bg_gate"), "close() left the gate in front of the compile"
print("BG_COMPILE_OK")
"""


def test_force_eager_never_reaches_dynamo_and_the_background_compile_does(monkeypatch):
    # Own interpreter: a dynamo compile on a worker thread breaks a later make_fx trace in-process.
    import os
    import subprocess
    from pathlib import Path

    monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE", raising = False)
    backend = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", PYTHONPATH = str(backend))
    r = subprocess.run(
        [sys.executable, "-c", _REAL_COMPILE],
        cwd = str(backend),
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert r.returncode == 0 and "BG_COMPILE_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_recording_dedups_by_input_shape_and_ignores_unforced_calls():
    counter = _Counter()
    net = _whole_compiled(counter)
    job = bg.arm(net)
    with torch.no_grad():
        with bg.force_eager():
            net(torch.randn(2, 8), return_dict = False)
            net(torch.randn(2, 8), return_dict = False)
            net(torch.randn(3, 8), return_dict = False)
    assert len(job.samples) == 2
    job.close()


def test_close_breaks_the_hook_job_cycle():
    net = _whole_compiled(_Counter())
    job = bg.arm(net)
    assert len(net._forward_pre_hooks) == 1
    job.close()
    assert len(net._forward_pre_hooks) == 0 and job.module is None


def test_kick_without_samples_keeps_recording():
    net = _whole_compiled(_Counter())
    job = bg.arm(net)
    assert job.kick() is False
    assert job.state == "recording"
    job.close()


def test_kill_switch_disables_arming(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_BG_COMPILE", "0")
    assert bg.arm(_whole_compiled(_Counter())) is None


@pytest.mark.parametrize(
    "raw, deferred, at_load",
    [
        (None, True, False),
        ("", True, False),
        ("1", True, True),
        ("on", True, True),
        ("0", False, False),
    ],
)
def test_load_time_background_compile_is_opt_in(monkeypatch, raw, deferred, at_load):
    # Eager then compiled renders would break same-seed repeats, so only deferred compiles in bg.
    if raw is None:
        monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_BG_COMPILE", raw)
    assert bg.enabled() is deferred
    assert bg.load_time_enabled() is at_load


def test_a_render_waits_for_an_in_flight_compile_and_cancel_releases_it():
    gate = threading.Event()

    class _Slow(torch.nn.Module):
        def forward(self, x):
            if threading.current_thread().name == "unsloth-diffusion-bg-compile":
                gate.wait(30)
            return x

    job = bg.BackgroundCompile(_Slow())
    assert job.install()
    with torch.no_grad(), bg.force_eager():
        job.module(torch.zeros(1))
    job.kick()
    assert job.compiling()
    cancel = threading.Event()
    cancel.set()
    assert job.wait(cancel) < 1.0, "a cancelled render kept waiting on the compile"
    threading.Timer(0.3, gate.set).start()
    assert job.wait() >= 0.2
    assert job.state == "done" and not job.compiling()
    job.close()


def test_the_background_compile_sees_the_recorded_compile_knobs():
    from core.inference import diffusion_compile_config as compile_config

    seen = {}

    class _Probe(torch.nn.Module):
        def forward(self, x):
            if threading.current_thread().name == "unsloth-diffusion-bg-compile":
                seen["emulate"] = torch._inductor.config.emulate_precision_casts
            return x

    module, attr = "torch._inductor.config", "emulate_precision_casts"
    before = compile_config.get_knob(module, attr)
    was_recorded = compile_config.is_recorded(module, attr)
    assert compile_config.set_knob(module, attr, True)
    try:
        job = bg.BackgroundCompile(_Probe())
        assert job.install()
        with torch.no_grad(), bg.force_eager():
            job.module(torch.zeros(1))
        job.kick()
        job._thread.join(30)
        assert job.state == "done", job.error
        assert (
            seen.get("emulate") is True
        ), "the background compile ran without the recorded inductor knobs"
        job.close()
    finally:
        if was_recorded:
            compile_config.set_knob(module, attr, before)
        else:
            with compile_config._LOCK:
                compile_config._KNOBS.pop((module, attr), None)
            torch._inductor.config.emulate_precision_casts = before


def test_a_failing_warm_forward_ends_the_attempt_without_raising():
    class _Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    net = _Boom()
    job = bg.BackgroundCompile(net)
    assert job.install()
    job.samples.append(
        (("l", True, [("l", True, [("t", 0)]), ("d", [])]), [torch.zeros(1)], False, False, None)
    )
    job.kick()
    job._thread.join(30)
    assert job.state == "failed" and "boom" in (job.error or "")
    assert not job.pending()


def test_compile_guard_routes_to_eager_under_force_eager():
    guard = speed._CompileGuard(None)
    calls = []
    wrapped = guard.wrap(
        lambda *a, **k: calls.append("compiled"), lambda *a, **k: calls.append("eager"), object()
    )
    with bg.force_eager():
        wrapped()
    wrapped()
    assert calls == ["eager", "compiled"]


def test_graphed_forward_neither_captures_nor_poisons_under_force_eager():
    net = _Net()
    graphed = cg.GraphedForward(net).enable()
    try:
        with torch.no_grad(), bg.force_eager():
            net(torch.randn(2, 8), return_dict = False)
        assert graphed.stats["captures"] == 0
        assert graphed.stats["eager_calls"] == 1
        assert not graphed.poisoned
        token = bg._NO_CAPTURE.set(True)
        try:
            with torch.no_grad():
                net(torch.randn(2, 8), return_dict = False)
        finally:
            bg._NO_CAPTURE.reset(token)
        assert graphed.stats["eager_calls"] == 1 and graphed.stats["captures"] == 0
    finally:
        graphed.free()


class _UNet2DConditionModel(torch.nn.Module):
    """Named like the class on the whole-compile list."""


@pytest.mark.parametrize(
    "override, expect",
    [
        ({}, True),
        ({"transformer_quant": "int8"}, False),
        ({"gguf_transformer": True}, False),
        ({"offload_policy": "group", "_hooked": True}, False),
        ({"offload_policy": "group"}, True),
        ({"speed_mode": "max"}, False),
        ({"speed_optims": ("cuda_graph",)}, False),
        ({"transformer_cache": "fbcache"}, False),
        ({"cache_auto": True}, False),
        ({"target": types.SimpleNamespace(device = "cuda", backend = "rocm")}, False),
        ({"target": types.SimpleNamespace(device = "mps", backend = "mps")}, False),
    ],
)
def test_only_dense_resident_default_tier_cuda_loads_compile_in_the_background(
    monkeypatch, override, expect
):
    from core.inference import diffusion as D

    monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE", raising = False)
    monkeypatch.setattr(
        speed, "_UNET_WHOLE_COMPILE", frozenset({"_UNet2DConditionModel"}), raising = False
    )
    unet = _UNet2DConditionModel()
    if override.pop("_hooked", False):
        unet._hf_hook = object()
    pipe = types.SimpleNamespace(unet = unet)
    kwargs = dict(
        speed_optims = ("compiled", "cuda_graph"),
        speed_mode = D.SPEED_DEFAULT,
        transformer_quant = None,
        gguf_transformer = False,
        offload_policy = D.OFFLOAD_NONE,
        transformer_cache = None,
        cache_auto = False,
        target = types.SimpleNamespace(device = "cuda", backend = "cuda"),
    )
    kwargs.update(override)
    got = D._bg_compile_module(pipe, **kwargs)
    assert (got is unet) is expect


@pytest.fixture
def probe_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROBE_CACHE", raising = False)
    fp = {"torch": "x", "gpu_uuid": "GPU-1", "env": {}}
    monkeypatch.setattr(probe_cache, "fingerprint", lambda card: dict(fp))
    return fp


def test_probe_table_round_trips_and_misses_on_any_stack_change(probe_home, monkeypatch):
    assert probe_cache.load("cuda:0") is None
    assert probe_cache.store("cuda:0", {"int8": True, "fp8": False, "mxfp8": None})
    assert probe_cache.load("cuda:0") == {
        "int8": True,
        "fp8": False,
    }, "an allocator failure (None) was persisted"

    monkeypatch.setattr(probe_cache, "fingerprint", lambda card: {**probe_home, "torch": "y"})
    assert probe_cache.load("cuda:0") is None


def test_an_all_negative_table_is_not_persisted(probe_home):
    assert probe_cache.store("cuda:0", {"int8": False, "fp8": False, "mxfp8": None}) is False
    assert probe_cache.load("cuda:0") is None


def test_probe_table_kill_switch_and_torn_file(probe_home, monkeypatch):
    probe_cache.store("cuda:0", {"int8": True})
    path = probe_cache._cache_file()
    path.write_text("{not json", encoding = "utf-8")
    assert probe_cache.load("cuda:0") is None
    probe_cache.store("cuda:0", {"int8": True})
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_CACHE", "0")
    assert probe_cache.load("cuda:0") is None
    assert json.loads(path.read_text(encoding = "utf-8"))


def test_a_persisted_table_answers_without_spawning_the_child(probe_home, monkeypatch):
    from core.inference import diffusion_transformer_quant as tq

    probe_cache.store("cuda:0", {"int8": True, "fp8": False})
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {}, raising = False)
    monkeypatch.setattr(tq, "_smoke_cache_device_key", lambda device, ordinal = None: "cuda:0")
    monkeypatch.setattr(tq, "nvfp4_blocked", lambda scheme: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _no_child(card):
        raise AssertionError("spawned the probe child despite a persisted table")

    monkeypatch.setattr(tq, "_child_probe_table", _no_child)
    assert tq._scheme_supported("int8", "cuda") is True
    assert tq._scheme_supported("fp8", "cuda") is False


def test_a_clean_child_table_is_persisted(monkeypatch):
    from core.inference import diffusion_transformer_quant as tq

    stored = {}
    monkeypatch.setattr(
        probe_cache, "store", lambda card, table: stored.setdefault(card, dict(table))
    )

    class _Q:
        def get(self, timeout = None):
            return {"int8": True}

    class _P:
        pid = 1

        def start(self):
            pass

        def is_alive(self):
            return True

    class _Ctx:
        def Queue(self):
            return _Q()

        def Process(self, **kwargs):
            return _P()

    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_context", lambda name: _Ctx())
    monkeypatch.setattr(tq, "_adopt_probe_pid", lambda pid: None, raising = False)
    monkeypatch.setattr(tq, "_close_probe_child", lambda proc, queue: None, raising = False)
    monkeypatch.setattr(tq, "_CHILD_PROBE_UNAVAILABLE", False, raising = False)
    assert tq._child_probe_table("cuda:0") == {"int8": True}
    assert stored == {"cuda:0": {"int8": True}}


def test_mapped_read_only_for_an_accelerator_destination(monkeypatch):
    from core.inference import diffusion_prequant as pq

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_MMAP", raising = False)
    assert pq.prequant_mmap_enabled("cuda")
    assert pq.prequant_mmap_enabled("cuda:1")
    assert not pq.prequant_mmap_enabled("cpu")
    assert not pq.prequant_mmap_enabled(None)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PREQUANT_MMAP", "0")
    assert not pq.prequant_mmap_enabled("cuda")


def test_mapped_read_matches_the_full_read_and_falls_back(tmp_path, monkeypatch):
    from core.inference import diffusion_prequant as pq

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_MMAP", raising = False)
    ckpt = {
        "format": pq.PREQUANT_FORMAT,
        "state_dict": {"w": torch.arange(64, dtype = torch.int8).reshape(8, 8)},
        "metadata": {"scheme": "int8"},
    }
    path = tmp_path / "x.pt"
    torch.save(ckpt, path)

    seen = []
    real = pq._load_prequant_checkpoint

    def spy(p, **kwargs):
        seen.append(kwargs.get("mmap"))
        return real(p, **kwargs)

    monkeypatch.setattr(pq, "_load_prequant_checkpoint", spy)
    mapped = pq._read_prequant_for(str(path), "cuda")
    assert seen == [True]
    assert torch.equal(mapped["state_dict"]["w"], ckpt["state_dict"]["w"])

    seen.clear()
    pq._read_prequant_for(str(path), "cpu")
    assert seen == [None]

    def mmap_refused(p, **kwargs):
        seen.append(kwargs.get("mmap"))
        if kwargs.get("mmap"):
            raise RuntimeError("mmap can only be used with files saved with the zipfile format")
        return real(p, **kwargs)

    seen.clear()
    monkeypatch.setattr(pq, "_load_prequant_checkpoint", mmap_refused)
    out = pq._read_prequant_for(str(path), "cuda")
    assert seen == [True, None]
    assert torch.equal(out["state_dict"]["w"], ckpt["state_dict"]["w"])


def _prewarm_ready(monkeypatch):
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_diffusers_prewarmed", False, raising = False)
    monkeypatch.delenv(torch_warmup.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv(torch_warmup.DISABLE_ENV_VAR, raising = False)
    monkeypatch.setattr(torch_warmup, "_a_local_model_would_load_through_diffusers", lambda: True)
    d = types.ModuleType("diffusers")
    hooks = types.ModuleType("diffusers.hooks")
    monkeypatch.setitem(sys.modules, "diffusers", d)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    imported = []
    real_import = torch_warmup.importlib.import_module

    def fake_import(name, *a, **k):
        if name.startswith("diffusers.models"):
            imported.append(name)
            return types.ModuleType(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(torch_warmup.importlib, "import_module", fake_import)
    return torch_warmup, imported


def test_the_prewarm_also_imports_the_model_classes(monkeypatch):
    torch_warmup, imported = _prewarm_ready(monkeypatch)
    monkeypatch.delenv(torch_warmup.DIFFUSERS_PREWARM_MODELS_ENV_VAR, raising = False)
    assert torch_warmup.prewarm_diffusers_if_image_models_exist() is True
    assert imported == list(torch_warmup._DIFFUSERS_PREWARM_MODEL_MODULES)


def test_the_model_prewarm_has_its_own_switch(monkeypatch):
    torch_warmup, imported = _prewarm_ready(monkeypatch)
    monkeypatch.setenv(torch_warmup.DIFFUSERS_PREWARM_MODELS_ENV_VAR, "0")
    assert torch_warmup.prewarm_diffusers_if_image_models_exist() is True
    assert imported == []


def test_load_progress_stops_rescanning_once_every_byte_is_on_disk(monkeypatch):
    from core.inference import diffusion as D

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROGRESS_LATCH", raising = False)
    scans = []
    loading = D._LoadingState(repo_id = "r", base_repo = "r", expected_bytes = 100)
    fake = types.SimpleNamespace(
        _loading = loading,
        _state = None,
        _cache_bytes = lambda repo: scans.append(repo) or 100,
        _cache_file_bytes = lambda repo, filename: 0,
    )
    first = D.DiffusionBackend.load_progress(fake)
    second = D.DiffusionBackend.load_progress(fake)
    assert first["phase"] == second["phase"] == "finalizing"
    assert len(scans) == 1, "a finished download was rescanned on every poll"
    loading.expected_bytes = 200
    D.DiffusionBackend.load_progress(fake)
    assert len(scans) == 2
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROGRESS_LATCH", "0")
    loading.expected_bytes = 100
    loading.finalized_scan = (100, 100)
    D.DiffusionBackend.load_progress(fake)
    assert len(scans) == 3


def test_load_progress_keeps_scanning_while_downloading():
    from core.inference import diffusion as D

    scans = []
    loading = D._LoadingState(repo_id = "r", base_repo = "r", expected_bytes = 100)
    fake = types.SimpleNamespace(
        _loading = loading,
        _state = None,
        _cache_bytes = lambda repo: scans.append(repo) or 40,
        _cache_file_bytes = lambda repo, filename: 0,
    )
    for _ in range(3):
        assert D.DiffusionBackend.load_progress(fake)["phase"] == "downloading"
    assert len(scans) == 3 and loading.finalized_scan is None


@pytest.fixture
def quant_probe(monkeypatch, probe_home):
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PROBE_PREWARM", raising = False)
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {}, raising = False)
    monkeypatch.setattr(tq, "_smoke_cache_device_key", lambda device, ordinal = None: "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    spawned = []

    def child(card):
        spawned.append(card)
        return {"int8": True, "fp8": None}

    monkeypatch.setattr(tq, "_child_probe_table", child)
    return tq, spawned


def test_boot_probe_fills_the_cache_once(quant_probe):
    tq, spawned = quant_probe
    assert tq.prewarm_probe_table() is True
    assert spawned == ["cuda:0"]
    assert tq._SMOKE_CACHE == {
        ("int8", "cuda:0"): True
    }, "an allocator failure (None) was cached as a verdict"
    assert tq.prewarm_probe_table() is False
    assert spawned == ["cuda:0"]


def test_boot_probe_skips_a_persisted_table_and_honours_its_switch(quant_probe, monkeypatch):
    tq, spawned = quant_probe
    probe_cache.store("cuda:0", {"int8": True})
    assert tq.prewarm_probe_table() is False
    probe_cache._cache_file().unlink()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_PREWARM", "0")
    assert tq.prewarm_probe_table() is False
    assert spawned == []
