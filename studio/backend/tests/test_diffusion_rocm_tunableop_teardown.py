# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unloading a diffusion pipeline restores the process-global TunableOp state (a fake torch.cuda.tunable; no GPU)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_rocm_tunableop as rt  # noqa: E402
from core.inference import diffusion_speed  # noqa: E402

_VALIDATORS = (("PT_VERSION", "2.11.0"), ("GCN_ARCH_NAME", "gfx1151"))


class FakeTunable:
    def __init__(self, enabled = False):
        self.enabled = enabled
        self.tuning = False

    def enable(self, v = True):
        self.enabled = bool(v)

    def is_enabled(self):
        return self.enabled

    def tuning_enable(self, v = True):
        self.tuning = bool(v)

    def set_filename(
        self,
        name,
        insert_device_ordinal = False,
    ):
        pass

    def set_max_tuning_duration(self, ms):
        pass

    def set_max_tuning_iterations(self, n):
        pass

    def read_file(self, name = None):
        return True

    def get_results(self):
        return ()

    def get_validators(self):
        return _VALIDATORS


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeTunable()
    for name in (
        "enable",
        "is_enabled",
        "tuning_enable",
        "set_filename",
        "set_max_tuning_duration",
        "set_max_tuning_iterations",
        "read_file",
        "get_results",
        "get_validators",
    ):
        monkeypatch.setattr(torch.cuda.tunable, name, getattr(f, name))
    monkeypatch.setattr(rt, "is_rocm_torch", lambda _t: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1151")
    monkeypatch.setattr(rt, "_prime", lambda _torch: None)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(rt.TUNABLEOP_ENV, raising = False)
    for k in [k for k in __import__("os").environ if k.startswith("PYTORCH_TUNABLEOP_")]:
        monkeypatch.delenv(k)
    tables = tmp_path / "tables"
    tables.mkdir()
    lines = [f"Validator,{k},{v}" for k, v in _VALIDATORS]
    lines.append("GemmTunableOp_BFloat16_TN,tn_8_8_8_ld_8_8_8,Gemm_Hipblaslt_1,1.0")
    (tables / "gfx1151_test.csv").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    monkeypatch.setattr(rt, "_TABLE_DIR", tables)
    rt._STATE.update(active = False, depth = 0, written = 0, file = None, tune = False)
    rt._STATE.get("owners", set()).clear()
    yield f
    rt._STATE.update(active = False, depth = 0, written = 0, file = None, tune = False)
    rt._STATE.get("owners", set()).clear()


def _pipe():
    return types.SimpleNamespace(transformer = nn.Linear(4, 4), vae = nn.Linear(4, 4))


def test_unload_restores_tunableop_only_after_the_last_pipeline(fake):
    image, video = _pipe(), _pipe()
    assert rt.install_for_pipe(image) and fake.enabled
    assert rt.install_for_pipe(video)
    diffusion_speed.release_rocm_tunableop(image)
    assert fake.enabled and rt.is_active()  # the co-resident pipeline still uses the table
    diffusion_speed.release_rocm_tunableop(video)
    assert not fake.enabled and not rt.is_active()


def test_unload_restores_a_previously_enabled_state(fake):
    fake.enabled = True
    pipe = _pipe()
    assert rt.install_for_pipe(pipe)
    diffusion_speed.release_rocm_tunableop(pipe)
    assert fake.enabled and not rt.is_active()


def test_user_owned_tunableop_is_left_alone(fake, monkeypatch):
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    fake.enabled = True
    pipe = _pipe()
    assert not rt.install_for_pipe(pipe)
    diffusion_speed.release_rocm_tunableop(pipe)
    assert fake.enabled


def test_image_and_video_teardown_release_tunableop():
    import ast
    from pathlib import Path

    root = Path(rt.__file__).resolve().parent
    for fname, method in (
        ("diffusion.py", "_unload_locked"),
        ("video.py", "_teardown_state_locked"),
    ):
        tree = ast.parse((root / fname).read_text(encoding = "utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method)
        called = {
            c.func.id
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "_release_rocm_tunableop" in called, fname


def test_a_load_racing_the_last_release_keeps_its_claim(fake, monkeypatch):
    # A load that lands between the last-owner check and the restore must stay installed.
    image, video = _pipe(), _pipe()
    assert rt.install_for_pipe(image)
    raced = []
    real_uninstall = rt.uninstall

    def racing_uninstall(pipe = None):
        if not raced:
            raced.append(rt.install_for_pipe(video))
        return real_uninstall(pipe)

    monkeypatch.setattr(rt, "uninstall", racing_uninstall)
    diffusion_speed.release_rocm_tunableop(image)
    if not raced:
        raced.append(rt.install_for_pipe(video))
    assert raced == [True]
    assert rt.is_active() and fake.enabled and rt._STATE["owners"]


def test_install_waits_for_a_release_in_progress(fake):
    import threading

    image, video = _pipe(), _pipe()
    assert rt.install_for_pipe(image)
    entered, go = threading.Event(), threading.Event()

    class SlowOwners(set):
        def discard(self, x):
            entered.set()
            go.wait(5)
            return super().discard(x)

    rt._STATE["owners"] = SlowOwners(rt._STATE["owners"])
    try:
        t = threading.Thread(target = diffusion_speed.release_rocm_tunableop, args = (image,))
        t.start()
        assert entered.wait(5)
        result = []
        u = threading.Thread(target = lambda: result.append(rt.install_for_pipe(video)))
        u.start()
        go.set()
        t.join(5)
        u.join(5)
        assert result == [True]
        assert rt.is_active() and fake.enabled and rt._STATE["owners"]
    finally:
        rt._STATE["owners"] = set(rt._STATE["owners"])


def test_a_failed_load_that_drops_its_pipe_releases_the_claim(fake):
    import gc

    class Pipe:  # weakref-able, like a diffusers pipeline
        def __init__(self):
            self.transformer = nn.Linear(4, 4)

    pipe = Pipe()
    assert rt.install_for_pipe(pipe) and fake.enabled
    del pipe  # a cancelled video load drops the unpublished pipe without release()
    gc.collect()
    assert not rt.is_active() and not fake.enabled and not rt._STATE["owners"]
