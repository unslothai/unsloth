# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm TunableOp scoping for diffusion denoisers (a fake torch.cuda.tunable; no GPU)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_rocm_tunableop as rt  # noqa: E402


class FakeTunable:
    def __init__(self):
        self.enabled = False
        self.tuning = False
        self.calls = []
        self.results = {}
        self.filename = None

    def enable(self, v = True):
        self.enabled = bool(v)
        self.calls.append(("enable", bool(v)))

    def tuning_enable(self, v = True):
        self.tuning = bool(v)
        self.calls.append(("tuning", bool(v)))

    def set_filename(
        self,
        name,
        insert_device_ordinal = False,
    ):
        self.filename = name

    def set_max_tuning_duration(self, ms):
        self.calls.append(("duration", ms))

    def set_max_tuning_iterations(self, n):
        self.calls.append(("iters", n))

    def read_file(self, name = None):
        self.calls.append(("read", name))
        return True

    def get_results(self):
        return tuple(
            ("GemmTunableOp_BFloat16_TN", k, "Gemm_Hipblaslt_1", v) for k, v in self.results.items()
        )

    def get_validators(self):
        return (("PT_VERSION", "2.11.0"), ("GCN_ARCH_NAME", "gfx1151"))


class Denoiser(nn.Module):
    def __init__(self, fake):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.fake = fake
        self.seen = []

    def forward(self, x):
        self.seen.append(self.fake.tuning)
        self.fake.results[f"shape{len(self.fake.results)}"] = 1.0  # a shape tuned inside the call
        return self.lin(x)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeTunable()
    for name in (
        "enable",
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
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(rt.TUNABLEOP_ENV, raising = False)
    for k in [k for k in __import__("os").environ if k.startswith("PYTORCH_TUNABLEOP_")]:
        monkeypatch.delenv(k)
    rt._STATE.update(active = False, depth = 0, written = 0, file = None)
    yield f
    rt._STATE.update(active = False, depth = 0, written = 0, file = None)


def _pipe(fake):
    return types.SimpleNamespace(transformer = Denoiser(fake), vae = nn.Linear(4, 4))


def test_tunes_only_inside_the_denoiser_and_persists(fake, tmp_path):
    pipe = _pipe(fake)
    assert rt.install_for_pipe(pipe)
    assert fake.enabled and not fake.tuning
    assert fake.filename.startswith(str(tmp_path / "tunableop"))
    assert ("duration", rt.MAX_TUNING_MS) in fake.calls
    pipe.transformer(torch.randn(2, 4))
    assert pipe.transformer.seen == [True]
    assert not fake.tuning  # back off for the text encoder / VAE
    from pathlib import Path

    text = Path(fake.filename).read_text(encoding = "utf-8").splitlines()
    assert text[0] == "Validator,PT_VERSION,2.11.0"
    assert text[-1] == "GemmTunableOp_BFloat16_TN,shape0,Gemm_Hipblaslt_1,1.0"
    mtime = Path(fake.filename).stat().st_mtime_ns
    pipe.transformer.forward = lambda x: pipe.transformer.lin(x)  # nothing new tuned: no rewrite
    pipe.transformer(torch.randn(2, 4))
    assert Path(fake.filename).stat().st_mtime_ns == mtime


def test_nested_calls_keep_tuning_on_until_the_outer_returns(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)
    rt._pre(None, ())
    pipe.transformer(torch.randn(1, 4))
    assert fake.tuning  # the outer call is still running
    rt._post(None, (), None)
    assert not fake.tuning


def test_exception_in_the_forward_turns_tuning_off(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)

    def boom(x):
        raise RuntimeError("x")

    pipe.transformer.forward = boom
    with pytest.raises(RuntimeError):
        pipe.transformer(torch.randn(1, 4))
    assert not fake.tuning and rt._STATE["depth"] == 0


@pytest.mark.parametrize("why", ["kill", "user", "arch", "not_rocm"])
def test_declines(fake, monkeypatch, why):
    if why == "kill":
        monkeypatch.setenv(rt.TUNABLEOP_ENV, "0")
    elif why == "user":
        monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    elif why == "arch":
        monkeypatch.setattr(rt, "_arch", lambda device = None: "gfx1100")
    else:
        monkeypatch.setattr(rt, "is_rocm_torch", lambda _t: False)
    pipe = _pipe(fake)
    assert not rt.install_for_pipe(pipe)
    assert not fake.enabled
    assert not pipe.transformer._forward_pre_hooks


def test_uninstall_restores_the_default_pick(fake):
    pipe = _pipe(fake)
    rt.install_for_pipe(pipe)
    rt.uninstall(pipe)
    assert not fake.enabled and not fake.tuning
    assert not pipe.transformer._forward_pre_hooks and not pipe.transformer._forward_hooks
    assert not rt.is_active()


def test_results_file_keys_on_arch_torch_and_hip(fake, tmp_path):
    p = rt.results_file("gfx1151")
    assert p.parent == tmp_path / "tunableop"
    assert p.name.startswith("gfx1151_") and p.suffix == ".csv"


def test_speed_layer_installs_on_rocm_tiers_and_removes_on_off(monkeypatch):
    from core.inference import diffusion_speed as ds

    seen = []
    monkeypatch.setattr(
        rt, "install_for_pipe", lambda pipe, device = None, logger = None: seen.append("install")
    )
    monkeypatch.setattr(rt, "uninstall", lambda pipe = None: seen.append("uninstall"))
    pipe = types.SimpleNamespace(transformer = nn.Linear(2, 2), vae = None)
    rocm = types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = torch.bfloat16)
    ds.apply_speed_optims(pipe, rocm, is_gguf = False, family = None, speed_mode = "off")
    assert seen == ["uninstall"]
    cuda = types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = torch.bfloat16)
    ds.apply_speed_optims(pipe, cuda, is_gguf = False, family = None, speed_mode = "off")
    assert seen == ["uninstall"]  # never on NVIDIA
