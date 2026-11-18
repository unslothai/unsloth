# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image first-render compile float seeds (CPU, ``aot_eager``): a load that compiles inside its first render (no
background compile) pre-seeds the SymFloat specialisations its key learned, so a returning load traces once."""

from __future__ import annotations

import dataclasses
import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch._dynamo.symbolic_convert")

from core.inference import diffusion as dmod  # noqa: E402
from core.inference import diffusion_bg_compile as bg  # noqa: E402

if (
    bg._tensorify_state() is None
):  # pragma: no cover - a torch without TensorifyState has nothing to seed
    pytest.skip("this torch has no TensorifyState", allow_module_level = True)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(16, eps = 1e-6)
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x):
        return x + self.lin(self.norm(x))


class Net(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self, dynamic = True) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])
        self._unsloth_regional_compile_kwargs = {"fullgraph": True, "dynamic": dynamic}
        self._unsloth_compile_guard = types.SimpleNamespace(error = None)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@pytest.fixture(autouse = True)
def _fresh_dynamo(monkeypatch):
    monkeypatch.delenv(bg.FLOAT_SEEDS_ENV, raising = False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def restarts(monkeypatch):
    count = {"n": 0}
    cls = torch._dynamo.exc.TensorifyScalarRestartAnalysis
    orig = cls.__init__

    def counting(self, *args, **kwargs):
        count["n"] += 1
        orig(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", counting)
    return count


def _state(
    net,
    seed_dir,
    *,
    speed_optims = ("compiled",),
    bg_compile = None,
):
    return types.SimpleNamespace(
        pipe = types.SimpleNamespace(transformer = net),
        speed_optims = speed_optims,
        compile_cache_ctx = types.SimpleNamespace(dir = seed_dir) if seed_dir is not None else None,
        bg_compile = bg_compile,
        inline_float_seeds = None,
    )


def _net(dynamic = True) -> Net:
    torch.manual_seed(0)
    net = Net(dynamic = dynamic).eval()
    for block in net.blocks:
        block.compile(backend = "aot_eager", dynamic = dynamic)
    return net


def _first_render(seed_dir, x):
    """One load's first compiled render as generate() runs it: seed at the denoise entry, keep at the decode."""
    net = _net()
    state = _state(net, seed_dir)
    seeds = dmod.DiffusionBackend()._inline_float_seeds(state)
    assert seeds is not None
    with bg.seed_before_first_forward(seeds), torch.no_grad():
        out = net(x)
    seeds.keep()
    return seeds, out


def test_load_state_carries_the_seeder():
    assert "inline_float_seeds" in {f.name for f in dataclasses.fields(dmod._LoadState)}


def test_returning_load_traces_its_first_render_once_into_the_same_output(tmp_path, restarts):
    x = torch.randn(2, 5, 16)
    first, out_first = _first_render(tmp_path, x)
    assert restarts["n"] >= 1  # a first-ever load still learns through the restart, as on main
    saved = json.loads((tmp_path / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))
    assert saved["names"] and first.float_seeds == 0

    torch._dynamo.reset()
    restarts["n"] = 0
    second, out_second = _first_render(tmp_path, x)
    assert restarts["n"] == 0
    assert second.float_seeds == len(saved["names"])
    assert torch.equal(out_first, out_second)


def test_seeds_file_is_the_background_compiles(tmp_path):
    """Same key dir and file as the deferred tier's background compile, so either path seeds the other."""
    x = torch.randn(2, 5, 16)
    _first_render(tmp_path, x)
    saved = json.loads((tmp_path / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))
    assert saved["signature"] == bg._modules_signature(_net())


def test_one_seeder_per_load_until_it_kept(tmp_path):
    backend = dmod.DiffusionBackend()
    state = _state(Net(), tmp_path)
    seeds = backend._inline_float_seeds(state)
    assert isinstance(seeds, bg.InlineFloatSeeds)
    assert backend._inline_float_seeds(state) is seeds
    seeds.seed()
    seeds.keep()
    assert backend._inline_float_seeds(state) is None
    assert seeds.modules == []  # the DiTs are not held past the keep


@pytest.mark.parametrize(
    "case",
    ["background_compile", "not_compiled", "static", "no_cache_ctx", "kill_switch"],
)
def test_no_inline_seeds(tmp_path, monkeypatch, case):
    backend = dmod.DiffusionBackend()
    if case == "background_compile":
        state = _state(Net(), tmp_path, bg_compile = object())
    elif case == "not_compiled":
        state = _state(Net(), tmp_path, speed_optims = ("channels_last",))
    elif case == "static":
        state = _state(Net(dynamic = False), tmp_path)
    elif case == "no_cache_ctx":
        state = _state(Net(), None)
    else:
        monkeypatch.setenv(bg.FLOAT_SEEDS_ENV, "0")
        state = _state(Net(), tmp_path)
    assert backend._inline_float_seeds(state) is None
    assert state.inline_float_seeds is None


def test_deferred_tier_is_asked_again_once_it_compiles(tmp_path):
    """Renders 1-2 of the deferred tier run eager; the render that engages an inline compile still seeds."""
    backend = dmod.DiffusionBackend()
    state = _state(Net(), tmp_path, speed_optims = ())
    assert backend._inline_float_seeds(state) is None
    state.speed_optims = ("compiled",)
    assert isinstance(backend._inline_float_seeds(state), bg.InlineFloatSeeds)
