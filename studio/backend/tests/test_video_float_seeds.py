# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video first-clip compile float seeds (CPU, ``aot_eager``): the inline counterpart of the background compile's."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch._dynamo.symbolic_convert")

from core.inference import diffusion_bg_compile as bg  # noqa: E402
from core.inference import video as video_mod  # noqa: E402

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


class OtherBlock(Block):
    def forward(self, x):
        return x - self.lin(self.norm(x))


class Net(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self, dynamic = True) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])
        self._unsloth_regional_compile_kwargs = {"fullgraph": True, "dynamic": dynamic}

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class OtherNet(Net):
    _repeated_blocks = ["OtherBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([OtherBlock() for _ in range(2)])


def _net() -> Net:
    torch.manual_seed(0)
    net = Net().eval()
    for block in net.blocks:
        block.compile(backend = "aot_eager", dynamic = True)
    return net


@pytest.fixture(autouse = True)
def _fresh_dynamo(monkeypatch, tmp_path):
    monkeypatch.delenv(bg.FLOAT_SEEDS_ENV, raising = False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", str(tmp_path / "cc"))
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


def _clip(seed_dir, x):
    """One load's first clip as video.py runs it: seed at the denoise entry, keep at the decode entry."""
    net = _net()
    seeds = bg.InlineFloatSeeds([net], seed_dir)
    seeds.seed()
    with torch.no_grad():
        out = net(x)
    seeds.keep()
    return seeds, net, out


def test_second_load_traces_once_into_the_same_graph(tmp_path, restarts):
    x = torch.randn(2, 5, 16)
    first, _, out_first = _clip(tmp_path, x)
    assert restarts["n"] >= 1  # the unseeded first clip restarts, as on main
    saved = json.loads((tmp_path / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))
    assert saved["names"] and saved["signature"] == bg._code_signature(_net())
    assert first.describe() == {"float_seeds": 0, "float_seeds_learned": len(saved["names"])}

    torch._dynamo.reset()
    restarts["n"] = 0
    second, _, out_second = _clip(tmp_path, x)
    assert restarts["n"] == 0
    assert second.float_seeds == len(saved["names"]) and second.done()
    assert torch.equal(out_first, out_second)


def test_first_forward_hook_seeds_before_the_compile(tmp_path, restarts):
    x = torch.randn(2, 5, 16)
    _clip(tmp_path, x)
    torch._dynamo.reset()
    restarts["n"] = 0
    net = _net()
    seeds = bg.InlineFloatSeeds([net], tmp_path)
    with video_mod._seed_before_first_forward(seeds), torch.no_grad():
        net(x)
    assert seeds.seeded and seeds.float_seeds > 0 and restarts["n"] == 0
    assert not net._forward_pre_hooks


def test_seeded_first_clip_does_not_recompile_on_later_clips(tmp_path):
    from torch._dynamo.utils import counters

    x = torch.randn(2, 5, 16)
    _clip(tmp_path, x)
    torch._dynamo.reset()
    _, net, _ = _clip(tmp_path, x)
    before = counters["stats"].get("unique_graphs", 0)
    with torch.no_grad():
        for length in (5, 7, 9):
            net(torch.randn(2, length, 16))
    assert counters["stats"].get("unique_graphs", 0) == before


def test_seed_and_keep_run_once_and_later_names_are_not_kept(tmp_path):
    x = torch.randn(2, 5, 16)
    seeds, _, _ = _clip(tmp_path, x)
    path = tmp_path / bg.FLOAT_SEEDS_FILE
    saved = path.read_text(encoding = "utf-8")
    # A later compile in the process (the VAE decode's, another backend's) adds names after the keep.
    bg._tensorify_state().specialize("zf99")
    assert seeds.seed() == 0
    seeds.keep()
    assert path.read_text(encoding = "utf-8") == saved
    assert seeds.modules == []  # the DiTs are not held past the keep


def test_kill_switch_keeps_the_restart(tmp_path, restarts, monkeypatch):
    monkeypatch.setenv(bg.FLOAT_SEEDS_ENV, "0")
    x = torch.randn(2, 5, 16)
    _clip(tmp_path, x)
    assert not (tmp_path / bg.FLOAT_SEEDS_FILE).exists()
    torch._dynamo.reset()
    restarts["n"] = 0
    seeds, _, _ = _clip(tmp_path, x)
    assert restarts["n"] >= 1 and seeds.float_seeds == 0


def test_a_single_module_list_signs_like_the_background_compile():
    a, b = Net(), OtherNet()
    assert bg._modules_signature([a]) == bg._code_signature(a) == bg._modules_signature(a)
    pair = bg._modules_signature([a, b])
    assert pair not in (None, bg._code_signature(a), bg._code_signature(b))
    assert bg._modules_signature([a, torch.nn.Linear(2, 2)]) is None


def test_only_dynamic_regional_compiles_take_seeds():
    dense, static, plain = Net(), Net(dynamic = False), torch.nn.Linear(2, 2)
    assert video_mod._float_seed_dits(types.SimpleNamespace(transformer = dense)) == [dense]
    # LTX-2 / 2.5 compile static: no SymFloat, no restart, nothing to seed.
    assert video_mod._float_seed_dits(types.SimpleNamespace(transformer = static)) == []
    assert video_mod._float_seed_dits(types.SimpleNamespace(transformer = plain)) == []
    both = types.SimpleNamespace(transformer = dense, transformer_2 = Net())
    assert video_mod._float_seed_dits(both) == [dense, both.transformer_2]


def _seed_dir(pipe, **over):
    kw = {
        "family": "wan2.1-t2v-1.3b",
        "dtype": "bfloat16",
        "quant": None,
        "attention_backend": None,
    }
    kw.update(over)
    return video_mod._video_float_seed_dir(pipe, **kw)


def test_seed_dir_is_keyed_like_the_compile_cache(tmp_path, monkeypatch):
    net = Net()
    pipe = types.SimpleNamespace(transformer = net)
    dits, path = _seed_dir(pipe)
    assert dits == [net]
    assert path.parent == tmp_path / "cc" / "video_float_seeds" and len(path.name) == 32
    assert _seed_dir(pipe, dtype = torch.bfloat16)[1] == path
    assert _seed_dir(pipe, dtype = "float16")[1] != path
    assert _seed_dir(pipe, quant = "int8")[1] != path
    assert _seed_dir(pipe, family = "wan2.2-ti2v-5b")[1] != path
    assert _seed_dir(types.SimpleNamespace(transformer = Net(dynamic = False))) == ([], None)
    monkeypatch.setenv(bg.FLOAT_SEEDS_ENV, "0")
    assert _seed_dir(pipe) == ([], None)


def _state(pipe, bg_compile = None):
    return types.SimpleNamespace(
        pipe = pipe,
        family = types.SimpleNamespace(name = "wan2.1-t2v-1.3b"),
        dtype = "bfloat16",
        transformer_quant = None,
        attention_backend = None,
        bg_compile = bg_compile,
    )


def test_backend_holds_one_seeder_per_load_until_it_kept():
    backend = video_mod.VideoBackend()
    state = _state(types.SimpleNamespace(transformer = Net()))
    seeds = backend._inline_float_seeds(state)
    assert isinstance(seeds, bg.InlineFloatSeeds)
    assert backend._inline_float_seeds(state) is seeds
    seeds.seed()
    seeds.keep()
    assert backend._inline_float_seeds(state) is None
    other = _state(types.SimpleNamespace(transformer = Net()))
    fresh = backend._inline_float_seeds(other)
    assert fresh is not None and fresh is not seeds


def test_backend_leaves_a_background_compile_its_own_seeds():
    backend = video_mod.VideoBackend()
    state = _state(types.SimpleNamespace(transformer = Net()), bg_compile = object())
    assert backend._inline_float_seeds(state) is None
    static = _state(types.SimpleNamespace(transformer = Net(dynamic = False)))
    assert backend._inline_float_seeds(static) is None
