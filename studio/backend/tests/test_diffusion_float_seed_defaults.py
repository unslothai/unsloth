# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Built-in float seeds (CPU, ``aot_eager``): a known denoiser's first compile traces once; saved names win."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch._dynamo.symbolic_convert")
pytest.importorskip("diffusers")

from core.inference import diffusion_bg_compile as bg  # noqa: E402

if (
    bg._tensorify_state() is None
):  # pragma: no cover - a torch without TensorifyState has nothing to seed
    pytest.skip(
        reason = "torch without TensorifyState: Dynamo has no float specialisations to seed",
        allow_module_level = True,
    )


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(16, eps = 1e-6)
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x):
        return x + self.lin(self.norm(x))


class Net(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(
        self,
        num_layers = 3,
        eps = 1e-6,
    ) -> None:
        super().__init__()
        self.config = {"num_layers": num_layers, "eps": eps, "_name_or_path": "x"}
        self.blocks = torch.nn.ModuleList([Block() for _ in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _net(**kwargs) -> Net:
    torch.manual_seed(0)
    net = Net(**kwargs).eval()
    for block in net.blocks:
        block.compile(backend = "aot_eager", dynamic = True)
    return net


@pytest.fixture(autouse = True)
def _fresh_dynamo(monkeypatch):
    monkeypatch.delenv(bg.FLOAT_SEEDS_ENV, raising = False)
    monkeypatch.delenv(bg.FLOAT_SEED_DEFAULTS_ENV, raising = False)
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


def _render(net, seed_dir, x):
    seeds = bg.InlineFloatSeeds([net], seed_dir)
    seeds.seed()
    with torch.no_grad():
        out = net(x)
    seeds.keep()
    return seeds, out


def _learned_names(tmp_path, x) -> tuple:
    _, out = _render(_net(), tmp_path / "learn", x)
    names = json.loads((tmp_path / "learn" / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))[
        "names"
    ]
    torch._dynamo.reset()
    return tuple(names), out


def test_builtin_names_trace_a_fresh_compile_once(tmp_path, restarts, monkeypatch):
    x = torch.randn(2, 5, 16)
    names, out_main = _learned_names(tmp_path, x)
    assert names and restarts["n"] >= 1
    key = bg._default_float_seed_key(_net())
    assert key is not None
    monkeypatch.setattr(bg, "_FLOAT_SEED_DEFAULTS", {key: " ".join(names)})
    restarts["n"] = 0
    seeds, out = _render(_net(), tmp_path / "fresh", x)
    assert restarts["n"] == 0
    assert seeds.float_seeds == len(names) and seeds.float_seeds_builtin
    assert seeds.float_seeds_learned == 0
    assert torch.equal(out, out_main)
    saved = json.loads((tmp_path / "fresh" / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))
    assert sorted(saved["names"]) == sorted(names)
    torch._dynamo.reset()
    restarts["n"] = 0
    again, out_again = _render(_net(), tmp_path / "fresh", x)
    assert restarts["n"] == 0 and again.float_seeds == len(names) and not again.float_seeds_builtin
    assert torch.equal(out_again, out_main)


def test_saved_names_win_over_builtin(tmp_path, monkeypatch):
    x = torch.randn(2, 5, 16)
    names, _ = _learned_names(tmp_path, x)
    net = _net()
    monkeypatch.setattr(bg, "_FLOAT_SEED_DEFAULTS", {bg._default_float_seed_key(net): "zf98 zf99"})
    seed_dir = tmp_path / "home"
    seed_dir.mkdir()
    sig = bg._modules_signature([net])
    (seed_dir / bg.FLOAT_SEEDS_FILE).write_text(
        json.dumps({"version": 1, "signature": sig, "names": list(names)}), encoding = "utf-8"
    )
    seeds = bg.InlineFloatSeeds([net], seed_dir)
    seeds.seed()
    assert seeds.float_seeds == len(names) and not seeds.float_seeds_builtin
    state = bg._tensorify_state()
    assert "zf98" not in state.force_specializations


@pytest.mark.parametrize("env", [bg.FLOAT_SEEDS_ENV, bg.FLOAT_SEED_DEFAULTS_ENV])
def test_kill_switches_keep_the_restart(tmp_path, restarts, monkeypatch, env):
    x = torch.randn(2, 5, 16)
    names, _ = _learned_names(tmp_path, x)
    monkeypatch.setattr(
        bg, "_FLOAT_SEED_DEFAULTS", {bg._default_float_seed_key(_net()): " ".join(names)}
    )
    monkeypatch.setenv(env, "0")
    restarts["n"] = 0
    seeds, _ = _render(_net(), tmp_path / "off", x)
    assert restarts["n"] >= 1 and seeds.float_seeds == 0


def test_unknown_and_stale_names_are_harmless(tmp_path, restarts, monkeypatch):
    """A name no float carries is never consulted; a missing one is learned by the restart, as on main."""
    x = torch.randn(2, 5, 16)
    names, out_main = _learned_names(tmp_path, x)
    monkeypatch.setattr(bg, "_FLOAT_SEED_DEFAULTS", {bg._default_float_seed_key(_net()): "zf97"})
    restarts["n"] = 0
    seeds, out = _render(_net(), tmp_path / "stale", x)
    assert seeds.float_seeds == 1
    assert restarts["n"] >= 1
    assert torch.equal(out, out_main)
    saved = json.loads((tmp_path / "stale" / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))
    assert set(names) <= set(saved["names"])


def test_key_ignores_layer_count_but_not_config_or_layout():
    base = bg._default_float_seed_key(_net())
    assert base is not None
    assert bg._default_float_seed_key(_net(num_layers = 1)) == base
    assert bg._default_float_seed_key(_net(eps = 1e-5)) != base
    swapped = _net()
    swapped.blocks[0].lin = torch.nn.Identity()
    assert bg._default_float_seed_key(swapped) != base
    patched = _net()
    patched.blocks[0].forward = lambda x: x
    assert bg._default_float_seed_key(patched) != base
    adapted = _net()
    adapted.peft_config = {"default": object()}
    assert bg._default_float_seed_key(adapted) is None
    no_config = _net()
    no_config.config = None
    assert bg._default_float_seed_key(no_config) is None


def _fused_layer_norm(self, x):
    return torch.nn.functional.layer_norm(
        x, self.normalized_shape, self.weight, self.bias, self.eps
    )


def test_key_moves_with_a_class_patch_inside_the_block(monkeypatch):
    base = bg._default_float_seed_key(_net())
    monkeypatch.setattr(torch.nn.LayerNorm, "forward", _fused_layer_norm)
    assert bg._default_float_seed_key(_net()) != base
    monkeypatch.undo()
    assert bg._default_float_seed_key(_net()) == base

    def calls_helper(self, x):
        return _helper(x)

    monkeypatch.setattr(Block, "forward", calls_helper)
    monkeypatch.setitem(globals(), "_helper", lambda x: x)
    first = bg._default_float_seed_key(_net())
    monkeypatch.setitem(globals(), "_helper", _fused_layer_norm)
    assert bg._default_float_seed_key(_net()) != first


def test_unknown_denoiser_gets_no_builtin_names():
    assert bg._default_float_seeds([_net()]) == set()


def test_table_is_well_formed():
    assert bg._FLOAT_SEED_DEFAULTS
    for key, value in bg._FLOAT_SEED_DEFAULTS.items():
        names = value.split()
        assert len(key) == 64 and int(key, 16) >= 0
        assert names and all(bg._SYMFLOAT_NAME.match(n) for n in names)
        assert len(set(names)) == len(names)
