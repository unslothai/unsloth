# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Built-in float seeds (CPU, ``aot_eager``): a first-ever compile of a known denoiser starts from the names a
returning load would read from its saved file, so it traces once; saved names win, and an unknown key seeds nothing."""

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
    """What a first-ever compile learns (main's restart), then a clean slate."""
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
    monkeypatch.setattr(bg, "_FLOAT_SEED_DEFAULTS", {key: names})
    restarts["n"] = 0
    seeds, out = _render(_net(), tmp_path / "fresh", x)
    assert restarts["n"] == 0
    assert seeds.float_seeds == len(names) and seeds.float_seeds_builtin
    assert seeds.float_seeds_learned == 0
    assert torch.equal(out, out_main)
    # Nothing new learned: no file, the next load reads the same built-in names.
    assert not (tmp_path / "fresh" / bg.FLOAT_SEEDS_FILE).exists()


def test_saved_names_win_over_builtin(tmp_path, monkeypatch):
    x = torch.randn(2, 5, 16)
    names, _ = _learned_names(tmp_path, x)
    net = _net()
    monkeypatch.setattr(
        bg, "_FLOAT_SEED_DEFAULTS", {bg._default_float_seed_key(net): ("zf98", "zf99")}
    )
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
    monkeypatch.setattr(bg, "_FLOAT_SEED_DEFAULTS", {bg._default_float_seed_key(_net()): names})
    monkeypatch.setenv(env, "0")
    restarts["n"] = 0
    seeds, _ = _render(_net(), tmp_path / "off", x)
    assert restarts["n"] >= 1 and seeds.float_seeds == 0


def test_unknown_and_stale_names_are_harmless(tmp_path, restarts, monkeypatch):
    """A name no float carries is never consulted; a missing one is learned by the restart, as on main."""
    x = torch.randn(2, 5, 16)
    names, out_main = _learned_names(tmp_path, x)
    monkeypatch.setattr(bg, "_FLOAT_SEED_DEFAULTS", {bg._default_float_seed_key(_net()): ("zf97",)})
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


def test_unknown_denoiser_gets_no_builtin_names():
    assert bg._default_float_seeds([_net()]) == set()


def test_table_is_well_formed():
    assert bg._FLOAT_SEED_DEFAULTS
    for key, names in bg._FLOAT_SEED_DEFAULTS.items():
        assert len(key) == 64 and int(key, 16) >= 0
        assert names and all(bg._SYMFLOAT_NAME.match(n) for n in names)
        assert len(set(names)) == len(names)


def _pinned_stack() -> bool:
    import diffusers
    return torch.__version__.split("+")[0].startswith("2.13.") and diffusers.__version__ == "0.42.1"


@pytest.mark.skipif(
    "not _pinned_stack()", reason = "the table is recorded on torch 2.13 + diffusers 0.42.1"
)
def test_flux2_names_match_a_fresh_compile(restarts):
    """Drift guard: the names do not depend on sizes, so a tiny FLUX.2 relearns the shipped FLUX.2 list."""
    import diffusers

    torch.manual_seed(0)
    model = diffusers.Flux2Transformer2DModel(
        in_channels = 16,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 128,
        num_attention_heads = 1,
        joint_attention_dim = 32,
        timestep_guidance_channels = 256,
        mlp_ratio = 3.0,
        axes_dims_rope = (32, 32, 32, 32),
        rope_theta = 2000,
        eps = 1e-6,
        guidance_embeds = False,
    ).eval()
    model.compile_repeated_blocks(fullgraph = True, dynamic = True, backend = "aot_eager")
    side = 4
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(side), torch.arange(side), torch.arange(1)
    )[None]
    with torch.no_grad():
        model(
            hidden_states = torch.randn(1, side * side, 16),
            encoder_hidden_states = torch.randn(1, 8, 32),
            timestep = torch.tensor([0.5]),
            img_ids = ids,
            txt_ids = torch.zeros(1, 8, 4, dtype = torch.long),
        )
    assert restarts["n"] >= 1
    assert sorted(bg._tensorify_state().force_specializations) == sorted(bg._FLUX2_SEEDS)
