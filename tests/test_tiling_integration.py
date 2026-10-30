# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Unsloth side of sequence tiling. CPU only; unsloth/tiling.py is loaded by path (no GPU init)."""

import ast
import importlib.util
import pathlib
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
registry = pytest.importorskip("unsloth_zoo.tiling_registry")


def _tiling():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_tiling_under_test", ROOT / "unsloth" / "tiling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FooMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))


def _model(n = 2):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(n)])
    for layer in model.layers:
        layer.mlp = FooMLP()
    return model


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TILED_MLP", raising = False)
    monkeypatch.delenv("UNSLOTH_TILED_MLP_MODULES", raising = False)
    yield


@pytest.mark.parametrize(
    ("env", "kwarg", "want"),
    [
        (None, False, None),
        (None, None, None),
        (None, True, True),
        (None, "auto", "auto"),
        (None, "rows:256", "rows:256"),
        (None, "0", None),
        ("0", False, None),
        ("0", True, True),  # old behaviour: the kwarg turns it on even with UNSLOTH_TILED_MLP=0
        ("arctic", False, "arctic"),
        ("1", False, "1"),
        ("target", False, "target"),
        ("target:2", True, "target:2"),
        ("target:2:mlp,ffn", False, "target:2:mlp,ffn"),
        ("auto", "rows:8", "auto"),  # the environment wins over the kwarg, as before
        ("false", True, True),
        ("off", True, True),
        ("No", True, True),
        ("none", True, True),
        ("FALSE", False, None),
        (None, 0, None),
        (None, "false", None),
        (None, "Off", None),
        (None, "no", None),
        (None, "none", None),
    ],
)
def test_loader_mode_spellings(monkeypatch, env, kwarg, want):
    if env is not None:
        monkeypatch.setenv("UNSLOTH_TILED_MLP", env)
    tiling = _tiling()
    assert tiling.resolve_loader_mode(kwarg) == want
    if want is not None:
        assert registry.parse_mode(want) is not None  # every accepted spelling parses


def test_loader_tile_mlp_installs_and_reports(capsys):
    tiling = _tiling()
    model = _model()
    assert tiling.loader_tile_mlp(model, False) is None and not registry.tiled_modules(model)
    report = tiling.loader_tile_mlp(model, True)
    assert len(report.tiled) == 2 and "Tiled MLP (auto) on 2 modules" in capsys.readouterr().out


def test_tile_mlp_and_untile_mlp():
    tiling = _tiling()
    model = _model()
    x = torch.randn(2, 9, 8)
    want = sum(layer.mlp(x) for layer in model.layers)
    report = tiling.tile_mlp(model, rows = 2, verbose = False)
    assert report.plan.mode == "rows" and len(registry.tiled_modules(model)) == 2
    torch.testing.assert_close(sum(layer.mlp(x) for layer in model.layers), want)
    assert tiling.tile_mlp(model, target_gb = 1.5, verbose = False).plan.target_gb == 1.5
    assert len(tiling.tile_mlp(model, rows = 2, modules = ["layers.0.mlp"], verbose = False).tiled) == 1
    assert registry.is_tiled(model.layers[0].mlp) and not registry.is_tiled(model.layers[1].mlp)
    with pytest.raises(ValueError):
        tiling.tile_mlp(model, rows = 2, target_gb = 1.0)
    assert tiling.untile_mlp(model) == 1
    assert not registry.tiled_modules(model)
    torch.testing.assert_close(sum(layer.mlp(x) for layer in model.layers), want)


@pytest.mark.parametrize("off", ["0", "false", "OFF", "no", "none"])
def test_tile_mlp_ignores_a_disabled_environment(monkeypatch, off):
    monkeypatch.setenv("UNSLOTH_TILED_MLP", off)
    assert _tiling().tile_mlp(_model(), verbose = False).plan.mode == "auto"
    monkeypatch.setenv("UNSLOTH_TILED_MLP", "rows:4")
    assert _tiling().tile_mlp(_model(), verbose = False).plan.rows == 4


def test_trainer_revalidation_covers_peft_added_modules():
    peft = pytest.importorskip("peft")
    tiling = _tiling()
    model = _model()
    tiling.loader_tile_mlp(model, "rows:2")
    model = peft.get_peft_model(
        model, peft.LoraConfig(r = 2, target_modules = ["up", "down"], lora_dropout = 0.1)
    )
    extra = torch.nn.Module()
    extra.mlp = FooMLP()
    model.base_model.model.layers.append(extra)
    report = tiling.revalidate_for_trainer(types.SimpleNamespace(model = model))
    assert len(report.tiled) == 3 and registry.is_tiled(extra.mlp)
    assert tiling.revalidate_for_trainer(types.SimpleNamespace(model = _model())) is None
    assert tiling.revalidate_for_trainer(types.SimpleNamespace(model = None)) is None
    model.train()
    x = torch.randn(2, 9, 8)
    out = sum(layer.mlp(x) for layer in model.base_model.model.layers)
    out.sum().backward()
    lora = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert lora and all(p.grad is not None for p in lora)


def _calls(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding = "utf-8"))
    return [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name
    ]


def test_loader_call_sites_use_the_new_install():
    src = (ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    assert "_tiled_mlp_ddp" not in src and "patch_tiled_mlp" not in src
    calls = _calls("unsloth/models/loader.py", "loader_tile_mlp")
    assert len(calls) == 2  # FastLanguageModel and FastModel
    for call in calls:
        assert [getattr(a, "id", None) for a in call.args] == ["model", "unsloth_tiled_mlp"]
    assert not (ROOT / "unsloth" / "models" / "_tiled_mlp_ddp.py").exists()


def test_trainer_init_revalidates():
    tree = ast.parse((ROOT / "unsloth" / "models" / "_utils.py").read_text(encoding = "utf-8"))
    init = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_trainer_init"
    )
    names = [getattr(n.func, "id", None) for n in ast.walk(init) if isinstance(n, ast.Call)]
    assert "revalidate_for_trainer" in names
    order = [n for n in names if n in ("_original_trainer_init", "revalidate_for_trainer")]
    assert order.index("_original_trainer_init") < order.index("revalidate_for_trainer")


def test_sentence_transformer_passes_the_kwarg_through():
    tree = ast.parse(
        (ROOT / "unsloth" / "models" / "sentence_transformer.py").read_text(encoding = "utf-8")
    )
    passed = [
        kw
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for kw in n.keywords
        if kw.arg == "unsloth_tiled_mlp" and getattr(kw.value, "id", None) == "unsloth_tiled_mlp"
    ]
    assert passed


def test_package_exports_tile_mlp_on_every_backend():
    tree = ast.parse((ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8"))
    top = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "tiling" and n.level == 1
    ]
    assert top and {a.name for a in top[0].names} == {"tile_mlp", "untile_mlp"}


@pytest.mark.parametrize("off", [False, None, 0, "0", "false", "no"])
def test_tiling_off_never_imports_the_registry(monkeypatch, off):
    import sys

    monkeypatch.setitem(sys.modules, "unsloth_zoo.tiling_registry", None)
    tiling = _tiling()
    tiling.check_tiling_available(off)
    assert tiling.loader_tile_mlp(_model(), off) is None


@pytest.mark.parametrize(
    ("env", "kwarg"), [(None, True), (None, "auto"), ("arctic", False), ("0", True)]
)
def test_tiling_on_without_the_registry_fails_fast(monkeypatch, env, kwarg):
    import sys

    if env is not None:
        monkeypatch.setenv("UNSLOTH_TILED_MLP", env)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.tiling_registry", None)
    with pytest.raises(ImportError, match = "newer unsloth_zoo"):
        _tiling().check_tiling_available(kwarg)


def test_loaders_check_the_registry_before_loading_weights():
    tree = ast.parse((ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8"))
    loaders = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "from_pretrained"
        and any(
            getattr(c.func, "id", None) == "loader_tile_mlp"
            for c in ast.walk(n)
            if isinstance(c, ast.Call)
        )
    ]
    assert len(loaders) == 2  # FastLanguageModel and FastModel
    for fn in loaders:
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
        check = [c.lineno for c in calls if getattr(c.func, "id", None) == "check_tiling_available"]
        loads = [c.lineno for c in calls if getattr(c.func, "attr", None) == "from_pretrained"]
        assert check and loads and min(check) < min(loads)


def test_model_init_revalidates_the_fresh_model():
    tree = ast.parse((ROOT / "unsloth" / "models" / "_utils.py").read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_call_model_init"
    )
    calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "revalidate_for_trainer"
    ]
    assert calls and [getattr(a, "id", None) for a in calls[0].args] == ["self", "model"]


def test_revalidate_for_trainer_takes_the_model_explicitly():
    tiling = _tiling()
    model = _model()
    tiling.tile_mlp(model, rows = 2, verbose = False)
    model.layers.append(torch.nn.Module())
    model.layers[-1].mlp = FooMLP()
    trainer = types.SimpleNamespace(model = _model())  # still the previous instance
    tiling.revalidate_for_trainer(trainer, model)
    assert len(registry.tiled_modules(model)) == 3


def test_tile_mlp_on_an_mlx_model_says_it_is_unsupported():
    from unsloth.tiling import tile_mlp, untile_mlp

    MlxModule = type("Module", (), {"__module__": "mlx.nn.layers.base"})
    model = type("Model", (MlxModule,), {})()
    for fn in (tile_mlp, untile_mlp):
        with pytest.raises(NotImplementedError, match = "not supported on MLX"):
            fn(model)
