"""Unit tests for the Gefen-X optimizer integration (``unsloth.optimizers.gefenx``).

These tests exercise the config→constructor mapping and parameter routing WITHOUT
requiring a GPU, the real ``gefen`` package, or a full ``import unsloth`` (which
triggers GPU init). The helper module has no relative imports, so it is loaded by
file path and handed a fake ``gefen`` module that records constructor arguments.
"""

import importlib.util
import pathlib
import sys
import types
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

# --- Load unsloth/optimizers/gefenx.py standalone (no unsloth package import) ---
_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "unsloth" / "optimizers" / "gefenx.py"
_spec = importlib.util.spec_from_file_location("_unsloth_gefenx_under_test", _MODULE_PATH)
gefenx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gefenx)


# --- Minimal stand-ins so the tests need neither torch nor unsloth.trainer -------
@dataclass
class _GefenXConfig:
    fused: bool = True
    factored_v_2d: bool = True
    force_1d_period_one: bool = False
    force_2d_period_one: bool = False
    period_one_substrings: tuple = ()
    codebook_refresh_every: int = 0
    stochastic_round: bool = False
    capturable: bool = False
    betas: Optional[tuple] = None
    eps: Optional[float] = None
    extra_kwargs: dict = field(default_factory = dict)


@dataclass
class _GefenXMuonConfig:
    fused: bool = True
    adjust_lr_fn: str = "match_rms_adamw"
    backup_lr_scale: Optional[float] = 0.5
    backup_lr: Optional[float] = None
    muon_lr: Optional[float] = None
    muon_weight_decay: Optional[float] = None
    backup_weight_decay: Optional[float] = None
    backup_1d_period_one: bool = True
    backup_2d_period_one: bool = False
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_schedule: str = "tuned3"
    sharded_mode: str = "exact"
    fp8_ns: bool = False
    stochastic_round: bool = False
    normuon: bool = True
    cautious: bool = False
    capturable: bool = False
    backup_substrings: Optional[List[str]] = None
    betas: Optional[tuple] = None
    eps: Optional[float] = None
    extra_kwargs: dict = field(default_factory = dict)


class _FakeParam:
    """Enough of an nn.Parameter for make_gefenx_param_groups / grouping."""

    def __init__(self, requires_grad = True):
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return list(self._named)


@pytest.fixture
def fake_gefen(monkeypatch):
    """Install a fake ``gefen`` module that records what got constructed."""
    captured = {}

    class _FakeGefen:
        def __init__(self, params, **kwargs):
            captured["gefen"] = {"params": params, "kwargs": kwargs}
            # Mimic torch.optim.Optimizer.param_groups shape for downstream code.
            self.param_groups = params

    def _from_model(
        model,
        *,
        backup_substrings = None,
        **kwargs,
    ):
        captured["muon"] = {
            "model": model,
            "backup_substrings": backup_substrings,
            "kwargs": kwargs,
        }
        return "MUON_OPTIMIZER"

    class _FakeHybrid:
        from_model = staticmethod(_from_model)

    module = types.ModuleType("gefen")
    module.Gefen = _FakeGefen
    module.GefenMuonHybrid = _FakeHybrid
    monkeypatch.setitem(sys.modules, "gefen", module)
    # The CUDA gate imports torch; these constructor-mapping tests are torch-free
    # by design, so stub it out here. The real gate is exercised by the dedicated
    # test_gate_* tests (which do require torch).
    monkeypatch.setattr(gefenx, "_require_nvidia_cuda", lambda: None)
    return captured


# --------------------------------------------------------------------------- #
# coerce_optim_arg
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("False", False),
        ("none", None),
        ("null", None),
        ("5", 5),
        ("6.0e-6", 6.0e-6),
        ("match_rms_adamw", "match_rms_adamw"),
        (True, True),  # non-strings pass through unchanged
        (0.5, 0.5),
        (None, None),
    ],
)
def test_coerce_optim_arg(raw, expected):
    assert gefenx.coerce_optim_arg(raw) == expected


# --------------------------------------------------------------------------- #
# make_gefenx_param_groups
# --------------------------------------------------------------------------- #
def test_param_groups_single_group_without_embedding_lr():
    model = _FakeModel(
        [
            ("model.layers.0.self_attn.q_proj.weight", _FakeParam()),
            ("model.embed_tokens.modules_to_save.default.weight", _FakeParam()),
            ("model.layers.0.frozen.weight", _FakeParam(requires_grad = False)),
        ]
    )
    groups = gefenx.make_gefenx_param_groups(model, lr = 1e-4, weight_decay = 0.01)
    # No embedding_lr => one group; frozen params excluded.
    assert len(groups) == 1
    assert len(groups[0]["params"]) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[0]["weight_decay"] == 0.01


def test_param_groups_splits_embedding_lr():
    model = _FakeModel(
        [
            ("model.layers.0.self_attn.q_proj.weight", _FakeParam()),
            ("model.embed_tokens.modules_to_save.default.weight", _FakeParam()),
        ]
    )
    groups = gefenx.make_gefenx_param_groups(model, lr = 1e-4, weight_decay = 0.0, embedding_lr = 5e-6)
    assert len(groups) == 2
    non_embed, embed = groups
    assert non_embed["lr"] == 1e-4 and len(non_embed["params"]) == 1
    assert embed["lr"] == 5e-6 and len(embed["params"]) == 1


def test_param_groups_emit_named_pairs():
    # Params must be (name, param) pairs so gefen keeps real names (period_one_substrings).
    p = _FakeParam()
    model = _FakeModel([("model.layers.0.self_attn.q_proj.weight", p)])
    groups = gefenx.make_gefenx_param_groups(model, lr = 1e-4, weight_decay = 0.0)
    entry = groups[0]["params"][0]
    assert isinstance(entry, tuple) and entry[0] == "model.layers.0.self_attn.q_proj.weight"
    assert entry[1] is p


def test_param_groups_embedding_only_omits_empty_group():
    # All trainable params are PEFT modules_to_save + embedding_lr => a single
    # embeddings group, NOT an empty non_embeddings group (gefen rejects empties).
    model = _FakeModel(
        [
            ("model.embed_tokens.modules_to_save.default.weight", _FakeParam()),
            ("lm_head.modules_to_save.default.weight", _FakeParam()),
        ]
    )
    groups = gefenx.make_gefenx_param_groups(model, lr = 1e-4, weight_decay = 0.0, embedding_lr = 5e-6)
    assert len(groups) == 1
    assert groups[0]["lr"] == 5e-6
    assert len(groups[0]["params"]) == 2
    assert all(len(g["params"]) > 0 for g in groups)


# --------------------------------------------------------------------------- #
# build_gefenx_optimizer
# --------------------------------------------------------------------------- #
def test_build_gefenx_forwards_config_and_falls_back_to_trainer_betas(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXConfig(fused = True, factored_v_2d = False, stochastic_round = True)
    opt = gefenx.build_gefenx_optimizer(
        model,
        config,
        lr = 1e-4,
        weight_decay = 0.01,
        betas = (0.9, 0.95),
        eps = 1e-8,
    )
    kw = fake_gefen["gefen"]["kwargs"]
    assert kw["fused"] is True
    assert kw["factored_v_2d"] is False
    assert kw["stochastic_round"] is True
    # betas/eps not set on the config => inherit the trainer's AdamW values.
    assert kw["betas"] == (0.9, 0.95)
    assert kw["eps"] == 1e-8
    # Empty period_one_substrings is dropped, not forwarded as ().
    assert "period_one_substrings" not in kw
    assert opt.param_groups == fake_gefen["gefen"]["params"]


def test_extra_kwargs_reserved_keys_are_dropped(fake_gefen):
    # Reserved keys (would collide with the builders' explicit args -> TypeError)
    # are dropped from extra_kwargs with a warning; non-reserved keys pass through.
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXConfig(
        extra_kwargs = {"lr": 9.9, "weight_decay": 9.9, "codebook_refresh_every": "50"}
    )
    with pytest.warns(UserWarning, match = "reserved key"):
        gefenx.build_gefenx_optimizer(
            model,
            config,
            lr = 1e-4,
            weight_decay = 0.01,
            betas = (0.9, 0.999),
            eps = 1e-8,
        )
    kw = fake_gefen["gefen"]["kwargs"]
    # Build succeeds (no duplicate-keyword TypeError) and the reserved extra_kwargs
    # values did NOT override the real builder args (9.9 dropped, 1e-4 / 0.01 win).
    assert kw["lr"] == 1e-4
    assert kw["weight_decay"] == 0.01
    assert kw["codebook_refresh_every"] == 50  # allowed key -> coerced


def test_muon_extra_kwargs_reserved_backup_substrings_dropped(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig(extra_kwargs = {"backup_substrings": ["x"], "ns_steps": "7"})
    with pytest.warns(UserWarning, match = "reserved key"):
        gefenx.build_gefenx_muon_optimizer(
            model,
            config,
            lr = 1e-4,
            weight_decay = 0.0,
            betas = (0.9, 0.999),
            eps = 1e-8,
        )
    kw = fake_gefen["muon"]["kwargs"]
    # backup_substrings is passed as an explicit arg, not via **kwargs.
    assert "backup_substrings" not in kw
    assert fake_gefen["muon"]["backup_substrings"] is None
    assert kw["ns_steps"] == 7


def test_extra_kwargs_reserved_model_and_embedding_keys_dropped(fake_gefen):
    # model / embedding_lr / embedding_learning_rate would collide (model) or be
    # unexpected keywords; they are dropped from extra_kwargs, and the real model
    # is still routed through from_model positionally.
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig(
        extra_kwargs = {"model": "oops", "embedding_lr": 1e-5, "embedding_learning_rate": 1e-5}
    )
    with pytest.warns(UserWarning, match = "reserved key"):
        gefenx.build_gefenx_muon_optimizer(
            model,
            config,
            lr = 1e-4,
            weight_decay = 0.0,
            betas = (0.9, 0.999),
            eps = 1e-8,
        )
    kw = fake_gefen["muon"]["kwargs"]
    assert "model" not in kw and "embedding_lr" not in kw
    assert "embedding_learning_rate" not in kw
    assert fake_gefen["muon"]["model"] is model


def test_none_config_fields_are_not_forwarded(fake_gefen):
    # None config fields are "unset" -> not passed, so gefen keeps its own defaults.
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig(muon_lr = None, muon_weight_decay = None, backup_weight_decay = None)
    gefenx.build_gefenx_muon_optimizer(
        model,
        config,
        lr = 1e-4,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    kw = fake_gefen["muon"]["kwargs"]
    assert "muon_lr" not in kw
    assert "muon_weight_decay" not in kw
    assert "backup_weight_decay" not in kw


def test_build_gefenx_config_betas_override_and_extra_kwargs(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXConfig(
        betas = (0.8, 0.9),
        eps = 1e-6,
        period_one_substrings = ("embed", "lm_head"),
        extra_kwargs = {"codebook_refresh_every": "100"},  # string coerced to int
    )
    gefenx.build_gefenx_optimizer(
        model,
        config,
        lr = 1e-4,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    kw = fake_gefen["gefen"]["kwargs"]
    assert kw["betas"] == (0.8, 0.9)
    assert kw["eps"] == 1e-6
    assert kw["period_one_substrings"] == ("embed", "lm_head")
    assert kw["codebook_refresh_every"] == 100


# --------------------------------------------------------------------------- #
# build_gefenx_muon_optimizer
# --------------------------------------------------------------------------- #
def test_build_gefenx_muon_applies_recommended_recipe(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig()  # defaults encode the recipe
    opt = gefenx.build_gefenx_muon_optimizer(
        model,
        config,
        lr = 1e-4,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    assert opt == "MUON_OPTIMIZER"
    call = fake_gefen["muon"]
    assert call["model"] is model
    kw = call["kwargs"]
    assert kw["backup_1d_period_one"] is True
    assert kw["adjust_lr_fn"] == "match_rms_adamw"
    assert kw["fused"] is True
    # backup_lr defaults to backup_lr_scale (0.5) * lr.
    assert kw["backup_lr"] == pytest.approx(0.5 * 1e-4)
    assert kw["lr"] == 1e-4


def test_build_gefenx_muon_explicit_backup_lr_wins(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig(backup_lr = 3e-5, backup_lr_scale = 0.5)
    gefenx.build_gefenx_muon_optimizer(
        model,
        config,
        lr = 1e-4,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    assert fake_gefen["muon"]["kwargs"]["backup_lr"] == 3e-5


def test_build_gefenx_muon_backup_lr_scale_none_leaves_backup_lr_unset(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig(backup_lr_scale = None)
    gefenx.build_gefenx_muon_optimizer(
        model,
        config,
        lr = 1e-4,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    # No scale and no explicit backup_lr => gefen uses its own default (None).
    assert fake_gefen["muon"]["kwargs"].get("backup_lr") is None


def test_build_gefenx_muon_passes_lr_weight_decay_and_backup_substrings(fake_gefen):
    model = _FakeModel([("w", _FakeParam())])
    config = _GefenXMuonConfig(backup_substrings = ["router", "gate"])
    gefenx.build_gefenx_muon_optimizer(
        model,
        config,
        lr = 2e-4,
        weight_decay = 0.05,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    call = fake_gefen["muon"]
    assert call["backup_substrings"] == ["router", "gate"]
    assert call["kwargs"]["lr"] == 2e-4
    assert call["kwargs"]["weight_decay"] == 0.05


# --------------------------------------------------------------------------- #
# Device gate: NVIDIA CUDA only (AMD/ROCm and Intel XPU rejected)
# --------------------------------------------------------------------------- #
def test_gate_rejects_rocm_hip_build(monkeypatch):
    # No fake_gefen here: this must exercise the REAL gate. gefen is never stubbed,
    # so a RuntimeError (not ModuleNotFoundError) proves the gate fires before any
    # `from gefen import ...`.
    torch = pytest.importorskip("torch")
    # Simulate an AMD/ROCm PyTorch build by tagging torch.version.hip.
    monkeypatch.setattr(torch.version, "hip", "6.0.0", raising = False)
    with pytest.raises(RuntimeError, match = "ROCm|HIP|CUDA"):
        gefenx._require_nvidia_cuda()
    model = _FakeModel([("w", _FakeParam())])
    with pytest.raises(RuntimeError, match = "ROCm|HIP|CUDA"):
        gefenx.build_gefenx_optimizer(
            model,
            _GefenXConfig(),
            lr = 1e-4,
            weight_decay = 0.0,
            betas = (0.9, 0.999),
            eps = 1e-8,
        )
    with pytest.raises(RuntimeError, match = "ROCm|HIP|CUDA"):
        gefenx.build_gefenx_muon_optimizer(
            model,
            _GefenXMuonConfig(),
            lr = 1e-4,
            weight_decay = 0.0,
            betas = (0.9, 0.999),
            eps = 1e-8,
        )


def test_gate_allows_non_hip(monkeypatch):
    torch = pytest.importorskip("torch")
    # A normal (non-HIP) build must pass the gate without raising.
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    gefenx._require_nvidia_cuda()  # should not raise


# --------------------------------------------------------------------------- #
# End-to-end against the REAL gefen package + REAL Unsloth trainer path.
#
# These construct real gefen optimizers on a real torch model, take a real step,
# and assert every trainable parameter actually moved (a no-op step would fail).
# --------------------------------------------------------------------------- #
def _cuda_available():
    try:
        import torch

        # NVIDIA CUDA only. On ROCm/HIP torch.cuda.is_available() is also True, but
        # the Gefen-X gate rejects HIP — so the fused tests must skip there too,
        # otherwise they'd error on _require_nvidia_cuda() instead of skipping.
        return torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
    except Exception:
        return False


_CUDA = _cuda_available()


def _tiny_model(torch, device = "cpu"):
    # Embedding + 2D Linears + LayerNorm exercises all of gefen's routing buckets:
    # 2D hidden weights (Muon half), embedding/1D norm/bias (Gefen backup half).
    return torch.nn.Sequential(
        torch.nn.Embedding(16, 8),
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 8),
    ).to(device)


def _forward_backward(
    torch,
    model,
    device = "cpu",
):
    ids = torch.arange(4, device = device)
    out = model[0](ids)
    out = model[1](out)
    out = model[2](out)
    out = model[3](out)
    out.sum().backward()


def _num_changed(torch, before, model):
    return sum(1 for a, p in zip(before, model.parameters()) if not torch.equal(a, p))


def test_real_gefenx_updates_all_params_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("gefen")
    from gefen import Gefen

    model = _tiny_model(torch)
    before = [p.detach().clone() for p in model.parameters()]
    opt = gefenx.build_gefenx_optimizer(
        model,
        _GefenXConfig(fused = False),
        lr = 1e-3,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    assert isinstance(opt, Gefen)
    _forward_backward(torch, model)
    opt.step()
    opt.zero_grad()
    assert _num_changed(torch, before, model) == len(before)


def test_real_gefenx_muon_updates_all_params_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("gefen")
    from gefen import GefenMuonHybrid

    model = _tiny_model(torch)
    before = [p.detach().clone() for p in model.parameters()]
    opt = gefenx.build_gefenx_muon_optimizer(
        model,
        _GefenXMuonConfig(fused = False),
        lr = 1e-3,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    assert isinstance(opt, GefenMuonHybrid)
    _forward_backward(torch, model)
    opt.step()
    opt.zero_grad()
    assert _num_changed(torch, before, model) == len(before)


def test_real_gefenx_preserves_param_names():
    # gefen must receive real names (not synthesized group_i_param_j), otherwise
    # GefenXConfig.period_one_substrings can never match.
    torch = pytest.importorskip("torch")
    pytest.importorskip("gefen")

    model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Linear(8, 8))
    opt = gefenx.build_gefenx_optimizer(
        model,
        _GefenXConfig(fused = False, period_one_substrings = ("embed",)),
        lr = 1e-3,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    names = [opt.state[p].get("name") for g in opt.param_groups for p in g["params"]]
    assert names and not any(str(n).startswith("group_") for n in names)


def test_real_gefenx_embedding_only_builds():
    # Regression: an embeddings-only run (all trainable params are modules_to_save)
    # with embedding_lr set must not crash on an empty non_embeddings group.
    torch = pytest.importorskip("torch")
    pytest.importorskip("gefen")
    from gefen import Gefen

    class _EmbedOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 4))

        def named_parameters(self, *a, **k):
            yield "base.embed_tokens.modules_to_save.default.weight", self.w

    model = _EmbedOnly()
    opt = gefenx.build_gefenx_optimizer(
        model,
        _GefenXConfig(fused = False),
        lr = 1e-3,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
        embedding_lr = 5e-6,
    )
    assert isinstance(opt, Gefen)
    assert all(len(g["params"]) > 0 for g in opt.param_groups)


@pytest.mark.skipif(not _CUDA, reason = "requires NVIDIA CUDA for the fused gefen kernels")
def test_real_gefenx_cuda_fused_updates_all_params():
    import torch

    pytest.importorskip("gefen")

    model = _tiny_model(torch, "cuda")
    before = [p.detach().clone() for p in model.parameters()]
    opt = gefenx.build_gefenx_optimizer(
        model,
        _GefenXConfig(fused = True),
        lr = 1e-3,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    _forward_backward(torch, model, "cuda")
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    assert _num_changed(torch, before, model) == len(before)


@pytest.mark.skipif(not _CUDA, reason = "requires NVIDIA CUDA for the fused gefen kernels")
def test_real_gefenx_muon_cuda_fused_updates_all_params():
    import torch

    pytest.importorskip("gefen")

    model = _tiny_model(torch, "cuda")
    before = [p.detach().clone() for p in model.parameters()]
    opt = gefenx.build_gefenx_muon_optimizer(
        model,
        _GefenXMuonConfig(fused = True),
        lr = 1e-3,
        weight_decay = 0.0,
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    _forward_backward(torch, model, "cuda")
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    assert _num_changed(torch, before, model) == len(before)


# --------------------------------------------------------------------------- #
# Full Unsloth trainer path: the REAL GefenXConfig / GefenXMuonConfig dataclasses
# carried through the REAL UnslothTrainingArguments and dispatched by the REAL
# UnslothTrainer.create_optimizer (not the standalone build_* helpers).
# --------------------------------------------------------------------------- #
def test_trainer_create_optimizer_dispatches_gefenx(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gefen")
    pytest.importorskip("unsloth")
    from unsloth import GefenXConfig
    from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
    from gefen import Gefen

    args = UnslothTrainingArguments(
        output_dir = str(tmp_path / "gx"),
        gefenx_config = GefenXConfig(fused = False),
        learning_rate = 1e-3,
        weight_decay = 0.0,
        report_to = "none",
    )
    # Config plumbing on the real dataclass-typed argument.
    assert args.gefenx_config is not None
    assert args.gefenx_muon_config is None

    trainer = UnslothTrainer.__new__(UnslothTrainer)  # skip heavy SFTTrainer.__init__
    trainer.model = _tiny_model(torch)
    trainer.args = args
    trainer.optimizer = None

    before = [p.detach().clone() for p in trainer.model.parameters()]
    opt = trainer.create_optimizer()
    assert isinstance(opt, Gefen)
    _forward_backward(torch, trainer.model)
    opt.step()
    opt.zero_grad()
    assert _num_changed(torch, before, trainer.model) == len(before)


def test_trainer_create_optimizer_dispatches_gefenx_muon(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gefen")
    pytest.importorskip("unsloth")
    from unsloth import GefenXMuonConfig
    from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
    from gefen import GefenMuonHybrid

    args = UnslothTrainingArguments(
        output_dir = str(tmp_path / "gm"),
        gefenx_muon_config = GefenXMuonConfig(fused = False),
        learning_rate = 1e-3,
        weight_decay = 0.0,
        report_to = "none",
    )
    assert args.gefenx_muon_config is not None

    trainer = UnslothTrainer.__new__(UnslothTrainer)
    trainer.model = _tiny_model(torch)
    trainer.args = args
    trainer.optimizer = None

    before = [p.detach().clone() for p in trainer.model.parameters()]
    opt = trainer.create_optimizer()
    assert isinstance(opt, GefenMuonHybrid)
    _forward_backward(torch, trainer.model)
    opt.step()
    opt.zero_grad()
    assert _num_changed(torch, before, trainer.model) == len(before)


def test_conflicting_optimizer_configs_raise(tmp_path):
    pytest.importorskip("unsloth")
    from unsloth import GefenXConfig, GefenXMuonConfig
    from unsloth.trainer import UnslothTrainingArguments

    # Setting both Gefen-X configs is ambiguous (dispatch would silently pick one).
    with pytest.raises(ValueError, match = "mutually exclusive"):
        UnslothTrainingArguments(
            output_dir = str(tmp_path / "conflict"),
            gefenx_config = GefenXConfig(),
            gefenx_muon_config = GefenXMuonConfig(),
            report_to = "none",
        )
