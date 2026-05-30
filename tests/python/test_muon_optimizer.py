"""Tests for Muon optimizer integration in Unsloth.
All tests run without GPU via the conftest GPU-free harness.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch


# -- Helpers ------------------------------------------------------------------

def _make_fake_model_with_params():
    model = torch.nn.Module()
    model.register_parameter("weight_2d", torch.nn.Parameter(torch.randn(4, 4)))
    model.register_parameter("bias_1d", torch.nn.Parameter(torch.randn(4)))
    model.register_parameter("embedding", torch.nn.Parameter(torch.randn(10, 4)))
    return model


def _skip_if_no_muon():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available (PyTorch < 2.9)")


# -- Tests: make_muon_param_groups --------------------------------------------


def test_make_muon_param_groups_splits_correctly():
    from unsloth.optimizers.muon import make_muon_param_groups

    model = _make_fake_model_with_params()
    muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.1)

    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert len(muon_params) == 2
    assert len(adamw_params) == 1


def test_is_muon_eligible_rejects_1d():
    from unsloth.optimizers.muon import _is_muon_eligible
    p = torch.nn.Parameter(torch.randn(4))
    assert not _is_muon_eligible("w", p, set())


def test_is_muon_eligible_accepts_2d():
    from unsloth.optimizers.muon import _is_muon_eligible
    p = torch.nn.Parameter(torch.randn(4, 4))
    assert _is_muon_eligible("w", p, set())


def test_is_muon_eligible_rejects_embedding():
    from unsloth.optimizers.muon import _is_muon_eligible, _get_embedding_param_names
    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    names = _get_embedding_param_names(model)
    p = model.emb.weight
    assert not _is_muon_eligible("emb.weight", p, names)


def test_make_muon_param_groups_excludes_embedding_module():
    from unsloth.optimizers.muon import make_muon_param_groups
    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    model.lin = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.0)
    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert model.lin.weight in muon_params, "Linear weight should be Muon-eligible"
    assert model.emb.weight in adamw_params, "Embedding weight should fall back to AdamW"


def test_make_muon_param_groups_splits_weight_decay():
    from unsloth.optimizers.muon import make_muon_param_groups
    model = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.1)

    no_decay_groups = [g for g in adamw_groups if g.get("weight_decay", 0.0) == 0.0]
    decay_groups = [g for g in adamw_groups if g.get("weight_decay", 0.0) > 0.0]
    all_adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert any(p is model.bias for p in no_decay_groups[0]["params"]), \
        "bias should have 0 weight decay"
    assert any(p is model.bias for p in all_adamw_params)
    assert not any(p is model.weight for p in all_adamw_params), \
        "weight should go to Muon, not AdamW"
    if decay_groups:
        assert not any(p is model.weight for p in decay_groups[0]["params"])


def test_no_requires_grad_excluded():
    from unsloth.optimizers.muon import make_muon_param_groups
    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad = False
    model.bias.requires_grad = False
    muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.0)
    total = sum(len(g["params"]) for g in muon_groups + adamw_groups)
    assert total == 0


def test_target_modules_filter():
    from unsloth.optimizers.muon import make_muon_param_groups
    model = torch.nn.Sequential()
    model.add_module("attn_q", torch.nn.Linear(4, 4))
    model.add_module("attn_v", torch.nn.Linear(4, 4))
    model.add_module("mlp_gate", torch.nn.Linear(4, 4))

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.0, target_modules=["attn"]
    )
    muon_names = set()
    for g in muon_groups:
        for p in g["params"]:
            for n, param in model.named_parameters():
                if param is p:
                    muon_names.add(n)

    for name in muon_names:
        assert "attn" in name, f"{name} should be filtered by target_modules"


# -- Tests: MuonConfig --------------------------------------------------------


def test_muon_config_defaults():
    from unsloth.trainer import MuonConfig
    cfg = MuonConfig()
    assert cfg.momentum == 0.95
    assert cfg.nesterov is True
    assert cfg.ns_steps == 5


def test_muon_config_custom_values():
    from unsloth.trainer import MuonConfig
    cfg = MuonConfig(momentum=0.9, nesterov=False, ns_steps=3, muon_lr_scale=0.5)
    assert cfg.momentum == 0.9
    assert cfg.nesterov is False
    assert cfg.ns_steps == 3
    assert cfg.muon_lr_scale == 0.5


# -- Tests: _MuonAdamWChained -------------------------------------------------


def test_chained_optimizer_step_calls_both():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]

    chained = _MuonAdamWChained(muon, adamw)
    chained.step()

    muon.step.assert_called_once()
    adamw.step.assert_called_once()


def test_chained_optimizer_zero_grad_calls_both():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]

    chained = _MuonAdamWChained(muon, adamw)
    chained.zero_grad()

    muon.zero_grad.assert_called_once()
    adamw.zero_grad.assert_called_once()


def test_chained_state_dict_roundtrip():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    muon.state_dict.return_value = {"state": "muon"}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]
    adamw.state_dict.return_value = {"state": "adamw"}

    chained = _MuonAdamWChained(muon, adamw)
    sd = chained.state_dict()

    assert sd == {"muon": {"state": "muon"}, "adamw": {"state": "adamw"}}

    chained.load_state_dict(sd)
    muon.load_state_dict.assert_called_once_with({"state": "muon"})
    adamw.load_state_dict.assert_called_once_with({"state": "adamw"})


def test_chained_merges_param_groups():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}, {"params": [], "lr": 5e-5}]

    chained = _MuonAdamWChained(muon, adamw)
    assert len(chained.param_groups) == 3


# -- Tests: _create_muon_optimizer (mocked) -----------------------------------


def test_create_muon_optimizer_smoke(monkeypatch):
    """Verify _create_muon_optimizer builds a _MuonAdamWChained instance."""
    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, _MuonAdamWChained, UnslothTrainer

    model = torch.nn.Linear(4, 4)
    args = MagicMock()
    args.learning_rate = 1e-3
    args.weight_decay = 0.1

    trainer = UnslothTrainer.__new__(UnslothTrainer)
    trainer.model = model
    trainer.args = args
    trainer.optimizer = None

    config = MuonConfig()
    result = trainer._create_muon_optimizer(config)

    assert isinstance(result, _MuonAdamWChained)
    assert hasattr(result, "step")
    assert hasattr(result, "zero_grad")
    assert hasattr(result, "state_dict")
    assert hasattr(result, "load_state_dict")


# -- Tests: __getstate__ / __setstate__ serialization -------------------------


def test_chained_state_dict_serialization(tmp_path):
    """state_dict serialized via torch.save must roundtrip through load_state_dict."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()

    sd_before = chained.state_dict()
    path = tmp_path / "optim_state.pt"
    torch.save(sd_before, path)

    sd_loaded = torch.load(path, weights_only=False)

    fresh_muon = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))], lr=1e-3)
    fresh_adamw = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr=1e-3)
    fresh_chained = _MuonAdamWChained(fresh_muon, fresh_adamw)
    fresh_chained.load_state_dict(sd_loaded)

    sd_after = fresh_chained.state_dict()
    for key in ["muon", "adamw"]:
        for k in sd_before[key]:
            if isinstance(sd_before[key][k], torch.Tensor):
                assert torch.equal(sd_before[key][k], sd_after[key][k]), \
                    f"{key}.{k} mismatch after state_dict serialization roundtrip"


def test_chained_state_dict_real_params():
    """state_dict / load_state_dict roundtrip with real PyTorch parameters."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))

    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()

    sd = chained.state_dict()

    fresh_muon = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))], lr=1e-3)
    fresh_adamw = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr=1e-3)
    fresh_chained = _MuonAdamWChained(fresh_muon, fresh_adamw)
    fresh_chained.load_state_dict(sd)

    restored_sd = fresh_chained.state_dict()
    for key in ["muon", "adamw"]:
        for k in sd[key]:
            if isinstance(sd[key][k], torch.Tensor):
                assert torch.equal(sd[key][k], restored_sd[key][k]), \
                    f"{key}.{k} mismatch after load_state_dict"


# -- Tests: surface API -------------------------------------------------------


def test_muon_config_exported_from_trainer():
    from unsloth.trainer import MuonConfig
    assert MuonConfig is not None


def test_muon_config_in_all():
    import unsloth.trainer as t
    assert "MuonConfig" in t.__all__


def test_muon_adamw_chained_in_all():
    import unsloth.trainer as t
    assert "_MuonAdamWChained" in t.__all__


def test_make_muon_param_groups_exported():
    from unsloth.optimizers import make_muon_param_groups
    assert make_muon_param_groups is not None


def test_muon_config_in_unsloth_training_arguments():
    from unsloth.trainer import UnslothTrainingArguments, MuonConfig
    cfg = MuonConfig()
    args = UnslothTrainingArguments(
        muon_config=cfg,
        output_dir="/tmp/unsloth_muon_test",
    )
    assert args.muon_config is cfg
