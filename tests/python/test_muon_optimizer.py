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

    assert any(p is model.lin.weight for p in muon_params), "Linear weight should be Muon-eligible"
    assert any(p is model.emb.weight for p in adamw_params), "Embedding weight should fall back to AdamW"


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


# -- Tests: serialization -----------------------------------------------------


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

    sd_loaded = torch.load(path, weights_only=True)

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

    # Verify param_groups structure matches (L2 fix)
    for i, (before_g, after_g) in enumerate(zip(chained.param_groups, fresh_chained.param_groups)):
        assert before_g.get("lr") == after_g.get("lr"), f"LR mismatch in group {i}"
        assert before_g.get("weight_decay") == after_g.get("weight_decay"), \
            f"weight_decay mismatch in group {i}"


def test_chained_pickle_roundtrip(tmp_path):
    """torch.save / torch.load of the optimizer object via pickle."""
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

    path = tmp_path / "optim_full.pt"
    torch.save(chained, path)

    loaded = torch.load(path, weights_only=False)
    assert isinstance(loaded, _MuonAdamWChained)
    assert hasattr(loaded, "muon")
    assert hasattr(loaded, "adamw")

    sd_orig = chained.state_dict()
    sd_loaded = loaded.state_dict()
    for key in ["muon", "adamw"]:
        for k in sd_orig[key]:
            if isinstance(sd_orig[key][k], torch.Tensor):
                assert torch.equal(sd_orig[key][k], sd_loaded[key][k]), \
                    f"{key}.{k} mismatch after pickle roundtrip"


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

    # Verify param_groups structure matches (L2)
    for i, (before_g, after_g) in enumerate(zip(chained.param_groups, fresh_chained.param_groups)):
        assert before_g.get("lr") == after_g.get("lr"), f"LR mismatch in group {i}"
        assert before_g.get("weight_decay") == after_g.get("weight_decay"), \
            f"weight_decay mismatch in group {i}"


# -- Tests: _sync_lr ---------------------------------------------------------


def test_sync_lr_propagates_from_chained_to_suboptimizers():
    """LR set on chained param_groups must reach sub-optimizers after step()."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))

    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    # Change LRs on the chained param_groups
    chained.param_groups[0]["lr"] = 5e-4  # Muon group
    chained.param_groups[1]["lr"] = 1e-4  # AdamW group

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()

    assert muon.param_groups[0]["lr"] == 5e-4, "Muon LR not synced"
    assert adamw.param_groups[0]["lr"] == 1e-4, "AdamW LR not synced"


def test_sync_lr_with_multiple_groups():
    """_sync_lr handles >1 group per sub-optimizer with correct index offsets."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q1 = torch.nn.Parameter(torch.randn(4))
    q2 = torch.nn.Parameter(torch.randn(8))

    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([
        {"params": [q1], "lr": 1e-4},
        {"params": [q2], "lr": 5e-5},
    ])

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    # Change LRs on chained groups
    chained.param_groups[0]["lr"] = 1e-3  # Muon
    chained.param_groups[1]["lr"] = 2e-4  # AdamW group 0
    chained.param_groups[2]["lr"] = 1e-4  # AdamW group 1

    p.grad = torch.randn_like(p)
    q1.grad = torch.randn_like(q1)
    q2.grad = torch.randn_like(q2)
    chained.step()

    assert muon.param_groups[0]["lr"] == 1e-3, "Muon LR not synced"
    assert adamw.param_groups[0]["lr"] == 2e-4, "AdamW group 0 LR not synced"
    assert adamw.param_groups[1]["lr"] == 1e-4, "AdamW group 1 LR not synced"


# -- Tests: defaults ----------------------------------------------------------


def test_chained_defaults_populated():
    """_MuonAdamWChained must populate a non-empty defaults dict."""
    from unsloth.trainer import _MuonAdamWChained

    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3, "weight_decay": 0.1}]
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4, "weight_decay": 0.0}]

    chained = _MuonAdamWChained(muon, adamw)
    assert "lr" in chained.defaults
    assert "weight_decay" in chained.defaults


# -- Tests: embedding_lr -----------------------------------------------------


def test_make_muon_param_groups_embedding_lr():
    """embedding_lr param creates a dedicated group with the correct LR."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    model.lin = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.0, embedding_lr=1e-4
    )

    emb_group = None
    other_adamw = []
    for g in adamw_groups:
        if any(p is model.emb.weight for p in g["params"]):
            emb_group = g
        else:
            other_adamw.append(g)

    assert emb_group is not None, "No embedding group found"
    assert emb_group["lr"] == 1e-4, f"Expected embedding_lr=1e-4, got {emb_group['lr']}"
    assert emb_group["weight_decay"] == 0.0, "Embedding group should have 0 weight decay"

    # Verify linear weight still goes to Muon
    muon_params = [p for g in muon_groups for p in g["params"]]
    assert model.lin.weight in muon_params


def test_make_muon_param_groups_embedding_lr_fallsback():
    """Without embedding_lr, embeddings go to decay group."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    model.lin = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.1
    )

    emb_in_no_decay = any(
        p is model.emb.weight
        for g in adamw_groups
        if g.get("weight_decay", 0.0) == 0.0
        for p in g["params"]
    )
    # Without embedding_lr, embeddings go to the no-decay group (weight_decay=0.0)
    # since they're routed as embeddings, not decay params
    assert emb_in_no_decay, "Embedding should be in no-decay group when no embedding_lr"


# -- Tests: weight-decay isolation -------------------------------------------


def test_weight_decay_isolation():
    """muon_weight_decay and adamw_weight_decay must be independent."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    # Muon gets weight, AdamW gets bias
    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.5, adamw_weight_decay=0.1
    )

    assert muon_groups[0]["weight_decay"] == 0.5, "Muon weight_decay should be 0.5"
    for g in adamw_groups:
        if any(p is model.bias for p in g["params"]):
            assert g["weight_decay"] == 0.0, "Bias should have 0 weight_decay (no-decay group)"
        else:
            assert g["weight_decay"] == 0.1, f"AdamW decay group weight_decay should be 0.1, got {g['weight_decay']}"


# -- Tests: MuonConfig validation --------------------------------------------


def test_muon_config_validates_ns_steps():
    from unsloth.trainer import MuonConfig
    with pytest.raises(ValueError, match="ns_steps must be < 100"):
        MuonConfig(ns_steps=100)
    with pytest.raises(ValueError, match="ns_steps must be < 100"):
        MuonConfig(ns_steps=200)


def test_muon_config_validates_adjust_lr_fn():
    from unsloth.trainer import MuonConfig
    with pytest.raises(ValueError, match="adjust_lr_fn"):
        MuonConfig(adjust_lr_fn="invalid")


def test_muon_config_valid_adjust_lr_fn():
    from unsloth.trainer import MuonConfig
    cfg = MuonConfig(adjust_lr_fn="original")
    assert cfg.adjust_lr_fn == "original"
    cfg = MuonConfig(adjust_lr_fn="match_rms_adamw")
    assert cfg.adjust_lr_fn == "match_rms_adamw"


# -- Tests: PEFT modules_to_save ---------------------------------------------


def test_peft_modules_to_save_excluded_from_muon():
    """PEFT-wrapped embedding copies must go to AdamW, not Muon."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.lin = torch.nn.Linear(4, 4)
    # Simulate PEFT-wrapped param by injecting into _parameters directly
    # (named_parameters iterates _parameters which can contain dots)
    peft_param = torch.nn.Parameter(torch.randn(10, 4))
    model._parameters["sub.modules_to_save.default.weight"] = peft_param

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.0
    )

    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert any(p is peft_param for p in adamw_params), "PEFT param should go to AdamW"
    assert not any(p is peft_param for p in muon_params), "PEFT param should NOT go to Muon"
    assert any(p is model.lin.weight for p in muon_params), "Linear weight should go to Muon"


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
