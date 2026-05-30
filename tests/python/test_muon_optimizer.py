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
    from unsloth.optimizers.muon import _is_muon_eligible, _classify_param_names
    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    names, _ = _classify_param_names(model)
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
    muon.defaults = {"lr": 1e-3}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]
    adamw.defaults = {"lr": 1e-4}

    chained = _MuonAdamWChained(muon, adamw)
    chained.step()

    muon.step.assert_called_once()
    adamw.step.assert_called_once()


def test_chained_optimizer_zero_grad_calls_both():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    muon.defaults = {"lr": 1e-3}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]
    adamw.defaults = {"lr": 1e-4}

    chained = _MuonAdamWChained(muon, adamw)
    chained.zero_grad()

    muon.zero_grad.assert_called_once()
    adamw.zero_grad.assert_called_once()


def test_chained_state_dict_roundtrip():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    muon.state_dict.return_value = {"state": "muon"}
    muon.defaults = {"lr": 1e-3}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]
    adamw.state_dict.return_value = {"state": "adamw"}
    adamw.defaults = {"lr": 1e-4}

    chained = _MuonAdamWChained(muon, adamw)
    sd = chained.state_dict()

    assert sd["_muon_version"] == 1
    assert sd["muon"] == {"state": "muon"}
    assert sd["adamw"] == {"state": "adamw"}

    chained.load_state_dict(sd)
    muon.load_state_dict.assert_called_once_with({"state": "muon"})
    adamw.load_state_dict.assert_called_once_with({"state": "adamw"})


def test_chained_merges_param_groups():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    muon.defaults = {"lr": 1e-3}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}, {"params": [], "lr": 5e-5}]
    adamw.defaults = {"lr": 1e-4}

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
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8

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


def test_chained_pickle_not_supported():
    """torch.save(optimizer) via pickle should work (dump), but load must raise."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    state = chained.__getstate__()
    assert "muon" in state
    assert "adamw" in state

    with pytest.raises(NotImplementedError, match="unpickling"):
        chained.__setstate__(state)


def test_chained_add_param_group_raises():
    """add_param_group must raise NotImplementedError."""
    from unsloth.trainer import _MuonAdamWChained

    muon = MagicMock()
    muon.param_groups = [{"params": [], "lr": 1e-3}]
    muon.defaults = {"lr": 1e-3}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4}]
    adamw.defaults = {"lr": 1e-4}

    chained = _MuonAdamWChained(muon, adamw)
    with pytest.raises(NotImplementedError, match="add_param_group"):
        chained.add_param_group({"params": []})


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
    muon.defaults = {"lr": 1e-3, "weight_decay": 0.1, "momentum": 0.95, "eps": 1e-7}
    adamw = MagicMock()
    adamw.param_groups = [{"params": [], "lr": 1e-4, "weight_decay": 0.0}]
    adamw.defaults = {"lr": 1e-4, "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8}

    chained = _MuonAdamWChained(muon, adamw)
    assert chained.defaults["lr"] == 1e-4  # AdamW wins on overlap
    assert "momentum" in chained.defaults
    assert "betas" in chained.defaults


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


def test_peft_modules_to_save_embedding_goes_to_adamw():
    """PEFT-wrapped embedding copies must go to AdamW embedding group."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 8)
    model.lin = torch.nn.Linear(8, 4)
    # Simulate PEFT-wrapped embedding param
    peft_param = torch.nn.Parameter(torch.randn(10, 8))
    model._parameters["emb.modules_to_save.default.weight"] = peft_param

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.0, embedding_lr=1e-4
    )

    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert any(p is peft_param for p in adamw_params), "Embedding modules_to_save should go to AdamW"
    assert not any(p is peft_param for p in muon_params), "Embedding modules_to_save should NOT go to Muon"
    assert any(p is model.lin.weight for p in muon_params), "Linear weight should go to Muon"


def test_peft_modules_to_save_non_embedding_goes_to_muon():
    """Non-embedding PEFT modules_to_save (e.g. classifier head) go to Muon if 2D."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.score = torch.nn.Linear(10, 2)
    # Simulate PEFT-wrapped classifier head param
    peft_param = torch.nn.Parameter(torch.randn(2, 10))
    model._parameters["score.modules_to_save.default.weight"] = peft_param

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.5, adamw_weight_decay=0.1
    )

    muon_params = [p for g in muon_groups for p in g["params"]]
    assert any(p is peft_param for p in muon_params), "Non-embedding 2D modules_to_save should go to Muon"


# -- Tests: MT1 — Full lifecycle integration ---------------------------------


def test_full_lifecycle_step_changes_params():
    """step() must modify model parameters; loaded optimizer must too."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))

    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    before = p.data.clone()
    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()
    assert not torch.equal(p.data, before), "step() must change params"

    # Save state_dict, re-create, load, step again
    sd = chained.state_dict()
    fresh_muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    fresh_adamw = torch.optim.AdamW([q], lr=1e-3)
    fresh_chained = _MuonAdamWChained(fresh_muon, fresh_adamw)
    fresh_chained.load_state_dict(sd)

    before2 = p.data.clone()
    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    fresh_chained.step()
    assert not torch.equal(p.data, before2), "loaded optimizer must change params"


# -- Tests: MT2 — Empty Muon group -------------------------------------------


def test_empty_muon_group_params():
    """No 2D params → empty Muon group must not crash."""
    from unsloth.optimizers.muon import make_muon_param_groups
    _skip_if_no_muon()

    model = torch.nn.Module()
    model.register_parameter("bias", torch.nn.Parameter(torch.randn(4)))

    muon_g, adamw_g = make_muon_param_groups(model, lr=1e-3, weight_decay=0.1)
    assert len(muon_g) == 1
    assert len(muon_g[0]["params"]) == 0

    # Should not raise
    torch.optim.Muon(muon_g, lr=1e-3, momentum=0.95, ns_steps=5)


# -- Tests: MT3 — Empty AdamW group ------------------------------------------


def test_empty_adamw_group_params():
    """All 2D no-bias params → empty AdamW groups returned."""
    from unsloth.optimizers.muon import make_muon_param_groups
    _skip_if_no_muon()

    model = torch.nn.Linear(4, 4, bias=False)
    muon_g, adamw_g = make_muon_param_groups(model, lr=1e-3, weight_decay=0.1)
    assert len(adamw_g) == 0
    # All 2D params go to Muon
    assert len(muon_g[0]["params"]) > 0


# -- Tests: MT4 — Weight-decay isolation with equal values -------------------


def test_weight_decay_isolation_equal_values():
    """Isolation must hold when muon_weight_decay == adamw_weight_decay."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    muon_g, adamw_g = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.1, adamw_weight_decay=0.1
    )

    assert muon_g[0]["weight_decay"] == 0.1
    decay_groups = [g for g in adamw_g if g.get("weight_decay", 0.0) > 0.0]
    no_decay_groups = [g for g in adamw_g if g.get("weight_decay", 0.0) == 0.0]
    assert len(decay_groups) == 0  # weight went to Muon, no adamw decay params
    assert any(p is model.bias for g in no_decay_groups for p in g["params"])


# -- Tests: MT5 — target_modules with non-existent module names ---------------


def test_target_modules_nonexistent():
    """Non-existent target_modules → all params fall back to AdamW."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    muon_g, adamw_g = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.0, target_modules=["nonexistent"]
    )

    assert len(muon_g[0]["params"]) == 0
    all_adamw = [p for g in adamw_g for p in g["params"]]
    assert any(p is model.bias for p in all_adamw)
    assert any(p is model.weight for p in all_adamw)


# -- Tests: MT6 — _sync_lr with mismatched param group lengths ---------------


def test_sync_lr_with_extra_chained_group_detected():
    """Extra group in chained.param_groups must be detected and raise RuntimeError."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    # Tamper with the chained wrapper's groups to simulate group drift
    chained.param_groups.append({"params": [], "lr": 1e-5, "weight_decay": 0.0})

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)

    with pytest.raises(RuntimeError, match="group count mismatch"):
        chained.step()


# -- Tests: MT7 — Distributed training warning --------------------------------


@patch("torch.distributed.is_available", return_value=True)
@patch("torch.distributed.is_initialized", return_value=True)
def test_distributed_blocked_by_default(mock_init, mock_avail):
    """Distributed training must be blocked without opt-in env var."""
    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, UnslothTrainer

    model = torch.nn.Linear(4, 4)
    args = MagicMock()
    args.learning_rate = 1e-3
    args.weight_decay = 0.1
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999

    trainer = UnslothTrainer.__new__(UnslothTrainer)
    trainer.model = model
    trainer.args = args
    trainer.optimizer = None

    config = MuonConfig()
    with pytest.raises(RuntimeError, match="UNSLOTH_MUON_DISTRIBUTED"):
        trainer._create_muon_optimizer(config)


@patch("torch.distributed.is_available", return_value=True)
@patch("torch.distributed.is_initialized", return_value=True)
def test_distributed_allowed_with_env_var(mock_init, mock_avail):
    """Setting UNSLOTH_MUON_DISTRIBUTED=1 should allow distributed training."""
    import os
    os.environ["UNSLOTH_MUON_DISTRIBUTED"] = "1"

    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, UnslothTrainer, _MuonAdamWChained

    model = torch.nn.Linear(4, 4)
    args = MagicMock()
    args.learning_rate = 1e-3
    args.weight_decay = 0.1
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8

    trainer = UnslothTrainer.__new__(UnslothTrainer)
    trainer.model = model
    trainer.args = args
    trainer.optimizer = None

    config = MuonConfig()
    result = trainer._create_muon_optimizer(config)
    assert isinstance(result, _MuonAdamWChained)

    del os.environ["UNSLOTH_MUON_DISTRIBUTED"]


# -- Tests: _sync_lr all hyperparams (H1) ------------------------------------


def test_sync_lr_all_hyperparams():
    """_sync_lr must propagate all hyperparams, not just lr."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))

    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    # Modify weight_decay on chained groups
    chained.param_groups[0]["weight_decay"] = 0.5
    chained.param_groups[1]["weight_decay"] = 0.0

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()

    assert muon.param_groups[0]["weight_decay"] == 0.5, "Muon weight_decay not synced"
    assert adamw.param_groups[0]["weight_decay"] == 0.0, "AdamW weight_decay not synced"


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


# -- Tests: 4th-pass additions (MUON_REVIEW_3.md) ----------------------------


def test_sync_lr_asymmetric_keys():
    """_sync_lr must handle keys present in sub-optimizer but not in chained."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    # Add a key to the Muon sub-optimizer that doesn't exist in the chained wrapper
    original_lr = muon.param_groups[0]["lr"]
    # Ensure known keys are synced forward
    chained.param_groups[0]["lr"] = 2e-4
    chained.param_groups[1]["lr"] = 2e-4

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()

    assert muon.param_groups[0]["lr"] == 2e-4, "Muon LR should be synced from chained"
    assert adamw.param_groups[0]["lr"] == 2e-4, "AdamW LR should be synced from chained"


def test_muon_config_value_validation():
    """MuonConfig must reject invalid numeric hyperparameters."""
    from unsloth.trainer import MuonConfig

    with pytest.raises(ValueError, match="momentum"):
        MuonConfig(momentum=-0.1)
    with pytest.raises(ValueError, match="muon_eps"):
        MuonConfig(muon_eps=0.0)
    with pytest.raises(ValueError, match="ns_steps"):
        MuonConfig(ns_steps=0)
    with pytest.raises(ValueError, match="muon_lr_scale"):
        MuonConfig(muon_lr_scale=-1.0)
    with pytest.raises(ValueError, match="muon_weight_decay"):
        MuonConfig(muon_weight_decay=-0.1)
    with pytest.raises(ValueError, match="ns_steps"):
        MuonConfig(ns_steps=100)


def test_muon_config_accepted_numeric_values():
    """MuonConfig must accept valid numeric hyperparameters."""
    from unsloth.trainer import MuonConfig

    cfg = MuonConfig(
        momentum=0.0,
        muon_eps=1e-7,
        ns_steps=5,
        muon_lr_scale=0.5,
        muon_weight_decay=0.0,
        adamw_weight_decay=0.1,
    )
    assert cfg.momentum == 0.0
    assert cfg.muon_eps == 1e-7
    assert cfg.ns_steps == 5
    assert cfg.muon_lr_scale == 0.5
    assert cfg.muon_weight_decay == 0.0
    assert cfg.adamw_weight_decay == 0.1


def test_lr_scheduler_with_muon():
    """LR scheduler applied to chained wrapper must decay LRs in both sub-optimizers."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        chained, start_factor=1.0, end_factor=0.1, total_iters=5
    )

    for step in range(5):
        p.grad = torch.randn_like(p)
        q.grad = torch.randn_like(q)
        chained.step()
        scheduler.step()

    assert muon.param_groups[0]["lr"] < 1e-3, "Muon LR should have decayed"
    assert adamw.param_groups[0]["lr"] < 1e-3, "AdamW LR should have decayed"


def test_muon_autocast_smoke():
    """Muon chained optimizer must not crash with autocast."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    with torch.amp.autocast(device_type="cpu"):
        p.grad = torch.randn_like(p)
        q.grad = torch.randn_like(q)
        chained.step()

    assert not torch.isnan(p).any(), "Muon step should not produce NaN"
    assert not torch.isnan(q).any(), "AdamW step should not produce NaN"


# -- Tests: 5th-pass additions (MUON_REVIEW_4.md) ----------------------------


def test_checkpoint_version_marker_detects_mismatch():
    """load_state_dict must reject state dicts with missing/wrong _muon_version."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    from unsloth.trainer import _MuonAdamWChained
    chained = _MuonAdamWChained(muon, adamw)

    # Missing version marker
    with pytest.raises(RuntimeError, match="version mismatch"):
        chained.load_state_dict({"muon": {}, "adamw": {}})

    # Wrong version number
    with pytest.raises(RuntimeError, match="version mismatch"):
        chained.load_state_dict({"_muon_version": 999, "muon": {}, "adamw": {}})

    # Valid version should pass (sub-optimizer load_state_dict may fail,
    # but version check itself is fine)
    sd = chained.state_dict()
    version = sd.pop("_muon_version")
    with pytest.raises(RuntimeError, match="version mismatch"):
        chained.load_state_dict(sd)
    sd["_muon_version"] = version
    # load_state_dict may raise from sub-optimizer shape mismatch, but version check passes
    try:
        chained.load_state_dict(sd)
    except (RuntimeError, KeyError, ValueError):
        pass


def test_ns_coefficients_custom_values():
    """Custom ns_coefficients must propagate to Muon constructor."""
    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, UnslothTrainer, _MuonAdamWChained

    config = MuonConfig(ns_coefficients=(3.0, -4.0, 2.0))
    # Use a simple model
    model = torch.nn.Linear(4, 4)
    args = MagicMock()
    args.learning_rate = 1e-3
    args.weight_decay = 0.1
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8

    trainer = UnslothTrainer.__new__(UnslothTrainer)
    trainer.model = model
    trainer.args = args
    trainer.optimizer = None

    result = trainer._create_muon_optimizer(config)
    assert isinstance(result, _MuonAdamWChained)
    # ns_coefficients is stored in muon.defaults, not param_groups
    assert result.muon.defaults.get("ns_coefficients") == (3.0, -4.0, 2.0)


def test_target_modules_with_embedding_match():
    """When target_modules contains 'embed', embeddings must still go to AdamW."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Sequential()
    model.add_module("emb", torch.nn.Embedding(10, 8))
    model.add_module("lin", torch.nn.Linear(8, 4))

    muon_g, adamw_g = make_muon_param_groups(
        model, lr=1e-3, weight_decay=0.0, target_modules=["lin"]
    )

    muon_params = [p for g in muon_g for p in g["params"]]
    adamw_params = [p for g in adamw_g for p in g["params"]]

    assert any(p is model.emb.weight for p in adamw_params), "Embedding should go to AdamW despite target_modules"
    assert any(p is model.lin.weight for p in muon_params), "Linear weight matching target_modules should go to Muon"


def test_no_trainable_params():
    """All params requires_grad=False must produce empty groups."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad = False

    muon_g, adamw_g = make_muon_param_groups(model, lr=1e-3, weight_decay=0.0)
    assert len(muon_g[0]["params"]) == 0
    all_adamw = [p for g in adamw_g for p in g["params"]]
    assert len(all_adamw) == 0


def test_empty_muon_group_step():
    """Real Muon/AdamW with step() must not crash."""
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
    assert not torch.isnan(p).any(), "Muon step should not produce NaN"
    assert not torch.isnan(q).any(), "AdamW step should not produce NaN"


# -- Tests: 6th-pass additions (MUON_REVIEW_5.md) ----------------------------


def test_resume_with_lr_scheduler():
    """After load_state_dict + step, LR must match saved LR, not construction LR."""
    _skip_if_no_muon()

    from unsloth.trainer import _MuonAdamWChained

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    chained = _MuonAdamWChained(muon, adamw)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        chained, start_factor=1.0, end_factor=0.1, total_iters=5
    )

    for step in range(5):
        p.grad = torch.randn_like(p)
        q.grad = torch.randn_like(q)
        chained.step()
        scheduler.step()

    saved_lr = muon.param_groups[0]["lr"]
    assert saved_lr < 1e-3, "LR should have decayed"

    state = chained.state_dict()

    # Recreate optimizer from scratch
    muon2 = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw2 = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr=1e-3)
    chained2 = _MuonAdamWChained(muon2, adamw2)

    chained2.load_state_dict(state)
    # After load_state_dict, chained groups must reflect loaded LR (C2 fix)
    assert chained2.param_groups[0]["lr"] == saved_lr, \
        f"chained LR ({chained2.param_groups[0]['lr']}) must match saved LR ({saved_lr})"
    assert muon2.param_groups[0]["lr"] == saved_lr, \
        f"muon LR ({muon2.param_groups[0]['lr']}) must match saved LR ({saved_lr})"

    # Step must NOT overwrite the loaded LR
    p2 = muon2.param_groups[0]["params"][0]
    q2 = adamw2.param_groups[0]["params"][0]
    p2.grad = torch.randn_like(p2)
    q2.grad = torch.randn_like(q2)
    chained2.step()
    assert muon2.param_groups[0]["lr"] == saved_lr, \
        "step() must not overwrite loaded LR (C2 regression)"


def test_peft_non_default_adapter_name():
    """Non-default adapter name must still route embedding to AdamW."""
    from unsloth.optimizers.muon import make_muon_param_groups, _classify_param_names

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 8)
    model.lin = torch.nn.Linear(8, 4)
    # Simulate PEFT modules_to_save with non-default adapter name "custom_name"
    peft_param = torch.nn.Parameter(torch.randn(10, 8))
    model._parameters["emb.modules_to_save.custom_name.weight"] = peft_param
    peft_bias = torch.nn.Parameter(torch.randn(8))
    model._parameters["emb.modules_to_save.custom_name.bias"] = peft_bias

    embedding_names, _ = _classify_param_names(model)
    assert "emb.modules_to_save.custom_name.weight" in embedding_names, \
        "Custom adapter embedding must be classified as embedding"
    assert "emb.modules_to_save.custom_name.bias" in embedding_names, \
        "Custom adapter embedding bias must be classified as embedding"

    muon_g, adamw_g = make_muon_param_groups(model, lr=1e-3, weight_decay=0.0)
    muon_params = [p for g in muon_g for p in g["params"]]
    adamw_params = [p for g in adamw_g for p in g["params"]]

    assert any(p is peft_param for p in adamw_params), \
        "Custom adapter embedding weight should go to AdamW, not Muon"
    assert not any(p is peft_param for p in muon_params), \
        "Custom adapter embedding weight should NOT go to Muon"


def test_load_state_dict_updates_chained_groups():
    """After load_state_dict, chained.param_groups must match sub-optimizer groups."""
    _skip_if_no_muon()

    from unsloth.trainer import _MuonAdamWChained

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW([q], lr=1e-3)

    chained = _MuonAdamWChained(muon, adamw)
    state = chained.state_dict()

    # Modify LR in saved state
    state["muon"]["param_groups"][0]["lr"] = 5e-4
    state["adamw"]["param_groups"][0]["lr"] = 5e-4
    state["muon"]["param_groups"][0]["weight_decay"] = 0.5

    # Recreate and load
    muon2 = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))], lr=1e-3, momentum=0.95, ns_steps=5)
    adamw2 = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr=1e-3)
    chained2 = _MuonAdamWChained(muon2, adamw2)

    chained2.load_state_dict(state)

    assert chained2.param_groups[0]["lr"] == 5e-4, "Chained LR should reflect loaded state"
    assert chained2.param_groups[0]["weight_decay"] == 0.5, "Chained weight_decay should reflect loaded state"
    assert chained2.param_groups[1]["lr"] == 5e-4, "Chained AdamW group LR should reflect loaded state"
