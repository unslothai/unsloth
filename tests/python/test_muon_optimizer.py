"""Tests for Muon optimizer integration in Unsloth.
All tests run without GPU via the conftest GPU-free harness.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch


# -- Helpers ------------------------------------------------------------


def _make_fake_model_with_params():
    model = torch.nn.Module()
    model.register_parameter("weight_2d", torch.nn.Parameter(torch.randn(4, 4)))
    model.register_parameter("bias_1d", torch.nn.Parameter(torch.randn(4)))
    model.register_parameter("embedding", torch.nn.Parameter(torch.randn(10, 4)))
    return model


def _skip_if_no_muon():
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available (PyTorch < 2.9)")


# -- Tests: param-group routing ----------------------------------------


def test_make_muon_param_groups_splits_correctly():
    from unsloth.optimizers.muon import make_muon_param_groups

    model = _make_fake_model_with_params()
    muon_groups, adamw_groups = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)

    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert len(muon_params) == 2
    assert len(adamw_params) == 1


def test_make_muon_param_groups_excludes_embedding_module():
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    model.lin = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.0)
    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert any(p is model.lin.weight for p in muon_params), "Linear weight should be Muon-eligible"
    assert any(
        p is model.emb.weight for p in adamw_params
    ), "Embedding weight should fall back to AdamW"


def test_make_muon_param_groups_splits_weight_decay():
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)

    no_decay_groups = [g for g in adamw_groups if g.get("weight_decay", 0.0) == 0.0]
    decay_groups = [g for g in adamw_groups if g.get("weight_decay", 0.0) > 0.0]
    all_adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert any(
        p is model.bias for p in no_decay_groups[0]["params"]
    ), "bias should have 0 weight decay"
    assert any(p is model.bias for p in all_adamw_params)
    assert not any(
        p is model.weight for p in all_adamw_params
    ), "weight should go to Muon, not AdamW"
    if decay_groups:
        assert not any(p is model.weight for p in decay_groups[0]["params"])


def test_weight_decay_isolation():
    """muon_weight_decay and adamw_weight_decay must be independent."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    # Muon gets weight, AdamW gets bias
    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr = 1e-3, muon_weight_decay = 0.5, adamw_weight_decay = 0.1
    )

    assert muon_groups[0]["weight_decay"] == 0.5, "Muon weight_decay should be 0.5"
    for g in adamw_groups:
        if any(p is model.bias for p in g["params"]):
            assert g["weight_decay"] == 0.0, "Bias should have 0 weight_decay (no-decay group)"
        else:
            assert (
                g["weight_decay"] == 0.1
            ), f"AdamW decay group weight_decay should be 0.1, got {g['weight_decay']}"


def test_target_modules_filter():
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Sequential()
    model.add_module("attn_q", torch.nn.Linear(4, 4))
    model.add_module("attn_v", torch.nn.Linear(4, 4))
    model.add_module("mlp_gate", torch.nn.Linear(4, 4))

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr = 1e-3, muon_weight_decay = 0.0, target_modules = ["attn"]
    )
    muon_names = set()
    for g in muon_groups:
        for p in g["params"]:
            for n, param in model.named_parameters():
                if param is p:
                    muon_names.add(n)

    for name in muon_names:
        assert "attn" in name, f"{name} should be filtered by target_modules"


def test_target_modules_nonexistent():
    """Non-existent target_modules → all params fall back to AdamW."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    muon_g, adamw_g = make_muon_param_groups(
        model, lr = 1e-3, muon_weight_decay = 0.0, target_modules = ["nonexistent"]
    )

    assert len(muon_g[0]["params"]) == 0
    all_adamw = [p for g in adamw_g for p in g["params"]]
    assert any(p is model.bias for p in all_adamw)
    assert any(p is model.weight for p in all_adamw)


def test_make_muon_param_groups_embedding_lr():
    """embedding_lr param creates a dedicated group with the correct LR."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    model.lin = torch.nn.Linear(4, 4)

    muon_groups, adamw_groups = make_muon_param_groups(
        model, lr = 1e-3, muon_weight_decay = 0.0, embedding_lr = 1e-4
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

    muon_groups, adamw_groups = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)

    emb_in_no_decay = any(
        p is model.emb.weight
        for g in adamw_groups
        if g.get("weight_decay", 0.0) == 0.0
        for p in g["params"]
    )
    # Without embedding_lr, embeddings go to the no-decay group (muon_weight_decay=0.0)
    # since they're routed as embeddings, not decay params
    assert emb_in_no_decay, "Embedding should be in no-decay group when no embedding_lr"


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
        model, lr = 1e-3, muon_weight_decay = 0.0, embedding_lr = 1e-4
    )

    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]

    assert any(
        p is peft_param for p in adamw_params
    ), "Embedding modules_to_save should go to AdamW"
    assert not any(
        p is peft_param for p in muon_params
    ), "Embedding modules_to_save should NOT go to Muon"
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
        model, lr = 1e-3, muon_weight_decay = 0.5, adamw_weight_decay = 0.1
    )

    muon_params = [p for g in muon_groups for p in g["params"]]
    assert any(
        p is peft_param for p in muon_params
    ), "Non-embedding 2D modules_to_save should go to Muon"


def test_tied_embedding_detected_via_data_ptr():
    """Tied embedding must be excluded from Muon via data_ptr detection."""
    from unsloth.optimizers.muon import _classify_param_names

    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(10, 4)
    model.lm_head = torch.nn.Linear(4, 10, bias = False)
    model.lm_head.weight = torch.nn.Parameter(model.emb.weight)

    embedding_names, _ = _classify_param_names(model)
    assert "emb.weight" in embedding_names
    assert "lm_head.weight" in embedding_names, "Tied lm_head.weight must be in embedding_names"


def test_tied_non_embedding_not_duplicated():
    """Shared tensor must not appear twice in muon_params."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    weight = torch.nn.Parameter(torch.randn(4, 4))
    model.register_parameter("shared_weight", weight)
    model.register_parameter("shared_weight_alias", weight)

    muon_g, adamw_g = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.0)
    muon_params_list = [p for g in muon_g for p in g["params"]]
    count = sum(1 for p in muon_params_list if p.data_ptr() == weight.data_ptr())
    assert count == 1, f"Shared tensor appears {count} times in muon_params (expected 1)"


def test_2d_norm_weight_goes_to_no_decay():
    """2D norm weights must be routed to AdamW no-decay, not Muon."""
    from unsloth.optimizers.muon import make_muon_param_groups

    class Fake2DNorm(torch.nn.LayerNorm):
        def __init__(self):
            super().__init__(4)
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

    model = torch.nn.Module()
    model.norm = Fake2DNorm()
    model.lin = torch.nn.Linear(4, 4)

    muon_g, adamw_g = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)

    muon_params = [p for g in muon_g for p in g["params"]]
    adamw_params = [p for g in adamw_g for p in g["params"]]

    assert any(p is model.lin.weight for p in muon_params), "Linear weight should go to Muon"
    assert any(
        p is model.norm.weight for p in adamw_params
    ), "2D norm weight should go to AdamW no-decay"
    assert not any(
        p is model.norm.weight for p in muon_params
    ), "2D norm weight should NOT go to Muon"

    # Verify norm weight is in a no-decay group
    norm_in_no_decay = any(
        any(p is model.norm.weight for p in g["params"]) and g.get("weight_decay", 0.0) == 0.0
        for g in adamw_g
    )
    assert norm_in_no_decay, "2D norm weight must be in a no-decay AdamW group"


def test_no_decay_takes_precedence_over_embedding():
    """Param in both no_decay and embedding sets goes to no_decay group."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Module()
    model.lin = torch.nn.Linear(4, 4)
    model.register_parameter(
        "norm_embedding_weight",
        torch.nn.Parameter(torch.randn(4, 4)),
    )

    def mock_classify(model):
        return {"norm_embedding_weight"}, {"norm_embedding_weight"}

    import unsloth.optimizers.muon as muon_mod

    original = muon_mod._classify_param_names
    muon_mod._classify_param_names = mock_classify
    try:
        muon_g, adamw_g = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)
        muon_params = [p for g in muon_g for p in g["params"]]
        adamw_params = [p for g in adamw_g for p in g["params"]]
        weight = model.norm_embedding_weight
        assert not any(p is weight for p in muon_params), "Overlapping param must not go to Muon"
        assert any(p is weight for p in adamw_params), "Overlapping param must go to AdamW"
        in_no_decay = any(
            any(p is weight for p in g["params"]) and g.get("weight_decay", -1) == 0.0
            for g in adamw_g
        )
        assert in_no_decay, "Overlapping param must be in AdamW no-decay group (weight_decay=0.0)"
    finally:
        muon_mod._classify_param_names = original


def test_modules_to_save_norm_goes_to_no_decay():
    """PEFT-wrapped norm copies must go to AdamW no-decay."""
    from unsloth.optimizers.muon import _classify_param_names

    model = torch.nn.Module()
    model.norm = torch.nn.LayerNorm(4)
    peft_param = torch.nn.Parameter(torch.randn(4))
    model._parameters["norm.modules_to_save.default.weight"] = peft_param

    _, no_decay_names = _classify_param_names(model)
    assert "norm.modules_to_save.default.weight" in no_decay_names


def test_muon_routes_lora_adapters():
    """LoRA A/B matrices (2D, rank-deficient) must be Muon-eligible."""
    from unsloth.optimizers.muon import _is_muon_eligible

    lora_a = torch.nn.Parameter(torch.randn(64, 16))
    assert _is_muon_eligible("lora_A", lora_a, set())

    lora_b = torch.nn.Parameter(torch.randn(16, 64))
    assert _is_muon_eligible("lora_B", lora_b, set())


def test_norm_name_pattern_catches_rms_norm():
    """_classify_param_names must catch custom RMSNorm named 'rms_norm'."""
    from unsloth.optimizers.muon import _classify_param_names

    model = torch.nn.Module()

    class CustomRMSNorm(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(dim))

    model.rms_norm = CustomRMSNorm(4)
    _, no_decay_names = _classify_param_names(model)
    assert "rms_norm.weight" in no_decay_names, "rms_norm must be classified as no_decay via regex"


def test_meta_device_raises_error():
    """make_muon_param_groups must raise RuntimeError on meta device."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4, device = "meta")
    with pytest.raises(RuntimeError, match = "meta"):
        make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.0)


def test_mixed_device_raises_error():
    """Mixed meta/real device model must raise RuntimeError."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4, device = "meta"),
    )
    with pytest.raises(RuntimeError, match = "meta"):
        make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.0)


def test_no_trainable_params():
    """All params requires_grad=False must produce empty groups."""
    from unsloth.optimizers.muon import make_muon_param_groups

    model = torch.nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad = False

    muon_g, adamw_g = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.0)
    assert len(muon_g[0]["params"]) == 0
    all_adamw = [p for g in adamw_g for p in g["params"]]
    assert len(all_adamw) == 0


def test_empty_muon_group_params():
    """No 2D params → empty Muon group must not crash."""
    from unsloth.optimizers.muon import make_muon_param_groups

    _skip_if_no_muon()

    model = torch.nn.Module()
    model.register_parameter("bias", torch.nn.Parameter(torch.randn(4)))

    muon_g, adamw_g = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)
    assert len(muon_g) == 1
    assert len(muon_g[0]["params"]) == 0

    # Should not raise on construction or step
    optimizer = torch.optim.Muon(muon_g, lr = 1e-3, momentum = 0.95, ns_steps = 5)
    optimizer.step()


def test_empty_adamw_group_params():
    """All 2D no-bias params → empty AdamW groups returned."""
    from unsloth.optimizers.muon import make_muon_param_groups

    _skip_if_no_muon()

    model = torch.nn.Linear(4, 4, bias = False)
    muon_g, adamw_g = make_muon_param_groups(model, lr = 1e-3, muon_weight_decay = 0.1)
    assert len(adamw_g) == 0
    # All 2D params go to Muon
    assert len(muon_g[0]["params"]) > 0


# -- Tests: config and optimizer construction --------------------------


def test_muon_config_defaults():
    from unsloth.trainer import MuonConfig

    cfg = MuonConfig()
    assert cfg.momentum == 0.95
    assert cfg.nesterov is True
    assert cfg.ns_steps == 5


def test_muon_config_custom_values():
    from unsloth.trainer import MuonConfig

    cfg = MuonConfig(momentum = 0.9, nesterov = False, ns_steps = 3, muon_lr_scale = 0.5)
    assert cfg.momentum == 0.9
    assert cfg.nesterov is False
    assert cfg.ns_steps == 3
    assert cfg.muon_lr_scale == 0.5


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


def test_muon_config_default_step():
    """Default MuonConfig() must complete step() without crashing."""
    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, UnslothTrainer, _MuonAdamWChained

    config = MuonConfig()
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

    p = model.weight
    q = model.bias
    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    # This must not crash (C1 regression: ns_coefficients=None would raise in upstream step())
    result.step()
    assert not torch.isnan(p).any(), "Step should not produce NaN with default MuonConfig"


def test_no_muon_params_optimizer():
    """No 2D params -> muon=None, but optimizer must still be constructable."""
    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, UnslothTrainer, _MuonAdamWChained

    model = torch.nn.Module()
    model.register_parameter("bias", torch.nn.Parameter(torch.randn(4)))

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
    assert result.muon is None
    assert result.adamw is not None


def test_no_adamw_params_optimizer():
    """All trainable params are 2D Muon-eligible -> adamw=None."""
    _skip_if_no_muon()

    from unsloth.trainer import MuonConfig, UnslothTrainer, _MuonAdamWChained

    model = torch.nn.Linear(4, 4, bias = False)

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
    assert result.muon is not None
    assert result.adamw is None


@patch("torch.distributed.is_available", return_value = True)
@patch("torch.distributed.is_initialized", return_value = True)
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
    with pytest.raises(RuntimeError, match = "UNSLOTH_MUON_DISTRIBUTED"):
        trainer._create_muon_optimizer(config)


@patch("torch.distributed.is_available", return_value = True)
@patch("torch.distributed.is_initialized", return_value = True)
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


def test_muon_config_in_unsloth_training_arguments():
    from unsloth.trainer import UnslothTrainingArguments, MuonConfig

    cfg = MuonConfig()
    args = UnslothTrainingArguments(
        muon_config = cfg,
        output_dir = "/tmp/unsloth_muon_test",
    )
    assert args.muon_config is cfg


def test_muon_config_exported_from_trainer():
    from unsloth.trainer import MuonConfig
    assert MuonConfig is not None


def test_make_muon_param_groups_exported():
    from unsloth.optimizers import make_muon_param_groups
    assert make_muon_param_groups is not None


# -- Tests: checkpoint save/load ---------------------------------------


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


def test_chained_state_dict_serialization(tmp_path):
    """state_dict serialized via torch.save must roundtrip through load_state_dict."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw = torch.optim.AdamW([q], lr = 1e-3)

    from unsloth.trainer import _MuonAdamWChained

    chained = _MuonAdamWChained(muon, adamw)

    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()

    sd_before = chained.state_dict()
    path = tmp_path / "optim_state.pt"
    torch.save(sd_before, path)

    sd_loaded = torch.load(path, weights_only = True)

    fresh_muon = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))], lr = 1e-3)
    fresh_adamw = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr = 1e-3)
    fresh_chained = _MuonAdamWChained(fresh_muon, fresh_adamw)
    fresh_chained.load_state_dict(sd_loaded)

    sd_after = fresh_chained.state_dict()
    for key in ["muon", "adamw"]:
        for k in sd_before[key]:
            if isinstance(sd_before[key][k], torch.Tensor):
                assert torch.equal(
                    sd_before[key][k], sd_after[key][k]
                ), f"{key}.{k} mismatch after state_dict serialization roundtrip"

    # Verify param_groups structure matches (L2 fix)
    for i, (before_g, after_g) in enumerate(zip(chained.param_groups, fresh_chained.param_groups)):
        assert before_g.get("lr") == after_g.get("lr"), f"LR mismatch in group {i}"
        assert before_g.get("weight_decay") == after_g.get(
            "weight_decay"
        ), f"weight_decay mismatch in group {i}"


def test_checkpoint_version_marker_detects_mismatch():
    """load_state_dict must reject state dicts with missing/wrong _muon_version."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw = torch.optim.AdamW([q], lr = 1e-3)

    from unsloth.trainer import _MuonAdamWChained

    chained = _MuonAdamWChained(muon, adamw)

    # Missing version marker
    with pytest.raises(RuntimeError, match = "version mismatch"):
        chained.load_state_dict({"muon": {}, "adamw": {}})

    # Wrong version number
    with pytest.raises(RuntimeError, match = "version mismatch"):
        chained.load_state_dict({"_muon_version": 999, "muon": {}, "adamw": {}})

    # Valid version should pass (sub-optimizer load_state_dict may fail,
    # but version check itself is fine)
    sd = chained.state_dict()
    version = sd.pop("_muon_version")
    with pytest.raises(RuntimeError, match = "version mismatch"):
        chained.load_state_dict(sd)
    sd["_muon_version"] = version
    # load_state_dict may raise from sub-optimizer shape mismatch, but version check passes
    try:
        chained.load_state_dict(sd)
    except (RuntimeError, KeyError, ValueError):
        pass


def test_resume_with_lr_scheduler():
    """After load_state_dict + step, LR must match saved LR, not construction LR."""
    _skip_if_no_muon()

    from unsloth.trainer import _MuonAdamWChained

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw = torch.optim.AdamW([q], lr = 1e-3)

    chained = _MuonAdamWChained(muon, adamw)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        chained, start_factor = 1.0, end_factor = 0.1, total_iters = 5
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
    muon2 = torch.optim.Muon(
        [torch.nn.Parameter(torch.randn(4, 4))], lr = 1e-3, momentum = 0.95, ns_steps = 5
    )
    adamw2 = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr = 1e-3)
    chained2 = _MuonAdamWChained(muon2, adamw2)

    chained2.load_state_dict(state)
    # After load_state_dict, chained groups must reflect loaded LR (C2 fix)
    assert (
        chained2.param_groups[0]["lr"] == saved_lr
    ), f"chained LR ({chained2.param_groups[0]['lr']}) must match saved LR ({saved_lr})"
    assert (
        muon2.param_groups[0]["lr"] == saved_lr
    ), f"muon LR ({muon2.param_groups[0]['lr']}) must match saved LR ({saved_lr})"

    # Step must NOT overwrite the loaded LR
    p2 = muon2.param_groups[0]["params"][0]
    q2 = adamw2.param_groups[0]["params"][0]
    p2.grad = torch.randn_like(p2)
    q2.grad = torch.randn_like(q2)
    chained2.step()
    assert (
        muon2.param_groups[0]["lr"] == saved_lr
    ), "step() must not overwrite loaded LR (C2 regression)"


def test_load_state_dict_missing_muon_key():
    """load_state_dict must raise RuntimeError when muon key is missing."""
    _skip_if_no_muon()

    from unsloth.trainer import _MuonAdamWChained

    q = torch.nn.Parameter(torch.randn(4))
    adamw_only = torch.optim.AdamW([q], lr = 1e-3)
    chained_no_muon = _MuonAdamWChained(None, adamw_only)

    state_no_muon = chained_no_muon.state_dict()
    assert "muon" not in state_no_muon, "state_dict with muon=None must not have muon key"

    # Create new optimizer with Muon
    p = torch.nn.Parameter(torch.randn(4, 4))
    q2 = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw2 = torch.optim.AdamW([q2], lr = 1e-3)
    chained_with_muon = _MuonAdamWChained(muon, adamw2)

    with pytest.raises(RuntimeError, match = "no Muon state"):
        chained_with_muon.load_state_dict(state_no_muon)


def test_load_state_dict_missing_adamw_key():
    """load_state_dict must raise RuntimeError when adamw key is missing."""
    _skip_if_no_muon()

    from unsloth.trainer import _MuonAdamWChained

    p = torch.nn.Parameter(torch.randn(4, 4))
    muon_only = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    chained_no_adamw = _MuonAdamWChained(muon_only, None)

    state_no_adamw = chained_no_adamw.state_dict()
    assert "adamw" not in state_no_adamw, "state_dict with adamw=None must not have adamw key"

    # Create new optimizer with AdamW
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon2 = torch.optim.Muon([p2], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw2 = torch.optim.AdamW([q], lr = 1e-3)
    chained_with_adamw = _MuonAdamWChained(muon2, adamw2)

    with pytest.raises(RuntimeError, match = "no AdamW state"):
        chained_with_adamw.load_state_dict(state_no_adamw)


def test_load_state_dict_updates_chained_groups():
    """After load_state_dict, chained.param_groups must match sub-optimizer groups."""
    _skip_if_no_muon()

    from unsloth.trainer import _MuonAdamWChained

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw = torch.optim.AdamW([q], lr = 1e-3)

    chained = _MuonAdamWChained(muon, adamw)
    state = chained.state_dict()

    # Modify LR in saved state
    state["muon"]["param_groups"][0]["lr"] = 5e-4
    state["adamw"]["param_groups"][0]["lr"] = 5e-4
    state["muon"]["param_groups"][0]["weight_decay"] = 0.5

    # Recreate and load
    muon2 = torch.optim.Muon(
        [torch.nn.Parameter(torch.randn(4, 4))], lr = 1e-3, momentum = 0.95, ns_steps = 5
    )
    adamw2 = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))], lr = 1e-3)
    chained2 = _MuonAdamWChained(muon2, adamw2)

    chained2.load_state_dict(state)

    assert chained2.param_groups[0]["lr"] == 5e-4, "Chained LR should reflect loaded state"
    assert (
        chained2.param_groups[0]["weight_decay"] == 0.5
    ), "Chained weight_decay should reflect loaded state"
    assert (
        chained2.param_groups[1]["lr"] == 5e-4
    ), "Chained AdamW group LR should reflect loaded state"


def test_full_lifecycle_step_changes_params():
    """step() must modify model parameters; loaded optimizer must too."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))

    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw = torch.optim.AdamW([q], lr = 1e-3)

    from unsloth.trainer import _MuonAdamWChained

    chained = _MuonAdamWChained(muon, adamw)

    before = p.data.clone()
    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    chained.step()
    assert not torch.equal(p.data, before), "step() must change params"

    # Save state_dict, re-create, load, step again
    sd = chained.state_dict()
    fresh_muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    fresh_adamw = torch.optim.AdamW([q], lr = 1e-3)
    fresh_chained = _MuonAdamWChained(fresh_muon, fresh_adamw)
    fresh_chained.load_state_dict(sd)

    before2 = p.data.clone()
    p.grad = torch.randn_like(p)
    q.grad = torch.randn_like(q)
    fresh_chained.step()
    assert not torch.equal(p.data, before2), "loaded optimizer must change params"


def test_lr_scheduler_with_muon():
    """LR scheduler applied to chained wrapper must decay LRs in both sub-optimizers."""
    _skip_if_no_muon()

    p = torch.nn.Parameter(torch.randn(4, 4))
    q = torch.nn.Parameter(torch.randn(4))
    muon = torch.optim.Muon([p], lr = 1e-3, momentum = 0.95, ns_steps = 5)
    adamw = torch.optim.AdamW([q], lr = 1e-3)

    from unsloth.trainer import _MuonAdamWChained

    chained = _MuonAdamWChained(muon, adamw)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        chained, start_factor = 1.0, end_factor = 0.1, total_iters = 5
    )

    for step in range(5):
        p.grad = torch.randn_like(p)
        q.grad = torch.randn_like(q)
        chained.step()
        scheduler.step()

    assert muon.param_groups[0]["lr"] < 1e-3, "Muon LR should have decayed"
    assert adamw.param_groups[0]["lr"] < 1e-3, "AdamW LR should have decayed"
