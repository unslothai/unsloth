"""LoRA on a block-diagonal grouped linear (DeepSeek-V4's `o_a_proj`)."""

import pytest
import torch

peft = pytest.importorskip("peft")


class GroupedLinear(torch.nn.Linear):
    """The shape contract of `DeepseekV4GroupedLinear` and `FP8GroupedLinear`."""

    def __init__(self, in_per_group, out_features, n_groups):
        super().__init__(in_per_group, out_features, bias = False)
        self.n_groups = n_groups

    def forward(self, x):
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o_a_proj = GroupedLinear(16, 4 * 8, 4)
        self.o_b_proj = torch.nn.Linear(4 * 8, 16, bias = False)

    def forward(self, x):
        return self.o_b_proj(self.o_a_proj(x).flatten(2))


def _peft_model(register):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = Block()
    config = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["o_a_proj", "o_b_proj"], init_lora_weights = False
    )
    if register:
        from unsloth.models.grouped_linear_lora import register_grouped_linear_lora
        assert register_grouped_linear_lora(config, model) == [GroupedLinear]
    return get_peft_model(model, config)


def test_is_grouped_linear_wants_n_groups_and_an_overridden_forward():
    from unsloth.models.grouped_linear_lora import is_grouped_linear

    assert is_grouped_linear(GroupedLinear(16, 32, 4))
    assert not is_grouped_linear(torch.nn.Linear(16, 32))
    plain = torch.nn.Linear(16, 32)
    plain.n_groups = 4
    assert not is_grouped_linear(plain)


def test_dense_lora_fails_on_the_grouped_linear():
    """The arm that fails on main."""
    model = _peft_model(register = False)
    with pytest.raises(RuntimeError, match = "must match the size"):
        model(torch.randn(2, 5, 4, 16))


def test_grouped_lora_trains_and_matches_the_merged_weight():
    model = _peft_model(register = True)
    x = torch.randn(2, 5, 4, 16)
    out = model(x)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    layer = model.base_model.model.o_a_proj
    assert type(layer).__name__ == "GroupedLinearLoRA"
    assert layer.lora_A["default"].weight.grad is not None
    assert layer.lora_B["default"].weight.grad is not None
    # Merging lora_B @ lora_A into the block-diagonal weight gives the same forward.
    with torch.no_grad():
        merged = model.merge_and_unload()
        merged_out = merged(x)
    torch.testing.assert_close(merged_out, out.detach(), atol = 1e-5, rtol = 1e-5)


def test_merge_into_an_fp8_grouped_weight_is_refused():
    """FP8GroupedLinear keeps an fp8 weight plus weight_scale_inv; B @ A cannot be added in place.
    PEFT merges layer by layer, so the dense layer ahead of it must not be merged either."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    from peft import LoraConfig, get_peft_model
    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora

    class DenseFirst(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias = False)  # merged first by PEFT
            self.o_a_proj = GroupedLinear(16, 4 * 8, 4)

        def forward(self, x):
            return self.o_a_proj(self.q_proj(x))

    torch.manual_seed(0)
    config = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["q_proj", "o_a_proj"], init_lora_weights = False
    )
    block = DenseFirst()
    register_grouped_linear_lora(config, block)
    model = get_peft_model(block, config)
    base = model.base_model.model.o_a_proj.get_base_layer()
    with torch.no_grad():
        base.weight = torch.nn.Parameter(base.weight.to(torch.float8_e4m3fn), requires_grad = False)
    before = base.weight.clone()
    dense = model.base_model.model.q_proj
    dense_before = dense.get_base_layer().weight.clone()
    for merge in (model.merge_and_unload, model.merge_adapter):
        with pytest.raises(NotImplementedError, match = "fp8 grouped linear"):
            merge()
        assert torch.equal(base.weight.view(torch.uint8), before.view(torch.uint8))
        assert not model.base_model.model.o_a_proj.merged
        assert model.base_model.model.q_proj is dense and not dense.merged
        assert torch.equal(dense.get_base_layer().weight, dense_before)


def test_grouped_lora_equals_the_diagonal_of_the_dense_delta():
    model = _peft_model(register = True)
    layer = model.base_model.model.o_a_proj
    x = torch.randn(3, 4, 16)
    with torch.no_grad():
        base = layer.base_layer(x)
        dense = layer.lora_B["default"](layer.lora_A["default"](x)) * layer.scaling["default"]
        dense = dense.view(3, 4, 4, 8)
        expected = base + torch.stack([dense[:, g, g] for g in range(4)], dim = 1)
        torch.testing.assert_close(layer(x), expected, atol = 1e-5, rtol = 1e-5)


def test_state_dict_is_a_plain_lora_checkpoint():
    model = _peft_model(register = True)
    state = peft.get_peft_model_state_dict(model)
    assert state["base_model.model.o_a_proj.lora_A.weight"].shape == (4, 16)
    assert state["base_model.model.o_a_proj.lora_B.weight"].shape == (32, 4)


def test_dora_is_refused_on_a_grouped_linear():
    """The grouped forward computes the plain LoRA sum only, so DoRA is refused."""
    from peft import LoraConfig
    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora

    model = Block()
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_a_proj"], use_dora = True)
    with pytest.raises(NotImplementedError, match = "DoRA"):
        register_grouped_linear_lora(config, model)


def test_a_variant_reaching_the_forward_is_refused():
    from peft import LoraConfig, get_peft_model

    model = Block()
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_a_proj"], init_lora_weights = False)
    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora

    register_grouped_linear_lora(config, model)
    peft_model = get_peft_model(model, config)
    layer = peft_model.base_model.model.o_a_proj
    if not hasattr(layer, "lora_variant"):
        pytest.skip("this PEFT has no LoRA variants")
    layer.lora_variant["default"] = object()
    with pytest.raises(NotImplementedError, match = "variants"):
        peft_model(torch.randn(2, 4, 16))


def test_dora_is_allowed_when_no_grouped_linear_is_targeted():
    """DoRA on the dense o_b_proj alone is valid: the grouped forward is never built."""
    from peft import LoraConfig
    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora

    model = Block()
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_b_proj"], use_dora = True)
    assert register_grouped_linear_lora(config, model) == []
    regex = LoraConfig(r = 4, lora_alpha = 8, target_modules = ".*o_b_proj", use_dora = True)
    assert register_grouped_linear_lora(regex, model) == []
    everything = LoraConfig(r = 4, lora_alpha = 8, target_modules = None, use_dora = True)
    with pytest.raises(NotImplementedError, match = "DoRA"):
        register_grouped_linear_lora(everything, model)


def test_a_saved_adapter_reloads_onto_the_grouped_forward(tmp_path):
    """The custom mapping is not in adapter_config.json, so a reload must register it again."""
    from peft import PeftModel
    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora_for_adapter

    peft_model = _peft_model(register = True)
    x = torch.randn(2, 5, 4, 16)
    with torch.no_grad():
        want = peft_model(x)
    peft_model.save_pretrained(str(tmp_path))

    base = Block()
    base_state = {
        k.replace(".base_layer", ""): v
        for k, v in peft_model.base_model.model.state_dict().items()
        if "lora_" not in k
    }
    base.load_state_dict(base_state, strict = True)
    config = register_grouped_linear_lora_for_adapter(base, str(tmp_path))
    assert config is not None
    reloaded = PeftModel.from_pretrained(base, str(tmp_path), config = config, torch_device = "cpu")
    assert type(reloaded.base_model.model.o_a_proj).__name__ == "GroupedLinearLoRA"
    with torch.no_grad():
        got = reloaded(x)
    torch.testing.assert_close(got, want)
    # A model with no grouped linear reloads exactly as before.
    assert register_grouped_linear_lora_for_adapter(torch.nn.Linear(4, 4), str(tmp_path)) is None
