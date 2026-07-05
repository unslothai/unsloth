from types import SimpleNamespace

import pytest
import torch


class _ExpertWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 4, 8))
        self.down_proj = torch.nn.Parameter(torch.zeros(2, 8, 4))


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _ExpertWeights()


class _FakeMoeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_experts = 2, model_type = "qwen3_moe")
        self.mlp = _Mlp()


@pytest.mark.parametrize(
    "target_modules",
    [
        ".*mlp.*proj",
        ".*ffn.*proj",
        r"(?:\bmodel\.layers\.[\d]{1,}\.(?:mlp)\.(?:gate_proj|up_proj|down_proj))",
    ],
)
def test_regex_mlp_targets_discover_moe_parameters(target_modules):
    from unsloth.models._utils import get_moe_target_parameters
    assert get_moe_target_parameters(_FakeMoeModel(), target_modules) == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_explicit_dotted_module_target_does_not_discover_moe_parameters():
    from unsloth.models._utils import get_moe_target_parameters
    assert (
        get_moe_target_parameters(
            _FakeMoeModel(),
            "model.layers.0.mlp.shared_expert.down_proj",
        )
        is None
    )


@pytest.mark.parametrize(
    "target_modules",
    [
        # Attention-only auto-regex lists every projection leaf (incl. gate/up/down)
        # but its path segment is attention-only, so experts must NOT be targeted.
        r"(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn|attention|attn|mixer)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))",
        ".*self_attn.*proj",
        # An mlp path alternative with attention-only leaves is still attention-only.
        r"model\.layers\.\d+\.(?:mlp|self_attn)\.(?:q_proj|k_proj|v_proj|o_proj)",
    ],
)
def test_attention_only_regex_does_not_discover_moe_parameters(target_modules):
    from unsloth.models._utils import get_moe_target_parameters
    assert get_moe_target_parameters(_FakeMoeModel(), target_modules) is None


def test_single_leaf_regex_targets_only_that_projection():
    from unsloth.models._utils import get_moe_target_parameters
    assert get_moe_target_parameters(_FakeMoeModel(), ".*experts.*down_proj") == [
        "mlp.experts.down_proj",
    ]
    assert get_moe_target_parameters(_FakeMoeModel(), ".*mlp.*gate_proj") == [
        "mlp.experts.gate_up_proj",
    ]


def test_auto_regex_mlp_tag_block_discovers_moe_on_fused_models():
    # get_peft_regex on a fused-expert model lists only attention Linears as
    # leaves; the mlp tag block is the remaining signal of MLP finetune intent.
    from unsloth.models._utils import get_moe_target_parameters
    both_auto = (
        r"(?:\bmodel\.layers\.[\d]{1,}\."
        r"(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)\."
        r"(?:(?:q_proj|k_proj|v_proj|o_proj)))"
    )
    assert get_moe_target_parameters(_FakeMoeModel(), both_auto) == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
