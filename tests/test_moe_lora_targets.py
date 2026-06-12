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
