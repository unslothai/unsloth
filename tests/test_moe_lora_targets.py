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


def test_explicit_attention_only_list_does_not_discover_moe_parameters():
    # An explicit attention-only leaf list names no MLP projection, so experts
    # must never be targeted. get_peft_model routes this ORIGINAL list (not the
    # scoped regex) into detection precisely because family scoping makes
    # get_peft_regex emit its full "mlp|feed_forward|ffn|dense" component block
    # even for an attention-only request (see the regex below), which the
    # string fallback cannot distinguish from the fused-expert auto regex.
    from unsloth.models._utils import get_moe_target_parameters

    attn_only_list = ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert get_moe_target_parameters(_FakeMoeModel(), attn_only_list) is None
    assert get_moe_target_parameters(_FakeMoeModel(), tuple(attn_only_list)) is None

    # The regex get_peft_regex emits for that same attention-only list under a
    # vision-off family scope carries the mlp component block, so the string
    # path would wrongly enable experts -- hence detection must use the list.
    scoped_regex = (
        r"(?:.*?(?:language|text).*?"
        r"(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer).*?"
        r"(?:q_proj|k_proj|v_proj|o_proj))"
    )
    assert get_moe_target_parameters(_FakeMoeModel(), scoped_regex) == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_frozen_mlp_full_list_does_not_discover_moe_parameters():
    # Regression: an explicit list that names MLP leaves together with
    # finetune_mlp_modules=False must NOT train experts. get_peft_regex scopes
    # the MLP leaves out (its emitted regex carries no mlp tag block), so
    # detection has to key on that SCOPED regex -- keying on the original list
    # would let its gate/up/down leaves silently re-enable the frozen experts.
    from unsloth.models._utils import (
        _select_moe_detection_targets,
        get_moe_target_parameters,
    )

    original_list = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    # Representative of what get_peft_regex emits for that list under
    # finetune_mlp_modules=False: attention-only path, no mlp component block.
    scoped_regex = (
        r"(?:.*?(?:language|text).*?"
        r"(?:self_attn|attention|attn|mixer).*?"
        r"(?:q_proj|k_proj|v_proj|o_proj))"
    )
    selected = _select_moe_detection_targets(
        original_list,
        scoped_regex,
        finetune_mlp_modules = False,
        finetune_language_layers = True,
    )
    assert selected is scoped_regex
    assert get_moe_target_parameters(_FakeMoeModel(), selected) is None


def test_frozen_language_full_list_does_not_discover_moe_parameters():
    # Vision-only request (finetune_language_layers=False) with a full leaf list
    # must not reach the language-model experts either.
    from unsloth.models._utils import (
        _select_moe_detection_targets,
        get_moe_target_parameters,
    )

    original_list = ["q_proj", "gate_proj", "up_proj", "down_proj"]
    scoped_regex = (
        r"(?:.*?(?:vision|visual|image).*?"
        r"(?:self_attn|attention|attn|mixer).*?"
        r"(?:q_proj|k_proj|v_proj|o_proj))"
    )
    selected = _select_moe_detection_targets(
        original_list,
        scoped_regex,
        finetune_mlp_modules = True,
        finetune_language_layers = False,
    )
    assert selected is scoped_regex
    assert get_moe_target_parameters(_FakeMoeModel(), selected) is None


def test_in_scope_mlp_full_list_still_discovers_moe_parameters():
    # With MLP and language both in scope, an explicit list that names MLP
    # leaves SHOULD enable the experts (unchanged behavior): the original list
    # is preferred and carries the gate/up/down intent.
    from unsloth.models._utils import (
        _select_moe_detection_targets,
        get_moe_target_parameters,
    )

    original_list = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    scoped_regex = r".*self_attn.*proj"  # unused: original list is preferred
    selected = _select_moe_detection_targets(
        original_list,
        scoped_regex,
        finetune_mlp_modules = True,
        finetune_language_layers = True,
    )
    assert selected is original_list
    assert get_moe_target_parameters(_FakeMoeModel(), selected) == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_attention_only_list_prefers_original_when_in_scope():
    # The case the PR originally fixed: an attention-only list routed through
    # get_peft_regex under a family scope (e.g. vision-off) still keeps experts
    # off, because with MLP+language in scope detection uses the original
    # attention-only list rather than the regex's spurious mlp component block.
    from unsloth.models._utils import (
        _select_moe_detection_targets,
        get_moe_target_parameters,
    )

    attn_only_list = ["q_proj", "k_proj", "v_proj", "o_proj"]
    scoped_regex = (  # carries the spurious mlp block get_peft_regex always adds
        r"(?:.*?(?:language|text).*?"
        r"(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense).*?"
        r"(?:q_proj|k_proj|v_proj|o_proj))"
    )
    selected = _select_moe_detection_targets(
        attn_only_list,
        scoped_regex,
        finetune_mlp_modules = True,
        finetune_language_layers = True,
    )
    assert selected is attn_only_list
    assert get_moe_target_parameters(_FakeMoeModel(), selected) is None
