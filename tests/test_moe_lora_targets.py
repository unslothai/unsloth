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
        self.config = SimpleNamespace(num_experts=2, model_type="qwen3_moe")
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
    # get_peft_regex on a fused-expert model lists only attention Linears as leaves; the mlp tag block is the remaining
    # signal of MLP finetune intent.
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
    # An explicit attention-only leaf list names no MLP projection, so experts must never be targeted.
    # get_peft_model routes this ORIGINAL list (not the scoped regex) into detection precisely because family scoping
    # makes get_peft_regex emit its full "mlp|feed_forward|ffn|dense" component block even for an attention-only request
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
    # Regression: an explicit list that names MLP leaves together with finetune_mlp_modules=False must NOT train
    # experts.
    # get_peft_regex scopes the MLP leaves out (its emitted regex carries no mlp tag block), so detection has to key on
    # that SCOPED regex -- keying on the original list would let its gate/up/down leaves silently re-enable the frozen
    # experts.
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
    # Representative of what get_peft_regex emits for that list under finetune_mlp_modules=False: attention-only
    # path, no mlp component block.
    scoped_regex = (
        r"(?:.*?(?:language|text).*?"
        r"(?:self_attn|attention|attn|mixer).*?"
        r"(?:q_proj|k_proj|v_proj|o_proj))"
    )
    selected = _select_moe_detection_targets(
        original_list,
        scoped_regex,
        finetune_mlp_modules=False,
        finetune_language_layers=True,
    )
    assert selected is scoped_regex
    assert get_moe_target_parameters(_FakeMoeModel(), selected) is None


def test_frozen_language_full_list_does_not_discover_moe_parameters():
    # Vision-only request (finetune_language_layers=False) with a full leaf list must not reach the language-model
    # experts either.
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
        finetune_mlp_modules=True,
        finetune_language_layers=False,
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
        finetune_mlp_modules=True,
        finetune_language_layers=True,
    )
    assert selected is original_list
    assert get_moe_target_parameters(_FakeMoeModel(), selected) == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_attention_only_list_prefers_original_when_in_scope():
    # The case the PR originally fixed: an attention-only list routed through get_peft_regex under a family scope (e.g.
    # vision-off) still keeps experts off, because with MLP+language in scope detection uses the original attention-only
    # list rather than the regex's spurious mlp component block.
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
        finetune_mlp_modules=True,
        finetune_language_layers=True,
    )
    assert selected is attn_only_list
    assert get_moe_target_parameters(_FakeMoeModel(), selected) is None


def test_unfused_expert_parameters_resolve_both_leaves():
    """NemotronH keeps the expert projections unfused as separate 3D Parameters, so
    up_proj must resolve alongside down_proj instead of being silently dropped
    (unsloth#4476)."""
    import torch
    from unsloth.models._utils import get_moe_target_parameters

    class _Cfg:
        model_type = "nemotron_h"
        n_routed_experts = 4

    class _Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.up = torch.nn.Parameter(torch.zeros(4, 8, 8))
            self.down = torch.nn.Parameter(torch.zeros(4, 8, 8))

        def named_parameters(self, *args, **kwargs):
            yield "model.layers.1.mixer.experts.up_proj", self.up
            yield "model.layers.1.mixer.experts.down_proj", self.down

    got = get_moe_target_parameters(_Fake(), target_modules=["up_proj", "down_proj"])
    assert got == ["experts.up_proj", "experts.down_proj"], got


# ---------------------------------------------------------------------------------------
# Unfused expert layouts and the no-experts-resolved warning (unsloth#4476).
#
# Four expert layouts are in the wild and only the first two were covered before:
#   fused 3D Parameter        experts.gate_up_proj / experts.down_proj (transformers 5.x)
#   per-expert Linear list    experts.gate_up_projs.<i> (gpt-oss bnb-4bit)
#   unfused 3D Parameter      experts.gate_proj / up_proj / down_proj (NemotronH)
#   per-expert submodule      experts.<i>.gate_proj (Qwen3-MoE on transformers 4.x)
# The third resolved only down_proj, silently. The fourth needs no help from us, so it
# must not trip the warning added for the third.
# ---------------------------------------------------------------------------------------

ALL_MLP_LEAVES = ["gate_proj", "up_proj", "down_proj"]


def _unfused_moe_model(leaves=("gate_proj", "up_proj", "down_proj"), container="mixer"):
    """A MoE layer whose expert projections are separate 3D Parameters, as NemotronH
    builds them. ``container`` picks the ``experts.*`` (mixer) or ``mlp.experts.*``
    spelling, which are the two paths _resolve_moe_parameter_name chooses between."""
    experts = torch.nn.Module()
    for leaf in leaves:
        experts.register_parameter(leaf, torch.nn.Parameter(torch.zeros(4, 8, 8)))
    holder = torch.nn.Module()
    holder.add_module("experts", experts)
    model = torch.nn.Module()
    model.config = SimpleNamespace(n_routed_experts=4, model_type="nemotron_h")
    model.add_module(container, holder)
    return model


def _per_expert_submodule_moe_model(num_experts=2):
    """transformers 4.x Qwen3-MoE / Mixtral-with-named-projections: each expert is its own
    submodule, so PEFT attaches LoRA by ordinary leaf name with no help from Unsloth."""
    experts = torch.nn.ModuleList()
    for _ in range(num_experts):
        expert = torch.nn.Module()
        for leaf in ALL_MLP_LEAVES:
            expert.add_module(leaf, torch.nn.Linear(8, 8, bias=False))
        experts.append(expert)
    mlp = torch.nn.Module()
    mlp.add_module("experts", experts)
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_experts=num_experts, model_type="qwen3_moe")
    model.add_module("mlp", mlp)
    return model


def _per_expert_linear_moe_model(num_experts=2):
    """gpt-oss bnb-4bit: ModuleLists of Linear directly under ``experts``, which
    get_moe_target_modules turns into gate_up_projs.<i> suffixes."""
    experts = torch.nn.Module()
    experts.add_module(
        "gate_up_projs",
        torch.nn.ModuleList([torch.nn.Linear(8, 16, bias=False) for _ in range(num_experts)]),
    )
    experts.add_module(
        "down_projs",
        torch.nn.ModuleList([torch.nn.Linear(16, 8, bias=False) for _ in range(num_experts)]),
    )
    mlp = torch.nn.Module()
    mlp.add_module("experts", experts)
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_local_experts=num_experts, model_type="gpt_oss")
    model.add_module("mlp", mlp)
    return model


def _unreachable_moe_model(num_experts=2):
    """transformers 4.x Mixtral: the expert projections are w1/w2/w3, which none of
    gate_proj/up_proj/down_proj match, so the experts genuinely go untrained."""
    experts = torch.nn.ModuleList()
    for _ in range(num_experts):
        expert = torch.nn.Module()
        for leaf in ("w1", "w2", "w3"):
            expert.add_module(leaf, torch.nn.Linear(8, 8, bias=False))
        experts.append(expert)
    block = torch.nn.Module()
    block.add_module("experts", experts)
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_local_experts=num_experts, model_type="mixtral")
    model.add_module("block_sparse_moe", block)
    return model


@pytest.mark.parametrize(
    "container,prefix",
    [("mixer", "experts"), ("mlp", "mlp.experts")],
)
def test_unfused_experts_resolve_every_requested_leaf(container, prefix):
    from unsloth.models._utils import get_moe_target_parameters

    got = get_moe_target_parameters(
        _unfused_moe_model(container=container),
        target_modules=ALL_MLP_LEAVES,
    )
    assert got == [f"{prefix}.gate_proj", f"{prefix}.up_proj", f"{prefix}.down_proj"], got


def test_unfused_experts_honour_a_down_only_request():
    from unsloth.models._utils import get_moe_target_parameters

    got = get_moe_target_parameters(_unfused_moe_model(), target_modules=["down_proj"])
    assert got == ["experts.down_proj"], got


def test_unfused_experts_honour_a_gate_only_request():
    from unsloth.models._utils import get_moe_target_parameters

    got = get_moe_target_parameters(_unfused_moe_model(), target_modules=["gate_proj"])
    assert got == ["experts.gate_proj"], got


def test_gate_up_proj_shorthand_expands_to_both_unfused_leaves():
    """A caller who asks for the fused name on a model that has no fused Parameter must
    still get both halves, not nothing."""
    from unsloth.models._utils import get_moe_target_parameters

    got = get_moe_target_parameters(_unfused_moe_model(), target_modules=["gate_up_proj"])
    assert got == ["experts.gate_proj", "experts.up_proj"], got


def test_unfused_experts_skip_a_leaf_the_model_does_not_have():
    """NemotronH as reported carries up_proj and down_proj only; asking for gate_proj too
    must not produce a dead path for PEFT to choke on."""
    from unsloth.models._utils import get_moe_target_parameters

    model = _unfused_moe_model(leaves=("up_proj", "down_proj"))
    got = get_moe_target_parameters(model, target_modules=ALL_MLP_LEAVES)
    assert got == ["experts.up_proj", "experts.down_proj"], got


def test_unfused_experts_are_not_touched_by_an_attention_only_request():
    from unsloth.models._utils import get_moe_target_parameters

    got = get_moe_target_parameters(
        _unfused_moe_model(),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    assert got is None


def test_fused_layout_is_unchanged_by_the_unfused_fallback():
    """The fused branch must still win, and must not pick up unfused leaves as well."""
    from unsloth.models._utils import get_moe_target_parameters

    assert get_moe_target_parameters(_FakeMoeModel(), ALL_MLP_LEAVES) == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def _warnings_from(target, *args, **kwargs):
    """Run get_moe_target_parameters and return what it logged."""
    from unsloth.models import _utils

    records = []
    original = _utils.logger.warning
    _utils.logger.warning = lambda message, *a, **k: records.append(str(message))
    try:
        result = _utils.get_moe_target_parameters(target, *args, **kwargs)
    finally:
        _utils.logger.warning = original
    return result, records


def test_warns_when_no_route_reaches_the_experts():
    result, records = _warnings_from(_unreachable_moe_model(), ALL_MLP_LEAVES)
    assert result is None
    assert len(records) == 1, records
    assert "will NOT be trained" in records[0]


def test_does_not_warn_when_the_unfused_fallback_resolved_the_experts():
    result, records = _warnings_from(_unfused_moe_model(), ALL_MLP_LEAVES)
    assert result == ["experts.gate_proj", "experts.up_proj", "experts.down_proj"]
    assert records == []


def test_does_not_warn_when_an_ordinary_suffix_match_reaches_the_experts():
    """transformers 4.x builds Qwen3-MoE experts as per-expert submodules. PEFT attaches
    to them by leaf name, so a warning here would be a false alarm on a model that trains
    its experts correctly. This is what keeps the fix identical on 4.57.6 and 5.x."""
    result, records = _warnings_from(_per_expert_submodule_moe_model(), ALL_MLP_LEAVES)
    assert result is None
    assert records == []


def test_does_not_warn_for_the_per_expert_linear_layout():
    """gpt-oss bnb-4bit is handled by get_moe_target_modules, not by target_parameters."""
    result, records = _warnings_from(_per_expert_linear_moe_model(), ALL_MLP_LEAVES)
    assert result is None
    assert records == []


def test_does_not_warn_for_an_attention_only_request():
    result, records = _warnings_from(
        _unreachable_moe_model(),
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    assert result is None
    assert records == []


def test_does_not_warn_for_a_non_moe_model():
    plain = torch.nn.Module()
    plain.config = SimpleNamespace(model_type="llama")
    result, records = _warnings_from(plain, ALL_MLP_LEAVES)
    assert result is None
    assert records == []


def _count_calls(model, method_name):
    calls = []
    original = getattr(model, method_name)

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    setattr(model, method_name, counting)
    return calls


def test_named_parameters_is_walked_once():
    """Resolving the fused name, the two unfused leaves and down_proj used to be one walk
    of named_parameters each. On a real MoE checkpoint that walk is the whole cost."""
    from unsloth.models._utils import get_moe_target_parameters

    model = _unfused_moe_model()
    calls = _count_calls(model, "named_parameters")
    get_moe_target_parameters(model, ALL_MLP_LEAVES)
    assert len(calls) == 1, len(calls)


def _count_module_attribute_calls(name):
    """Count calls to a module level helper in unsloth.models._utils, keeping its result."""
    from unsloth.models import _utils

    calls = []
    original = getattr(_utils, name)

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    setattr(_utils, name, counting)
    return calls, lambda: setattr(_utils, name, original)


def test_supplied_module_targets_are_not_recomputed():
    """get_peft_model computes get_moe_target_modules anyway, so passing it in means the
    warning path costs no extra walk of the model."""
    from unsloth.models import _utils

    calls, restore = _count_module_attribute_calls("get_moe_target_modules")
    try:
        result = _utils.get_moe_target_parameters(
            _unreachable_moe_model(),
            ALL_MLP_LEAVES,
            moe_module_targets=["gate_up_projs.0"],
        )
    finally:
        restore()
    assert result is None
    assert calls == [], len(calls)


def test_module_targets_are_computed_when_the_caller_does_not_supply_them():
    """Standalone callers (and the tests above) must keep working unchanged."""
    from unsloth.models import _utils

    calls, restore = _count_module_attribute_calls("get_moe_target_modules")
    try:
        result = _utils.get_moe_target_parameters(_unreachable_moe_model(), ALL_MLP_LEAVES)
    finally:
        restore()
    assert result is None
    assert len(calls) == 1, len(calls)


def test_reachability_probe_runs_once_when_module_targets_are_empty():
    from unsloth.models import _utils

    calls, restore = _count_module_attribute_calls("_moe_experts_reachable_by_module_name")
    try:
        result = _utils.get_moe_target_parameters(
            _unreachable_moe_model(),
            ALL_MLP_LEAVES,
            moe_module_targets=[],
        )
    finally:
        restore()
    assert result is None
    assert len(calls) == 1, len(calls)


def test_reachability_probe_is_skipped_when_module_targets_resolved():
    from unsloth.models import _utils

    calls, restore = _count_module_attribute_calls("_moe_experts_reachable_by_module_name")
    try:
        _utils.get_moe_target_parameters(
            _per_expert_linear_moe_model(),
            ALL_MLP_LEAVES,
            moe_module_targets=["gate_up_projs.0"],
        )
    finally:
        restore()
    assert calls == []


def test_supplied_module_targets_suppress_the_warning():
    """A per-expert Linear model whose targets the caller already resolved must stay quiet
    even though no Parameter path matched."""
    model = _per_expert_linear_moe_model()
    result, records = _warnings_from(
        model,
        ALL_MLP_LEAVES,
        moe_module_targets=["gate_up_projs.0", "gate_up_projs.1"],
    )
    assert result is None
    assert records == []


def _non_mlp_submodule_moe_model(num_experts=2):
    """NemotronH-style per-expert submodules under ``mixer``, so a regex anchored on an
    ``mlp`` path segment reaches none of them."""
    experts = torch.nn.ModuleList()
    for _ in range(num_experts):
        expert = torch.nn.Module()
        for leaf in ALL_MLP_LEAVES:
            expert.add_module(leaf, torch.nn.Linear(8, 8, bias=False))
        experts.append(expert)
    mixer = torch.nn.Module()
    mixer.add_module("experts", experts)
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_experts=num_experts, model_type="nemotron_h")
    model.add_module("mixer", mixer)
    return model


def test_a_regex_that_peft_cannot_match_still_warns():
    """A str target_modules is a regex PEFT full-matches against the whole module path, so
    a leaf-only reachability answer suppressed the warning on the very case it exists for:
    ``.*mlp.*down_proj`` derives {"down_proj"} but matches no ``mixer.experts.N.down_proj``."""
    result, records = _warnings_from(_non_mlp_submodule_moe_model(), ".*mlp.*down_proj")
    assert result is None
    assert len(records) == 1, records
    assert "will NOT be trained" in records[0]


def test_a_regex_that_peft_does_match_stays_quiet():
    """The control: the same layout with a regex that really reaches the experts."""
    result, records = _warnings_from(_non_mlp_submodule_moe_model(), ".*experts.*down_proj")
    assert result is None
    assert records == []


def test_an_uncompilable_regex_does_not_claim_reachability():
    result, records = _warnings_from(_non_mlp_submodule_moe_model(), ".*mlp.*(down_proj")
    assert result is None
    assert len(records) == 1, records


def test_a_list_target_modules_still_matches_by_suffix():
    """PEFT matches a list by whole key or dotted suffix, which is the leaf test; the regex
    arm must not change that."""
    result, records = _warnings_from(_non_mlp_submodule_moe_model(), ALL_MLP_LEAVES)
    assert result is None
    assert records == []
