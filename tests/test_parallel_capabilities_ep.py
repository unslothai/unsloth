# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expert parallel (UNSLOTH_EP_SIZE) in the registry: its refusals, family eligibility and cells. CPU, meta models."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import parallel_capabilities as caps
from unsloth.parallel_env import ParallelLayout

_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_EP_SIZE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_PARALLEL_STRICT",
    "WORLD_SIZE",
)


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    from unsloth.models import tp_env

    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    tp_env._reset_env_layout()
    yield monkeypatch
    tp_env._reset_env_layout()


_SMALL = dict(vocab_size = 128, hidden_size = 64, num_hidden_layers = 2, num_attention_heads = 4)


def _meta(family):
    from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, Qwen3MoeConfig

    if family == "jetmoe":
        config = AutoConfig.for_model("jetmoe")
        for key, value in dict(_SMALL, intermediate_size = 32, num_key_value_heads = 2).items():
            setattr(config, key, value)
    elif family == "llama":
        config = LlamaConfig(**_SMALL, intermediate_size = 32)
    else:
        config = Qwen3MoeConfig(
            **_SMALL,
            intermediate_size = 32,
            num_key_value_heads = 4,
            num_experts = 4,
            num_experts_per_tok = 2,
            moe_intermediate_size = 32,
            head_dim = 16,
        )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return model.requires_grad_(False)


def _ep_layout(ep = 2, **kw):
    return ParallelLayout(1, ep, ep, kw.pop("fsdp2", False), ep = ep, **kw)


def _ep_query(**kw):
    base = dict(trainer = "sft", family = "qwen3_moe", moe = True, dp = 2, ep = 2, torch = (2, 13))
    base.update(kw)
    return caps.Query(**base)


def _refusal(query):
    with pytest.raises(NotImplementedError) as info:
        caps.check(query)
    return str(info.value)


def test_ep_with_vllm_is_refused():
    message = _refusal(_ep_query(trainer = "grpo", vllm = True))
    assert message.startswith("Unsloth: ep x vLLM: "), message
    assert "DTensor" in message and "fast_inference = False" in message, message
    caps.check(_ep_query(trainer = "grpo"), cells = False)
    caps.check(_ep_query(ep = 1, trainer = "grpo", vllm = True), cells = False)


@pytest.mark.parametrize(
    "axis",
    [dict(tp = 2), dict(pp = 2), dict(fsdp2 = True), dict(cp = 2, cp_backend = "ulysses", fsdp2 = True)],
    ids = ["tp", "pp", "fsdp2", "cp_fsdp2"],
)
def test_ep_with_another_model_axis_or_fsdp2_is_refused(axis):
    message = _refusal(_ep_query(**axis))
    assert message.startswith("Unsloth: ep x tp / pp / fsdp2: "), message
    assert "ep 2" in message, message


def test_ep_with_ulysses_cp_has_a_supported_cell_and_ring_is_refused():
    ulysses = _ep_query(dp = 1, cp = 2, cp_backend = "ulysses", expert_lora = True, mode = "lora")
    status, cell = caps.cell_status(ulysses)
    assert status == "supported" and "x ep_cp" in cell.name, (status, cell)
    message = _refusal(_ep_query(dp = 1, cp = 2, cp_backend = "ring", expert_lora = True, mode = "lora"))
    assert "no supported cell covers it" in message, message


@pytest.mark.parametrize(
    "family, reason",
    [
        ("jetmoe", "its experts forward takes ('layer_input',)"),
        ("llama", "no module has an `experts` child"),
    ],
)
def test_models_ep_cannot_shard_are_refused_by_the_registry(family, reason):
    model = _meta(family)
    query = caps.query_from(_ep_layout(), None, model)
    assert query.ep == 2 and reason in query.ep_blocker, query
    with pytest.raises(NotImplementedError) as info:
        caps.check_layout(_ep_layout(), None, model, types.SimpleNamespace(packing = False))
    message = str(info.value)
    assert message.startswith("Unsloth: ep x model: "), message
    assert reason in message and "Instead: drop ep" in message, message
    assert all(p.is_meta for p in model.parameters())


def test_a_model_ep_shards_is_not_refused_and_no_ep_asks_nothing():
    model = _meta("qwen3_moe")
    caps.check_layout(_ep_layout(), None, model, types.SimpleNamespace(packing = False))
    assert caps.query_from(_ep_layout(), None, model).ep_blocker == ""
    jetmoe = _meta("jetmoe")
    assert caps.query_from(ParallelLayout(1, 2, 2, False), None, jetmoe).ep_blocker == ""


def test_swapped_expert_lora_factors_shard_under_ep_and_stay_replicated_under_tp(monkeypatch):
    # zoo flags Llama4TextExperts (is_transposed) and Qwen3VLMoeTextExperts (transformers 5.x) as training expert
    # LoRA factors swapped against vanilla PEFT's: expert parallel cuts and saves them (gloo parity in
    # test_ep_moe_families_gloo.py), tensor parallel keeps such experts whole.
    from unsloth.models import tp_native

    for family in ("llama4", "qwen3_vl_moe"):
        assert caps._ep_blocker(family, 2) == "", family
        status, cell = caps.cell_status(_ep_query(family = family, expert_lora = True))
        assert status == "supported", (family, status, cell)
        assert (
            cell.parity
            == f"tests/test_ep_moe_families_gloo.py::test_ep_family_matches_one_process[{family}]"
        )
    model = _meta("qwen3_moe")
    block = tp_native.moe_blocks_with_experts(model)[0]
    experts = tp_native._experts_chain(block)[-1]
    if isinstance(experts, torch.nn.ModuleList):
        pytest.skip("transformers 4.x per-expert modules: no fused experts class to flag")
    monkeypatch.setattr(type(experts), "_unsloth_expert_lora_factors_swapped", True, raising = False)
    assert tp_native.expert_shard_problem(block, 2) is None
    assert tp_native.expert_shard_problem(block, 2, packed = False) == "swapped expert LoRA factors"
    assert caps.query_from(_ep_layout(), None, model).ep_blocker == ""
    caps.check_layout(_ep_layout(), None, model, types.SimpleNamespace(packing = False))


def test_family_table_answers_when_only_the_model_type_is_known():
    assert "layer_input" in caps._ep_blocker("jetmoe", 2)
    assert caps._ep_blocker("qwen3_moe", 2) == "" and caps._ep_blocker("jetmoe", 1) == ""
    message = _refusal(_ep_query(family = "jetmoe", ep_blocker = caps._ep_blocker("jetmoe", 2)))
    assert message.startswith("Unsloth: ep x model: "), message


def test_ep_cells_cite_the_gloo_tests_and_leave_dp_cells_alone():
    query = _ep_query(expert_lora = True)
    status, cell = caps.cell_status(query)
    assert (
        status == "supported" and cell.name == "MoE qwen3_moe x lora+experts x aux off x ep"
    ), cell
    assert cell.parity == (
        "tests/test_ep_moe_families_gloo.py::test_ep_family_matches_one_process[qwen3_moe]"
    )
    ddp = caps.cell_status(_ep_query(ep = 1, expert_lora = True))[1]
    assert ddp.name.endswith(" x ddp"), ddp.name
    assert "refused by the ep x model rule" in caps._moe_note("jetmoe", None, False, "ep")
    inkling = caps.cell_status(_ep_query(family = "inkling2", expert_lora = True))[1]
    assert inkling.name == "Inkling 2 x ep" and not inkling.missing(), inkling


def test_trainer_precheck_runs_for_an_ep_only_environment(clean_env):
    from unsloth.models import _utils, tp_env

    clean_env.setenv("UNSLOTH_EP_SIZE", "2")
    clean_env.setenv("WORLD_SIZE", "2")
    assert tp_env.env_layout() is None

    class Trainer:
        def __init__(
            self,
            model = None,
            args = None,
        ):
            pass

    trainer = object.__new__(Trainer)
    args = types.SimpleNamespace(packing = False)
    with pytest.raises(NotImplementedError, match = "ep x model: .*layer_input"):
        _utils._parallel_capabilities_pre(
            trainer, Trainer.__init__, (), {"args": args}, _meta("jetmoe")
        )
    vllm = types.SimpleNamespace(packing = False, use_vllm = True)
    with pytest.raises(NotImplementedError, match = "ep x vLLM"):
        _utils._parallel_capabilities_pre(
            trainer, Trainer.__init__, (), {"args": vllm}, _meta("qwen3_moe")
        )
    _utils._parallel_capabilities_pre(
        trainer, Trainer.__init__, (), {"args": args}, _meta("qwen3_moe")
    )


def test_fast_inference_is_refused_at_load_under_ep(clean_env):
    from unsloth.models import tp_env

    clean_env.setenv("UNSLOTH_EP_SIZE", "2")
    calls = []

    def from_pretrained(
        model_name = None,
        fast_inference = False,
        load_in_4bit = True,
        **kwargs,
    ):
        calls.append(model_name)

    load = tp_env.tensor_parallel_env_load(from_pretrained)
    with pytest.raises(NotImplementedError) as info:
        load("unsloth/Qwen3-30B-A3B", fast_inference = True)
    message = str(info.value)
    assert message.startswith(
        "Unsloth: UNSLOTH_EP_SIZE=2 (expert parallel) does not support vLLM"
    ), message
    assert "fast_inference = False" in message and not calls, message


def _hunyuan_meta():
    """Hunyuan-A13B's layout shrunk: 64 experts, a top-k per layer, a shared expert."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "hunyuan_v1_moe",
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 16,
        num_hidden_layers = 3,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        num_experts = 64,
        moe_topk = [8, 8, 8],
        num_shared_expert = [1, 1, 1],
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return model.requires_grad_(False)


def test_hunyuan_v1_moe_is_not_blocked_by_its_default_config():
    """The family table once held hunyuan_v1_moe because its default config has num_experts = 1; a real
    (64-expert) model shards, so only the loaded model's own num_experts decides."""
    assert caps._ep_blocker("hunyuan_v1_moe", 2) == ""
    assert "hunyuan_v1_moe" not in caps._EP_BLOCKED_FAMILIES
    model = _hunyuan_meta()
    # transformers 4.x keeps per-expert modules: they shard as a per-expert list (models/ep_expert_lists.py).
    assert caps._ep_blocker(model, 2) == ""
    caps.check_layout(_ep_layout(), None, model, types.SimpleNamespace(packing = False))
    assert "refused" not in caps._moe_note("hunyuan_v1_moe", "lora", False, "ep")
    assert "refused" not in caps._moe_note("gpt_oss", "qlora", False, "ep")
    assert "refused" not in caps._moe_note("gpt_oss", "lora", False, "ep")


def test_gpt_oss_4bit_is_planned_when_the_experts_split_evenly():
    """load_in_4bit gpt-oss is refused only for an ep not dividing the released expert counts (32, 128)."""
    assert caps._ep_blocker("gpt_oss", 2) == ""
    assert caps._ep_blocker("gpt_oss", 2, load_in_4bit = False) == ""
    for ep in (2, 4, 8):
        assert caps._ep_blocker("gpt_oss", ep, load_in_4bit = True) == "", ep
    assert caps._ep_blocker("gpt_oss", 1, load_in_4bit = True) == ""
    assert caps._ep_blocker("qwen3_moe", 2, load_in_4bit = True) == ""
    assert caps._ep_blocker("gpt_oss", 3, load_in_4bit = False) == ""
    reason = caps._ep_blocker("gpt_oss", 3, load_in_4bit = True)
    assert "per-expert Linear lists" in reason and "32 experts over 3" in reason, reason
    message = _refusal(_ep_query(family = "gpt_oss", mode = "qlora", ep_blocker = reason))
    assert (
        message.startswith("Unsloth: ep x model: ") and "per-expert Linear lists" in message
    ), message
    # The checkpoint's own expert count when the caller read its config (Studio's model facts).
    assert "30 experts over 4" in caps._ep_blocker("gpt_oss", 4, load_in_4bit = True, experts = 30)
    assert caps._ep_blocker("gpt_oss", 3, load_in_4bit = True, experts = 6) == ""
    status, cell = caps.cell_status(_ep_query(family = "gpt_oss", mode = "qlora", attention = "eager"))
    assert status == "supported" and cell.name == "MoE gpt_oss x qlora x aux off x ep", (
        status,
        cell,
    )


def test_gpt_oss_with_zoo_bnb4bit_experts_shards_by_structure():
    """zoo's GptOssExpertsBnb4bit (what every load_in_4bit gpt-oss load installs), built unquantized on meta: its
    per-expert Linear lists now shard (models/ep_expert_lists.py), so the loaded model is not refused."""
    zoo_gpt_oss = pytest.importorskip("unsloth_zoo.temporary_patches.gpt_oss")
    bnb_experts = getattr(zoo_gpt_oss, "GptOssExpertsBnb4bit", None)
    if bnb_experts is None:
        pytest.skip("this unsloth_zoo has no GptOssExpertsBnb4bit")
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss
    from transformers import AutoConfig, AutoModelForCausalLM
    from unsloth.models import ep_native

    config = AutoConfig.for_model(
        "gpt_oss",
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        layer_types = ["full_attention", "full_attention"],
    )
    stock = gpt_oss.GptOssExperts
    try:
        gpt_oss.GptOssExperts = bnb_experts
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    finally:
        gpt_oss.GptOssExperts = stock
    assert type(model.model.layers[0].mlp.experts) is bnb_experts
    assert caps._ep_blocker(model, 2) == ""
    caps.check_layout(_ep_layout(), None, model, types.SimpleNamespace(packing = False))
    ep_native.ep_refuse_per_expert_lists(model)
    # 4 experts over 3 ranks do not split evenly: still refused, with the reason.
    assert "per-expert Linear lists" in caps._ep_blocker(model, 3)
    with torch.device("meta"):
        plain = AutoModelForCausalLM.from_config(config)
    assert caps._ep_blocker(plain, 2) == ""
    ep_native.ep_refuse_per_expert_lists(plain)
