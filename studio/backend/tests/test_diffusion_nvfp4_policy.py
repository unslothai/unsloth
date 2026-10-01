# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from core.inference.diffusion_nvfp4_policy import (  # noqa: E402
    FLUX_R420,
    NVFP4_POLICY_KEY,
    NVFP4_POLICIES,
    NVFP4_POLICY_KIND,
    QWEN21_R020,
    QWEN2512_M120_ATTN8,
    QWEN_P02,
    ZIMG_RG76,
    Admit,
    NVFP4Policy,
    PolicyMismatch,
    Rule,
    assign_precisions,
    declares_policy,
    policy_by_id,
    policy_metadata,
    policy_metadata_error,
    quantize_with_policy,
    resolve_policy,
)

NVFP4 = "nvfp4"
FP8 = "fp8"
BF16 = "bf16"


def _zimage_rows() -> list:
    """Z-Image-Turbo's 276 linears: 30 transformer layers, 2 noise refiners, 2 context refiners."""
    rows = [
        ("all_x_embedder.2-1", 64, 3840),
        ("all_final_layer.2-1.linear", 3840, 64),
        ("all_final_layer.2-1.adaLN_modulation.1", 256, 3840),
        ("cap_embedder.1", 2560, 3840),
        ("t_embedder.mlp.0", 256, 1024),
        ("t_embedder.mlp.2", 1024, 256),
    ]

    def block(
        prefix: str,
        index: int,
        modulation: bool = True,
    ) -> list:
        out = []
        if modulation:
            out.append((f"{prefix}.{index}.adaLN_modulation.0", 256, 15360))
        for leaf in ("to_q", "to_k", "to_v"):
            out.append((f"{prefix}.{index}.attention.{leaf}", 3840, 3840))
        out.append((f"{prefix}.{index}.attention.to_out.0", 3840, 3840))
        out.append((f"{prefix}.{index}.feed_forward.w1", 3840, 10240))
        out.append((f"{prefix}.{index}.feed_forward.w2", 10240, 3840))
        out.append((f"{prefix}.{index}.feed_forward.w3", 3840, 10240))
        return out

    for index in range(30):
        rows += block("layers", index)
    for index in range(2):
        rows += block("noise_refiner", index)
    for index in range(2):
        rows += block("context_refiner", index, modulation = False)
    return rows


def _flux_rows() -> list:
    """FLUX.1-schnell's 502 linears: 19 double blocks and 38 single blocks."""
    rows = [
        ("x_embedder", 64, 3072),
        ("context_embedder", 4096, 3072),
        ("proj_out", 3072, 64),
        ("norm_out.linear", 3072, 6144),
        ("time_text_embed.text_embedder.linear_1", 768, 3072),
        ("time_text_embed.text_embedder.linear_2", 3072, 3072),
        ("time_text_embed.timestep_embedder.linear_1", 256, 3072),
        ("time_text_embed.timestep_embedder.linear_2", 3072, 3072),
    ]
    for index in range(19):
        prefix = f"transformer_blocks.{index}"
        for leaf in (
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ):
            rows.append((f"{prefix}.attn.{leaf}", 3072, 3072))
        rows.append((f"{prefix}.attn.to_out.0", 3072, 3072))
        rows.append((f"{prefix}.ff.net.0.proj", 3072, 12288))
        rows.append((f"{prefix}.ff.net.2", 12288, 3072))
        rows.append((f"{prefix}.ff_context.net.0.proj", 3072, 12288))
        rows.append((f"{prefix}.ff_context.net.2", 12288, 3072))
        rows.append((f"{prefix}.norm1.linear", 3072, 18432))
        rows.append((f"{prefix}.norm1_context.linear", 3072, 18432))
    for index in range(38):
        prefix = f"single_transformer_blocks.{index}"
        for leaf in ("to_q", "to_k", "to_v"):
            rows.append((f"{prefix}.attn.{leaf}", 3072, 3072))
        rows.append((f"{prefix}.norm.linear", 3072, 9216))
        rows.append((f"{prefix}.proj_mlp", 3072, 12288))
        rows.append((f"{prefix}.proj_out", 15360, 3072))
    return rows


def _qwen_rows() -> list:
    """Qwen-Image's 846 linears: 60 joint blocks, each with an img and a txt modulation."""
    rows = [
        ("img_in", 64, 3072),
        ("txt_in", 3584, 3072),
        ("proj_out", 3072, 64),
        ("norm_out.linear", 3072, 6144),
        ("time_text_embed.timestep_embedder.linear_1", 256, 3072),
        ("time_text_embed.timestep_embedder.linear_2", 3072, 3072),
    ]
    for index in range(60):
        prefix = f"transformer_blocks.{index}"
        for leaf in (
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ):
            rows.append((f"{prefix}.attn.{leaf}", 3072, 3072))
        rows.append((f"{prefix}.attn.to_out.0", 3072, 3072))
        rows.append((f"{prefix}.img_mlp.net.0.proj", 3072, 12288))
        rows.append((f"{prefix}.img_mlp.net.2", 12288, 3072))
        rows.append((f"{prefix}.txt_mlp.net.0.proj", 3072, 12288))
        rows.append((f"{prefix}.txt_mlp.net.2", 12288, 3072))
        rows.append((f"{prefix}.img_mod.1", 3072, 18432))
        rows.append((f"{prefix}.txt_mod.1", 3072, 18432))
    return rows


def _qwen21_rows() -> list:
    """Qwen-Image-2.1's 232 linears: 32 single-stream blocks and one global modulation."""
    rows = [
        ("img_in", 64, 4096),
        ("txt_in.in_layer", 4096, 4096),
        ("txt_in.out_layer", 4096, 4096),
        ("modulation.1", 4096, 16384),
        ("norm_out.linear", 4096, 4096),
        ("proj_out", 4096, 64),
        ("time_text_embed.timestep_embedder.linear_1", 256, 4096),
        ("time_text_embed.timestep_embedder.linear_2", 4096, 4096),
    ]
    for index in range(32):
        prefix = f"transformer_blocks.{index}"
        for leaf in ("to_q", "to_k", "to_v"):
            rows.append((f"{prefix}.attn.{leaf}", 4096, 4096))
        rows.append((f"{prefix}.attn.to_out.0", 4096, 4096))
        rows.append((f"{prefix}.img_mlp.proj", 4096, 12288))
        rows.append((f"{prefix}.img_mlp.gate_layer", 4096, 12288))
        rows.append((f"{prefix}.img_mlp.out", 12288, 4096))
    return rows


class _Tree:
    def __init__(self, rows) -> None:
        self.linears = {
            fqn: nn.Linear(in_features, out_features, device = "meta", dtype = torch.bfloat16)
            for fqn, in_features, out_features in rows
        }

    def named_modules(self):
        yield "", self
        yield from self.linears.items()


def _zimage() -> _Tree:
    return _Tree(_zimage_rows())


def _flux() -> _Tree:
    return _Tree(_flux_rows())


def _qwen() -> _Tree:
    return _Tree(_qwen_rows())


def _qwen21() -> _Tree:
    return _Tree(_qwen21_rows())


def test_the_synthetic_trees_reproduce_the_census_they_came_from():
    from core.inference.diffusion_transformer_quant import make_filter_fn
    admitted = make_filter_fn(512, ("lora_",), require_bf16 = True, require_divisible = 16)
    for tree, total, admits in (
        (_zimage(), 276, 239),
        (_flux(), 502, 499),
        (_qwen(), 846, 843),
        (_qwen21(), 232, 229),
    ):
        assert len(tree.linears) == total
        assert sum(1 for fqn, m in tree.linears.items() if admitted(m, fqn)) == admits


def test_a_policy_resolves_for_the_bases_it_was_solved_on_and_no_others():
    assert resolve_policy("z-image", "Tongyi-MAI/Z-Image-Turbo") is ZIMG_RG76
    assert resolve_policy("flux.1", "black-forest-labs/FLUX.1-schnell") is FLUX_R420
    assert resolve_policy("qwen-image", "Qwen/Qwen-Image") is QWEN_P02
    assert resolve_policy("qwen-image", "Qwen/Qwen-Image-2512") is QWEN2512_M120_ATTN8
    assert resolve_policy("qwen-image", "unsloth/Qwen-Image-2512") is QWEN2512_M120_ATTN8
    assert resolve_policy("qwen-image-2.1", "Qwen/Qwen-Image-2.1") is QWEN21_R020
    assert resolve_policy("qwen-image-2.1", "unsloth/Qwen-Image-2.1") is QWEN21_R020
    assert resolve_policy("qwen-image", "Qwen/Qwen-Image-2.1") is None
    assert resolve_policy("qwen-image-2.1", "Qwen/Qwen-Image-2512") is None
    assert resolve_policy("z-image", "unsloth/Z-Image-Turbo") is ZIMG_RG76
    assert resolve_policy("flux.1", "  UNSLOTH/FLUX.1-schnell ") is FLUX_R420
    assert resolve_policy("flux.1", "black-forest-labs/FLUX.1-dev") is None
    assert resolve_policy("qwen-image", "Qwen/Qwen-Image-Edit-2511") is None
    assert resolve_policy("z-image", None) is None
    assert resolve_policy("z-image", "") is None
    assert resolve_policy("wan2.2-ti2v-5b", "Wan-AI/Wan2.2-TI2V-5B-Diffusers") is None
    assert resolve_policy(None, "Tongyi-MAI/Z-Image-Turbo") is None
    assert policy_by_id("qwen_p02_v1") is QWEN_P02
    assert policy_by_id("qwen_p02_v2") is None


@pytest.mark.parametrize(
    ("policy", "tree_fn", "expected"),
    [
        (ZIMG_RG76, _zimage, {NVFP4: 76, FP8: 195, BF16: 5}),
        (FLUX_R420, _flux, {NVFP4: 187, FP8: 312, BF16: 3}),
        (QWEN21_R020, _qwen21, {NVFP4: 48, FP8: 181, BF16: 3}),
        (QWEN2512_M120_ATTN8, _qwen, {NVFP4: 600, FP8: 243, BF16: 3}),
        (QWEN_P02, _qwen, {NVFP4: 120, FP8: 723, BF16: 3}),
    ],
)
def test_each_policy_assigns_the_layer_counts_it_was_gated_at(policy, tree_fn, expected):
    from collections import Counter

    assignment = assign_precisions(tree_fn(), policy)
    assert dict(Counter(assignment.values())) == expected
    assert dict(policy.expected_counts) == expected


def test_the_superseded_picks_are_gone_rather_than_left_resolvable():
    assert policy_by_id("zimg_f8mod_toq34_v1") is None
    assert policy_by_id("flux_mod_single_v1") is None
    bases = {}
    for policy in NVFP4_POLICIES:
        for base in policy.base_repos:
            assert (policy.family, base) not in bases, (
                policy.policy_id,
                bases.get((policy.family, base)),
            )
            bases[(policy.family, base)] = policy.policy_id


def test_zimage_takes_both_query_and_key_the_early_ffn_gate_and_admits_the_modulation():
    tree = _zimage()
    assignment = assign_precisions(tree, ZIMG_RG76)
    nvfp4 = sorted(fqn for fqn, precision in assignment.items() if precision == NVFP4)
    assert len(nvfp4) == 76
    assert sum(1 for fqn in nvfp4 if fqn.endswith(".attention.to_q")) == 34
    assert sum(1 for fqn in nvfp4 if fqn.endswith(".attention.to_k")) == 34
    assert "context_refiner.0.attention.to_k" in nvfp4
    w1 = sorted(fqn for fqn in nvfp4 if fqn.endswith(".feed_forward.w1"))
    assert w1 == [f"layers.{i}.feed_forward.w1" for i in range(6)] + [
        "noise_refiner.0.feed_forward.w1",
        "noise_refiner.1.feed_forward.w1",
    ]
    assert assignment["layers.6.feed_forward.w1"] == FP8
    assert assignment["context_refiner.0.feed_forward.w1"] == FP8
    assert assignment["layers.0.attention.to_v"] == FP8
    admits = [fqn for fqn in assignment if fqn.endswith(".adaLN_modulation.0")]
    assert len(admits) == 32
    assert {assignment[fqn] for fqn in admits} == {FP8}
    assert assignment["all_final_layer.2-1.adaLN_modulation.1"] == BF16


def test_qwen_takes_both_modulation_streams_and_nothing_else():
    assignment = assign_precisions(_qwen(), QWEN_P02)
    nvfp4 = sorted(fqn for fqn, precision in assignment.items() if precision == NVFP4)
    assert len(nvfp4) == 120
    assert sum(1 for fqn in nvfp4 if fqn.endswith(".img_mod.1")) == 60
    assert sum(1 for fqn in nvfp4 if fqn.endswith(".txt_mod.1")) == 60
    assert assignment["transformer_blocks.0.img_mlp.net.0.proj"] == FP8


def test_a_renamed_layer_raises_rather_than_shipping_a_different_model():
    rows = [
        (fqn.replace("attention.to_q", "attention.q_proj"), i, o) for fqn, i, o in _zimage_rows()
    ]
    with pytest.raises(PolicyMismatch, match = "attention.to_q"):
        assign_precisions(_Tree(rows), ZIMG_RG76)
    extra = _zimage_rows() + [("layers.30.attention.to_q", 3840, 3840)]
    with pytest.raises(PolicyMismatch, match = "expects 34 layers, found 35"):
        assign_precisions(_Tree(extra), ZIMG_RG76)


def test_an_admitted_layer_of_another_width_is_refused_not_quantised():
    rows = [
        (fqn, i, 7680 if fqn.endswith(".adaLN_modulation.0") else o) for fqn, i, o in _zimage_rows()
    ]
    with pytest.raises(PolicyMismatch, match = r"\(256, 15360\)"):
        assign_precisions(_Tree(rows), ZIMG_RG76)
    fewer = [row for row in _zimage_rows() if not row[0].startswith("noise_refiner.0.adaLN")]
    with pytest.raises(PolicyMismatch, match = "expecting 32 layers, found 31"):
        assign_precisions(_Tree(fewer), ZIMG_RG76)


def test_a_layer_that_changed_width_moves_the_totals_and_raises():
    rows = [
        (fqn, 256 if fqn == "transformer_blocks.0.attn.to_k" else i, o)
        for fqn, i, o in _qwen_rows()
    ]
    with pytest.raises(PolicyMismatch, match = "expected"):
        assign_precisions(_Tree(rows), QWEN_P02)


def test_a_policy_whose_totals_do_not_add_up_cannot_be_applied_at_all():
    bogus = NVFP4Policy(
        policy_id = "bogus_v1",
        version = 1,
        family = "z-image",
        base_repos = ("tongyi-mai/z-image-turbo",),
        rules = (Rule(suffix = "attention.to_q", precision = NVFP4, expect = 34),),
        admit = (Admit(suffix = "adaLN_modulation.0", shape = (256, 15360), expect = 32),),
        expected_counts = {NVFP4: 34, FP8: 236, BF16: 6},
    )
    with pytest.raises(PolicyMismatch, match = "assigned"):
        assign_precisions(_zimage(), bogus)


def test_a_table_that_spells_out_a_zero_total_still_applies():
    spelled = NVFP4Policy(
        policy_id = "spelled_v1",
        version = 1,
        family = "flux.1",
        base_repos = ("black-forest-labs/flux.1-schnell",),
        rules = (
            Rule(
                suffix = "norm.linear",
                precision = NVFP4,
                expect = 38,
                prefix = "single_transformer_blocks.",
            ),
        ),
        expected_counts = {NVFP4: 38, FP8: 461, BF16: 3, "int8": 0},
    )
    assert assign_precisions(_flux(), spelled)


def test_a_per_block_rule_takes_exactly_the_blocks_it_lists():
    flux = assign_precisions(_flux(), FLUX_R420)
    single = [i for i in range(38) if flux[f"single_transformer_blocks.{i}.norm.linear"] == NVFP4]
    assert single == list(range(11, 38))
    assert flux["single_transformer_blocks.14.attn.to_k"] == FP8
    assert flux["single_transformer_blocks.15.attn.to_k"] == NVFP4
    assert flux["single_transformer_blocks.20.attn.to_v"] == FP8
    assert flux["single_transformer_blocks.21.attn.to_v"] == NVFP4
    assert flux["transformer_blocks.5.norm1.linear"] == NVFP4
    assert flux["transformer_blocks.3.norm1.linear"] == FP8
    assert flux["transformer_blocks.17.ff_context.net.2"] == FP8
    assert flux["norm_out.linear"] == NVFP4
    assert all(flux[f"transformer_blocks.{i}.attn.to_out.0"] == FP8 for i in range(19))
    qwen21 = assign_precisions(_qwen21(), QWEN21_R020)
    assert qwen21["transformer_blocks.5.attn.to_k"] == NVFP4
    assert qwen21["transformer_blocks.3.attn.to_k"] == FP8
    assert qwen21["transformer_blocks.29.attn.to_q"] == NVFP4
    assert qwen21["transformer_blocks.28.attn.to_q"] == FP8
    assert all(qwen21[f"transformer_blocks.{i}.img_mlp.out"] == FP8 for i in range(32))
    assert not any(
        qwen21[f"transformer_blocks.{i}.{leaf}"] == NVFP4
        for i in (30, 31)
        for leaf in ("attn.to_q", "attn.to_k", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out")
    )
    assert qwen21["modulation.1"] == FP8 and qwen21["txt_in.in_layer"] == FP8


def test_the_2512_policy_takes_modulation_and_all_eight_attention_projections():
    assignment = assign_precisions(_qwen(), QWEN2512_M120_ATTN8)
    nvfp4 = {fqn for fqn, precision in assignment.items() if precision == NVFP4}
    leaves = {fqn.split(".", 2)[2] for fqn in nvfp4}
    assert leaves == {
        "img_mod.1",
        "txt_mod.1",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    }
    assert assignment["transformer_blocks.0.img_mlp.net.0.proj"] == FP8
    assert assignment["norm_out.linear"] == FP8


def test_a_suffix_never_matches_a_longer_leaf_name():
    rule = Rule(suffix = "attention.to_q", precision = NVFP4, expect = 1)
    assert rule.matches("layers.0.attention.to_q")
    assert rule.matches("attention.to_q")
    assert not rule.matches("layers.0.attention.to_q_extra")
    assert not rule.matches("layers.0.self_attention.to_q")
    assert not rule.matches("layers.0.attention.to_qkv")
    prefixed = Rule(
        suffix = "norm.linear",
        precision = NVFP4,
        expect = 1,
        prefix = "single_transformer_blocks.",
    )
    assert prefixed.matches("single_transformer_blocks.7.norm.linear")
    assert not prefixed.matches("transformer_blocks.7.norm.linear")
    assert not prefixed.matches("single_transformer_blocks.7.norm1.linear")


def test_the_timestep_embedder_mlp_stays_dense_under_every_image_policy():
    assignment = assign_precisions(_zimage(), ZIMG_RG76)
    assert assignment["t_embedder.mlp.0"] == BF16
    assert assignment["t_embedder.mlp.2"] == BF16
    flux = assign_precisions(_flux(), FLUX_R420)
    assert flux["time_text_embed.timestep_embedder.linear_1"] == BF16
    assert flux["x_embedder"] == BF16 and flux["proj_out"] == BF16
    for policy, tree in (
        (QWEN_P02, _qwen()),
        (QWEN2512_M120_ATTN8, _qwen()),
        (QWEN21_R020, _qwen21()),
    ):
        qwen = assign_precisions(tree, policy)
        assert qwen["time_text_embed.timestep_embedder.linear_1"] == BF16, policy.policy_id
        assert qwen["img_in"] == BF16 and qwen["proj_out"] == BF16, policy.policy_id


class _FakeQuantized:
    def __init__(self, name: str) -> None:
        self.__class__ = type(name, (_FakeQuantized,), {})


def _stub_quantize(
    monkeypatch,
    *,
    skip = (),
    produced = None,
):
    calls: list = []
    classes = produced or {"cfg:nvfp4": "NVFP4Tensor", "cfg:fp8": "Float8Tensor"}

    def _quantize_(
        module,
        config,
        filter_fn = None,
    ):
        selected = [
            fqn
            for fqn, sub in module.named_modules()
            if isinstance(sub, nn.Linear) and (filter_fn is None or filter_fn(sub, fqn))
        ]
        calls.append({"config": config, "selected": selected, "filter_fn": filter_fn})
        modules = dict(module.named_modules())
        for fqn in selected:
            if fqn in skip:
                continue
            modules[fqn]._parameters["weight"] = _FakeQuantized(classes[config])

    quantization = types.ModuleType("torchao.quantization")
    quantization.quantize_ = _quantize_
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)
    dtq = sys.modules["core.inference.diffusion_transformer_quant"]
    monkeypatch.setattr(dtq, "_make_quant_config", lambda scheme, fast_accum = None: f"cfg:{scheme}")
    return calls


def test_the_two_passes_are_disjoint_and_nvfp4_runs_first(monkeypatch):
    import core.inference.diffusion_transformer_quant  # noqa: F401 - the stub patches the module

    calls = _stub_quantize(monkeypatch)
    tree = _flux()
    assignment = quantize_with_policy(tree, FLUX_R420)
    assert [call["config"] for call in calls] == ["cfg:nvfp4", "cfg:fp8"]
    nvfp4, fp8 = set(calls[0]["selected"]), set(calls[1]["selected"])
    assert len(nvfp4) == 187 and len(fp8) == 312
    assert not (nvfp4 & fp8)
    assert nvfp4 == {fqn for fqn, p in assignment.items() if p == NVFP4}
    assert fp8 == {fqn for fqn, p in assignment.items() if p == FP8}
    fp8_filter = calls[1]["filter_fn"]
    already = tree.linears["single_transformer_blocks.11.norm.linear"]
    assert not fp8_filter(already, "single_transformer_blocks.11.norm.linear")
    victim = "single_transformer_blocks.0.attn.to_q"
    tree.linears[victim]._parameters["weight"] = _FakeQuantized("NVFP4Tensor")
    assert not fp8_filter(tree.linears[victim], victim)


def test_a_layer_the_quantiser_silently_declined_fails_the_build(monkeypatch):
    import core.inference.diffusion_transformer_quant  # noqa: F401 - the stub patches the module
    _stub_quantize(monkeypatch, skip = ("transformer_blocks.3.attn.to_v",))
    with pytest.raises(PolicyMismatch, match = "transformer_blocks.3.attn.to_v"):
        quantize_with_policy(_qwen(), QWEN_P02)


def test_a_pass_that_produced_the_wrong_tensor_class_fails_the_build(monkeypatch):
    import core.inference.diffusion_transformer_quant  # noqa: F401 - the stub patches the module
    _stub_quantize(monkeypatch, produced = {"cfg:nvfp4": "Float8Tensor", "cfg:fp8": "Float8Tensor"})
    with pytest.raises(PolicyMismatch, match = "wanted NVFP4Tensor"):
        quantize_with_policy(_zimage(), ZIMG_RG76)


def test_the_metadata_block_records_the_set_that_was_built():
    assignment = assign_precisions(_zimage(), ZIMG_RG76)
    fragment = policy_metadata(ZIMG_RG76, assignment)
    block = fragment[NVFP4_POLICY_KEY]
    assert block["kind"] == NVFP4_POLICY_KIND
    assert block["policy_id"] == "zimg_rg76_v1" and block["policy_version"] == 1
    assert block["counts"] == {BF16: 5, FP8: 195, NVFP4: 76}
    assert len(block["nvfp4_fqns"]) == 76
    assert block["nvfp4_fqns"] == sorted(block["nvfp4_fqns"])
    assert block["activation_scales_baked"] is False and block["gptq"] is False
    assert declares_policy({"scheme": "nvfp4", **fragment})
    assert policy_metadata_error({"scheme": "nvfp4", **fragment}) is None
    assert policy_metadata(ZIMG_RG76, assign_precisions(_zimage(), ZIMG_RG76)) == fragment


def test_an_unreadable_policy_block_reads_as_declared_so_it_can_be_refused():
    assert not declares_policy({"scheme": "nvfp4"})
    assert not declares_policy(None)
    assert not declares_policy({NVFP4_POLICY_KEY: None})
    assert declares_policy({NVFP4_POLICY_KEY: {}})
    assert policy_metadata_error({"scheme": "nvfp4"}) is None


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({}, "unsupported nvfp4 policy"),
        ("zimg_rg76_v1", "not a dict"),
        ({"kind": "unsloth_nvfp4_layer_policy_v2"}, "unsupported nvfp4 policy"),
        ({"kind": NVFP4_POLICY_KIND}, "no policy_id"),
        ({"kind": NVFP4_POLICY_KIND, "policy_id": "p"}, "policy_version"),
        (
            {"kind": NVFP4_POLICY_KIND, "policy_id": "p", "policy_version": 1},
            "no counts",
        ),
        (
            {
                "kind": NVFP4_POLICY_KIND,
                "policy_id": "p",
                "policy_version": 1,
                "counts": {NVFP4: 2},
            },
            "no nvfp4_fqns",
        ),
        (
            {
                "kind": NVFP4_POLICY_KIND,
                "policy_id": "p",
                "policy_version": 1,
                "counts": {NVFP4: 2},
                "nvfp4_fqns": ["a", "a"],
            },
            "duplicates",
        ),
        (
            {
                "kind": NVFP4_POLICY_KIND,
                "policy_id": "p",
                "policy_version": 1,
                "counts": {NVFP4: 3},
                "nvfp4_fqns": ["a", "b"],
            },
            "lists 2",
        ),
    ],
)
def test_every_malformed_policy_block_says_why(block, expected):
    problem = policy_metadata_error({NVFP4_POLICY_KEY: block})
    assert problem is not None and expected in problem
