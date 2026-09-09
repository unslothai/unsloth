# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the per-layer NVFP4 image policies (``diffusion_nvfp4_policy.py``).

The module trees here are SYNTHETIC but census-exact: every fqn and every ``(in_features,
out_features)`` pair is the one the linear census read off the real diffusers module for that base
repo (z-image 276 linears / 239 admitted, flux.1 502 / 499, qwen-image 846 / 843), built as
``nn.Linear`` on the meta device so a 60-block DiT costs no memory. The counts the policies assert
were verified against those same trees instantiated from the real configs, so a test passing here
means the policy selects the layers the campaign measured.

What is being tested is fail-closed behaviour. A policy that silently selects a different set of
layers produces an artifact that loads clean, renders, and is not the model any gate ran on, so
every rule carries a count and every count is asserted; the tests below are mostly the ways that
assertion has to fire.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from core.inference.diffusion_nvfp4_policy import (  # noqa: E402
    FLUX_MOD_SINGLE,
    NVFP4_POLICY_KEY,
    NVFP4_POLICY_KIND,
    QWEN_P02,
    ZIMAGE_F8MOD_TOQ34,
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


# ── the census-derived module trees ──────────────────────────────────────────────


def _zimage_rows() -> list:
    """Z-Image-Turbo's 276 linears: 30 transformer layers, 2 noise refiners, 2 context refiners."""
    rows = [
        ("all_x_embedder.2-1", 64, 3840),
        ("all_final_layer.2-1.linear", 3840, 64),
        ("all_final_layer.2-1.adaLN_modulation.1", 256, 3840),
        ("cap_embedder.1", 2560, 3840),
        # TimestepEmbedder.forward reads mlp[0].weight.dtype, so these two can never be swapped.
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
        # The context refiners carry no modulation projection, which is why 32 and not 34.
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
        # The double blocks spell their modulation projection norm1.linear, at twice the width of
        # the single blocks' norm.linear. Nothing gated them.
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


class _Tree:
    """A DiT as far as everything under test looks at one: a flat ``named_modules`` walk.

    The Linears are real ``nn.Linear`` (the shared filter asks ``isinstance``, ``in_features``
    and ``weight.dtype``) on the meta device, so the whole 846-linear qwen tree allocates nothing.
    """

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


def test_the_synthetic_trees_reproduce_the_census_they_came_from():
    """The trees are the fixture everything else rests on, so their totals are asserted first."""
    from core.inference.diffusion_transformer_quant import make_filter_fn

    admitted = make_filter_fn(512, ("lora_",), require_bf16 = True, require_divisible = 16)
    for tree, total, admits in ((_zimage(), 276, 239), (_flux(), 502, 499), (_qwen(), 846, 843)):
        assert len(tree.linears) == total
        assert sum(1 for fqn, m in tree.linears.items() if admitted(m, fqn)) == admits


# ── T-1 resolution per base ──────────────────────────────────────────────────────


def test_a_policy_resolves_for_the_bases_it_was_solved_on_and_no_others():
    assert resolve_policy("z-image", "Tongyi-MAI/Z-Image-Turbo") is ZIMAGE_F8MOD_TOQ34
    assert resolve_policy("flux.1", "black-forest-labs/FLUX.1-schnell") is FLUX_MOD_SINGLE
    assert resolve_policy("qwen-image", "Qwen/Qwen-Image") is QWEN_P02
    # A mirror is the same weights under another name, so it canonicalises to the same verdict.
    assert resolve_policy("z-image", "unsloth/Z-Image-Turbo") is ZIMAGE_F8MOD_TOQ34
    assert resolve_policy("flux.1", "  UNSLOTH/FLUX.1-schnell ") is FLUX_MOD_SINGLE
    # A sibling checkpoint is a different set of weights whose gate has not run.
    assert resolve_policy("flux.1", "black-forest-labs/FLUX.1-dev") is None
    assert resolve_policy("qwen-image", "Qwen/Qwen-Image-Edit-2511") is None
    # An unnamed base never inherits the family's only policy: today's "only" is a fact about the
    # table, not about the model the caller is holding.
    assert resolve_policy("z-image", None) is None
    assert resolve_policy("z-image", "") is None
    # An unlisted family has no policy at all, and neither has a missing one.
    assert resolve_policy("wan2.2-ti2v-5b", "Wan-AI/Wan2.2-TI2V-5B-Diffusers") is None
    assert resolve_policy(None, "Tongyi-MAI/Z-Image-Turbo") is None
    # The id round-trips, which is how a checkpoint's declaration is re-resolved.
    assert policy_by_id("qwen_p02_v1") is QWEN_P02
    assert policy_by_id("qwen_p02_v2") is None


# ── T-2 exact counts on the census trees ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("policy", "tree_fn", "expected"),
    [
        (ZIMAGE_F8MOD_TOQ34, _zimage, {NVFP4: 34, FP8: 237, BF16: 5}),
        (FLUX_MOD_SINGLE, _flux, {NVFP4: 38, FP8: 461, BF16: 3}),
        (QWEN_P02, _qwen, {NVFP4: 120, FP8: 723, BF16: 3}),
    ],
)
def test_each_policy_assigns_the_layer_counts_it_was_gated_at(policy, tree_fn, expected):
    from collections import Counter

    assignment = assign_precisions(tree_fn(), policy)
    assert dict(Counter(assignment.values())) == expected
    assert dict(policy.expected_counts) == expected


def test_zimage_takes_the_to_q_projections_and_admits_the_modulation_the_floor_rejects():
    tree = _zimage()
    assignment = assign_precisions(tree, ZIMAGE_F8MOD_TOQ34)
    nvfp4 = sorted(fqn for fqn, precision in assignment.items() if precision == NVFP4)
    assert len(nvfp4) == 34
    assert all(fqn.endswith(".attention.to_q") for fqn in nvfp4)
    # Every to_q in the model, refiners included, not just the 30 main layers.
    assert "context_refiner.0.attention.to_q" in nvfp4
    assert "noise_refiner.1.attention.to_q" in nvfp4
    # The 32 (256, 15360) modulation projections are below the 512 floor and quantised anyway,
    # which is where most of this policy's saving comes from.
    admits = [fqn for fqn in assignment if fqn.endswith(".adaLN_modulation.0")]
    assert len(admits) == 32
    assert {assignment[fqn] for fqn in admits} == {FP8}
    # The final layer's adaLN_modulation.1 is a different layer of a different width: not admitted.
    assert assignment["all_final_layer.2-1.adaLN_modulation.1"] == BF16


def test_qwen_takes_both_modulation_streams_and_nothing_else():
    assignment = assign_precisions(_qwen(), QWEN_P02)
    nvfp4 = sorted(fqn for fqn, precision in assignment.items() if precision == NVFP4)
    assert len(nvfp4) == 120
    assert sum(1 for fqn in nvfp4 if fqn.endswith(".img_mod.1")) == 60
    assert sum(1 for fqn in nvfp4 if fqn.endswith(".txt_mod.1")) == 60
    # The mlp projections that share the mod prefix stay fp8.
    assert assignment["transformer_blocks.0.img_mlp.net.0.proj"] == FP8


# ── T-3 fail closed on a rename or a shape drift ─────────────────────────────────


def test_a_renamed_layer_raises_rather_than_shipping_a_different_model():
    rows = [
        (fqn.replace("attention.to_q", "attention.q_proj"), i, o) for fqn, i, o in _zimage_rows()
    ]
    with pytest.raises(PolicyMismatch, match = "attention.to_q"):
        assign_precisions(_Tree(rows), ZIMAGE_F8MOD_TOQ34)
    # One extra layer wearing the rule's name is just as wrong as one fewer.
    extra = _zimage_rows() + [("layers.30.attention.to_q", 3840, 3840)]
    with pytest.raises(PolicyMismatch, match = "expects 34 layers, found 35"):
        assign_precisions(_Tree(extra), ZIMAGE_F8MOD_TOQ34)


def test_an_admitted_layer_of_another_width_is_refused_not_quantised():
    # The min_features floor is being overridden for one measured layer, so a layer of another
    # width wearing the same name is not that layer.
    rows = [
        (fqn, i, 7680 if fqn.endswith(".adaLN_modulation.0") else o) for fqn, i, o in _zimage_rows()
    ]
    with pytest.raises(PolicyMismatch, match = r"\(256, 15360\)"):
        assign_precisions(_Tree(rows), ZIMAGE_F8MOD_TOQ34)
    # A missing admit is caught by its own count before any shape is looked at.
    fewer = [row for row in _zimage_rows() if not row[0].startswith("noise_refiner.0.adaLN")]
    with pytest.raises(PolicyMismatch, match = "expecting 32 layers, found 31"):
        assign_precisions(_Tree(fewer), ZIMAGE_F8MOD_TOQ34)


def test_a_layer_that_changed_width_moves_the_totals_and_raises():
    # Every rule count still holds; what changed is that one fp8 layer dropped below the floor.
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
    # A Counter never records a precision no layer took, so a policy whose model has nothing left
    # dense may write bf16: 0 or leave it out and mean the same thing.
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


# ── T-4 suffix, never substring ──────────────────────────────────────────────────


def test_the_flux_rule_takes_the_single_blocks_modulation_and_no_other_linear():
    assignment = assign_precisions(_flux(), FLUX_MOD_SINGLE)
    nvfp4 = sorted(fqn for fqn, precision in assignment.items() if precision == NVFP4)
    assert len(nvfp4) == 38
    assert nvfp4[0] == "single_transformer_blocks.0.norm.linear"
    # norm1.linear ENDS WITH "linear" and CONTAINS "norm", and is a different layer at twice the
    # width. A substring match would have taken all 38 of these too.
    assert assignment["transformer_blocks.0.norm1.linear"] == FP8
    assert assignment["transformer_blocks.0.norm1_context.linear"] == FP8
    # And the top-level norm_out.linear, which the prefix excludes as well.
    assert assignment["norm_out.linear"] == FP8
    # The rule's prefix is a subtree, not a name fragment: a double block never matches it.
    assert not any(fqn.startswith("transformer_blocks.") for fqn in nvfp4)


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


# ── T-5 the layers that can never be swapped ─────────────────────────────────────


def test_the_timestep_embedder_mlp_stays_dense_under_every_image_policy():
    # TimestepEmbedder.forward reads mlp[0].weight.dtype, so a quantised weight there is a raise
    # on the first denoise. They sit below the 512 floor, and this is the reason no policy is
    # allowed to lower it (256 would admit them).
    assignment = assign_precisions(_zimage(), ZIMAGE_F8MOD_TOQ34)
    assert assignment["t_embedder.mlp.0"] == BF16
    assert assignment["t_embedder.mlp.2"] == BF16
    # The same layer in the other two families, under their own policies.
    flux = assign_precisions(_flux(), FLUX_MOD_SINGLE)
    assert flux["time_text_embed.timestep_embedder.linear_1"] == BF16
    qwen = assign_precisions(_qwen(), QWEN_P02)
    assert qwen["time_text_embed.timestep_embedder.linear_1"] == BF16
    # And the layers the GEMM floor rejects for shape rather than size.
    assert flux["x_embedder"] == BF16 and flux["proj_out"] == BF16
    assert qwen["img_in"] == BF16 and qwen["proj_out"] == BF16


# ── T-6 the two quantise passes ──────────────────────────────────────────────────


class _FakeQuantized:
    """Stands in for a torchao weight subclass: only its class NAME is ever read."""

    def __init__(self, name: str) -> None:
        self.__class__ = type(name, (_FakeQuantized,), {})


def _stub_quantize(
    monkeypatch,
    *,
    skip = (),
    produced = None,
):
    """Record every ``quantize_`` pass and swap the selected weights for the class it produces.

    ``skip`` names fqns the stub selects but silently leaves dense, which is the shape of a
    torchao that declines a layer and the reason the post-pass walk exists."""
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
            # nn.Module.__setattr__ refuses a non-Parameter over a Parameter, and the real
            # torchao writes the subclass straight into _parameters.
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
    assignment = quantize_with_policy(tree, FLUX_MOD_SINGLE)
    assert [call["config"] for call in calls] == ["cfg:nvfp4", "cfg:fp8"]
    nvfp4, fp8 = set(calls[0]["selected"]), set(calls[1]["selected"])
    assert len(nvfp4) == 38 and len(fp8) == 461
    assert not (nvfp4 & fp8)
    assert nvfp4 == {fqn for fqn, p in assignment.items() if p == NVFP4}
    assert fp8 == {fqn for fqn, p in assignment.items() if p == FP8}
    # NVFP4 first is what lets pass 2 require a plain Parameter: the layers pass 1 took are no
    # longer holding one, so no fqn-set bug can quantise a layer twice.
    fp8_filter = calls[1]["filter_fn"]
    already = tree.linears["single_transformer_blocks.0.norm.linear"]
    assert not fp8_filter(already, "single_transformer_blocks.0.norm.linear")
    # Even for a layer the fp8 SET claims, an already-quantised weight is refused.
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

    # An NVFP4 pass that quietly produced fp8 weights is an artifact whose metadata lies about
    # the precision of 34 layers.
    _stub_quantize(monkeypatch, produced = {"cfg:nvfp4": "Float8Tensor", "cfg:fp8": "Float8Tensor"})
    with pytest.raises(PolicyMismatch, match = "wanted NVFP4Tensor"):
        quantize_with_policy(_zimage(), ZIMAGE_F8MOD_TOQ34)


# ── the metadata contract ────────────────────────────────────────────────────────


def test_the_metadata_block_records_the_set_that_was_built():
    assignment = assign_precisions(_zimage(), ZIMAGE_F8MOD_TOQ34)
    fragment = policy_metadata(ZIMAGE_F8MOD_TOQ34, assignment)
    block = fragment[NVFP4_POLICY_KEY]
    assert block["kind"] == NVFP4_POLICY_KIND
    assert block["policy_id"] == "zimg_f8mod_toq34_v1" and block["policy_version"] == 1
    assert block["counts"] == {BF16: 5, FP8: 237, NVFP4: 34}
    assert len(block["nvfp4_fqns"]) == 34
    assert block["nvfp4_fqns"] == sorted(block["nvfp4_fqns"])
    # Both flags are off until the passes that set them run (PR 2 commits 7 and later).
    assert block["activation_scales_baked"] is False and block["gptq"] is False
    # The fragment is what a builder merges into its metadata, so it declares a policy there.
    assert declares_policy({"scheme": "nvfp4", **fragment})
    assert policy_metadata_error({"scheme": "nvfp4", **fragment}) is None
    # Two builds of the same model produce the same block, so artifacts can be diffed.
    assert (
        policy_metadata(ZIMAGE_F8MOD_TOQ34, assign_precisions(_zimage(), ZIMAGE_F8MOD_TOQ34))
        == fragment
    )


def test_an_unreadable_policy_block_reads_as_declared_so_it_can_be_refused():
    # "Declared" is keyed on the KEY, so a block this build cannot parse is refused rather than
    # read as a whole-model artifact and loaded at the wrong precisions.
    assert not declares_policy({"scheme": "nvfp4"})
    assert not declares_policy(None)
    assert not declares_policy({NVFP4_POLICY_KEY: None})
    assert declares_policy({NVFP4_POLICY_KEY: {}})
    assert policy_metadata_error({"scheme": "nvfp4"}) is None


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({}, "unsupported nvfp4 policy"),
        ("zimg_f8mod_toq34_v1", "not a dict"),
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
