# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The registry follows the measured composition evidence. CPU only."""

import dataclasses
import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import parallel_capabilities as caps
from unsloth.parallel_env import ParallelLayout

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_PARALLEL_STRICT",
    "WORLD_SIZE",
)


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    yield monkeypatch


def _tiny():
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(TINY)
    except Exception as e:  # offline host without the tiny repo cached
        pytest.skip(f"{TINY}: {e}")
    config.num_hidden_layers = 2
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config)


def _packed(**kw):
    base = dict(
        trainer = "sft",
        family = "qwen3",
        packing = True,
        cp = 2,
        dp = 1,
        cp_backend = "ulysses",
        attention = "sdpa",
        compile = "eager",
        mode = "lora",
        torch = (2, 13),
    )
    base.update(kw)
    return caps.Query(**base)


# Hybrid (Gated DeltaNet + attention) cp x native FSDP2.


def _hybrid(**kw):
    base = dict(
        trainer = "sft",
        family = "qwen3_5",
        hybrid = True,
        cp = 2,
        dp = 2,
        fsdp2 = True,
        cp_backend = "ulysses",
        attention = "sdpa",
        compile = "eager",
        mode = "lora",
        torch = (2, 13),
    )
    base.update(kw)
    return caps.Query(**base)


def test_hybrid_qwen3_5_cp_x_native_fsdp2_is_supported_under_strict(clean_env):
    status, cell = caps.cell_status(_hybrid())
    assert status == "supported", cell
    assert "hybrid" in cell.name and not cell.missing()
    assert "test_hybrid_qwen3_5_cp_with_native_fsdp2_matches_one_process" in cell.parity
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    caps.check(_hybrid())


@pytest.mark.parametrize(
    "kw",
    [
        {"family": "qwen3_next"},
        {"attention": "eager"},
        {"dp": 4},
        {"mode": "full"},
        {"memory": "gc"},
        {"compile": "block"},
    ],
)
def test_hybrid_cp_x_native_fsdp2_outside_the_measured_cell_stays_unverified(kw, clean_env):
    assert caps.cell_status(_hybrid(**kw))[0] == "unverified"
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    with pytest.raises(NotImplementedError, match = "no numeric parity"):
        caps.check(_hybrid(**kw))


# Packed / padding-free rows x cp Ulysses.


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"family": "gemma3", "windowed": True, "attention": "eager"},
        {"family": "mistral", "windowed": True},
        {"family": "gemma2", "windowed": True, "attention": "eager"},
        {"mode": "full"},
    ],
)
def test_packed_rows_under_generic_ulysses_eager_are_supported(kw):
    status, cell = caps.cell_status(_packed(**kw))
    assert status == "supported", cell
    assert "packed rows x cp Ulysses" in cell.name and not cell.missing()
    assert "test_generic_ulysses_matches_one_process" in cell.parity
    caps.check(_packed(**kw))


def test_packed_rows_through_query_from_reach_the_eager_cell():
    model = _tiny()
    layout = types.SimpleNamespace(dp = 1, tp = 1, pp = 1, cp = 2, fsdp2 = False, cp_backend = "ulysses")
    args = types.SimpleNamespace(packing = False, padding_free = True, gradient_checkpointing = False)
    trl = pytest.importorskip("trl")
    query = caps.query_from(
        layout, trl.SFTTrainer, model, args, cp_backend = "ulysses", peft_config = object()
    )
    assert query.packing and query.family == "qwen3" and query.memory == "none"
    status, cell = caps.cell_status(query)
    assert status == "supported" and "packed rows" in cell.name, (status, cell)


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_packed_rows_under_block_compile_are_supported_on_torch_2_13(family, clean_env):
    query = _packed(compile = "block", family = family)
    status, cell = caps.cell_status(query)
    assert (
        status == "supported" and cell.name.endswith("Unsloth block compile") and not cell.missing()
    )
    for kind in (cell.parity, cell.engagement, cell.compile):
        assert "test_block_compile_packed_rows_under_cp_match_one_process" in kind
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    caps.check(query)


@pytest.mark.parametrize(
    "kw",
    [
        {"torch": (2, 12)},  # the compile test passes on torch >= 2.13 only
        {"family": "gemma3", "windowed": True, "attention": "eager"},
        {"family": "llama", "torch": (2, 12)},
        {"mode": "full"},
        {"attention": "eager"},
    ],
)
def test_packed_rows_under_block_compile_outside_the_measured_cell_stay_unverified(kw, clean_env):
    query = _packed(compile = "block", **kw)
    status, detail = caps.cell_status(query)
    assert status == "unverified", detail
    assert "lacks engagement, compile evidence" in detail and "torch >= 2.13 only" in detail, detail
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    with pytest.raises(NotImplementedError):
        caps.check(query)


@pytest.mark.parametrize(
    "kw",
    [
        {"dp": 2},  # dp 2 x cp 2 was not measured
        {"fsdp2": True},
        {"family": "llama"},
        {"kernels": True, "loss": "fused_ce"},  # Unsloth's fast attention is not the generic path
        {"memory": "unsloth_offload"},
    ],
)
def test_packed_rows_outside_the_measured_cell_are_not_supported(kw):
    assert caps.cell_status(_packed(**kw))[0] != "supported"


def test_kept_refusals_flex_and_ring_with_packing_or_window():
    with pytest.raises(NotImplementedError, match = "flex attention block masks"):
        caps.check(_packed(attention = "flex"))
    with pytest.raises(NotImplementedError, match = "ring context parallel with packed rows"):
        caps.check(_packed(cp_backend = "ring", sm = 9))
    with pytest.raises(NotImplementedError, match = "sliding window"):
        caps.check(_packed(cp_backend = "ring", packing = False, windowed = True, sm = 9))


# Memory: Unsloth's checkpoint and tiling.


@pytest.mark.parametrize(
    "fn_name, offloads, want",
    [
        ("unsloth_checkpoint", None, "unsloth_offload"),  # path A (UnslothCheckpointFunction)
        ("_nonre_checkpoint", True, "unsloth_offload"),  # path B with offload hooks
        # path B without offload: a plain non-reentrant checkpoint
        ("_nonre_checkpoint", False, "gc"),
        ("unsloth_offloaded_gradient_checkpoint", None, "unsloth_offload_legacy"),  # not measured
        ("unsloth_gradient_checkpoint", None, "unsloth_gc"),
        ("checkpoint", None, "gc"),
    ],
)
def test_memory_reports_unsloth_checkpoints(monkeypatch, fn_name, offloads, want):
    import torch.utils.checkpoint as checkpoint

    def fn(function, *args, **kwargs):
        return function(*args, **kwargs)

    fn.__name__ = fn_name
    if offloads is not None:
        fn._unsloth_offloads = offloads
    monkeypatch.setattr(checkpoint, "checkpoint", fn)
    args = types.SimpleNamespace(gradient_checkpointing = True)
    assert caps._memory(None, args) == want
    assert caps._memory(None, types.SimpleNamespace(gradient_checkpointing = False)) == "none"


def test_memory_reports_the_zoo_unsloth_checkpoint(monkeypatch):
    zoo = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    import torch.utils.checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "checkpoint", zoo.unsloth_checkpoint)
    assert (
        caps._memory(None, types.SimpleNamespace(gradient_checkpointing = True)) == "unsloth_offload"
    )


def test_unsloth_checkpoint_is_seen_and_covered_by_the_cells_that_measured_it(monkeypatch):
    zoo = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    import torch.utils.checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "checkpoint", zoo.unsloth_checkpoint)
    memory = caps._memory(None, types.SimpleNamespace(gradient_checkpointing = True))
    assert memory == "unsloth_offload"
    for kw, name in (
        (dict(family = "llama", tp = 2), "cp x tp (DDP dp), eager"),
        (dict(family = "llama", tp = 2, dp = 2, fsdp2 = True), "cp x tp (FSDP2 dp), eager"),
        (dict(family = "llama", tp = 2, compile = "block"), "cp x tp, Unsloth block compile"),
        (dict(family = "llama", fsdp2 = True), "cp x native FSDP2 (dp_shard x cp), eager"),
        (
            dict(family = "qwen3_moe", moe = True, expert_lora = True),
            "MoE qwen3_moe x lora+experts x aux off x cp",
        ),
        (
            dict(family = "qwen3_moe", moe = True, mode = "full", compile = "block"),
            "MoE qwen3_moe x full x aux off x cp, block compile",
        ),
    ):
        query = caps.Query(
            trainer = "sft", cp = 2, cp_backend = "ulysses", memory = memory, torch = (2, 13), **kw
        )
        status, cell = caps.cell_status(query)
        assert status == "supported" and cell.name == name, (kw, status, cell)
        assert "unsloth_checkpointing" in cell.note, cell.note
        caps.check(query)


@pytest.mark.parametrize(
    "mode, expert_lora",
    [("full", False), ("lora", False), ("lora", True)],
    ids = ["full", "lora", "lora+experts"],
)
def test_kimi_linear_moe_under_cp_follows_its_gloo_evidence(mode, expert_lora):
    # tests/test_ulysses_kimi.py::test_kimi_moe_under_ulysses_matches_one_process: sdpa / eager, none / HF
    # checkpointing, cp 2 and dp 2 x cp 2. Unsloth's checkpoint, FSDP2 x cp and the router loss stay refused.
    kw = dict(
        trainer = "sft",
        family = "kimi_linear",
        moe = True,
        mode = mode,
        expert_lora = expert_lora,
        cp = 2,
        cp_backend = "ulysses",
        torch = (2, 13),
    )
    for extra in ({}, {"memory": "gc"}, {"attention": "eager"}, {"dp": 2}):
        query = caps.Query(**kw, **extra)
        status, cell = caps.cell_status(query)
        assert status == "supported" and "test_kimi_moe_under_ulysses" in cell.parity, (extra, cell)
        caps.check(query)
    for extra, match in (
        ({"memory": "unsloth_offload"}, "Unsloth checkpoint"),
        ({"fsdp2": True}, "fsdp2_cp"),
        ({"router_aux_loss": True}, "router auxiliary loss"),
        ({"compile": "block"}, "no numeric parity"),
    ):
        with pytest.raises(NotImplementedError, match = match):
            caps.check(caps.Query(**kw, **extra))


@pytest.mark.parametrize("family", ["minimax", "minimax_m2"])
@pytest.mark.parametrize(
    "mode, expert_lora",
    [("full", False), ("lora", False), ("lora", True)],
    ids = ["full", "lora", "lora+experts"],
)
def test_minimax_moe_under_cp_follows_its_gloo_evidence(family, mode, expert_lora):
    # tests/test_ulysses_minimax_parity.py::test_minimax_model_matches_upstream: sdpa / eager, none / HF
    # checkpointing, cp 2 and dp 2 x cp 2. Unsloth's checkpoint, FSDP2 x cp and the router loss stay refused.
    kw = dict(
        trainer = "sft",
        family = family,
        moe = True,
        mode = mode,
        expert_lora = expert_lora,
        cp = 2,
        cp_backend = "ulysses",
        torch = (2, 13),
    )
    for extra in ({}, {"memory": "gc"}, {"attention": "eager"}, {"dp": 2}):
        query = caps.Query(**kw, **extra)
        status, cell = caps.cell_status(query)
        assert status == "supported" and "test_minimax_model_matches_upstream" in cell.parity, (
            extra,
            cell,
        )
        caps.check(query)
    for extra, match in (
        ({"memory": "unsloth_offload"}, "Unsloth checkpoint"),
        ({"fsdp2": True}, "fsdp2_cp"),
        ({"router_aux_loss": True}, "router auxiliary loss"),
        ({"compile": "block"}, "no numeric parity"),
        ({"mode": "qlora"}, "4-bit experts"),
    ):
        with pytest.raises(NotImplementedError, match = match):
            caps.check(caps.Query(**{**kw, **extra}))


def test_minimax_evidence_ids_are_real_parity_cases():
    # Every case a MiniMax cell names exists in the parity test, so a renamed case cannot leave a cell
    # pointing at nothing.
    import re

    from tests import test_ulysses_minimax_parity as parity
    for cell in caps.CELLS:
        if cell.parity and "test_minimax_model_matches_upstream[" in cell.parity:
            ids = re.search(r"\[(.*)\]", cell.parity).group(1).split("|")
            assert set(ids) <= set(parity.MODEL_CASES), (cell.name, ids)


def test_unsloth_checkpoint_keeps_an_unverified_moe_row_refused_under_strict(
    clean_env, monkeypatch
):
    zoo = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    import torch.utils.checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "checkpoint", zoo.unsloth_checkpoint)
    memory = caps._memory(None, types.SimpleNamespace(gradient_checkpointing = True))
    query = caps.Query(trainer = "sft", family = "qwen3_moe", moe = True, dp = 2, memory = memory)
    assert caps.cell_status(query)[0] == "unverified"
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    with pytest.raises(NotImplementedError, match = "MoE qwen3_moe x lora x aux off x ddp"):
        caps.check(query)


def test_tiling_is_its_own_flag():
    pytest.importorskip("unsloth_zoo.tiling_registry")
    from unsloth.tiling import tile_mlp

    model = _tiny()
    args = types.SimpleNamespace(gradient_checkpointing = True)
    assert not caps.query_from(None, None, model, args).tiled
    tile_mlp(model)
    query = caps.query_from(None, None, model, args)
    assert query.tiled and query.memory != "tiled_mlp"
    assert {"tiled_mlp", "gradient_checkpointing"} <= set(caps.check_features(query))


def test_every_feature_module_path_exists():
    import os
    import re

    root = os.path.dirname(caps.__file__)
    for feature in caps.FEATURES:
        local = feature.module.split("unsloth_zoo")[0]
        for path in re.findall(r"[\w/]+\.py", local):
            assert os.path.exists(os.path.join(root, path)), (feature.name, path)


# Block swap.


def _swap(model, variant):
    layers = model.model.layers
    if variant == "count":
        swapper = types.SimpleNamespace(indices = [1])
        layers._unsloth_block_swap = swapper
        model._unsloth_block_swap = swapper
    elif variant == "empty":  # a swapper that swapped nothing (not auto): nothing to refuse
        layers._unsloth_block_swap = types.SimpleNamespace(indices = [])
    elif variant == "auto":
        model._unsloth_offload_layers_auto = "auto"
    elif variant == "auto_ddp":
        model._unsloth_offload_layers_auto_ddp = True
    return model


@pytest.mark.parametrize("variant", ["none", "count", "empty", "auto", "auto_ddp"])
def test_block_swap_flag_agrees_with_the_fsdp2_shard_refusal(variant):
    from unsloth.models import fsdp2_native

    model = _swap(_tiny(), variant)
    try:
        fsdp2_native._refuse_block_swap(model)
        refused = False
    except ValueError:
        refused = True
    assert caps._block_swap(model) == refused == (variant in ("count", "auto", "auto_ddp"))


@pytest.mark.parametrize("variant", ["count", "auto"])
def test_block_swap_under_fsdp2_is_refused_by_check_layout_before_weights_move(variant):
    model = _swap(_tiny(), variant)
    before = {n: (p.data_ptr(), p.detach().clone()) for n, p in model.named_parameters()}
    layout = ParallelLayout(tp = 2, dp = 2, world = 4, fsdp2 = True)
    with pytest.raises(NotImplementedError, match = "block swap"):
        caps.check_layout(layout, None, model, None)
    for name, param in model.named_parameters():
        assert param.data_ptr() == before[name][0] and torch.equal(param.detach(), before[name][1])
    assert "block_swap" in caps.markdown_table("features")
    assert "block swap (offload_layers) under native FSDP2" in caps.markdown_table("rules")


def test_block_swap_without_fsdp2_passes_the_rule():
    query = caps.Query(dp = 2, block_swap = True)
    assert "block_swap" in caps.check_features(query)
    caps.check(query)


@pytest.mark.parametrize("size", [2, 3, 16])
@pytest.mark.parametrize("axis", ["tp", "pp"])
def test_block_swap_under_tp_or_pp_is_unverified(clean_env, axis, size):
    query = caps.Query(trainer = "sft", block_swap = True, **{axis: size})
    status, detail = caps.cell_status(query)
    assert status == "unverified" and f"block swap x {axis}" in detail
    caps.check(query)  # a single axis: runs outside strict mode
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    with pytest.raises(NotImplementedError, match = "block swap check"):
        caps.check(query)


@pytest.mark.parametrize(
    "layout",
    [
        dict(family = "llama", cp = 2, tp = 2, cp_backend = "ulysses"),
        dict(family = "qwen3_moe", moe = True, expert_lora = True, pp = 2, pp_schedule = "gpipe"),
    ],
)
def test_block_swap_refusal_on_a_gated_layout_names_offload_layers(clean_env, layout):
    query = caps.Query(trainer = "sft", torch = (2, 13), **layout)
    assert caps.cell_status(query)[0] == "supported"
    query = dataclasses.replace(query, block_swap = True)
    with pytest.raises(NotImplementedError) as e:
        caps.check(query)
    assert "block swap x" in str(e.value) and "offload_layers = 0" in str(e.value)


# MoE qwen3_moe x qlora x pp: measured with one micro-batch only, which the registry cannot see.


def test_qwen3_moe_qlora_pp_keeps_its_refusal_with_the_micro_batch_evidence():
    query = caps.Query(
        trainer = "sft",
        family = "qwen3_moe",
        moe = True,
        router_aux_loss = True,
        mode = "qlora",
        pp = 2,
        pp_schedule = "gpipe",
        torch = (2, 13),
    )
    with pytest.raises(NotImplementedError, match = "UNSLOTH_PP_MICROBATCHES=1") as e:
        caps.check(query)
    assert "qwen3_moe x qlora x aux on x pp" in str(e.value)
