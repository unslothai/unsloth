# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma 4 E2B / E4B per-layer inputs and KV-shared layers under pipeline parallel: gloo vs one process, fp32."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _tiny_mm as tiny  # noqa: E402
import test_pp_vlm_text_only_gloo as vlm  # noqa: E402

pytest.importorskip("transformers.models.gemma4")

ppn = vlm.ppn
LAYERS = 6  # layer types s f s f | s f, the last two KV-shared: layer 4 reads layer 2, layer 5 reads layer 3
_NO_PROJECTION = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
    r"|per_layer_input_gate|per_layer_projection)"
)


def _shift_received(
    ctx,
    decoder,
    original,
    inputs_embeds,
    per_layer_inputs = None,
):
    """Fault: a later stage reads the projection columns of the next layer instead of its own."""
    out = ppn.__dict__["_true_ple_project"](ctx, decoder, original, inputs_embeds, None)
    if ctx.active and not ctx.is_first:
        out = out.roll(1, dims = 2)
    return (
        out
        if per_layer_inputs is None
        else (out + per_layer_inputs) * decoder.per_layer_input_scale
    )


def _zero_received(
    ctx,
    decoder,
    original,
    inputs_embeds,
    per_layer_inputs = None,
):
    """Fault: a later stage gets no projection (only its token table columns)."""
    out = ppn.__dict__["_true_ple_project"](ctx, decoder, original, inputs_embeds, None)
    if ctx.active and not ctx.is_first:
        out = out * 0
    return (
        out
        if per_layer_inputs is None
        else (out + per_layer_inputs) * decoder.per_layer_input_scale
    )


FAULTS = {"shift": _shift_received, "zero": _zero_received}


def _case_parity(
    rank,
    world,
    tmp,
    causal_lm = False,
    target_modules = "all-linear",
    schedule = "gpipe",
    layers_per_stage = None,
    fault = None,
    n_mb = 2,
    keep_replica = False,
):
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    if fault is not None:
        ppn._true_ple_project = ppn._ple_project
        ppn._ple_project = FAULTS[fault]
    build = dict(layers = LAYERS, target_modules = target_modules, causal_lm = causal_lm)
    ref = tiny.build("gemma4_e2b", **build)
    ids, labels = tiny.batch()
    ref_res = vlm.reference(ref, ids, labels)
    if rank == 0:
        ref.save_pretrained(os.path.join(tmp, "ref"))
    model = tiny.build("gemma4_e2b", **build)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    config = ppn.PPConfig(
        text_only = not causal_lm or None,
        layers_per_stage = layers_per_stage,
        keep_replica = keep_replica,
    )
    pp = ppn.apply_pipeline_parallel(
        model, mesh["pp"], n_microbatches = n_mb, schedule = schedule, config = config
    )
    decoder = pp.parts.decoder
    table = decoder.embed_tokens_per_layer
    pp_res = vlm.pipeline(model, pp, ids, labels)
    pp.save_adapter(os.path.join(tmp, "pp"))
    out = {
        "ref": ref_res,
        "pp": pp_res,
        "bounds": pp.bounds,
        "table": (type(table).__name__, tuple(table.weight.shape), table._pp_layers[0]),
        "projection_trains": [
            n for n, p in decoder.per_layer_model_projection.named_parameters() if p.requires_grad
        ],
    }
    if rank == 0:
        out["saved_ref"] = load_file(os.path.join(tmp, "ref", "adapter_model.safetensors"))
        out["saved_pp"] = load_file(os.path.join(tmp, "pp", "adapter_model.safetensors"))
    if keep_replica:
        vlm._load("pp_env").gather_replica(model, "generate")
        with torch.no_grad():
            out["replica_loss"] = float(model(input_ids = ids, labels = labels, use_cache = False).loss)
            out["ref_loss"] = float(ref(input_ids = ids, labels = labels, use_cache = False).loss)
        out["replica_table"] = type(decoder.embed_tokens_per_layer).__name__
    return out


vlm.CASES["ple_parity"] = _case_parity


def _worker(*args):
    vlm._worker(*args)


def spawn(
    case,
    world = 2,
    **kwargs,
):
    worker = vlm._worker
    vlm._worker = _worker
    try:
        return vlm.spawn(case, world, **kwargs)
    finally:
        vlm._worker = worker


def _check(
    results,
    world,
    projection = True,
):
    worst = vlm.check_parity(results)
    vlm.check_saved(results)
    bounds = results[0]["bounds"]
    for rank, res in enumerate(results):
        name, shape, owned = res["table"]
        start, end = bounds[rank]
        assert name == "_PPLayerTableSlice" and owned == tuple(range(start, end)), res["table"]
        assert shape == (tiny.VOCAB, (end - start) * 8), shape
        assert bool(res["projection_trains"]) == (projection and rank == 0), (
            rank,
            res["projection_trains"],
        )
    assert all(start <= 2 for start, _ in bounds[1:]), bounds
    return worst


@pytest.mark.parametrize("causal_lm", [False, True])
def test_pp2_per_layer_inputs_match_one_process(causal_lm):
    results = spawn("ple_parity", causal_lm = causal_lm)
    print("worst gradient error", _check(results, 2))


def test_pp2_per_layer_inputs_keep_replica_restores_the_whole_model():
    results = spawn("ple_parity", keep_replica = True)
    _check(results, 2)
    for res in results:
        assert res["replica_table"] == "Gemma4TextScaledWordEmbedding", res["replica_table"]
        assert abs(res["replica_loss"] - res["ref_loss"]) < 1e-5, (
            res["replica_loss"],
            res["ref_loss"],
        )


def test_pp2_per_layer_inputs_frozen_projection_match_one_process():
    results = spawn("ple_parity", target_modules = _NO_PROJECTION)
    _check(results, 2, projection = False)


def test_pp3_middle_stage_forwards_the_later_columns_1f1b():
    results = spawn("ple_parity", world = 3, schedule = "1f1b", n_mb = 4)
    _check(results, 3)
    assert results[0]["bounds"] == [(0, 1), (1, 2), (2, 6)]


@pytest.mark.parametrize("fault", ["shift", "zero"])
def test_parity_detects_a_wrong_per_layer_feed(fault):
    results = spawn("ple_parity", fault = fault)
    with pytest.raises(AssertionError):
        vlm.check_parity(results)
    ref_grads = results[0]["ref"][1][0]
    got = {n: g for res in results for n, g in res["pp"][1][0].items()}
    worst = vlm.grad_error(got, ref_grads)
    print(fault, "fault gradient error", worst)
    assert worst[0] > 0.05, worst


def test_kv_shared_cuts_keep_source_and_shared_layers_together():
    model = tiny.build("gemma4_e2b", layers = LAYERS, lora = False)
    parts = ppn._find_parts(model)
    assert ppn._kv_shared_reads(parts) == [(2, 4), (3, 5)]
    assert ppn._cut_points(parts) == [1, 2]
    assert ppn.derive_pp_split(model, 2)[1][0] <= 2
    assert ppn.derive_pp_split(model, 2, ppn.PPConfig(layers_per_stage = [2, 4])) == [(0, 2), (2, 6)]
    for sizes in ([3, 3], [4, 2]):
        with pytest.raises(ValueError, match = "not safe cut points"):
            ppn.derive_pp_split(model, 2, ppn.PPConfig(layers_per_stage = sizes))
    with pytest.raises(ValueError, match = "safe cut points"):
        ppn.derive_pp_split(model, 4)  # 3 cuts needed, 2 allowed
    assert parts.ple is not None and (parts.ple.layers, parts.ple.width) == (LAYERS, 8)
    assert not any(name.startswith("per_layer") for name, _ in parts.norms), parts.norms
    assert {name for _, name, _ in parts.towers} == {
        "vision_tower",
        "audio_tower",
        "embed_vision",
        "embed_audio",
    }


def test_trainable_per_layer_table_is_refused():
    model = tiny.build("gemma4_e2b", layers = LAYERS, causal_lm = True)
    model.get_base_model().model.embed_tokens_per_layer.weight.requires_grad_(True)
    with pytest.raises(ValueError, match = "per-layer embedding table"):
        ppn._refuse_unsupported(model, ppn._find_parts(model))


def test_table_slice_matches_the_full_lookup():
    model = tiny.build("gemma4_e2b", layers = LAYERS, lora = False, causal_lm = True)
    decoder = model.model
    ids, _ = tiny.batch()
    full = decoder.get_per_layer_inputs(ids, None)
    sliced = ppn._PPLayerTableSlice(decoder.embed_tokens_per_layer, [2, 3, 4, 5], LAYERS, 8)(ids)
    assert torch.equal(sliced[:, :, 2:], full[:, :, 2:])
    assert not sliced[:, :, :2].any()
