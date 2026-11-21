# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel over Gemma 4 E2B / E4B style KV-shared layers (num_kv_shared_layers): the last layers have no
key / value projections and read the keys / values of an earlier layer of their type. With one key / value head
(E2B) at tp 2 the source layer replicates it on every rank, so the shared layers shard only their query heads.
gloo CPU world vs one process, fp32."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402

pytest.importorskip("transformers.models.gemma4")

STEPS = 4
_TARGETS = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
    r"|per_layer_input_gate|per_layer_projection)"
)


def tiny_e2b(
    kv_heads = 1,
    layers = 6,
    shared = 2,
    seed = 0,
    lora = True,
    dtype = torch.float32,
):
    """Gemma4 with E2B's shape: per-layer inputs, last `shared` layers KV-shared, vision/audio towers."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    text = dict(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = kv_heads,
        head_dim = 8,
        global_head_dim = 16,
        layer_types = (["sliding_attention", "full_attention"] * layers)[:layers],
        sliding_window = 4,
        hidden_size_per_layer_input = 8,
        num_kv_shared_layers = shared,
        attention_k_eq_v = False,
        pad_token_id = 0,
        tie_word_embeddings = True,
    )
    vision = dict(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
        patch_size = 4,
    )
    audio = dict(hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 2)
    config = Gemma4Config(
        text_config = text,
        vision_config = vision,
        audio_config = audio,
        image_token_id = 120,
        audio_token_id = 121,
        video_token_id = 122,
        boi_token_id = 123,
        eoi_token_id = 124,
    )
    config._attn_implementation = "eager"
    config.get_text_config()._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Gemma4ForConditionalGeneration(config).to(dtype)
    with torch.no_grad():
        # Unit-variance activations: at the default std 0.02 the per-layer-input and value RMSNorms rescale
        # ~1e-3 vectors to unit size, so fp32 rounding alone moves the gradients by O(1) within a few steps.
        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in name:
                p.normal_(0, p.shape[1] ** -0.5)
            elif "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + 1)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = _TARGETS, init_lora_weights = False)
    return get_peft_model(model, config).train()


def _batch(seed = 1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 110, (2, 8), generator = g)
    labels = ids.clone()
    labels[0, :2] = -100
    return ids, labels


def _clip_fp64(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt().float()
    coef = torch.clamp(max_norm / (total + 1e-6), max = 1.0)
    for g in grads:
        g.mul_(coef)
    return total


def _train(model, steps, clip):
    ids, labels = _batch()
    params = [p for p in model.parameters() if p.requires_grad]
    # SGD: Adam's per-element normalisation turns ~1e-8 differences in near-zero gradients into full steps.
    opt = torch.optim.SGD(params, lr = 0.01)
    losses, grads = [], []
    for _ in range(steps):
        out = model(input_ids = ids, labels = labels, use_cache = False)
        out.loss.backward()
        grads.append(
            {
                n: tpn._full(p.grad, p).detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        )
        clip(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return losses, grads


def _drop_shared_regions(monkeypatch_target):
    """Fault: the KV-shared attentions are left out of the plan (whole query heads on every rank)."""
    derive = monkeypatch_target.derive_tp_plan

    def derive_without_shared(
        model,
        tp_size,
        plan = None,
    ):
        out = derive(model, tp_size, plan)
        out.regions = [
            r
            for r in out.regions
            if not (r.kind == "attention" and getattr(r.module, "is_kv_shared_layer", False))
        ]
        return out

    monkeypatch_target.derive_tp_plan = derive_without_shared


def _case_parity(
    rank,
    world,
    tmp,
    kv_heads,
    fault = None,
):
    from torch.distributed.device_mesh import init_device_mesh

    ref = tiny_e2b(kv_heads)
    model = tiny_e2b(kv_heads)
    if fault == "drop_shared_q":
        _drop_shared_regions(tpn)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    shared = [
        r
        for r in model._unsloth_tp_plan.regions
        if r.kind == "attention" and getattr(r.module, "is_kv_shared_layer", False)
    ]
    layout = [
        (
            r.path,
            r.kv_replicated,
            sorted((p.name, p.style) for p in r.projs),
            (r.module.num_heads, r.module.num_key_value_heads, r.module.num_key_value_groups),
        )
        for r in shared
    ]
    return {
        "ref": _train(ref, STEPS, _clip_fp64),
        "tp": _train(model, STEPS, tpn.tp_clip_grad_norm_),
        "layout": layout,
    }


def _worker(rank, world, port, case, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    finished = False
    try:
        res = {"parity": _case_parity}[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def _grad_error(got, want):
    """Largest |difference| over a tensor's largest |entry|: fp32 TP reorders the head / row reductions, which on
    this model alone moves gradients by up to ~2e-4 of their scale (one process, o_proj summed in two halves)."""
    assert set(got) == set(want) and want, sorted(set(got) ^ set(want))[:6]
    return max(
        (float((got[n] - want[n]).abs().max() / want[n].abs().max().clamp_min(1e-12)), n)
        for n in want
    )


def _check_parity(
    results,
    grad_tol = 1e-3,
    loss_tol = 1e-5,
):
    worst = (0.0, None)
    for res in results:
        (ref_losses, ref_grads), (losses, grads) = res["ref"], res["tp"]
        for got, want in zip(grads, ref_grads):
            worst = max(worst, _grad_error(got, want))
        assert worst[0] < grad_tol, worst
        assert losses == pytest.approx(ref_losses, rel = 0, abs = loss_tol), (losses, ref_losses)
        assert len(set(ref_losses)) == len(ref_losses)  # the steps train
    return worst


def test_kv_shared_layers_are_planned_as_replicated_kv():
    model = tiny_e2b(lora = False)
    plan = tpn.derive_tp_plan(model, 2)
    attn = {r.path.split(".")[-2]: r for r in plan.regions if r.kind == "attention"}
    assert len(attn) == 6
    for index in ("4", "5"):  # the KV-shared layers: no key / value projections of their own
        region = attn[index]
        assert region.kv_replicated and (region.heads, region.kv_heads) == (4, 1)
        assert {p.name: p.style for p in region.projs} == {
            "q_proj": tpn.COLWISE,
            "o_proj": tpn.ROWWISE,
        }
    for index in ("2", "3"):  # their sources (last sliding / full layer before the shared ones)
        assert attn[index].kv_replicated and attn[index].module.store_full_length_kv
        assert {p.name: p.style for p in attn[index].projs}["k_proj"] == tpn.KV_REPLICATED


def test_kv_shared_layer_without_a_matching_source_is_refused():
    model = tiny_e2b(lora = False)
    layers = model.model.language_model.layers
    layers[
        2
    ].self_attn.store_full_length_kv = (
        False  # the sliding source no longer stores its keys / values
    )
    with pytest.raises(ValueError, match = "KV-shared attention"):
        tpn.derive_tp_plan(model, 2)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_tp2_kv_shared_matches_one_process(kv_heads):
    results = _spawn("parity", kv_heads = kv_heads)
    print("worst gradient error", _check_parity(results))
    for rank, res in enumerate(results):
        assert len(res["layout"]) == 2
        for path, replicated, styles, heads in res["layout"]:
            assert replicated == (kv_heads < 2), path
            assert styles == [("o_proj", tpn.ROWWISE), ("q_proj", tpn.COLWISE)], styles
            assert heads == (2, 1, 2), (rank, path, heads)  # 4 / 2 query heads over 1 local KV head


def test_tp2_kv_shared_parity_detects_unsharded_query_heads():
    results = _spawn("parity", kv_heads = 1, fault = "drop_shared_q")
    assert all(not res["layout"] for res in results)
    with pytest.raises(AssertionError):
        _check_parity(results)
    worst = max(_grad_error(g, w) for res in results for g, w in zip(res["tp"][1], res["ref"][1]))
    print("fault gradient error", worst)
    assert worst[0] > 0.05, worst  # far outside the fp32 tolerance, not a near miss
