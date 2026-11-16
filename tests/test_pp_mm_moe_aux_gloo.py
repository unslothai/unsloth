# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallelism over the mixture-of-experts ConditionalGeneration wrappers (Qwen3.5-MoE, Qwen4Exp),
text only with the towers out: the router auxiliary loss is carried between stages with num_experts / top-k /
coef from text_config, as the wrapper's forward reads them. gloo CPU world of 2 vs gradient accumulation over
the same micro-batches in one process, fp32."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import _load  # noqa: E402

pytest.importorskip("transformers.models.qwen3_5_moe")
pytest.importorskip(
    "transformers.models.qwen4_exp", reason = "transformers without qwen4_exp (5.22+)"
)

ppn = _load("pp_native")
VOCAB = 512
MEDIA = dict(
    image_token_id = 508, video_token_id = 509, vision_start_token_id = 510, vision_end_token_id = 511
)
VISION = dict(
    depth = 1,
    hidden_size = 16,
    intermediate_size = 32,
    num_heads = 2,
    out_hidden_size = 64,
    patch_size = 4,
    temporal_patch_size = 2,
    spatial_merge_size = 2,
)
TEXT = dict(
    vocab_size = VOCAB,
    hidden_size = 64,
    num_hidden_layers = 4,
    layer_types = ["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 16,
    linear_conv_kernel_dim = 4,
    linear_key_head_dim = 8,
    linear_num_key_heads = 2,
    linear_num_value_heads = 4,
    linear_value_head_dim = 8,
    num_experts = 8,
    num_experts_per_tok = 2,
    moe_intermediate_size = 32,
    shared_expert_intermediate_size = 32,
    tie_word_embeddings = False,
    pad_token_id = 0,
)
# unsloth/Qwen3.8-Flash-Next's text layout, shrunk (tests/test_qwen4exp_parallel_gloo.py).
QWEN4EXP = dict(
    bos_token_id = 1,
    eos_token_id = 2,
    full_attention_interval = 4,
    hc_count = 4,
    hc_lowrank = 8,
    indexer_budget = 64,
    indexer_compress_ratio = 4,
    indexer_head_dim = 16,
    indexer_kv_heads = 1,
    indexer_n_heads = 2,
    ple_layer_ids = [2],
    ple_embed_dim = 64,
    heads_per_ngram = 8,
    ngram_size = 3,
    ngram_vocab_size_base = 509,
    split_ngram_parts = 4,
    make_ngram_vocab_size_divisible_by = 8,
    ple_conv_kernel_size = 4,
    mtp_num_hidden_layers = 0,
    max_position_embeddings = 512,
)
SEQ = {
    "qwen3_5_moe": 8,
    "qwen4_exp": 80,
}  # Qwen4Exp: > indexer_budget, the QSA indexer selects blocks


def build(
    family,
    aux = True,
    coef = 0.01,
    lora = False,
    seed = 0,
):
    import transformers

    text = dict(TEXT, output_router_logits = aux, router_aux_loss_coef = coef)
    if family == "qwen3_5_moe":
        cls, config_cls = (
            transformers.Qwen3_5MoeForConditionalGeneration,
            transformers.Qwen3_5MoeConfig,
        )
    else:
        cls, config_cls = transformers.Qwen4ExpForConditionalGeneration, transformers.Qwen4ExpConfig
        text.update(QWEN4EXP)
    config = config_cls(text_config = text, vision_config = VISION, **MEDIA)
    config._attn_implementation = "eager"
    config.get_text_config()._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = cls(config).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
            elif name.endswith("mlp.gate.weight"):
                p.normal_(
                    0.0, 0.5
                )  # routers start at zero: a load-balancing loss far from its minimum
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + 1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = r"^(?!.*\.(indexer|visual)\.).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|out_proj)$",
        target_parameters = ["mlp.gate.weight"],
        init_lora_weights = False,
        lora_dropout = 0.0,
    )
    return get_peft_model(model, config).train()


def batch(family, bsz = 4):
    seq = SEQ[family]
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 500, (bsz, seq), generator = g)  # text tokens only
    labels = ids.clone()
    labels[0, :3] = -100
    mask = torch.ones(bsz, seq, dtype = torch.long)
    mask[1, -3:] = 0  # right padding: the auxiliary loss weights tokens by the mask
    mask[-1, -1:] = 0
    labels[mask == 0] = -100
    return ids, labels, mask


def _reference(model, ids, labels, mask, n_mb, steps):
    """The wrapper's forward per micro-batch (lm loss + text_config coef x its load-balancing loss)."""
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr = 1e-2, foreach = False
    )
    n_items = int((labels[:, 1:] != -100).sum())
    losses, auxes, grads = [], [], None
    for step in range(steps):
        total, aux_total = 0.0, 0.0
        for i, l, m in zip(ids.chunk(n_mb), labels.chunk(n_mb), mask.chunk(n_mb)):
            out = model(
                input_ids = i, labels = l, attention_mask = m, num_items_in_batch = n_items, use_cache = False
            )
            out.loss.backward()
            total += float(out.loss)
            aux_total += float(out.aux_loss) if out.aux_loss is not None else 0.0
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(total)
        auxes.append(aux_total)
    return {"losses": losses, "aux": auxes, "grads": grads}


def _case_parity(
    rank,
    world,
    tmp,
    family,
    n_mb = 2,
    schedule = "gpipe",
    steps = 2,
    **build_kw,
):
    from torch.distributed.device_mesh import init_device_mesh

    ids, labels, mask = batch(family)
    ref = _reference(build(family, **build_kw), ids, labels, mask, n_mb, steps)
    out = {"ref": {k: ref[k] for k in ("losses", "aux")}}
    if build_kw.get("aux", True):
        # The router gradients differ by far more than the tolerance without the auxiliary term.
        dropped = _reference(
            build(family, **{**build_kw, "aux": False}), ids, labels, mask, n_mb, 1
        )["grads"]
        out["aux_effect"] = max(
            float((ref["grads"][n] - g).abs().max()) for n, g in dropped.items() if "mlp.gate" in n
        )
    model = build(family, **build_kw)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    pp = ppn.apply_pipeline_parallel(
        model, mesh, n_microbatches = n_mb, schedule = schedule, config = ppn.PPConfig(text_only = True)
    )
    opt = torch.optim.AdamW(pp.parameters(), lr = 1e-2, foreach = False)
    losses, auxes, worst, owned, missing = [], [], 0.0, [], []
    for step in range(steps):
        loss = pp.step(ids, labels, attention_mask = mask)
        if step == 0:
            for name, p in model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                want = ref["grads"].get(name)
                if want is None:
                    missing.append(name)
                    continue
                owned.append(name)
                err = float((p.grad - want).abs().max() / want.abs().max().clamp_min(1e-12))
                worst = max(worst, err)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
        auxes.append(float(pp.last_aux_loss) if pp.last_aux_loss is not None else None)
    out.update(losses = losses, aux = auxes, worst = worst, owned = owned, missing = missing)
    return out


def _case_refusals(rank, world, tmp, family):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    text_only = ppn.PPConfig(text_only = True)
    out = {}

    def catch(key, fn):
        try:
            fn()
            out[key] = None
        except (ValueError, RuntimeError) as exc:
            out[key] = str(exc)

    catch("no_text_only", lambda: ppn.apply_pipeline_parallel(build(family), mesh))
    catch(
        "interleaved",
        lambda: ppn.apply_pipeline_parallel(
            build(family), mesh, n_microbatches = 4, schedule = "interleaved1f1b", config = text_only
        ),
    )
    module = sys.modules[type(build(family)).__module__]
    original = module.load_balancing_loss_func
    module.load_balancing_loss_func = lambda *a, **k: original(*a, **k) * 2
    try:
        catch("formula", lambda: ppn.apply_pipeline_parallel(build(family), mesh, config = text_only))
    finally:
        module.load_balancing_loss_func = original
    return out


CASES = {"parity": _case_parity, "refusals": _case_refusals}


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

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    finished = False
    try:
        torch.save(CASES[case](rank, world, tmp, **kwargs), os.path.join(tmp, f"result_{rank}.pt"))
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
    try:
        ppn._check_torch_pipelining()
    except RuntimeError as exc:
        pytest.skip(str(exc))
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


CELLS = {
    "qwen3_5_moe_full_coef1_gpipe": dict(family = "qwen3_5_moe", coef = 1.0),
    "qwen3_5_moe_lora_coef0.01_1f1b": dict(family = "qwen3_5_moe", lora = True, schedule = "1f1b"),
    "qwen3_5_moe_full_aux_off_1f1b": dict(family = "qwen3_5_moe", aux = False, schedule = "1f1b"),
    "qwen4_exp_full_coef1_gpipe": dict(family = "qwen4_exp", coef = 1.0),
    "qwen4_exp_lora_coef0.01_1f1b": dict(family = "qwen4_exp", lora = True, schedule = "1f1b"),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_pp2_wrapper_moe_aux_matches_gradient_accumulation(cell):
    # Failing before: "Qwen3_5MoeForConditionalGeneration is not proven" / Qwen4Exp likewise.
    kwargs = CELLS[cell]
    results = _spawn("parity", **kwargs)
    ref = results[0]["ref"]
    aux_on = kwargs.get("aux", True)
    for res in results:
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5, abs = 1e-6), (
            res["losses"],
            ref,
        )
        if aux_on:
            assert min(ref["aux"]) > 0
            assert res["aux"] == pytest.approx(ref["aux"], rel = 1e-5, abs = 1e-6), (res["aux"], ref)
        else:
            assert res["aux"] == [0.0, 0.0] and ref["aux"] == [0.0, 0.0]
        assert not res["missing"], res["missing"][:6]
        assert res["worst"] < 1e-4, (cell, res["worst"])
    owned = results[0]["owned"] + results[1]["owned"]
    assert len(owned) == len(set(owned))
    assert any("mlp.gate" in n for n in results[0]["owned"]) and any(
        "mlp.gate" in n for n in results[1]["owned"]
    )
    if "aux_effect" in results[0] and kwargs.get("coef", 0.01) >= 1.0:
        assert results[0]["aux_effect"] > 1e-3, results[0]["aux_effect"]


@pytest.mark.parametrize("family", ["qwen3_5_moe", "qwen4_exp"])
def test_pp2_wrapper_moe_refusals_kept(family):
    out = _spawn("refusals", family = family)[0]
    assert "text_only = True" in (out["no_text_only"] or ""), out["no_text_only"]
    assert "'gpipe' / '1f1b'" in (out["interleaved"] or ""), out["interleaved"]
    assert "is not proven" in (out["formula"] or ""), out["formula"]


@pytest.mark.parametrize("family", ["qwen3_5_moe", "qwen4_exp"])
def test_wrapper_aux_plan_reads_text_config(family):
    model = build(family, coef = 0.25)
    model.config.get_text_config().num_experts_per_tok = 3
    moe = ppn._moe_aux_plan(ppn._find_parts(model))
    assert moe is not None
    assert (moe.num_experts, moe.top_k, ppn._router_aux_coef(moe)) == (8, 3, 0.25)
    assert moe.config is model.config.get_text_config() and moe.config.output_router_logits
