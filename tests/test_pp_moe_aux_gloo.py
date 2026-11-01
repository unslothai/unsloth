# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallelism over a mixture-of-experts model carries the router auxiliary (load-balancing) loss
between stages: a gloo CPU world of 2 stages vs gradient accumulation over the same micro-batches in one
process (transformers adds router_aux_loss_coef x aux per forward), fp32, tiny random Qwen3-MoE."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    """unsloth/models/<name>.py without importing the unsloth package (which needs a GPU)."""
    full = f"unsloth.models.{name}"
    if full in sys.modules:
        return sys.modules[full]
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


ppn = _load("pp_native")
assert os.path.realpath(ppn.__file__).startswith(os.path.realpath(REPO))  # this checkout's tree

VOCAB = 128
DEEPSEEK_V3 = "trl-internal-testing/tiny-DeepseekV3ForCausalLM"


def build(
    aux = True,
    coef = 0.01,
    lora = False,
    layers = 4,
    mlp_only_layers = (),
    family = "qwen3_moe",
    seed = 0,
):
    import transformers

    common = dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
        output_router_logits = aux,
        router_aux_loss_coef = coef,
        attention_dropout = 0.0,
    )
    if family == "qwen3_moe":
        config = transformers.Qwen3MoeConfig(
            head_dim = 16,
            moe_intermediate_size = 32,
            num_experts = 8,
            num_experts_per_tok = 2,
            norm_topk_prob = True,
            mlp_only_layers = list(mlp_only_layers),
            **common,
        )
    elif family == "mixtral":
        config = transformers.MixtralConfig(
            head_dim = 16, num_local_experts = 8, num_experts_per_tok = 2, **common
        )
    elif family == "olmoe":
        config = transformers.OlmoeConfig(num_experts = 8, num_experts_per_tok = 2, **common)
    else:
        raise ValueError(family)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
            if name.endswith("mlp.gate.weight"):
                p.normal_(0.0, 0.5)  # uneven routing: a load-balancing loss far from its minimum
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + 1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters = [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
            "mlp.gate.weight",
        ],
        init_lora_weights = False,
        lora_dropout = 0.0,
    )
    return get_peft_model(model, config).train()


def batch(
    bsz = 4,
    seq = 8,
    seed = 1,
    mask = True,
):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (bsz, seq), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100  # uneven token counts per micro-batch: num_items normalisation matters
    attention_mask = None
    if mask:
        attention_mask = torch.ones(bsz, seq, dtype = torch.long)
        attention_mask[1, -3:] = 0  # right padding: the auxiliary loss weights tokens by the mask
        attention_mask[-1, -1:] = 0
        labels[attention_mask == 0] = -100
    return ids, labels, attention_mask


def _chunks(t, n):
    return [None] * n if t is None else list(t.chunk(n))


def _reference(model, ids, labels, attention_mask, n_mb, steps):
    """Gradient accumulation over the pipeline's micro-batches in one process: each forward adds
    router_aux_loss_coef x its own load-balancing loss (transformers), the CE loss is normalised by the token
    count of the whole batch."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    n_items = int((labels[:, 1:] != -100).sum())
    losses, auxes, grads = [], [], None
    for step in range(steps):
        total, aux_total = 0.0, 0.0
        for i, l, m in zip(ids.chunk(n_mb), labels.chunk(n_mb), _chunks(attention_mask, n_mb)):
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


def _group(name):
    prefix = "lora_" if "lora_" in name else ""
    if "mlp.gate." in name:
        return prefix + "router"
    if "mlp.experts" in name:
        return prefix + "experts"
    if "self_attn" in name:
        return prefix + "attention"
    return prefix + "other"


def _case_parity(
    rank,
    world,
    tmp,
    n_mb = 2,
    schedule = "gpipe",
    mask = True,
    steps = 2,
    **build_kw,
):
    from torch.distributed.device_mesh import init_device_mesh

    ids, labels, attention_mask = batch(mask = mask)
    ref = _reference(build(**build_kw), ids, labels, attention_mask, n_mb, steps)
    out = {"ref": {k: ref[k] for k in ("losses", "aux")}}
    if build_kw.get("aux", True) and build_kw.get("coef", 0.01) >= 1.0:
        # The auxiliary loss is visible: dropping it changes the router / adapter gradients by far more than
        # the parity tolerance.
        dropped = _reference(
            build(**{**build_kw, "aux": False}), ids, labels, attention_mask, n_mb, 1
        )["grads"]
        out["aux_effect"] = max(
            float((ref["grads"][n] - g).abs().max())
            for n, g in dropped.items()
            if _group(n) in ("router", "lora_router", "lora_attention")
        )
    model = build(**build_kw)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = n_mb, schedule = schedule)
    params = pp.parameters()
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, auxes, diffs, missing = [], [], {}, []
    for step in range(steps):
        loss = pp.step(ids, labels, attention_mask = attention_mask)
        if step == 0:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                want = ref["grads"].get(name)
                if p.grad is None:
                    if want is not None and float(want.abs().max()) > 0:
                        missing.append(name)
                    continue
                d = float((p.grad - want).abs().max())
                scale = float(want.abs().max())
                group = _group(name)
                prev = diffs.get(group, (0.0, 0.0, 0))
                diffs[group] = (max(prev[0], d), max(prev[1], scale), prev[2] + 1)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
        auxes.append(float(pp.last_aux_loss) if pp.last_aux_loss is not None else None)
    out.update(losses = losses, aux = auxes, diffs = diffs, missing = missing, stage = (pp.start, pp.end))
    return out


def _case_refusals(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {}

    def catch(key, fn):
        try:
            fn()
            out[key] = None
        except (ValueError, RuntimeError) as exc:
            out[key] = str(exc)

    catch("mixtral", lambda: ppn.apply_pipeline_parallel(build(family = "mixtral"), mesh))
    catch("olmoe", lambda: ppn.apply_pipeline_parallel(build(family = "olmoe"), mesh))

    def deepseek():
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(DEEPSEEK_V3)
        config.num_hidden_layers = 2
        ppn.apply_pipeline_parallel(AutoModelForCausalLM.from_config(config), mesh)

    catch("deepseek_v3", deepseek)
    for schedule in ("interleaved1f1b", "loopedbfs", "zbvzerobubble"):
        catch(
            schedule,
            lambda: ppn.apply_pipeline_parallel(build(), mesh, n_microbatches = 4, schedule = schedule),
        )
    # A changed transformers formula refuses instead of training on another loss.
    module = sys.modules[type(build()).__module__]
    original = module.load_balancing_loss_func
    module.load_balancing_loss_func = lambda *a, **k: original(*a, **k) * 2
    try:
        catch("formula", lambda: ppn.apply_pipeline_parallel(build(), mesh))
    finally:
        module.load_balancing_loss_func = original
    # pipeline_sharded_loading keeps refusing mixture-of-experts models.
    model = build()
    catch(
        "sharded_loading",
        lambda: ppn._refuse_unsupported(model, ppn._find_parts(model), allow_moe = False),
    )
    # pp x dp over a mixture-of-experts stage (shard_model checks it before sharding).
    pp = ppn.apply_pipeline_parallel(build(), mesh)
    dp = init_device_mesh("cpu", (world, 1), mesh_dim_names = ("pp", "dp"))["dp"]
    catch("pp_dp", lambda: ppn.check_pp_dp_composition(pp.model, dp))
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
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


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
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


CELLS = {
    "full_coef0.01_gpipe_mask": dict(coef = 0.01),
    "full_coef1_1f1b_mask_dense_layer": dict(coef = 1.0, schedule = "1f1b", mlp_only_layers = (1,)),
    "full_coef1_gpipe_nomask_mb4": dict(coef = 1.0, mask = False, n_mb = 4),
    "lora_coef1_gpipe_mask": dict(coef = 1.0, lora = True),
    "lora_coef0.01_1f1b_nomask": dict(coef = 0.01, lora = True, schedule = "1f1b", mask = False),
    "full_aux_off_gpipe_mask": dict(aux = False, coef = 1.0),
    "lora_aux_off_1f1b_mask": dict(aux = False, lora = True, schedule = "1f1b"),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_pp2_moe_aux_matches_gradient_accumulation(cell):
    kwargs = CELLS[cell]
    results = _spawn("parity", 2, **kwargs)
    ref = results[0]["ref"]
    aux_on = kwargs.get("aux", True)
    for res in results:
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5, abs = 1e-6), (
            res["losses"],
            ref,
        )
        if aux_on:
            assert res["aux"] == pytest.approx(ref["aux"], rel = 1e-5, abs = 1e-6), (res["aux"], ref)
            assert min(ref["aux"]) > 0
        else:
            assert res["aux"] == [0.0, 0.0] and ref["aux"] == [0.0, 0.0]
        assert not res["missing"], res["missing"]
        for group, (diff, scale, count) in res["diffs"].items():
            assert diff <= 1e-6 + 1e-4 * scale, (cell, res["stage"], group, diff, scale)
    groups = set(results[0]["diffs"]) | set(results[1]["diffs"])
    want = {"router", "experts", "attention"}
    want = {"lora_" + g for g in want} if kwargs.get("lora") else want | {"other"}
    assert want <= groups, groups
    if "aux_effect" in results[0]:
        assert results[0]["aux_effect"] > 1e-3, results[0]["aux_effect"]
    print(cell, {r["stage"]: r["diffs"] for r in results}, results[0].get("aux_effect"))


def test_pp2_moe_refusals_for_unproven_cases():
    out = _spawn("refusals", 2)[0]
    for key in ("mixtral", "olmoe", "deepseek_v3", "formula"):
        assert "mixture-of-experts" in (out[key] or ""), (key, out[key])
        assert "Qwen3MoeForCausalLM" in out[key], (key, out[key])
    for key in ("interleaved1f1b", "loopedbfs", "zbvzerobubble"):
        assert "'gpipe' / '1f1b'" in (out[key] or ""), (key, out[key])
    assert "pipeline_sharded_loading" in (out["sharded_loading"] or ""), out["sharded_loading"]
    assert "pp x dp" in (out["pp_dp"] or ""), out["pp_dp"]


def test_router_statistics_sum_over_stages_to_transformers_loss():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func

    g = torch.Generator().manual_seed(3)
    logits = tuple(torch.randn(10, 8, generator = g, requires_grad = True) for _ in range(4))
    mask = torch.ones(2, 5, dtype = torch.long)
    mask[1, 2:] = 0
    for m in (None, mask):
        want = load_balancing_loss_func(logits, 8, 2, m)
        (want_grads) = torch.autograd.grad(want, logits)
        stats = sum(ppn._router_statistics(logits[i : i + 1], 8, 2, m) for i in range(4))
        got = ppn._load_balancing_from_statistics(stats, 8)
        got_grads = torch.autograd.grad(got, logits)
        assert torch.allclose(want, got, rtol = 1e-6, atol = 1e-7)
        for a, b in zip(want_grads, got_grads):
            assert torch.allclose(a, b, rtol = 1e-5, atol = 1e-8)


@pytest.mark.parametrize(
    "aux, reentrant, refused",
    [(True, True, True), (True, False, False), (False, True, False), (True, None, False)],
)
def test_aux_loss_refuses_reentrant_checkpointing(aux, reentrant, refused):
    # Failing before: reentrant checkpointing captured router logits under no_grad, so the auxiliary loss added
    # its value to the pipeline loss but no gradient.
    model = build(aux = aux)
    if reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": reentrant}
        )
    if refused:
        with pytest.raises(ValueError, match = "non-reentrant"):
            ppn._refuse_aux_with_reentrant_checkpointing(model, model.config)
    else:
        ppn._refuse_aux_with_reentrant_checkpointing(model, model.config)
