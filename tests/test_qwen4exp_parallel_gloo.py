# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen4Exp under TP / PP / FSDP2 on a gloo CPU world, LoRA on every linear, fp32, vs one process."""

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

pytest.importorskip(
    "transformers.models.qwen4_exp", reason = "transformers without qwen4_exp (5.22+)"
)


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


tpn = _load("tp_native")
ppn = _load("pp_native")

VOCAB = 512
SEQ = 80  # > indexer_budget + indexer_compress_ratio: the QSA indexer selects blocks
# unsloth/Qwen3.8-Flash-Next's text config, shrunk (layer pattern, streams, PLE and indexer layout kept).
CONFIG = dict(
    vocab_size = VOCAB,
    bos_token_id = VOCAB - 1,
    eos_token_id = VOCAB - 1,
    hidden_size = 64,
    num_hidden_layers = 4,
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    full_attention_interval = 4,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 16,
    hc_count = 4,
    hc_lowrank = 8,
    indexer_budget = 64,
    indexer_compress_ratio = 4,
    indexer_head_dim = 16,
    indexer_kv_heads = 1,
    indexer_n_heads = 2,
    linear_conv_kernel_dim = 4,
    linear_key_head_dim = 16,
    linear_num_key_heads = 2,
    linear_num_value_heads = 4,
    linear_value_head_dim = 16,
    num_experts = 4,
    num_experts_per_tok = 2,
    moe_intermediate_size = 32,
    shared_expert_intermediate_size = 32,
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
    output_router_logits = True,
    router_aux_loss_coef = 0.01,
    tie_word_embeddings = False,
)


def build(seed = 0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(**CONFIG)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(config).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
            elif name.endswith("gate.weight") and p.ndim == 2:
                p.normal_(0, 0.5)  # routers start at zero: spread the routing
    torch.manual_seed(seed + 1)
    # Unsloth's default targets: every linear but the QSA indexer (integer top-k, no gradient).
    names = {
        n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    names.discard("lm_head")
    regex = r"^(?!.*\.indexer\.).*\.(" + "|".join(sorted(names)) + r")$"
    return get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = regex, init_lora_weights = False)
    ).eval()


def batch(bsz = 2):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB - 1, (bsz, SEQ), generator = g)
    return ids, ids.clone()


def _lora(model, full = lambda g, p: g):
    return {
        n.replace(".default.", "."): full(p.grad, p).detach().clone().double()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def _reference(mode, ids, labels, parts, steps):
    from peft import get_peft_model_state_dict

    model = build()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    per = ids.shape[0] // parts
    chunks = [(ids[i * per : (i + 1) * per], labels[i * per : (i + 1) * per]) for i in range(parts)]
    n_items = int((labels[:, 1:] != -100).sum())
    losses, grads = [], None
    for step in range(steps):
        if mode == "tp":
            loss = model(input_ids = ids, labels = labels, use_cache = False).loss
        elif mode == "fsdp2":
            loss = (
                sum(model(input_ids = a, labels = b, use_cache = False).loss for a, b in chunks) / parts
            )
        else:
            loss = sum(
                model(input_ids = a, labels = b, use_cache = False, num_items_in_batch = n_items).loss
                for a, b in chunks
            )
        loss.backward()
        if step == 0:
            grads = _lora(model)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    trainable = {
        n.replace(".default.", ".") for n, p in model.named_parameters() if p.requires_grad
    }
    state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    return {"losses": losses, "grads": grads, "trainable": trainable, "state": state}


def _case_parity(
    rank,
    world,
    tmp,
    mode,
    schedule = "gpipe",
    n_mb = 2,
    steps = 2,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    bsz = max(2, world) if mode == "fsdp2" else (max(2, n_mb) if mode == "pp" else 2)
    ids, labels = batch(bsz)
    parts = world if mode == "fsdp2" else n_mb
    ref = _reference(mode, ids, labels, parts, steps)
    model = build()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = (mode,))[mode]
    save_dir = os.path.join(tmp, "adapter")
    if mode == "tp":
        model = tpn.apply_tensor_parallel(model, mesh)
        params = [p for p in model.parameters() if p.requires_grad]
        step = lambda: model(input_ids = ids, labels = labels, use_cache = False).loss
        full = tpn._full
    elif mode == "fsdp2":
        native = _load("fsdp2_native")
        native.shard_model(
            model, native.FSDP2Config(device = "cpu", param_dtype = torch.float32, mesh = mesh)
        )
        per = bsz // world
        mine = slice(rank * per, (rank + 1) * per)
        params = [p for p in model.parameters() if p.requires_grad]
        step = lambda: model(input_ids = ids[mine], labels = labels[mine], use_cache = False).loss
        full = lambda g, p: g.full_tensor() if hasattr(g, "full_tensor") else g
    else:
        pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = n_mb, schedule = schedule)
        params = pp.parameters()
        step = lambda: pp.step(ids, labels)
        full = lambda g, p: g
    opt = torch.optim.SGD(params, lr = 1e-2)
    losses, grads = [], None
    for s in range(steps):
        loss = step()
        if mode != "pp":
            loss.backward()
        if s == 0:
            grads = _lora(model, full)
        opt.step()
        opt.zero_grad(set_to_none = True)
        if mode == "fsdp2":
            value = torch.tensor([float(loss)], dtype = torch.float64)
            dist.all_reduce(value)
            losses.append(float(value) / world)
        else:
            losses.append(float(loss))
    if mode == "pp":
        pp.save_adapter(save_dir)
    elif mode == "tp":
        tpn.save_tp_adapter(model, save_dir)
    else:
        with _load("fsdp2_save").fsdp2_gathered(model, trainable_only = True) as writer:
            if writer:
                model.save_pretrained(save_dir)
        dist.barrier()
    saved = None
    if rank == 0:
        from safetensors.torch import load_file
        saved = load_file(os.path.join(save_dir, "adapter_model.safetensors"))
    return {"ref": ref, "losses": losses, "grads": grads, "saved": saved}


CASES = {"parity": _case_parity}


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
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
    if kwargs.get("mode") == "pp":
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
            p.join(900)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


CELLS = {
    "tp2": dict(world = 2, mode = "tp"),
    "tp4_kv_replicated": dict(world = 4, mode = "tp"),
    "fsdp2_2": dict(world = 2, mode = "fsdp2"),
    "pp2_gpipe": dict(world = 2, mode = "pp"),
    "pp2_1f1b": dict(world = 2, mode = "pp", schedule = "1f1b"),
    "pp4_gpipe_mb4": dict(world = 4, mode = "pp", n_mb = 4),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_qwen4exp_parallel_matches_one_process(cell):
    kwargs = dict(CELLS[cell])
    world = kwargs.pop("world")
    results = _spawn("parity", world, **kwargs)
    ref = results[0]["ref"]
    assert results[0]["losses"] == pytest.approx(ref["losses"], rel = 1e-6), (
        cell,
        results[0]["losses"],
    )
    merged = {}
    for res in results:
        for name, grad in res["grads"].items():
            merged.setdefault(name, grad)
    # Every adapter the reference trains gets a gradient on its owning rank(s) (PP: one stage each).
    assert set(ref["grads"]) == set(merged), sorted(set(ref["grads"]) ^ set(merged))[:6]
    assert not any(".indexer." in n for n in ref["trainable"])
    assert len(merged) == len(ref["trainable"])
    for name, want in ref["grads"].items():
        got = merged[name]
        scale = float(want.abs().max())
        assert scale > 0, name
        assert float((got - want).abs().max()) <= 1e-7 + 1e-4 * scale, (cell, name)
    saved = results[0]["saved"]
    assert set(saved) == set(ref["state"]), sorted(set(saved) ^ set(ref["state"]))[:6]
    for key, want in ref["state"].items():
        diff = float((saved[key].double() - want.double()).abs().max())
        assert diff <= 1e-7 + 1e-5 * float(want.abs().max()), (cell, key, diff)


def test_tp_plan_shards_qsa_heads_and_keeps_the_indexer_whole():
    model = build()
    plan = tpn.derive_tp_plan(model, 2)
    (region,) = plan.regions  # the one QSA layer; GDN layers, MoE and the PLE layer stay replicated
    assert region.path.endswith("layers.3.self_attn")
    # q_proj emits [q, gate] per head: 4 heads, not 8.
    assert (region.heads, region.kv_heads, region.head_dim) == (4, 2, 16)
    assert [name for name, _ in region.replicated] == ["indexer"]
    assert len(plan.moe_blocks) == 4


def test_pp_splits_streams_and_runs_the_mixer_last():
    model = build()
    parts = ppn._find_parts(model)
    assert [name for name, _ in parts.norms] == ["hyper_connection_mixer"]
    assert (
        ppn._stage_width(model, 64) == 64 * 4
    )  # hidden states between layers: 4 flattened streams
    assert (
        ppn._refuse_unsupported(model, parts) is not None
    )  # router aux loss carried (Qwen3-MoE path)


def test_pp_still_refuses_unproven_hybrid_stacks():
    model = build()
    parts = ppn._find_parts(model)
    family = type(ppn._unwrap(model)).__name__
    saved = ppn._PP_HYBRID_FAMILIES.pop(family)
    try:
        with pytest.raises(ValueError, match = "hybrid layer stacks"):
            ppn._refuse_unsupported(model, parts)
    finally:
        ppn._PP_HYBRID_FAMILIES[family] = saved
