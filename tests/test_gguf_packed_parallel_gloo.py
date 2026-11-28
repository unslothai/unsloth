# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`gguf_packed = True` models under DDP, native FSDP2 and pipeline parallel, gloo world of 2 on the CPU: the
packed weights (UnslothGGUFLinear uint8 blocks, GgufExpertParam expert stacks) stay packed and run, and the loss
and LoRA gradients of one step match the same model in one process.

Two models: a GGUF file written here and loaded through Unsloth's packed loader (transformers' llama), and a
Qwen3-MoE built in memory whose attention Linears are UnslothGGUFLinear and whose expert stacks are
GgufExpertParam (the expert hook runs them packed, LoRA on the stacks included)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    os.environ.setdefault(
        "UNSLOTH_GGUF_KERNELS", "reference"
    )  # the pure torch decode: faster on the CPU

import importlib.machinery  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "_shared"))

ROUTES = ("ddp", "fsdp2", "pp")
_SEQ = (9, 7)


def _stub_packages():
    """unsloth / unsloth.kernels / unsloth.models as bare packages: a spawned rank must not run unsloth/__init__."""
    for name, path in (
        ("unsloth", os.path.join(_REPO, "unsloth")),
        ("unsloth.kernels", os.path.join(_REPO, "unsloth", "kernels")),
        ("unsloth.models", os.path.join(_REPO, "unsloth", "models")),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [path]
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package = True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module


def _gp():
    from unsloth.models import gguf_packed
    return gguf_packed


# ------------------------------------------------------------------------------------------ the two models


def _llama_from_gguf(path):
    from transformers import AutoModelForCausalLM
    gp = _gp()
    with gp.gguf_packed_load(path):
        return AutoModelForCausalLM.from_pretrained(
            os.path.dirname(path),
            gguf_file = os.path.basename(path),
            dtype = torch.float32,
            attn_implementation = "eager",
            quantization_config = gp.gguf_packed_quantization_config(),
        )


def _moe_backend():
    """unsloth_zoo's MoE backend on the transformers 5 Qwen3-MoE experts, as Unsloth installs it."""
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mod
    from unsloth_zoo.temporary_patches import moe_utils as mu
    from unsloth_zoo.temporary_patches import mxfp4 as mx

    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    mu._TORCH_GROUPED_MM_SUPPORTED = True

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return mu.forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)

    mod.Qwen3MoeExperts.forward = forward


def _blocks(type_name, lead, rows, cols, seed):
    from test_gguf_packed_lora import _blocks as blocks
    out = [blocks(type_name, rows, cols, seed + i, scale = 1e-3) for i in range(lead or 1)]
    return torch.stack(out) if lead else out[0]


def _packed_moe():
    """Qwen3-MoE with packed GGUF attention projections (Q4_K / Q8_0) and packed GGUF expert stacks."""
    from test_gguf_packed_lora import ref
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    gp = _gp()
    Expert = gp.gguf_expert_param_class()
    config = Qwen3MoeConfig(
        vocab_size = 128,
        hidden_size = 256,
        intermediate_size = 256,
        moe_intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        num_experts = 4,
        num_experts_per_tok = 2,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
        tie_word_embeddings = False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(config).float()
    q4k, q8 = ref.GGML_TYPE_IDS["Q4_K"], ref.GGML_TYPE_IDS["Q8_0"]
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for j, name in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
            dense = getattr(attn, name)
            kind, t = ("Q8_0", q8) if name == "o_proj" else ("Q4_K", q4k)
            blocks = _blocks(kind, 0, dense.out_features, dense.in_features, 100 * i + 10 * j)
            setattr(
                attn,
                name,
                gp.UnslothGGUFLinear.from_packed(
                    blocks, t, dense.in_features, dense.out_features, compute_dtype = torch.float32
                ),
            )
        experts = layer.mlp.experts
        for j, (name, kind, t) in enumerate(
            (("gate_up_proj", "Q4_K", q4k), ("down_proj", "Q8_0", q8))
        ):
            E, N, K = experts._parameters[name].shape
            experts._parameters[name] = Expert(
                _blocks(kind, E, N, K, 1000 + 100 * i + 10 * j),
                ggml_type = t,
                packed_cols = K,
                packed_dtype = torch.float32,
            )
    return model


_BUILT = {}


def _build(kind, path):
    """A fresh LoRA model per call: the first is built, later ones are deep copies of it (PEFT's wrap reads every
    packed base weight, a dequantization each, which is most of a CPU build)."""
    import copy

    if kind not in _BUILT:
        _BUILT[kind] = _build_new(kind, path)
    return copy.deepcopy(_BUILT[kind])


def _build_new(kind, path):
    from peft import LoraConfig, get_peft_model

    gp = _gp()
    gp.patch_peft_for_gguf_packed_linears()
    if kind == "llama":
        model = _llama_from_gguf(path)
        lora = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    else:
        _moe_backend()
        model = _packed_moe()
        lora = LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            init_lora_weights = False,
        )
    torch.manual_seed(1)
    model = get_peft_model(model, lora)
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    return model.train()


def _packed_census(model):
    """(packed GGUF Linears still holding uint8 blocks, GgufExpertParam stacks) on this rank."""
    gp = _gp()
    Expert = gp.gguf_expert_param_class()
    linears = sum(
        isinstance(m, gp.UnslothGGUFLinear) and m._parameters["weight_packed"].dtype == torch.uint8
        for m in model.modules()
    )
    stacks = sum(Expert is not None and isinstance(p, Expert) for p in model.parameters())
    return linears, stacks


_CALLS = {"linear": 0, "experts": 0}


def _count_packed_calls():
    """Engagement: forwards of a packed GGUF Linear, and packed GGUF expert stack calls (hook or decode)."""
    gp = _gp()
    if not getattr(gp.UnslothGGUFLinear.forward, "_counted", False):
        forward = gp.UnslothGGUFLinear.forward

        def counted(self, *args, **kwargs):
            _CALLS["linear"] += 1
            return forward(self, *args, **kwargs)

        counted._counted = True
        gp.UnslothGGUFLinear.forward = counted
        Expert = gp.gguf_expert_param_class()
        if Expert is not None:
            decode = Expert._decode

            def counted_decode(self, *args, **kwargs):
                _CALLS["experts"] += 1
                return decode(self, *args, **kwargs)

            Expert._decode = counted_decode
    calls = dict(_CALLS)
    try:
        from unsloth_zoo.packed_experts import PACKED_EXPERT_STATS
        calls["experts"] += PACKED_EXPERT_STATS[
            "calls"
        ]  # the zoo's packed grouped linear (expert hook)
    except ImportError:
        pass
    return calls


# ------------------------------------------------------------------------------------------------ the routes


def _rows(vocab, rank):
    ids = torch.randint(
        0, vocab, (2, _SEQ[rank]), generator = torch.Generator().manual_seed(20 + rank)
    )
    labels = ids.clone()
    labels[0, :2] = -100
    return ids, labels


def _sum_loss(model, ids, labels):
    logits = model(input_ids = ids, use_cache = False).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1).float(),
        labels[:, 1:].flatten(),
        ignore_index = -100,
        reduction = "sum",
    )


def _clean(name):
    name = name.replace("_checkpoint_wrapped_module.", "")
    return name[len("module.") :] if name.startswith("module.") else name


def _grads(model):
    return {
        _clean(k): p.grad.detach().clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }


def _compare(got, want):
    keys = sorted(set(got) | set(want))
    missing = [k for k in keys if k not in got or k not in want]
    diffs = {k: float((got[k] - want[k]).abs().max()) for k in keys if k not in missing}
    worst = max(diffs, key = diffs.get, default = None)
    return {
        "missing": missing[:6],
        "n": len(keys),
        "maxdiff": diffs.get(worst),
        "scale": max((float(want[k].abs().max()) for k in want), default = 0.0),
        "worst": worst,
        "worst_scale": None if worst is None else float(want[worst].abs().max()),
    }


def _data_parallel_reference(kind, path, world):
    """One process over both ranks' rows: summed loss / all supervised tokens."""
    ref = _build(kind, path)
    vocab = ref.config.vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    with torch.no_grad():
        loss = sum(float(_sum_loss(ref, ids, lab)) for ids, lab in rows) / n
    return rows, n, loss, _grads(ref)


def _run_ddp(kind, path, rank, world, rows, n):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    model = _build(kind, path)
    census = _packed_census(model)
    wrapped = DistributedDataParallel(
        model
    )  # broadcasts every parameter, the frozen packed ones included
    before = _count_packed_calls()
    loss = _sum_loss(wrapped, *rows[rank]) * world / n
    loss.backward()
    after = _count_packed_calls()
    value = loss.detach() / world
    dist.all_reduce(value)
    return model, float(value), _grads(model), census, before, after


def _run_fsdp2(kind, path, rank, world, rows, n):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from unsloth_models_loader import load_models_module

    native = load_models_module("fsdp2_native")
    model = _build(kind, path)
    model = native.shard_model(
        model, native.FSDP2Config(param_dtype = torch.float32, shard_frozen = True)
    )
    census = _packed_census(model)
    before = _count_packed_calls()
    loss = _sum_loss(model, *rows[rank]) * world / n
    loss.backward()
    after = _count_packed_calls()
    value = loss.detach() / world
    dist.all_reduce(value)
    got = {}
    for k, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()
        else:
            g = g.clone()
            dist.all_reduce(g)
        got[_clean(k)] = g.detach().clone()
    sharded = sum(isinstance(p, DTensor) for p in model.parameters())
    return model, float(value), got, census + (sharded,), before, after


def _run_pp(kind, path, rank, world):
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth_models_loader import load_models_module

    ppn = load_models_module("pp_native")
    ref = _build(kind, path)
    vocab = ref.config.vocab_size
    ids = torch.cat([_rows(vocab, r)[0][:, :7] for r in range(world)])
    labels = torch.cat([_rows(vocab, r)[1][:, :7] for r in range(world)])
    # The reference runs pp.step's two micro-batches (rows 0-1, 2-3) itself: the packed expert grouped GEMM
    # rounds like bf16 even in float32, so a whole-batch reference would differ from any micro-batched run.
    n = int((labels[:, 1:] != -100).sum())
    ref_loss = 0.0
    for half in (slice(0, 2), slice(2, 4)):
        loss = _sum_loss(ref, ids[half], labels[half]) / n
        loss.backward()
        ref_loss += float(loss.detach())
    want = _grads(ref)
    model = _build(kind, path)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    census = _packed_census(model)
    before = _count_packed_calls()
    loss = float(pp.step(ids, labels))
    after = _count_packed_calls()
    got = _grads(model)
    want = {k: v for k, v in want.items() if k in got}  # this stage's parameters
    return ref_loss, loss, got, want, census + (tuple(pp.bounds),), before, after


def _case_routes(rank, world, tmp, kind, path, routes):
    import torch.distributed as dist

    if kind == "moe":
        _moe_backend()
    out = {}
    rows, n, want_loss, want = _data_parallel_reference(kind, path, world)
    for route in routes:
        dist.barrier()
        try:
            if route == "pp":
                ref_loss, loss, got, want_pp, census, before, after = _run_pp(
                    kind, path, rank, world
                )
                cmp = _compare(got, want_pp)
                out[route] = dict(loss = (loss, ref_loss), **cmp)
            else:
                run = _run_ddp if route == "ddp" else _run_fsdp2
                _, loss, got, census, before, after = run(kind, path, rank, world, rows, n)
                out[route] = dict(loss = (loss, want_loss), **_compare(got, want))
            out[route].update(
                census = census,
                linear_calls = after["linear"] - before["linear"],
                expert_calls = after["experts"] - before["experts"],
                lora_expert_grads = sum(".experts." in k and "lora_" in k for k in got),
            )
        except Exception as exc:  # noqa: BLE001 - one route's failure must not hide the others
            import traceback
            out[route] = {
                "error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-2500:]
            }
    return out


def _worker(rank, world, tmp, case, kwargs):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    _stub_packages()
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 600),
    )
    try:
        res = globals()[case](rank, world, tmp, **kwargs)
    except BaseException as exc:  # a rank must report, not vanish
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-4000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(
    case,
    tmp,
    world = 2,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target = _worker, args = (r, world, tmp, case, kwargs)) for r in range(world)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 900
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        time.sleep(0.2)
    for p in procs:
        if p.is_alive():
            p.terminate()
        p.join(30)
    out = []
    for r in range(world):
        path = os.path.join(tmp, f"result_{r}.pt")
        out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
    return out


# --------------------------------------------------------------------------------------------------- tests


def _requirements(kind):
    pytest.importorskip("peft")
    from unsloth.models import gguf_packed as gp

    if gp.gguf_packed_supported() is not None:
        pytest.skip(f"packed GGUF loading needs transformers >= {gp.GGUF_PACKED_MIN_TRANSFORMERS}")
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch without gloo")
    if kind == "moe":
        if gp.gguf_expert_param_class() is None:
            pytest.skip("unsloth_zoo without the packed GGUF expert hook")
        try:
            torch._grouped_mm(
                torch.randn(4, 32),
                torch.randn(2, 32, 16),
                offs = torch.tensor([2, 4], dtype = torch.int32),
            )
        except Exception:
            pytest.skip("no CPU torch._grouped_mm")


def _routes():
    """FSDP2 keeps packed weights replicated through fully_shard(ignored_params) (torch 2.7+); pipeline parallel
    needs torch.distributed.pipelining. A route this torch cannot run is checked elsewhere (its refusal)."""
    import inspect

    routes = ["ddp"]
    try:
        from torch.distributed.fsdp import fully_shard
        if "ignored_params" in inspect.signature(fully_shard).parameters:
            routes.append("fsdp2")
    except ImportError:
        pass
    from unsloth_models_loader import load_models_module

    try:
        load_models_module("pp_native")._check_torch_pipelining()
        routes.append("pp")
    except RuntimeError:
        pass
    return tuple(routes)


def _check(results, routes, kind):
    for rank, res in enumerate(results):
        assert res is not None and "error" not in res, (rank, res)
        for route in routes:
            r = res[route]
            assert "error" not in r, (rank, route, r["error"] if "error" in r else r)
            got, want = r["loss"]
            assert got == pytest.approx(want, rel = 1e-5, abs = 1e-6), (rank, route, r["loss"])
            assert not r["missing"] and r["n"] > 0, (rank, route, r["missing"])
            assert r["maxdiff"] <= 1e-4 * max(r["scale"], 1e-3) + 1e-7, (
                rank,
                route,
                r["worst"],
                r["maxdiff"],
                r["worst_scale"],
            )
            linears, stacks = r["census"][:2]
            # The packed weights stay packed on every rank (pp: on this rank's stage) and run.
            assert linears > 0 and r["linear_calls"] > 0, (
                rank,
                route,
                r["census"],
                r["linear_calls"],
            )
            if kind == "moe":
                assert stacks > 0 and r["expert_calls"] > 0, (
                    rank,
                    route,
                    r["census"],
                    r["expert_calls"],
                )
                assert r["lora_expert_grads"] > 0, (rank, route)
            if route == "fsdp2":
                assert r["census"][2] > 0, (rank, "nothing sharded")


def test_packed_gguf_llama_from_the_loader_matches_one_process(tmp_path):
    _requirements("llama")
    from test_gguf_packed_lora import _write_llama_gguf

    path = tmp_path / "tiny.gguf"
    _write_llama_gguf(path)
    routes = _routes()
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn("_case_routes", tmp, kind = "llama", path = str(path), routes = routes)
    _check(results, routes, "llama")


def test_packed_gguf_linears_and_expert_stacks_match_one_process():
    _requirements("moe")
    routes = _routes()
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn("_case_routes", tmp, kind = "moe", path = None, routes = routes)
    _check(results, routes, "moe")
