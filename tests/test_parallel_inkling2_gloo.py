# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 LoRA (attention Linears + routed experts through PEFT target_parameters) on a 2-rank gloo CPU world
vs one process, fp32: DDP and native FSDP2 (fsdp2_native.shard_model, every decoder layer a unit; Inkling-2-Mini's
looped layers run three times per forward under the FSDP2 hooks) match the reference losses and step-0 LoRA
gradients; tensor / pipeline parallel refuse the model with a ValueError instead of training it."""

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
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERTS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def _load(name):
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


def build(loop, reentrant = None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    text = dict(
        vocab_size = 256,
        unpadded_vocab_size = 250,
        hidden_size = 32,
        num_hidden_layers = 12,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        swa_num_attention_heads = 2,
        swa_num_key_value_heads = 1,
        swa_head_dim = 16,
        d_rel = 4,
        rel_extent = 8,
        sliding_window_size = 4,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        moe_intermediate_size = 32,
    )
    vision = dict(
        model_type = "inkling2_vision",
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        projection_hidden_dim = 32,
        text_hidden_size = 32,
    )
    if loop:
        text.update(
            loop_layer_index_start = 6, loop_layer_index_end = 12, loop_num_passes = 3, loop_gate_rank = 4
        )
        config = Inkling2LoopConfig(text_config = text, vision_config = vision)
    else:
        config = Inkling2Config(text_config = text, vision_config = vision)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).float()
    with torch.no_grad():  # init leaves residual queries / gates at zero: give every path signal
        g = torch.Generator().manual_seed(0)
        for name, p in model.named_parameters():
            if (
                name.endswith(("query", "static_weight", "proj"))
                or "e_score_correction_bias" in name
            ):
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    model.config._attn_implementation = "eager"
    targets = [
        n
        for n, m in model.named_modules()
        if ".language_model.layers." in n
        and n.endswith(
            ("q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "o_proj.gate_proj")
        )
    ]
    lora = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = 0.0,
        target_modules = targets,
        target_parameters = EXPERTS,
        init_lora_weights = False,
    )
    torch.manual_seed(1)
    model = get_peft_model(model, lora)
    if reentrant is not None:
        # Gradient checkpointing with Unsloth's checkpointed loop message residual (Mini).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": reentrant}
        )
        if reentrant:
            model.enable_input_require_grads()
        assert _load("inkling2").checkpoint_inkling2_message_inputs(model) == int(loop)
    model.train()
    return model


def batch(rows = 4, seq = 12):
    ids = torch.randint(1, 250, (rows, seq), generator = torch.Generator().manual_seed(3))
    return ids, ids.clone()


def _grads(model, full = lambda g: g):
    out = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            key = n.replace(".default", "").replace(
                "module.", "", 1 if n.startswith("module.") else 0
            )
            out[key] = full(p.grad).detach().clone().double()
    return out


def _train(
    model,
    step,
    params,
    steps,
    full = lambda g: g,
    mean = None,
):
    opt = torch.optim.SGD(params, lr = 0.05)
    losses, grads = [], None
    for s in range(steps):
        loss = step()
        loss.backward()
        if s == 0:
            grads = _grads(model, full)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(mean(loss) if mean else float(loss))
    return losses, grads


def _case(
    rank,
    world,
    mode,
    loop,
    steps = 2,
    reentrant = None,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    ids, labels = batch()
    recomputed = [0]
    if reentrant is not None:
        ink = _load("inkling2")
        math = ink._message_input_math

        def counted(*args):
            recomputed[0] += 1
            return math(*args)

        ink._message_input_math = counted
    ref_model = build(loop, reentrant)
    ref = _train(
        ref_model,
        lambda: ref_model(input_ids = ids, labels = labels).loss,
        [p for p in ref_model.parameters() if p.requires_grad],
        steps,
    )
    model = build(loop, reentrant)
    per = ids.shape[0] // world
    mine = slice(rank * per, (rank + 1) * per)

    def mean(loss):
        value = torch.tensor([float(loss)], dtype = torch.float64)
        dist.all_reduce(value)
        return float(value) / world

    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        wrapped = DistributedDataParallel(model)
        got = _train(
            model,
            lambda: wrapped(input_ids = ids[mine], labels = labels[mine]).loss,
            [p for p in model.parameters() if p.requires_grad],
            steps,
            mean = mean,
        )
        return {"ref": ref, "got": got, "recomputed": recomputed[0]}
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = (mode,))[mode]
    if mode == "fsdp2":
        native = _load("fsdp2_native")
        native.shard_model(
            model, native.FSDP2Config(device = "cpu", param_dtype = torch.float32, mesh = mesh)
        )
        units = sum(1 for m in model.modules() if type(m).__name__.startswith("FSDP"))
        got = _train(
            model,
            lambda: model(input_ids = ids[mine], labels = labels[mine]).loss,
            [p for p in model.parameters() if p.requires_grad],
            steps,
            full = lambda g: g.full_tensor() if hasattr(g, "full_tensor") else g,
            mean = mean,
        )
        return {"ref": ref, "got": got, "units": units, "recomputed": recomputed[0]}
    try:
        if mode == "tp":
            _load("tp_native").apply_tensor_parallel(model, mesh)
        else:
            ppn = _load("pp_native")
            ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2, schedule = "gpipe")
    except ValueError as exc:
        return {"refused": str(exc)}
    return {"refused": None}


def _worker(rank, world, port, tmp, kwargs):
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
        res = _case(rank, world, **kwargs)
    except Exception as exc:
        res = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(world = 2, **kwargs):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, kwargs)) for r in range(world)
        ]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
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


@pytest.mark.parametrize("loop", [True, False], ids = ["mini_loop", "small"])
@pytest.mark.parametrize("mode", ["ddp", "fsdp2"])
def test_data_parallel_matches_one_process(mode, loop):
    results = _spawn(mode = mode, loop = loop)
    assert all(r is not None and "error" not in r for r in results), results
    (ref_losses, ref_grads), (losses, grads) = results[0]["ref"], results[0]["got"]
    assert losses == pytest.approx(ref_losses, rel = 1e-5)
    assert set(grads) == set(ref_grads) and any(".experts." in n for n in grads)
    for name, want in ref_grads.items():
        scale = float(want.abs().max())
        assert scale > 0, name
        assert float((grads[name] - want).abs().max()) <= 1e-7 + 1e-4 * scale, name
    if mode == "fsdp2":
        assert results[0]["units"] > 12  # one unit per decoder layer plus the root


# DDP with reentrant checkpointing is left out: the looped layers run each parameter several times per forward, which
# stock DDP rejects ("mark a variable ready only once") with or without the message checkpoint.
@pytest.mark.parametrize("mode,reentrant", [("ddp", False), ("fsdp2", False), ("fsdp2", True)])
def test_data_parallel_message_checkpoint_matches_one_process(mode, reentrant):
    # Inkling-2-Mini under gradient checkpointing, with the loop message residual checkpointed (recomputed in
    # backward) on both ranks and in the reference.
    results = _spawn(mode = mode, loop = True, reentrant = reentrant)
    assert all(r is not None and "error" not in r for r in results), results
    (ref_losses, ref_grads), (losses, grads) = results[0]["ref"], results[0]["got"]
    # 6 loop layers x 2 later passes, forward + backward recompute, 2 steps, reference and parallel model.
    assert results[0]["recomputed"] >= 2 * 2 * 6 * 2 * 2, results[0]["recomputed"]
    assert losses == pytest.approx(ref_losses, rel = 1e-5)
    assert set(grads) == set(ref_grads) and any(".experts." in n for n in grads)
    for name, want in ref_grads.items():
        scale = float(want.abs().max())
        assert scale > 0, name
        assert float((grads[name] - want).abs().max()) <= 1e-7 + 1e-4 * scale, name


@pytest.mark.parametrize("loop", [True, False], ids = ["mini_loop", "small"])
@pytest.mark.parametrize(
    "mode", ["tp"]
)  # pipeline parallel trains it: tests/test_pp_inkling2_gloo.py
def test_tensor_and_pipeline_parallel_refuse(mode, loop):
    results = _spawn(mode = mode, loop = loop)
    assert all(r is not None and "error" not in r for r in results), results
    refused = results[0]["refused"]
    assert refused and refused.startswith("Unsloth:"), refused
    if mode == "tp":
        assert "QueryProjection" in refused
    else:
        assert "final_input" in refused
