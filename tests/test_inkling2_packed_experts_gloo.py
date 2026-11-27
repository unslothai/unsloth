# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""UNSLOTH_INKLING2_PACKED_EXPERTS=1: Inkling 2's released stacked NVFP4 / MXFP8 experts (``mlp.experts.w13_weight``
interleaved gate / up rows, ``w2_weight``, ``_weight_scale``, ``_weight_amax``, ``_weight_outer_scale_denominator``)
load as unsloth_zoo's packed stacks.

One process: every stack decodes bitwise to what transformers' own dequantizing converter loads, and the flag off
leaves transformers' path as it is. Expert parallel (gloo CPU, 2 ranks): each rank reads only its own experts'
slices of every stack from the checkpoint, its stacks are bitwise the full packed load's rows, marked, logical
shape whole, and one LoRA step after apply_expert_parallel matches the full load's (loss and every gradient)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
import test_tp_inkling2_gloo as ink  # noqa: E402

E = 4
STACKS = ("experts.gate_up_proj", "experts.down_proj")
ENV = "UNSLOTH_INKLING2_PACKED_EXPERTS"


def _requirements():
    pe = pytest.importorskip("unsloth_zoo.packed_experts")
    pytest.importorskip("peft")
    if not hasattr(pe, "Nvfp4ExpertParam"):
        pytest.skip("unsloth_zoo without packed expert stacks")
    try:
        torch._grouped_mm(
            torch.randn(4, 32), torch.randn(2, 32, 16), offs = torch.tensor([2, 4], dtype = torch.int32)
        )
    except Exception:
        pytest.skip("no CPU torch._grouped_mm")


def _write_checkpoint(path):
    """A tiny Inkling-2-Mini (inkling2_loop) saved by transformers, its experts replaced by packed stacks in the
    released layout: NVFP4 on even layers, MXFP8 on odd ones (random codes, finite scales)."""
    from safetensors.torch import load_file, save_file

    model = ink.build(loop = True, lora = False)
    model.generation_config.pad_token_id = None
    model.save_pretrained(path, safe_serialization = True)
    file = os.path.join(path, "model.safetensors")
    state = load_file(file)
    g = torch.Generator().manual_seed(0)
    safe = torch.tensor([0.25, 0.5, 1.0, 2.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    for key in [k for k in state if k.endswith(("experts.w13_weight", "experts.w2_weight"))]:
        n_experts, rows, cols = state.pop(key).shape
        layer = int(key.split(".layers.")[1].split(".")[0])
        if layer % 2 == 0:
            state[key] = torch.randint(
                0, 256, (n_experts, rows, cols // 2), dtype = torch.uint8, generator = g
            )
            pick = torch.randint(0, len(safe), (n_experts, rows, cols // 16), generator = g)
            state[key + "_scale"] = safe[pick].view(torch.float8_e4m3fn)
            state[key + "_amax"] = torch.rand(n_experts, generator = g) + 0.5
            state[key + "_outer_scale_denominator"] = torch.tensor([6.0 * 448.0])
        else:
            codes = torch.randint(0, 0x7E, (n_experts, rows, cols), dtype = torch.uint8, generator = g)
            codes |= (
                torch.randint(0, 2, (n_experts, rows, cols), dtype = torch.uint8, generator = g) << 7
            )
            state[key] = codes.view(torch.float8_e4m3fn)
            state[key + "_scale"] = torch.randint(
                120, 126, (n_experts, rows, cols // 32), dtype = torch.uint8, generator = g
            )
    save_file(state, file, metadata = {"format": "pt"})


def _load(path):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(path, dtype = torch.float32)
    model.config._attn_implementation = "eager"
    return model


def _loader():
    return ink._load("inkling2_packed")


def _zoo_backend():
    """unsloth_zoo's MoE backend on the Inkling 2 experts (its softcapped SwiGLU), as Unsloth installs it."""
    from transformers.models.inkling2_loop import modeling_inkling2_loop as mod
    from unsloth_zoo.temporary_patches import moe_utils as mu
    from unsloth_zoo.temporary_patches import mxfp4 as mx
    from unsloth_zoo.temporary_patches.inkling2 import patch_inkling2_moe_experts

    patch_inkling2_moe_experts()
    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    mu._TORCH_GROUPED_MM_SUPPORTED = True

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return mu.forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)

    mod.Inkling2LoopExperts.forward = forward


def test_packed_inkling2_experts_decode_like_transformers(tmp_path, monkeypatch):
    _requirements()
    _write_checkpoint(str(tmp_path))
    monkeypatch.delenv(ENV, raising = False)
    loader = _loader()
    assert loader.install_inkling2_packed_expert_loading()
    dense = _load(str(tmp_path))  # flag off: transformers' own dequantizing converter
    dense_stacks = {k: p for k, p in dense.named_parameters() if k.endswith(STACKS)}
    assert dense_stacks and all(type(p) is torch.nn.Parameter for p in dense_stacks.values())
    monkeypatch.setenv(ENV, "1")
    packed = _load(str(tmp_path))
    got = {k: p for k, p in packed.named_parameters() if k.endswith(STACKS)}
    assert set(got) == set(dense_stacks) and len(got) == 24
    kinds = {type(p).__name__ for p in got.values()}
    assert kinds == {"Nvfp4ExpertParam", "Mxfp8ExpertParam"}, kinds
    for k, p in got.items():
        assert torch.equal(p.dequantize(torch.float32), dense_stacks[k]), k
    # Everything else loads exactly as before.
    other = {k: p for k, p in dense.named_parameters() if not k.endswith(STACKS)}
    for k, p in packed.named_parameters():
        if k in other:
            assert torch.equal(p, other[k]), k


def _read_ranges(loader):
    seen = []
    original = loader._read

    def read(parts, suffix, lo, hi, files):
        seen.append((parts[suffix][3], lo, hi))
        return original(parts, suffix, lo, hi, files)

    loader._read = read
    return seen, lambda: setattr(loader, "_read", original)


def _lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    targets = [
        n
        for n, m in model.named_modules()
        if ".language_model.layers." in n and n.endswith(ink.ATTENTION)
    ]
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = targets,
        target_parameters = ink.EXPERTS,
        init_lora_weights = False,
    )
    return get_peft_model(model, config).train()


def _step(epn, model, rank, world, mesh):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    rows = []
    for r in range(world):
        ids = torch.randint(
            1, 250, (2, (12, 8)[r]), generator = torch.Generator().manual_seed(10 + r)
        )
        rows.append(ids)
    n = sum(int(ids[:, 1:].numel()) for ids in rows)
    epn.apply_expert_parallel(model, mesh)
    ddp = DistributedDataParallel(model)
    ids = rows[rank]
    logits = ddp(input_ids = ids, use_cache = False).logits
    loss = (
        torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction = "sum"
        )
        * world
        / n
    )
    loss.backward()
    epn.ep_sync_expert_grads(model)
    value = loss.detach() / world
    dist.all_reduce(value)
    group = epn._mesh(model).get_group()
    grads = {
        k: (epn._gather_expert(p, group, p.grad) if getattr(p, epn._EP, None) else p.grad)
        .detach()
        .clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }
    return float(value), grads


def _case_ep(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    os.environ[ENV] = "1"
    _zoo_backend()
    tpn = ink._load("tp_native")
    epn = ink._load("ep_native")
    loader = _loader()
    assert loader.install_inkling2_packed_expert_loading()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        _write_checkpoint(ckpt)
    dist.barrier()
    full = _load(ckpt)
    want = {k: p for k, p in full.named_parameters() if k.endswith(STACKS)}
    seen, restore = _read_ranges(loader)
    with epn.expert_parallel_loading(mesh) as state:
        model = _load(ckpt)
    restore()
    per = E // world
    lo, hi = rank * per, (rank + 1) * per
    have = {k: p for k, p in model.named_parameters() if k.endswith(STACKS)}

    def rows_of(
        p,
        a = None,
        b = None,
    ):
        kw = p._init_kwargs()
        parts = [torch.Tensor.detach(p), kw["packed_scales"], kw.get("packed_global")]
        return [None if t is None else (t if a is None else t[a:b]) for t in parts]

    out = {
        "models": len(state["models"]),
        "stats": dict(model.__dict__.get("_unsloth_tp_load_stats") or {}),
        "marks": sorted({getattr(p, tpn._EXPERT_PRELOAD, None) for p in have.values()}),
        "logical": sorted({tuple(p._original_shape)[0] for p in have.values()}),
        "sliced": set(have) == set(want)
        and all(
            all(
                (x is None and y is None) or torch.equal(x, y)
                for x, y in zip(rows_of(have[k]), rows_of(want[k], lo, hi))
            )
            for k in want
        ),
        # Every E-major tensor of every stack was read as this rank's slice, never whole.
        "reads": sorted({(a, b) for key, a, b in seen if not key.endswith("denominator")}),
        "n_reads": len(seen),
    }
    try:
        model(input_ids = torch.zeros(1, 3, dtype = torch.long))
        out["unapplied_forward"] = "ran"
    except RuntimeError as exc:
        out["unapplied_forward"] = str(exc)
    full, model = _lora(full), _lora(model)
    lora = lambda m: {k: p.detach().clone() for k, p in m.named_parameters() if "lora_" in k}  # noqa: E731
    want_lora, got_lora = lora(full), lora(model)
    out["lora_equal"] = set(want_lora) == set(got_lora) and all(
        torch.equal(want_lora[k], got_lora[k]) for k in want_lora
    )
    out["full"] = _step(epn, full, rank, world, mesh)
    out["sharded"] = _step(epn, model, rank, world, mesh)
    return out


def _worker(rank, world, tmp, case, kwargs):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 300),
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
    world = 2,
    **kwargs,
):
    _requirements()
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, tmp, case, kwargs)) for r in range(world)
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


def test_ep_packed_inkling2_reads_only_local_experts_and_matches_full_load():
    results = _spawn("_case_ep")
    assert all(r is not None and "error" not in r for r in results), results
    for rank, r in enumerate(results):
        per = E // 2
        assert r["models"] == 1, r
        assert r["stats"].get("sharded_expert_stacks") == 24, r["stats"]
        assert r["marks"] == [(E, 2, (0, 1))], r["marks"]
        assert r["logical"] == [E], r["logical"]
        assert r["sliced"], "a rank's packed stacks are not the full load's rows [lo, hi)"
        assert r["reads"] == [(rank * per, (rank + 1) * per)], r["reads"]
        assert "apply_expert_parallel" in r["unapplied_forward"], r["unapplied_forward"]
        assert r["lora_equal"]
        (got_loss, got), (want_loss, want) = r["sharded"], r["full"]
        assert got_loss == want_loss, (got_loss, want_loss)
        assert set(got) == set(want) and any(".experts." in k and "lora_" in k for k in got)
        for k in want:
            assert torch.equal(got[k], want[k]), k


def test_packed_stacks_follow_the_device_map():
    """Each experts module's stacks go where the device map puts that module, not where the first parameter is."""
    pytest.importorskip("transformers.integrations.accelerate")
    loader = _loader()
    device_map = {"model.language_model.layers.0": 0, "model.language_model.layers.1": 1, "": "cpu"}
    model = torch.nn.Module()
    name = "model.language_model.layers.1.mlp.experts"
    assert loader._device_of(model, name, device_map) == torch.device("cuda:1")
    assert loader._device_of(
        model, "model.language_model.layers.0.mlp.experts", device_map
    ) == torch.device("cuda:0")
    assert loader._device_of(
        model, "model.language_model.layers.5.mlp.experts", device_map
    ) == torch.device("cpu")
    assert loader._offloads_to_disk({"": "disk"}) and not loader._offloads_to_disk({"": 0})


def test_packed_load_keeps_the_expert_converters_for_saving(tmp_path, monkeypatch):
    """save_pretrained reverses the converters transformers recorded: the packed load records the same ones."""
    _requirements()
    _write_checkpoint(str(tmp_path))
    loader = _loader()
    assert loader.install_inkling2_packed_expert_loading()
    monkeypatch.delenv(ENV, raising = False)
    dense = _load(str(tmp_path))
    monkeypatch.setenv(ENV, "1")
    packed = _load(str(tmp_path))
    names = lambda m: [
        type(c).__name__ + str(getattr(c, "_original_source_patterns", ""))
        for c in m._weight_conversions
    ]  # noqa: E731
    assert names(packed) == names(dense) and any("w13_weight" in n for n in names(packed)), (
        names(packed),
        names(dense),
    )
