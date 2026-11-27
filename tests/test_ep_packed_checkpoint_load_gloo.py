# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""expert_parallel_loading of a pre-quantized packed MoE checkpoint (compressed-tensors NVFP4 / MXFP8 per-expert
tensors, gloo CPU, 2 ranks, fp32): each rank reads only its own experts' packed tensors, its stacks are bitwise
the rows of a full load, marked as expert parallel loading's cut, and a LoRA step after apply_expert_parallel
matches the full load's (loss and every gradient bitwise). UNSLOTH_PACKED_EXPERTS=0 (16-bit stacks) is cut alike."""

import json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

import test_ep_packed_experts_gloo as packed_ep  # noqa: E402

E, H, I = 4, 64, 32
STACKS = ("gate_up_proj", "down_proj")
PRESET = {"nvfp4": ("NVFP4A16", "nvfp4-pack-quantized"), "mxfp8": ("MXFP8A16", "mxfp8-quantized")}


def _write_checkpoint(path, fmt):
    """Per-expert packed tensors under the released key layout (``experts.<e>.<proj>.weight_packed`` ...)."""
    import transformers
    import unsloth_zoo.packed_experts as pe
    from compressed_tensors.quantization import QuantizationConfig, preset_name_to_scheme
    from safetensors.torch import save_file

    torch.manual_seed(0)
    config = packed_ep._base().config
    config.tie_word_embeddings = False
    model = transformers.Qwen3MoeForCausalLM(config).float()
    state = {
        k: v.contiguous()
        for k, v in model.state_dict().items()
        if not k.endswith(tuple(f".mlp.experts.{s}" for s in STACKS))
    }
    for layer, block in enumerate(model.model.layers):
        gate_up, down = block.mlp.experts.gate_up_proj.data, block.mlp.experts.down_proj.data
        n = down.shape[2]
        for e in range(gate_up.shape[0]):
            for proj, w in (
                ("gate_proj", gate_up[e, :n]),
                ("up_proj", gate_up[e, n:]),
                ("down_proj", down[e]),
            ):
                key = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                if fmt == "nvfp4":
                    q = pe.quantize_experts_nvfp4(w[None])
                    state[f"{key}.weight_packed"] = q.data[0].contiguous()
                    state[f"{key}.weight_scale"] = q.packed_scales[0].view(torch.float8_e4m3fn)
                    state[f"{key}.weight_global_scale"] = q.packed_global[0].reshape(1).clone()
                else:
                    q = pe.quantize_experts_mxfp8(w[None])
                    state[f"{key}.weight"] = q.data[0].view(torch.float8_e4m3fn).contiguous()
                    state[f"{key}.weight_scale"] = q.packed_scales[0].contiguous()
    os.makedirs(path, exist_ok = True)
    save_file(state, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})
    preset, ct_format = PRESET[fmt]
    scheme = preset_name_to_scheme(preset, [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"])
    quant = QuantizationConfig(
        config_groups = {"group_0": scheme},
        format = ct_format,
        quantization_status = "compressed",
        ignore = ["lm_head"],
    )
    q = json.loads(json.dumps(quant.model_dump(exclude_none = False), default = str))
    q["quant_method"] = "compressed-tensors"
    data = model.config.to_dict()
    data["quantization_config"] = q
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(data, f)


def _load(path):
    from transformers import AutoModelForCausalLM

    # No device_map: packed_expert_loader leaves a cpu / disk offload map to transformers.
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype = torch.float32, attn_implementation = "eager"
    )
    model.config._attn_implementation = "eager"
    return model


def _stacks(model):
    """{name: (codes or 16-bit values, scales, global)} of every routed expert stack."""
    out = {}
    for name, p in model.named_parameters():
        if name.rpartition(".")[2] in STACKS and ".experts." in name:
            kwargs = p._init_kwargs() if callable(getattr(p, "_init_kwargs", None)) else {}
            out[name] = (
                torch.Tensor.detach(p).clone(),
                kwargs.get("packed_scales"),
                kwargs.get("packed_global"),
            )
    return out


def _nbytes(stacks):
    return sum(t.numel() * t.element_size() for v in stacks.values() for t in v if t is not None)


def _step(epn, model, rank, world, mesh, rows, n):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    epn.apply_expert_parallel(model, mesh)
    ddp = DistributedDataParallel(model)
    calls = packed_ep._uses()
    loss = packed_ep._sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
    engaged = packed_ep._uses() - calls
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
    return float(value), grads, engaged


def _case_load(
    rank,
    world,
    tmp,
    fmt,
    packed = True,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not packed:
        os.environ["UNSLOTH_PACKED_EXPERTS"] = "0"
    packed_ep._zoo_backend()
    tpn, epn, loader = packed_ep._modules()
    assert loader.install_packed_expert_loading()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        _write_checkpoint(ckpt, fmt)
    dist.barrier()

    reads = []
    original_get = loader._Shards.get

    def get(self, entries, idx, proj, suffix):
        reads.append(idx)
        return original_get(self, entries, idx, proj, suffix)

    loader._Shards.get = get
    out = {"tp_native": tpn.__file__}
    full = _load(ckpt)
    out["full_reads"] = sorted(set(reads))
    want = _stacks(full)
    reads.clear()
    with epn.expert_parallel_loading(mesh) as state:
        model = _load(ckpt)
    out["ep_reads"] = sorted(set(reads))
    loader._Shards.get = original_get
    out["models"] = len(state["models"])
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    have = _stacks(model)
    per = E // world
    lo, hi = rank * per, (rank + 1) * per
    out["types"] = sorted({type(p).__name__ for n, p in model.named_parameters() if n in have})
    out["marks"] = sorted(
        {getattr(p, tpn._EXPERT_PRELOAD, None) for n, p in model.named_parameters() if n in have}
    )
    out["logical"] = sorted(
        {
            tuple(getattr(p, "_original_shape", None) or ())[:1]
            for n, p in model.named_parameters()
            if n in have
        }
    )
    out["sliced"] = set(have) == set(want) and all(
        all((a is None and b is None) or torch.equal(a, b[lo:hi]) for a, b in zip(have[k], want[k]))
        for k in want
    )
    out["held"], out["whole"] = _nbytes(have), _nbytes(want)
    try:
        model(input_ids = torch.zeros(1, 3, dtype = torch.long))
        out["unapplied_forward"] = "ran"
    except RuntimeError as exc:
        out["unapplied_forward"] = str(exc)

    rows = [packed_ep._rows(128, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    full, model = _peft(full), _peft(model)
    lora = lambda m: {k: p.detach().clone() for k, p in m.named_parameters() if "lora_" in k}  # noqa: E731
    want_lora, got_lora = lora(full), lora(model)
    out["lora_equal"] = set(want_lora) == set(got_lora) and all(
        torch.equal(want_lora[k], got_lora[k]) for k in want_lora
    )
    out["full"] = _step(epn, full, rank, world, mesh, rows, n)
    out["sharded"] = _step(epn, model, rank, world, mesh, rows, n)
    return out


def _peft(model):
    """PEFT's own (seeded) expert LoRA init: equal only when it sizes the cut stacks for every expert."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        init_lora_weights = False,
    )
    return get_peft_model(model, config).train()


def _case_bad_header(rank, world, tmp):
    """A malformed expert owned by rank 1 fails the load on every rank, not only on its owner."""
    import torch.distributed as dist
    from safetensors.torch import load_file, save_file
    from torch.distributed.device_mesh import init_device_mesh

    packed_ep._zoo_backend()
    tpn, epn, loader = packed_ep._modules()
    assert loader.install_packed_expert_loading()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        _write_checkpoint(ckpt, "nvfp4")
        path = os.path.join(ckpt, "model.safetensors")
        state = load_file(path)
        key = f"model.layers.1.mlp.experts.{E - 1}.gate_proj.weight_scale"
        state[key] = state[key][:, :1].contiguous()
        save_file(state, path, metadata = {"format": "pt"})
    dist.barrier()
    try:
        with epn.expert_parallel_loading(mesh):
            _load(ckpt)
        return {"loaded": True}
    except RuntimeError as exc:
        return {"loaded": False, "message": str(exc)}


def _check(results, packed = True):
    assert all(r is not None and "error" not in r for r in results), results
    for rank, r in enumerate(results):
        per = E // 2
        assert r["full_reads"] == list(range(E)), r["full_reads"]
        # Only this rank's experts were read from the checkpoint.
        assert r["ep_reads"] == list(range(rank * per, (rank + 1) * per)), r["ep_reads"]
        assert r["models"] == 1, r
        assert r["stats"].get("sharded_expert_stacks") == 4 and r["stats"].get("prequantized"), r[
            "stats"
        ]
        if packed:
            assert r["types"] in (["Nvfp4ExpertParam"], ["Mxfp8ExpertParam"]), r["types"]
            assert r["logical"] == [(E,)], r[
                "logical"
            ]  # PEFT sizes the expert LoRA for every expert
        else:
            assert r["types"] == ["Parameter"], r["types"]
        assert r["marks"] == [(E, 2, (0, 1))], r["marks"]
        assert r["sliced"], "a rank's stacks are not the full load's rows [lo, hi)"
        assert r["held"] * 2 == r["whole"], (r["held"], r["whole"])
        assert "apply_expert_parallel" in r["unapplied_forward"], r["unapplied_forward"]
        assert r["lora_equal"]
        (got_loss, got, engaged), (want_loss, want, want_engaged) = r["sharded"], r["full"]
        if packed:
            assert engaged > 0 and want_engaged > 0, (engaged, want_engaged)
        assert got_loss == want_loss, (got_loss, want_loss)
        assert set(got) == set(want) and any(".experts." in k and "lora_" in k for k in got)
        for k in want:
            assert torch.equal(got[k], want[k]), k


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
    packed_ep._requirements()
    pytest.importorskip("compressed_tensors")
    pytest.importorskip("transformers.core_model_loading")
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


@pytest.mark.parametrize("fmt", ("nvfp4", "mxfp8"))
def test_ep_packed_checkpoint_reads_only_local_experts_and_matches_full_load(fmt):
    _check(_spawn("_case_load", fmt = fmt), packed = True)


def test_ep_packed_checkpoint_kill_switch_16bit_stacks_are_cut_alike():
    _check(_spawn("_case_load", fmt = "nvfp4", packed = False), packed = False)


def test_ep_packed_checkpoint_malformed_remote_expert_fails_every_rank():
    results = _spawn("_case_bad_header")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["loaded"] is False and "does not match the model" in r["message"], r
