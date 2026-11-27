# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""expert_parallel_loading of a quantized checkpoint whose routed expert stacks transformers' own converters build
(gpt-oss native MXFP4, gloo CPU, 2 ranks): each stack is cut to this rank's experts as soon as it is set, so a rank
never holds every expert. Dequantized (no MXFP4 kernels; here to the fp32 load dtype): bitwise the full load's rows, and one LoRA
step after apply_expert_parallel matches the full load's (loss and every gradient). Kept packed (zoo's
Mxfp4ExpertParam): the codes and scales are the full stack's rows, marked, logical shape whole, and
apply_expert_parallel shards it."""

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from transformers5_loading import require_core_model_loading  # noqa: E402

E, H, I = 4, 64, 64
STACKS = ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias")
_SEQ = (7, 5)


def _modules():
    from unsloth_models_loader import load_models_module
    return load_models_module("tp_native"), load_models_module("ep_native")


def _write_checkpoint(path):
    """gpt-oss's released MXFP4 layout: ``experts.{gate_up,down}_proj_{blocks,scales}`` and 16-bit biases."""
    from safetensors.torch import save_file
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        vocab_size = 128,
        hidden_size = H,
        intermediate_size = I,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        num_local_experts = E,
        num_experts_per_tok = 2,
        layer_types = ["full_attention"] * 2,
        tie_word_embeddings = False,
    )
    with torch.device("meta"):
        skeleton = GptOssForCausalLM(config)
    g = torch.Generator().manual_seed(0)
    state = {}
    for key, p in skeleton.state_dict().items():
        if key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            continue
        state[key] = (torch.randn(p.shape, generator = g) * 0.05).to(torch.bfloat16)
    for layer in range(2):
        pre = f"model.layers.{layer}.mlp.experts"
        for stack, (rows, cols) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
            state[f"{pre}.{stack}_blocks"] = torch.randint(
                0, 256, (E, rows, cols // 32, 16), dtype = torch.uint8, generator = g
            )
            state[f"{pre}.{stack}_scales"] = torch.randint(
                122, 127, (E, rows, cols // 32), dtype = torch.uint8, generator = g
            )
    os.makedirs(path, exist_ok = True)
    save_file(state, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})
    data = config.to_dict()
    data["quantization_config"] = {
        "quant_method": "mxfp4",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(data, f)


def _keep_packed():
    """zoo's keep-packed branch of transformers' MXFP4 converter (what Unsloth installs where grouped_mm decodes)."""
    import transformers.integrations.mxfp4 as mx
    from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam
    from unsloth_zoo.temporary_patches import moe_utils
    from unsloth_zoo.temporary_patches.mxfp4 import (
        patch_mxfp4_packed_expert_shape_check,
        patch_peft_param_wrapper_mxfp4,
    )

    patch_mxfp4_packed_expert_shape_check()  # packed (E, N, G, 16) blocks against the logical stack shape
    patch_peft_param_wrapper_mxfp4()  # PEFT's expert LoRA over the packed stack's logical shape
    moe_utils.patch_param_wrapper_for_moe()

    def dequantize_convertops(blocks, scales, *args, **kwargs):
        return Mxfp4ExpertParam(blocks, mxfp4_scales = scales)

    mx.dequantize_convertops = dequantize_convertops


def _load(path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype = torch.float32, attn_implementation = "eager"
    )
    model.config._attn_implementation = "eager"
    return model


def _stacks(model):
    out = {}
    for name, p in model.named_parameters():
        if name.rpartition(".")[2] in STACKS and ".experts." in name:
            extra = getattr(p, "mxfp4_scales", None)
            out[name] = (torch.Tensor.detach(p).clone(), None if extra is None else extra.clone())
    return out


def _held(stacks):
    return sum(t.numel() * t.element_size() for v in stacks.values() for t in v if t is not None)


def _lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "v_proj"],
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        init_lora_weights = False,
    )
    return get_peft_model(model, config).train()


def _step(epn, model, rank, world, mesh):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    rows = [
        torch.randint(0, 128, (2, _SEQ[r]), generator = torch.Generator().manual_seed(20 + r))
        for r in range(world)
    ]
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


def _case(rank, world, tmp, packed):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    tpn, epn = _modules()
    if packed:
        _keep_packed()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        _write_checkpoint(ckpt)
    dist.barrier()
    full = _load(ckpt)
    want = _stacks(full)
    with epn.expert_parallel_loading(mesh) as state:
        model = _load(ckpt)
    have = _stacks(model)
    per = E // world
    lo, hi = rank * per, (rank + 1) * per
    params = dict(model.named_parameters())
    out = {
        "models": len(state["models"]),
        "stats": dict(model.__dict__.get("_unsloth_tp_load_stats") or {}),
        "types": sorted({type(params[n]).__name__ for n in have if not n.endswith("_bias")}),
        "marks": sorted({getattr(params[n], tpn._EXPERT_PRELOAD, None) for n in have}),
        "sliced": set(have) == set(want)
        and all(
            all(
                (a is None and b is None) or torch.equal(a, b[lo:hi])
                for a, b in zip(have[k], want[k])
            )
            for k in want
        ),
        "held": _held(have),
        "whole": _held(want),
    }
    # Saving the whole model before apply would write only this rank's experts: refused before anything is written.
    dest = tempfile.mkdtemp()
    with open(os.path.join(dest, "keep.txt"), "w") as f:
        f.write("x")
    try:
        model.save_pretrained(dest)
        out["unapplied_save"] = "ran"
    except RuntimeError as exc:
        out["unapplied_save"] = str(exc)
    out["save_dir"] = sorted(os.listdir(dest))
    # A one-token no-grad eval forward (zoo's fused gpt-oss decode reads the stacks directly) is refused too.
    try:
        with torch.no_grad():
            model.eval()(input_ids = torch.zeros(1, 1, dtype = torch.long))
        out["unapplied_decode"] = "ran"
    except RuntimeError as exc:
        out["unapplied_decode"] = str(exc)
    model.train()
    if packed:
        out["logical"] = sorted(
            {
                tuple(params[n]._original_shape)[0]
                for n in have
                if hasattr(params[n], "_original_shape")
            }
        )
        # A forward before apply is refused up front; PEFT sizes the expert LoRA as for the full load; PEFT and
        # apply_expert_parallel accept the cut packed stacks (no training step: zoo's grouped decode is GPU only).
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
        epn.apply_expert_parallel(model, mesh)
        out["applied"] = sorted(
            {
                tuple(p.shape)[0]
                for n, p in model.named_parameters()
                if n.endswith(("gate_up_proj", "down_proj"))
            }
        )
        return out
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
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("requires a PyTorch build with Gloo")
    require_core_model_loading()
    pytest.importorskip("peft")
    try:
        from transformers import GptOssConfig  # noqa: F401
        import transformers.integrations.mxfp4 as mx
    except ImportError:
        pytest.skip("transformers without gpt-oss MXFP4")
    if not hasattr(mx, "dequantize_convertops"):
        pytest.skip("transformers without the MXFP4 weight converter")
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


def _common(results, kind):
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["models"] == 1, r
        assert r["stats"].get("sharded_expert_stacks") == 8 and r["stats"].get("prequantized"), r[
            "stats"
        ]
        assert r["types"] == [kind], r["types"]
        assert r["marks"] == [(E, 2, (0, 1))], r["marks"]
        assert r["sliced"], "a rank's stacks are not the full load's rows [lo, hi)"
        assert r["held"] * 2 == r["whole"], (r["held"], r["whole"])
        assert "apply_expert_parallel" in r["unapplied_save"], r["unapplied_save"]
        assert r["save_dir"] == ["keep.txt"], r["save_dir"]
        assert "apply_expert_parallel" in r["unapplied_decode"], r["unapplied_decode"]


def test_ep_mxfp4_dequantized_stacks_cut_on_load_match_full_load():
    results = _spawn("_case", packed = False)
    _common(results, "Parameter")
    for r in results:
        assert "apply_expert_parallel" in r["unapplied_forward"], r["unapplied_forward"]
        assert r["lora_equal"]
        (got_loss, got), (want_loss, want) = r["sharded"], r["full"]
        assert got_loss == want_loss, (got_loss, want_loss)
        assert set(got) == set(want) and any(".experts." in k and "lora_" in k for k in got)
        for k in want:
            assert torch.equal(got[k], want[k]), k


def test_ep_mxfp4_packed_stacks_cut_on_load():
    pytest.importorskip("unsloth_zoo.mxfp4_dequant")
    results = _spawn("_case", packed = True)
    _common(results, "Mxfp4ExpertParam")
    for r in results:
        assert r["logical"] == [E], r["logical"]
        assert r["applied"] == [E // 2], r["applied"]
        assert "apply_expert_parallel" in r["unapplied_forward"], r["unapplied_forward"]
        assert r["lora_equal"]


def test_converter_cut_setter_is_installed_once_and_skips_offloading_maps():
    """One permanent set_param_for_module wrapper (overlapping loads never restore it out of order); a device map
    that offloads keeps the whole stacks for its dispatch."""
    require_core_model_loading()
    import transformers.core_model_loading as loading

    tpn, _ = _modules()
    assert tpn._install_converter_cut_setter()
    first = loading.set_param_for_module
    assert tpn._install_converter_cut_setter() and loading.set_param_for_module is first
    assert getattr(first, "_unsloth_ep_converter_cut", False)
    keeps = tpn._keeps_experts_on_device
    assert keeps(None) and keeps({"": "cuda:1"}) and keeps({"": 0}) and keeps("cuda:0")
    assert (
        not keeps({"": "cpu"})
        and not keeps({"model.layers.1": "disk", "": 0})
        and not keeps("auto")
    )
    assert not keeps({"": torch.device("cpu")})


def test_held_bytes_counts_mxfp4_scales():
    """Zoo's Mxfp4ExpertParam keeps its scales beside the codes: the per-rank byte counters include them."""
    zoo = pytest.importorskip("unsloth_zoo.mxfp4_dequant")
    tpn, _ = _modules()
    blocks = torch.zeros(E, 2 * I, H // 32, 16, dtype = torch.uint8)
    scales = torch.zeros(E, 2 * I, H // 32, dtype = torch.uint8)
    param = zoo.Mxfp4ExpertParam(blocks, mxfp4_scales = scales)
    assert tpn._held_bytes(param) == blocks.numel() + scales.numel()


def test_quantizing_load_counts_packed_scales():
    """The quantize-on-load setter (expert parallel) counts a packed stack's scales in loaded and kept bytes, so
    a cut never reports less than the rank holds."""
    zoo = pytest.importorskip("unsloth_zoo.mxfp4_dequant")
    require_core_model_loading()
    import transformers.core_model_loading as loading
    from transformers import GptOssConfig, GptOssForCausalLM
    from unsloth_zoo.temporary_patches.mxfp4 import patch_peft_param_wrapper_mxfp4

    patch_peft_param_wrapper_mxfp4()
    tpn, _ = _modules()
    config = GptOssConfig(
        vocab_size = 128,
        hidden_size = H,
        intermediate_size = I,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        num_local_experts = E,
        num_experts_per_tok = 2,
        layer_types = ["full_attention"],
    )
    pre = "model.layers.0.mlp.experts"
    values = {}
    for stack, (rows, cols) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
        blocks = torch.zeros(E, rows, cols // 32, 16, dtype = torch.uint8)
        scales = torch.full((E, rows, cols // 32), 127, dtype = torch.uint8)
        values[f"{pre}.{stack}"] = zoo.Mxfp4ExpertParam(blocks, mxfp4_scales = scales)
        values[f"{pre}.{stack}_bias"] = torch.nn.Parameter(torch.zeros(E, rows))

    def plain_set(model, target_name, value, *rest, **more):
        path, _, pname = target_name.rpartition(".")
        model.get_submodule(path)._parameters[pname] = value

    def original(cls, name, *args, **kwargs):
        with torch.device("meta"):
            model = cls(config)
        for key, value in values.items():
            loading.set_param_for_module(model, key, value, None, None)
        return model

    class Mesh:
        mesh = torch.arange(2)

        def size(self):
            return 2

        def get_local_rank(self):
            return 0

    real = loading.set_param_for_module
    loading.set_param_for_module = plain_set
    try:
        state = {"experts_only": True, "models": [], "plan": None}
        model = tpn._tp_load_quantizing(original, GptOssForCausalLM, "x", (), {}, Mesh(), state)
    finally:
        loading.set_param_for_module = real
    stats = model.__dict__["_unsloth_tp_load_stats"]
    whole = sum(tpn._held_bytes(v) for v in values.values())
    held = sum(tpn._held_bytes(p) for p in model.model.layers[0].mlp.experts._parameters.values())
    assert stats["sharded_expert_stacks"] == 4, stats
    assert stats["loaded_bytes"] == whole and stats["kept_bytes"] == held == whole // 2, (
        stats,
        whole,
        held,
    )


class _Mesh:
    mesh = torch.arange(2)

    def size(self):
        return 2

    def get_local_rank(self):
        return 0


def test_tensor_parallel_cut_uses_tensor_parallel_eligibility(monkeypatch):
    """The load-time cut asks the same eligibility question apply asks: packed (zoo stacks, swapped LoRA factors)
    only under expert parallel. Tensor parallel 4-bit loading keeps a block apply_tensor_parallel would refuse."""
    tpn, _ = _modules()
    asked = []

    def problem(
        block,
        tp,
        packed = True,
    ):
        asked.append(packed)
        return None if packed else "tensor parallel cannot regather this block"

    monkeypatch.setattr(tpn, "expert_shard_problem", problem)
    module = torch.nn.Module()
    module.num_experts = E
    module.gate_up_proj = torch.nn.Parameter(torch.zeros(E, H, 2 * I))
    module.down_proj = torch.nn.Parameter(torch.zeros(E, I, H))
    assert tpn._cut_expert_stacks(module, None, 0, 2, (0, 1), experts_only = False) == (0, 0)
    assert tuple(module.gate_up_proj.shape)[0] == E and asked == [False]
    cut, _ = tpn._cut_expert_stacks(module, None, 0, 2, (0, 1), experts_only = True)
    assert cut == 2 and tuple(module.gate_up_proj.shape)[0] == E // 2 and asked == [False, True]


def test_ep_packed_load_runs_under_the_sharded_load_memory_guard(monkeypatch):
    """transformers' load inside _ep_packed_load materializes synchronously (no async prefetch of every layer's
    blocks onto the card) and with the rank-sized allocator warmup, as the quantize-on-load path does."""
    from transformers import PreTrainedModel

    from unsloth_models_loader import load_models_module

    tpn, _ = _modules()
    fsdp2_load = load_models_module("fsdp2_load")
    monkeypatch.delenv(fsdp2_load._ASYNC_LOAD_ENV, raising = False)
    seen = []

    def original(cls, name, *args, **kwargs):
        seen.append(os.environ.get(fsdp2_load._ASYNC_LOAD_ENV))
        return torch.nn.Linear(2, 2)

    state = {"models": [], "active": False}
    tpn._ep_packed_load(original, PreTrainedModel, "x", (), {}, _Mesh(), state)
    assert seen == ["1"] and fsdp2_load._ASYNC_LOAD_ENV not in os.environ
