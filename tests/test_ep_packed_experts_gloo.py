# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel (gloo CPU, 2 ranks, fp32) over unsloth_zoo's frozen packed expert stacks (NVFP4, MXFP8, MXFP4)
vs one process: loss, every gradient and the saved adapter after one SGD step.

The stacks are cut by expert index: the codes, block scales and per-expert global scales are all expert-major.
Also covered: UNSLOTH_MOE_EXPERTS_QUANTIZE on stacks expert parallel loading already cut to this rank's experts
(the quantized stack keeps the cut's mark, so the block still shards), and tensor parallel still refusing packed
stacks (it cannot regather them)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts
    os.environ.setdefault(
        "UNSLOTH_ALLOW_CPU", "1"
    )  # unsloth_zoo's import on a host without an accelerator
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

FORMATS = ("nvfp4", "mxfp8", "mxfp4")
_SEQ = (7, 5)


def _requirements():
    pe = pytest.importorskip("unsloth_zoo.packed_experts")
    pytest.importorskip("peft")
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts  # noqa: F401
    except ImportError:
        pytest.skip("transformers 5 fused expert stacks")
    if not hasattr(pe, "quantize_experts_mxfp8"):
        pytest.skip("unsloth_zoo without packed NVFP4 / MXFP8 expert stacks")
    try:
        torch._grouped_mm(
            torch.randn(4, 32), torch.randn(2, 32, 16), offs = torch.tensor([2, 4], dtype = torch.int32)
        )
    except Exception:
        pytest.skip("no CPU torch._grouped_mm")


def _zoo_backend(device = "cpu"):
    """unsloth_zoo's MoE backend on the transformers 5 Qwen3-MoE experts, as Unsloth installs it."""
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mod
    from unsloth_zoo.temporary_patches import moe_utils as mu
    from unsloth_zoo.temporary_patches import mxfp4 as mx

    mx.patch_peft_param_wrapper_mxfp4()
    assert mu.patch_param_wrapper_for_moe()
    if device == "cpu":
        mu._TORCH_GROUPED_MM_SUPPORTED = True

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return mu.forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)

    mod.Qwen3MoeExperts.forward = forward


_DECODES = [0]


def _uses():
    """Packed expert calls so far: zoo's packed grouped op (NVFP4 / MXFP8), or an MXFP4 stack decode."""
    import unsloth_zoo.packed_experts as pe
    from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam

    if not getattr(Mxfp4ExpertParam.dequantize, "_counted", False):
        original = Mxfp4ExpertParam.dequantize

        def dequantize(self, *args, **kwargs):
            _DECODES[0] += 1
            return original(self, *args, **kwargs)

        dequantize._counted = True
        Mxfp4ExpertParam.dequantize = dequantize
    return pe.PACKED_EXPERT_STATS["calls"] + _DECODES[0]


def _modules():
    from unsloth_models_loader import load_models_module
    return (
        load_models_module("tp_native"),
        load_models_module("ep_native"),
        load_models_module("packed_expert_loader"),
    )


def _base(seed = 0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        num_experts = 4,
        num_experts_per_tok = 2,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    return Qwen3MoeForCausalLM(config).float()


def _quantize(model, fmt):
    import unsloth_zoo.packed_experts as pe

    if fmt == "bf16_stacks":  # the same backend on the plain 16-bit stacks: the precision baseline
        return
    if fmt.endswith("_dense"):  # the packed values, dequantized to plain stacks
        _quantize(model, fmt[: -len("_dense")])
        for module in model.modules():
            for name in ("gate_up_proj", "down_proj"):
                p = module._parameters.get(name)
                if p is not None and pe.is_packed_expert_param(p):
                    module._parameters[name] = torch.nn.Parameter(
                        p.dequantize(torch.float32), requires_grad = False
                    )
        return
    if fmt in ("nvfp4", "mxfp4"):
        assert pe.quantize_moe_expert_stacks(model, fmt) == 4
        return
    for module in model.modules():
        for name in ("gate_up_proj", "down_proj"):
            p = module._parameters.get(name)
            if p is not None and p.dim() == 3:
                module._parameters[name] = pe.quantize_experts_mxfp8(p.data, p.dtype)


def _lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    model = get_peft_model(
        model,
        LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            init_lora_weights = False,
        ),
    )
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    return model.train()


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


def _adapter(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(path, "adapter_model.safetensors"))


def _packed_names(model):
    import unsloth_zoo.packed_experts as pe
    return sorted(n for n, p in model.named_parameters() if pe.is_packed_expert_param(p))


def _case_packed(
    rank,
    world,
    tmp,
    fmt,
    preload = False,
    device = "cpu",
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    _zoo_backend(device)
    tpn, epn, loader = _modules()

    mesh = init_device_mesh(device, (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _base().to(device)
    _quantize(ref, fmt)
    ref = _lora(ref)
    rows = [tuple(t.to(device) for t in _rows(128, r)) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    calls = _uses()
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    ref_engaged = _uses() - calls
    want = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}
    with torch.no_grad():
        want_loss = sum(float(_sum_loss(ref, i, lab)) for i, lab in rows) / n
        for p in ref.parameters():
            if p.grad is not None:
                p.sub_(0.5 * p.grad)
    want_dir = os.path.join(tmp, f"ref_{rank}")
    ref.save_pretrained(want_dir)

    model = _base().to(device)
    if preload:
        # What expert parallel loading leaves: each 16-bit stack already cut to this rank's experts, marked; then
        # from_pretrained's UNSLOTH_MOE_EXPERTS_QUANTIZE pass quantizes the cut stacks.
        ranks = tuple(mesh.mesh.tolist())
        for module in model.modules():
            for name in ("gate_up_proj", "down_proj"):
                p = module._parameters.get(name)
                if p is not None and p.dim() == 3:
                    module._parameters[name] = tpn._preload_expert_plain(
                        module, p, rank, world, ranks
                    )
        os.environ["UNSLOTH_MOE_EXPERTS_QUANTIZE"] = fmt
        try:
            quantized = loader.quantize_moe_experts_if_requested(model)
        finally:
            os.environ.pop("UNSLOTH_MOE_EXPERTS_QUANTIZE", None)
        assert quantized == 4, quantized
    else:
        _quantize(model, fmt)
    model = _lora(model)
    try:
        epn.apply_expert_parallel(model, mesh)
    except Exception as exc:  # noqa: BLE001
        return {"refused": f"{type(exc).__name__}: {exc}"}
    packed = _packed_names(model)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters() if n in packed}
    whole = dict(ref.named_parameters())
    # Bitwise: this rank's decoded experts are the one-process stack's rows [lo, hi) (codes and scales cut alike).
    bitwise = all(
        torch.equal(
            p.dequantize(torch.float32), whole[n].dequantize(torch.float32)[2 * rank : 2 * rank + 2]
        )
        for n, p in model.named_parameters()
        if n in packed
    )
    ddp = DistributedDataParallel(
        model, device_ids = [torch.cuda.current_device()] if device == "cuda" else None
    )
    calls = _uses()
    loss = _sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
    engaged = _uses() - calls
    epn.ep_sync_expert_grads(model)
    value = loss.detach() / world
    dist.all_reduce(value)
    group = epn._mesh(model).get_group()
    got = {
        k: (epn._gather_expert(p, group, p.grad) if getattr(p, epn._EP, None) else p.grad)
        .detach()
        .clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(0.5 * p.grad)
    got_dir = os.path.join(tmp, "ep_adapter")
    model.save_pretrained(got_dir)
    dist.barrier()
    return {
        "loss": (float(value), want_loss),
        "grads": ({k: v.cpu() for k, v in got.items()}, {k: v.cpu() for k, v in want.items()}),
        "packed": packed,
        "shapes": shapes,
        "engaged": (engaged, ref_engaged),
        "bitwise": bitwise,
        "tags": sorted(
            {getattr(p, epn._EP)[0] for n, p in model.named_parameters() if n in packed}
        ),
        "adapter": (_adapter(got_dir), _adapter(want_dir)) if rank == 0 else None,
    }


def _case_tp_refuses(rank, world, tmp, fmt):
    _zoo_backend()
    tpn, _, _ = _modules()
    model = _base()
    _quantize(model, fmt)
    block = tpn.moe_blocks_with_experts(model)[0]
    return {
        "ep": tpn.expert_shard_problem(block, world),
        "tp": tpn.expert_shard_problem(block, world, packed = False),
    }


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


def _check(results):
    assert all(r is not None and "error" not in r and "refused" not in r for r in results), results
    for rank, r in enumerate(results):
        assert len(r["packed"]) == 4, r["packed"]
        assert r["tags"] == ["4bit"], r[
            "tags"
        ]  # frozen quantized stacks: never gathered, never in the adapter
        for name, shape in r["shapes"].items():
            assert shape[0] == 2, (name, shape)  # 2 of the 4 experts per rank
        assert r["bitwise"], "a rank's packed experts are not the one-process stack's rows"
        engaged, ref_engaged = r["engaged"]
        assert engaged > 0 and ref_engaged > 0, r[
            "engaged"
        ]  # the packed grouped op ran, not a dense copy
        # The packed GEMM groups experts in chunks of the stack it holds (4 in one process, 2 per rank here), so
        # fp32 sums round differently: about 1e-6 relative on the loss, the size of the gap between the packed
        # path and the same values as plain stacks in one process. A mis-cut scale or expert is orders larger.
        got, want = r["loss"]
        assert got == pytest.approx(want, rel = 1e-5), r["loss"]
        got, want = r["grads"]
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        assert any(".experts." in k and "lora_" in k for k in want)
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol = 1e-3, atol = 5e-5, msg = k)
    got, want = results[0]["adapter"]
    assert set(got) == set(want) and any(".experts." in k for k in want), sorted(
        set(got) ^ set(want)
    )[:6]
    for k in want:
        assert got[k].shape == want[k].shape, k
        torch.testing.assert_close(got[k], want[k], rtol = 1e-4, atol = 1e-5, msg = k)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ep_packed_experts_match_one_process(fmt):
    _check(_spawn("_case_packed", fmt = fmt))


@pytest.mark.parametrize("fmt", ("nvfp4", "mxfp4"))
def test_ep_experts_quantize_after_sharded_load_matches_one_process(fmt):
    _check(_spawn("_case_packed", fmt = fmt, preload = True))


def test_tp_still_refuses_packed_experts():
    for r in _spawn("_case_tp_refuses", fmt = "nvfp4"):
        assert r is not None and "error" not in r, r
        assert r["ep"] is None, r
        assert r["tp"] is not None and "expert parallel only" in r["tp"], r
