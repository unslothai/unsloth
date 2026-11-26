# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_EP_EXPERT_ROWS chunks match one experts call (loss, logits, grads) and save fewer bytes."""

import multiprocessing
import os

if os.environ.get("UNSLOTH_TEST_RANKS_ON_GPU") != "1" and (
    multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port, _import_checkout_unsloth  # noqa: E402

import transformers  # noqa: E402

SEQ, ROWS = 64, 16


def _tiny_moe(device):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_experts = 8,
        num_experts_per_tok = 2,
        decoder_sparse_step = 1,
        norm_topk_prob = True,
        router_aux_loss_coef = 0.0,
    )
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(config).to(device = device, dtype = torch.float32)


def _saved_bytes(fn):
    seen = {}

    def pack(t):
        seen[(t.untyped_storage().data_ptr(), t.device)] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(seen.values())


def _run(rank, device):
    from peft import LoraConfig, get_peft_model

    from unsloth.models import ep_native

    model = _tiny_moe(device)
    torch.manual_seed(1)
    model = get_peft_model(
        model,
        LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False),
    )
    ep_native.apply_expert_parallel(model)
    assert ep_native.is_expert_parallel(model)
    ids = torch.randint(0, 128, (1, SEQ), generator = torch.Generator().manual_seed(rank)).to(device)
    out = {}
    for arm in ("one_call", "chunked", "gc_one_call", "gc_chunked"):
        if arm == "gc_one_call":
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            )
            model.enable_input_require_grads()
            model.train()
        if arm.endswith("chunked"):
            os.environ["UNSLOTH_EP_EXPERT_ROWS"] = str(ROWS)
        else:
            os.environ.pop("UNSLOTH_EP_EXPERT_ROWS", None)
        model.zero_grad(set_to_none = True)
        res, saved = _saved_bytes(lambda: model(input_ids = ids, labels = ids))
        res.loss.backward()
        with torch.no_grad():
            nograd = model(input_ids = ids).logits
        out[arm] = {
            "loss": res.loss.detach().cpu(),
            "logits": res.logits.detach().cpu(),
            "nograd": nograd.cpu(),
            "saved": saved,
            "grads": {
                n: p.grad.detach().cpu().clone()
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            },
        }
    os.environ.pop("UNSLOTH_EP_EXPERT_ROWS", None)
    out["chunked_calls"] = _chunked_calls(model)
    out["expert_lora"] = [n for n in out["one_call"]["grads"] if ".experts." in n and "lora_" in n]
    return out


def _chunked_calls(model):
    shards = [m.__dict__.get("_unsloth_tp_expert_shard") for m in model.modules()]
    return sum(getattr(s, "chunked_calls", 0) for s in shards if s is not None)


def _worker(rank, world, port, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in ("UNSLOTH_EP_SIZE", "UNSLOTH_PARALLEL", "UNSLOTH_EP_EXPERT_ROWS"):
        os.environ.pop(key, None)
    torch.set_num_threads(1)
    import torch.distributed as dist

    try:
        error = _import_checkout_unsloth()
        if error is not None:
            res = {"skip": error}
        else:
            on_gpu = os.environ.get("UNSLOTH_TEST_RANKS_ON_GPU") == "1"
            device = f"cuda:{rank}" if on_gpu else "cpu"
            if on_gpu:
                torch.cuda.set_device(rank)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            else:
                from unsloth import context_parallel as cpm
                cpm.DEVICE_TYPE_TORCH = "cpu"
            dist.init_process_group("nccl" if on_gpu else "gloo")
            res = _run(rank, device)
            res["backend"] = dist.get_backend()
    except Exception as e:
        import traceback
        res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    torch.save(res, os.path.join(tmp, f"r{rank}.pt"))


def _spawn(world = 2):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, port, tmp)) for r in range(world)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
            if p.is_alive():
                p.kill()
        results = []
        for r in range(world):
            path = os.path.join(tmp, f"r{r}.pt")
            results.append(
                torch.load(path, weights_only = False)
                if os.path.exists(path)
                else {"error": f"rank {r} died"}
            )
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


@pytest.mark.skipif(
    int(transformers.__version__.split(".")[0]) < 5,
    reason = "expert parallel needs fused MoE expert stacks (transformers >= 5)",
)
def test_ep_expert_rows_chunks_match_one_call_and_save_less():
    for rank, res in enumerate(_spawn()):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["chunked_calls"] > 0, ("the chunked path never ran", rank)
        assert res["expert_lora"], rank
        one, chunked = res["one_call"], res["chunked"]
        torch.testing.assert_close(chunked["loss"], one["loss"], rtol = 1e-5, atol = 1e-6)
        torch.testing.assert_close(chunked["logits"], one["logits"], rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(chunked["nograd"], one["nograd"], rtol = 1e-5, atol = 1e-5)
        assert chunked["grads"].keys() == one["grads"].keys()
        for name in res["expert_lora"]:
            assert one["grads"][name].abs().sum() > 0, name
        for name, g in one["grads"].items():
            torch.testing.assert_close(chunked["grads"][name], g, rtol = 1e-4, atol = 1e-6, msg = name)
        assert chunked["saved"] < one["saved"], (rank, chunked["saved"], one["saved"])
        for name, g in res["gc_one_call"]["grads"].items():
            torch.testing.assert_close(
                res["gc_chunked"]["grads"][name], g, rtol = 1e-4, atol = 1e-6, msg = name
            )
            torch.testing.assert_close(g, one["grads"][name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_ep_expert_rows_env_is_validated():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
    from unsloth_models_loader import load_models_module

    epn = load_models_module("ep_native")
    old = os.environ.pop("UNSLOTH_EP_EXPERT_ROWS", None)
    try:
        assert epn._ep_expert_rows() is None
        os.environ["UNSLOTH_EP_EXPERT_ROWS"] = "0"
        assert epn._ep_expert_rows() is None
        os.environ["UNSLOTH_EP_EXPERT_ROWS"] = "4096"
        assert epn._ep_expert_rows() == 4096
        for bad in ("-1", "lots"):
            os.environ["UNSLOTH_EP_EXPERT_ROWS"] = bad
            with pytest.raises(ValueError, match = "UNSLOTH_EP_EXPERT_ROWS"):
                epn._ep_expert_rows()
    finally:
        os.environ.pop("UNSLOTH_EP_EXPERT_ROWS", None)
        if old is not None:
            os.environ["UNSLOTH_EP_EXPERT_ROWS"] = old
