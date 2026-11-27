# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 gradient accumulation, sharded LoRA layout with trainable embeddings / lm_head, on a
gloo CPU world of 2.

UNSLOTH_FSDP2_GA_SYNC=auto picks "skip" for a sharded LoRA model. A unit whose own parameters are mostly
trainable (the root holding trainable embed_tokens / lm_head, plain or as modules_to_save copies) must
still reduce-scatter every micro-batch: skipping it keeps its whole fp32 gradient (2 * vocab * hidden * 4
bytes) on every rank between micro-batches instead of the 1 / world shard. LoRA-only block units keep
skipping, and the gradients match the plain "skip" path."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_ga_sync_gloo import _unsharded_grad_bytes  # noqa: E402
from test_fsdp2_native_layout_agree_gloo import REPO_ID, _init, _spawn  # noqa: E402
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

MICRO = 3


def _model():
    """Tiny Llama whose frozen body outweighs the embed + head tables, as in a real model (Llama-8B: the
    tables are 1.05B of 8B), so the model as a whole is not mostly trainable and auto picks "skip"."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.update(
        {
            "vocab_size": 1024,
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "tie_word_embeddings": False,
        }
    )
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32).eval()


def _peft(variant):
    from peft import LoraConfig, get_peft_model

    extra = {"modules_to_save": ["embed_tokens", "lm_head"]} if variant == "modules_to_save" else {}
    model = get_peft_model(
        _model(), LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", **extra)
    )
    if variant == "trainable":  # trainable embeddings, as with embedding_learning_rate
        for module in (model.get_input_embeddings(), model.get_output_embeddings()):
            module.weight.requires_grad_(True)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02, generator = g)
    return model


def _mostly(native, variant):
    return native._mostly_trainable(_peft(variant))


def _run(native, variant, mode):
    from torch.distributed.tensor import DTensor

    os.environ["UNSLOTH_FSDP2_GA_SYNC"] = mode  # read on every no_sync entry
    rank = int(os.environ["RANK"])
    model = _peft(variant)
    native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, fit_layout = False))
    held = []
    for i in range(MICRO):
        ids = torch.randint(
            3, 1024, (2, 16), generator = torch.Generator().manual_seed(100 * rank + i)
        )
        ctx = native._no_grad_sync(model) if i < MICRO - 1 else torch.enable_grad()
        with ctx:
            model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        held.append(_unsharded_grad_bytes(model))
    grads = {
        n: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    table_bytes = sum(
        p.numel() * 4
        for n, p in model.named_parameters()
        if p.requires_grad and ("embed_tokens" in n or "lm_head" in n)
    )
    return {
        "held": held,
        "grads": grads,
        "fits": model._unsloth_fsdp2_layout["fits"],
        "table_bytes": table_bytes,
    }


def _nested_units(native):
    """A mostly trainable unit below the root is found too, a frozen one and the frozen root are not."""
    from torch import nn
    from torch.distributed.fsdp import fully_shard

    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(8, 8))
    for p in list(model[0].parameters()) + list(model[2].parameters()):
        p.requires_grad_(False)
    fully_shard(model[0])
    fully_shard(model[1])
    fully_shard(model)
    return [
        i for i, m in enumerate(model) if any(u is m for u in native._mostly_trainable_units(model))
    ], [u is model for u in native._mostly_trainable_units(model)]


def _worker(rank, world, port, out, path):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        assert os.path.realpath(native.__file__).startswith(os.path.realpath(REPO)), native.__file__
        res = {
            ("mostly_trainable", variant): _mostly(native, variant)
            for variant in ("trainable", "modules_to_save")
        }
        res.update(
            {
                (variant, mode): _run(native, variant, mode)
                for variant in ("trainable", "modules_to_save")
                for mode in ("auto", "skip", "each")
            }
        )
        res["nested"] = _nested_units(native)
        if rank == 0:  # a file: a large queued payload blocks the child's exit (join deadlock)
            torch.save(res, path)
            out.put("ok")
    except Exception as exc:
        import traceback
        out.put({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def test_auto_reduces_mostly_trainable_units_every_micro_batch(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    path = str(tmp_path / "ga_trainable_units.pt")
    (payload,) = _spawn(_worker, path)
    assert payload == "ok", payload["error"]
    res = torch.load(path)
    assert res.pop("nested") == ([1], [False]), "only the trainable child unit"
    for variant in ("trainable", "modules_to_save"):
        auto, skip, each = (res[(variant, m)] for m in ("auto", "skip", "each"))
        print(
            "GA_HELD",
            variant,
            "tables",
            auto["table_bytes"],
            "auto",
            auto["held"],
            "skip",
            skip["held"],
            "each",
            each["held"],
        )
        assert auto["fits"] is False and auto["table_bytes"] > 0
        assert res[("mostly_trainable", variant)] is False
        # Explicit skip keeps every whole gradient, the trainable tables included.
        assert all(b >= skip["table_bytes"] for b in skip["held"][:-1]), skip["held"]
        # auto: the tables reduce-scatter every micro-batch; only the LoRA units' gradients are held.
        lora_bytes = skip["held"][0] - skip["table_bytes"]
        assert all(0 < b <= lora_bytes for b in auto["held"][:-1]), (auto["held"], lora_bytes)
        assert each["held"][:-1] == [0] * (MICRO - 1), each["held"]
        g_auto, g_skip, g_each = auto["grads"], skip["grads"], each["grads"]
        assert g_skip and g_auto.keys() == g_skip.keys() == g_each.keys()
        assert any("embed_tokens" in n for n in g_auto) and any("lm_head" in n for n in g_auto)
        for n in g_skip:
            torch.testing.assert_close(g_auto[n], g_skip[n], rtol = 1e-5, atol = 1e-7, msg = n)
            torch.testing.assert_close(g_each[n], g_skip[n], rtol = 1e-5, atol = 1e-7, msg = n)
