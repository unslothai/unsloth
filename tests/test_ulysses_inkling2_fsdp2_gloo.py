# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inkling 2 Ulysses context parallelism over native FSDP2 on the same two ranks (UNSLOTH_PARALLEL="cp=2,fsdp=1":
the parameters are sharded over the cp group, each rank holds its own sequence shard), through the Unsloth-patched
SFTTrainer on two gloo CPU ranks against one process in float32: losses of two SGD steps and every step-0 gradient.

This is the layout that lets a model whose weights do not fit one GPU train under cp (Inkling-2-Mini QLoRA on
2x T4): the FSDP2 sharded load and shard_model cut the weights over the same ranks that split the sequence. The
tiny models are the ones of tests/test_ulysses_inkling2_gloo.py (sliding window and relative bias across the shard
seam, kernel 4 short convolutions with a halo, the Mini's looped layers, the MoE)."""

import dataclasses
import multiprocessing
import os
import sys
import tempfile
import time

import pytest
import torch

pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")

TESTS = os.path.dirname(os.path.abspath(__file__))
SEQ = 22  # 11 tokens per rank, not a multiple of the convolution kernel


def _canon(name):
    for x in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."):
        name = name.replace(x, "")
    return name


def _train(
    loop,
    lora,
    gc,
    world,
    rank = 0,
):
    """SFTTrainer, 2 steps of 2 rows (cp peers share them). Returns losses, step-0 grads and engagement."""
    sys.path.insert(0, TESTS)
    import datasets
    import transformers
    import trl

    import test_ulysses_inkling2_gloo as T
    from unsloth import context_parallel as cpm
    from unsloth.models._utils import patch_gradient_accumulation_fix
    from unsloth.utils import ulysses_inkling2 as ui

    patch_gradient_accumulation_fix(transformers.Trainer)
    cpm.patch_sft_trainer()
    cpm.DEVICE_TYPE_TORCH = "cpu"
    model = T._tiny(loop)
    if lora:
        model = T._lora(model)
    else:
        model.model.vision_tower.requires_grad_(
            False
        )  # a text batch: no vision gradient on either side
    forwards = []
    model.register_forward_pre_hook(lambda module, args: forwards.append(1))
    ids = torch.randint(3, T.VOCAB, (4, SEQ), generator = torch.Generator().manual_seed(1))
    data = datasets.Dataset.from_dict({"input_ids": ids.tolist()})

    def collate(rows):
        x = torch.tensor([r["input_ids"] for r in rows])
        labels = x.clone()
        labels[labels % 7 == 0] = -100
        return {"input_ids": x, "labels": labels}

    class Sequential(trl.SFTTrainer):
        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    grads = {}

    class Grab(transformers.TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kw):
            if grads:
                return
            for n, p in trainer.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                    grads[_canon(n)] = g.detach().double().clone()

    args = trl.SFTConfig(
        output_dir = tempfile.mkdtemp(),
        per_device_train_batch_size = 2,
        max_steps = 2,
        learning_rate = 1e-2,
        optim = "sgd",
        lr_scheduler_type = "constant",
        max_grad_norm = 1e9,
        logging_steps = 1,
        save_strategy = "no",
        report_to = [],
        use_cpu = True,
        seed = 0,
        remove_unused_columns = False,
        disable_tqdm = True,
        dataset_kwargs = {"skip_prepare_dataset": True},
        gradient_checkpointing = gc,
    )
    if hasattr(args, "router_aux_loss_coef"):
        args.router_aux_loss_coef = 0.0
    calls = {"attention": 0, "conv": 0, "lengths": set()}
    attention, conv = ui.ulysses_inkling2_attention_forward, ui.ulysses_inkling2_conv_forward

    def counted_attention(module, hidden_states, *a, **k):
        calls["attention"] += 1
        calls["lengths"].add(int(hidden_states.shape[1]))
        return attention(module, hidden_states, *a, **k)

    def counted_conv(*a, **k):
        calls["conv"] += 1
        return conv(*a, **k)

    ui.ulysses_inkling2_attention_forward, ui.ulysses_inkling2_conv_forward = (
        counted_attention,
        counted_conv,
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object = __import__("tokenizers").Tokenizer(
            __import__("tokenizers").models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token = "<unk>")
        ),
        pad_token = "<pad>",
    )
    trainer = Sequential(
        model = model,
        args = args,
        train_dataset = data,
        data_collator = collate,
        processing_class = tok,
        callbacks = [Grab()],
    )
    trainer.train()
    from unsloth.models.fsdp2_native import is_native_fsdp2

    sharded = [
        (p.to_local().numel(), p.numel())
        for p in trainer.model.parameters()
        if type(p).__name__ == "DTensor" and not p.requires_grad
    ]
    return {
        "losses": [h["loss"] for h in trainer.state.log_history if "loss" in h],
        "grads": grads,
        "native_fsdp2": bool(is_native_fsdp2(trainer.model)),
        "manager": getattr(trainer, "_context_parallel_manager", None) is not None,
        "calls": calls,
        "forwards": len(forwards),
        "frozen_dtensors": len(sharded),
        "frozen_local": sum(a for a, _ in sharded),
        "frozen_full": sum(b for _, b in sharded),
    }


def _worker(rank, world, port, out, loop, lora, gc):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        CUDA_VISIBLE_DEVICES = "",
        UNSLOTH_PARALLEL = "cp=2,cp_backend=ulysses,fsdp=1",
    )
    torch.set_num_threads(1)
    try:
        # conftest's device spoof first (a GPU-less host), then `import unsloth`, which parses UNSLOTH_PARALLEL.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_cpfsdp_conftest", os.path.join(TESTS, "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth  # noqa: F401
        from unsloth.models import fsdp2_load

        # A QLoRA load under this launch takes the FSDP2 sharded quantize-on-load over the world, the shard mesh.
        auto_sharded = bool(fsdp2_load._auto_sharded_load())
        res = _train(loop, lora, gc, world, rank)
        res["auto_sharded_load"] = auto_sharded
    except Exception as exc:
        import traceback
        res = {
            "error": f"{type(exc).__name__}: {exc}"[:3000],
            "trace": traceback.format_exc()[-3000:],
        }
    torch.save(res, os.path.join(out, f"r{rank}.pt"))


def _spawn(
    loop,
    lora,
    gc,
    world = 2,
):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as out:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, out, loop, lora, gc))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 300
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.1)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        paths = [os.path.join(out, f"r{r}.pt") for r in range(world)]
        return [torch.load(x, weights_only = False) if os.path.exists(x) else None for x in paths]


def _rel(a, b):
    diff = float((a - b).norm())
    return 0.0 if diff == 0 else diff / max(float(b.norm()), 1e-30)


@pytest.mark.parametrize(
    "loop, lora, gc",
    [(True, True, True), (False, False, False)],
    ids = ["mini-lora-gc", "small-full"],
)
def test_inkling2_ulysses_fsdp2_matches_one_process(loop, lora, gc):
    results = _spawn(loop, lora, gc)
    os.environ.pop("UNSLOTH_PARALLEL", None)
    ref = _train(loop, lora, gc, 1)
    assert ref["calls"]["attention"] == 0 and not ref["native_fsdp2"]
    layers = 4 + (2 if loop else 0)  # the Mini runs its two loop layers twice
    for rank, res in enumerate(results):
        assert res is not None and "error" not in res, (
            rank,
            res and res.get("error"),
            res and res.get("trace"),
        )
        assert res["native_fsdp2"] and res["manager"] and res["auto_sharded_load"]
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (
            res["losses"],
            ref["losses"],
        )
        assert set(res["grads"]) == set(ref["grads"]) and len(ref["grads"]) > 20
        worst = max((_rel(res["grads"][n], g), n) for n, g in ref["grads"].items())
        print(
            f"rank {rank} forwards {res['forwards']} losses {res['losses']} ref {ref['losses']} worst grad {worst} calls {res['calls']} "
            f"frozen {res['frozen_dtensors']} {res['frozen_local']} / {res['frozen_full']}"
        )
        assert worst[0] < 1e-4, worst
        # FSDP2 shards over the cp group: each rank holds about half of the sharded frozen weights.
        assert (
            res["frozen_dtensors"] > 10 and 0.45 < res["frozen_local"] / res["frozen_full"] < 0.55
        ), res
        # Per model forward (recomputed once more under gradient checkpointing): one Ulysses attention and four
        # halo convolutions per layer call, on 11 local tokens.
        assert res["forwards"] == ref["forwards"] > 0
        calls = res["forwards"] * layers * (2 if gc else 1)
        assert res["calls"]["attention"] == calls and res["calls"]["conv"] == 4 * calls, (
            res["calls"],
            calls,
        )
        assert res["calls"]["lengths"] == {SEQ // 2}


def test_inkling2_cp_fsdp2_capability():
    """The registry has a cell for Ulysses cp x native FSDP2 on Inkling 2; ring and packing stay refused."""
    from unsloth import parallel_capabilities as pc

    q = pc.Query(
        trainer = "sft",
        family = "inkling2_loop",
        hybrid = True,
        multimodal = True,
        windowed = True,
        mode = "qlora",
        cp_backend = "ulysses",
        cp = 2,
        fsdp2 = True,
        attention = "eager",
        memory = "unsloth_offload",
        moe = True,
        torch = (2, 10),
        sm = 7,
    )
    pc.check(q)
    assert pc.cell_status(q)[0] == "supported"
    pc.check(dataclasses.replace(q, family = "inkling2", mode = "full", memory = "gc"))
    with pytest.raises(NotImplementedError, match = "ring"):
        pc.check(dataclasses.replace(q, cp_backend = "ring", sm = 8))
    with pytest.raises(NotImplementedError, match = "composition"):
        pc.check(dataclasses.replace(q, packing = True))
    with pytest.raises(NotImplementedError, match = "composition"):
        pc.check(dataclasses.replace(q, dp = 2))
