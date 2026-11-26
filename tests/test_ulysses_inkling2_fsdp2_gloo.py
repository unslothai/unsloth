# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inkling 2 Ulysses context parallelism over native FSDP2 on the same two ranks
(UNSLOTH_PARALLEL="cp=2,cp_backend=ulysses,fsdp=1": the parameters are sharded over the cp group, each rank holds its
own sequence shard), through the Unsloth-patched SFTTrainer on two gloo CPU ranks against the same trainer in one
process, in float32: losses of two SGD steps and every step-0 gradient.

This is the layout that lets a model whose weights do not fit one GPU train under cp (Inkling-2-Mini QLoRA on
2x T4): the FSDP2 sharded load and shard_model cut the weights over the same ranks that split the sequence. The
tiny models are the ones of tests/test_ulysses_inkling2_gloo.py (sliding window and relative bias across the shard
seam, kernel 4 short convolutions with a halo, the Mini's looped layers, the MoE).

Every training run (the one-process reference too) runs in its own spawned interpreter, so the pytest process keeps
its environment, sys.path and module state."""

import dataclasses
import math
import multiprocessing
import os
import tempfile
import time
import types
from datetime import timedelta

import pytest
import torch

SEQ = 22  # 11 tokens per rank, not a multiple of the convolution kernel
INKLING2_MODULE = "transformers.models.inkling2_loop.modeling_inkling2_loop"
# Launch settings a child must not inherit from the caller: an inherited alias (UNSLOTH_CP_SIZE=4) would change the
# layout, and the one-process reference must see none of them.
_LAUNCH_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_CP_BACKEND",
    "UNSLOTH_PP_MICROBATCHES",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "UNSLOTH_FSDP2_SHARDED_LOAD",
    "ACCELERATE_USE_FSDP",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)
LAUNCH = "cp=2,cp_backend=ulysses,fsdp=1"


def _canon(name):
    for x in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."):
        name = name.replace(x, "")
    return name


def _train(loop, lora, gc, flex, patch, output_dir):
    """SFTTrainer, 2 steps of 2 rows (cp peers share them). Returns losses, step-0 grads and engagement."""
    import datasets
    import transformers
    import trl

    from tests import test_ulysses_inkling2_gloo as T
    from unsloth import context_parallel as cpm
    from unsloth.models._utils import patch_gradient_accumulation_fix
    from unsloth.models.fsdp2_native import is_native_fsdp2
    from unsloth.utils import ulysses_inkling2 as ui

    patch_gradient_accumulation_fix(transformers.Trainer)
    cpm.patch_sft_trainer()
    patch.setattr(cpm, "DEVICE_TYPE_TORCH", "cpu")
    model = T._tiny(loop)
    flex_stats = None
    if flex:
        from torch.nn.attention import flex_attention as flex_mod

        from unsloth.models import inkling2_flex as fx

        patch.setattr(fx, "_ALLOW_CPU", True)
        # flex's reference math runs backward on CPU too
        patch.setattr(flex_mod, "_validate_device", lambda *a, **k: None)
        assert fx.enable_inkling2_flex(model, verbose = False), fx._FAILED
        flex_stats = fx._STATS
    if lora:
        model = T._lora(model)
    # Text batches give a trainable vision tower no grads; native FSDP2 copes, DDP does not.
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
        output_dir = output_dir,
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
        # SFTConfig defaults bf16 to True (CPU autocast included): pin float32 on both sides.
        bf16 = False,
        fp16 = False,
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

    patch.setattr(ui, "ulysses_inkling2_attention_forward", counted_attention)
    patch.setattr(ui, "ulysses_inkling2_conv_forward", counted_conv)
    import tokenizers

    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object = tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token = "<unk>")
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
    flex_before = dict(flex_stats) if flex_stats is not None else {}
    trainer.train()
    from torch.distributed.tensor import DTensor

    # The decoder's own weights: frozen bases under LoRA, the trained weights under full finetuning.
    decoder = [
        (p.to_local().numel(), p.numel())
        for n, p in trainer.model.named_parameters()
        if "language_model.layers." in n
        and isinstance(p, DTensor)
        and any(placement.is_shard() for placement in p.placements)
        and p.requires_grad == (not lora)
    ]
    trainable_tower = [
        n for n, p in trainer.model.named_parameters() if p.requires_grad and "vision_tower" in n
    ]
    return {
        "losses": [h["loss"] for h in trainer.state.log_history if "loss" in h],
        "grads": grads,
        "native_fsdp2": bool(is_native_fsdp2(trainer.model)),
        "manager": getattr(trainer, "_context_parallel_manager", None) is not None,
        "calls": {**calls, "lengths": sorted(calls["lengths"])},
        "forwards": len(forwards),
        "decoder_shards": decoder,
        "trainable_tower": len(trainable_tower),
        "flex_calls": (flex_stats or {}).get("flex", 0) - flex_before.get("flex", 0),
    }


def _worker(rank, world, out, loop, lora, gc, flex):
    for name in _LAUNCH_ENV:
        os.environ.pop(name, None)
    if world > 1:
        os.environ.update(
            RANK = str(rank),
            WORLD_SIZE = str(world),
            LOCAL_RANK = str(rank),
            LOCAL_WORLD_SIZE = str(world),
            UNSLOTH_PARALLEL = LAUNCH,
        )
    torch.set_num_threads(1)
    dist = torch.distributed
    try:
        if world > 1:
            # A private file rendezvous: no TCP listener, no free-port race.
            dist.init_process_group(
                "gloo",
                store = dist.FileStore(os.path.join(out, "rendezvous"), world),
                rank = rank,
                world_size = world,
                timeout = timedelta(seconds = 120),
            )
        # conftest's device spoof first (a GPU-less host), then `import unsloth`, which parses UNSLOTH_PARALLEL.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_cpfsdp_conftest",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.py"),
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth  # noqa: F401
        from unsloth.models import fsdp2_load

        # Launcher picks FSDP2 sharded QLoRA load; dense fakes come from config, not a load.
        auto_sharded = bool(fsdp2_load._auto_sharded_load())
        output_dir = os.path.join(out, f"trainer-{world}-{rank}")
        os.makedirs(output_dir, exist_ok = True)
        with pytest.MonkeyPatch.context() as patch:
            res = _train(loop, lora, gc, flex, patch, output_dir)
        res["auto_sharded_load"] = auto_sharded
    except Exception as exc:
        import traceback
        res = {
            "error": f"{type(exc).__name__}: {exc}"[:3000],
            "trace": traceback.format_exc()[-3000:],
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    torch.save(res, os.path.join(out, f"r{world}-{rank}.pt"))


def _spawn(loop, lora, gc, flex, world):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as out:
        procs = [
            ctx.Process(target = _worker, args = (r, world, out, loop, lora, gc, flex))
            for r in range(world)
        ]
        # The children inherit the mask from their first instruction on, before they import anything.
        with pytest.MonkeyPatch.context() as env:
            env.setenv("CUDA_VISIBLE_DEVICES", "")
            for p in procs:
                p.start()
        deadline = time.monotonic() + 300
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.1)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        paths = [os.path.join(out, f"r{world}-{r}.pt") for r in range(world)]
        return [
            torch.load(x, map_location = "cpu", weights_only = True) if os.path.exists(x) else None
            for x in paths
        ]


def _rel(a, b):
    diff = float((a - b).norm())
    return 0.0 if diff == 0 else diff / max(float(b.norm()), 1e-30)


def _ok(res, rank):
    assert res is not None and "error" not in res, (
        rank,
        res and res.get("error"),
        res and res.get("trace"),
    )


@pytest.mark.parametrize(
    "loop, lora, gc, flex",
    [(True, True, True, False), (False, False, False, False), (True, True, False, True)],
    ids = ["mini-lora-gc", "small-full", "mini-flex-lora"],
)
def test_inkling2_ulysses_fsdp2_matches_one_process(loop, lora, gc, flex):
    pytest.importorskip(INKLING2_MODULE)
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("needs a gloo-enabled PyTorch build")
    results = _spawn(loop, lora, gc, flex, world = 2)
    (ref,) = _spawn(loop, lora, gc, flex, world = 1)
    _ok(ref, "reference")
    assert ref["calls"]["attention"] == 0 and not ref["native_fsdp2"] and not ref["manager"]
    layers = 4 + (2 if loop else 0)  # the Mini runs its two loop layers twice
    for rank, res in enumerate(results):
        _ok(res, rank)
        assert res["native_fsdp2"] and res["manager"] and res["auto_sharded_load"]
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (
            res["losses"],
            ref["losses"],
        )
        assert set(res["grads"]) == set(ref["grads"]) and len(ref["grads"]) > 20
        errors = []
        for name, want in ref["grads"].items():
            error = _rel(res["grads"][name], want)
            assert math.isfinite(error), (rank, name, error)
            errors.append((error, name))
        worst = max(errors)
        shards = res["decoder_shards"]
        print(
            f"rank {rank} forwards {res['forwards']} losses {res['losses']} ref {ref['losses']} worst grad {worst} "
            f"calls {res['calls']} flex {res['flex_calls']} decoder shards {len(shards)} "
            f"{sum(a for a, _ in shards)} / {sum(b for _, b in shards)} trainable tower {res['trainable_tower']}"
        )
        assert worst[0] < 1e-4, worst
        # FSDP2 shards over the cp group: each rank holds about half of the decoder's weights.
        assert len(shards) > 10, shards
        assert 0.45 < sum(a for a, _ in shards) / sum(b for _, b in shards) < 0.55, shards
        if not lora:
            # The text batch leaves the trainable tower without gradients on both sides.
            assert res["trainable_tower"] > 0 and not any("vision_tower" in n for n in ref["grads"])
        # Per model forward (recomputed once more under gradient checkpointing): one Ulysses attention and four
        # halo convolutions per layer call, on 11 local tokens.
        assert res["forwards"] == ref["forwards"] > 0
        calls = res["forwards"] * layers * (2 if gc else 1)
        assert res["calls"]["attention"] == calls and res["calls"]["conv"] == 4 * calls, (
            res["calls"],
            calls,
        )
        assert res["calls"]["lengths"] == [SEQ // 2]
        # Inkling 2 flex runs inside the Ulysses attention (and in the reference), not eager.
        assert (
            (res["flex_calls"] == calls and ref["flex_calls"] > 0)
            if flex
            else res["flex_calls"] == 0
        )


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


def _fake_inkling2(text_attention):
    text = types.SimpleNamespace(
        model_type = "inkling2_loop_text",
        layer_types = ["hybrid_sliding", "hybrid"],
        sliding_window = 5,
        _attn_implementation = text_attention,
    )
    config = types.SimpleNamespace(
        model_type = "inkling2_loop",
        text_config = text,
        vision_config = types.SimpleNamespace(),
        _attn_implementation = "eager",
    )
    return types.SimpleNamespace(config = config)


@pytest.mark.parametrize("fsdp2", [True, False], ids = ["fsdp2", "ddp"])
def test_inkling2_cp_capability_reads_the_text_attention(fsdp2):
    """Only the text config goes flex; the registry must classify the query as flex, not eager."""
    from unsloth import parallel_capabilities as pc

    layout = types.SimpleNamespace(dp = 1, tp = 1, pp = 1, cp = 2, cp_backend = "ulysses", fsdp2 = fsdp2)
    q = pc.query_from(layout, model = _fake_inkling2("unsloth_inkling2_flex"))
    assert (q.family, q.cp, q.fsdp2, q.attention) == ("inkling2_loop", 2, fsdp2, "flex")
    q = dataclasses.replace(
        q, trainer = "sft", mode = "lora", moe = True, memory = "gc", torch = (2, 10), sm = 8
    )
    pc.check(q)
    assert pc.cell_status(q)[0] == "supported"
    assert pc.query_from(layout, model = _fake_inkling2("eager")).attention == "eager"
    with pytest.raises(NotImplementedError, match = "flex"):
        pc.check(dataclasses.replace(q, family = "llama", hybrid = False, moe = False))
    with pytest.raises(NotImplementedError):
        pc.check(dataclasses.replace(q, cp_backend = "ring"))


@pytest.mark.parametrize("native", [True, False], ids = ["native-fsdp2", "ddp"])
def test_cp_trainable_tower_refusal_is_ddp_only(monkeypatch, native):
    """Trainable tower with no grads breaks DDP at step 2, so cp refuses; native FSDP2 is not."""
    from unsloth import context_parallel as cpm
    from unsloth.models import fsdp2_native

    monkeypatch.setattr(cpm, "_check_cp_model", lambda *a, **k: None)
    model = torch.nn.Module()
    model.vision_tower = torch.nn.Linear(2, 2)
    model.language_model = torch.nn.Linear(2, 2)
    if native:
        setattr(model, fsdp2_native._NATIVE_FLAG, True)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(
            distributed_type = types.SimpleNamespace(name = "MULTI_CPU"),
            num_processes = 2,
            dispatch_batches = None,
        ),
        train_dataset = None,
        eval_dataset = None,
    )
    check = lambda: cpm._check_cp_trainer(trainer, 2, "ulysses", [model], allow_fsdp2 = True)
    if native:
        assert check() is False  # not accelerate's FSDP2: the native wrapper owns the sharding
    else:
        with pytest.raises(NotImplementedError, match = "vision / audio tower"):
            check()
    # DPO / GRPO (allow_fsdp2 = False) keep the refusal either way.
    with pytest.raises(NotImplementedError, match = "vision / audio tower"):
        cpm._check_cp_trainer(trainer, 2, "ulysses", [model])
