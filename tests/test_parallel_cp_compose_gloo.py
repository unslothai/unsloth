# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses CP beside DDP, FSDP2 and TP through SFTTrainer on gloo CPU ranks vs one process (losses, grads,
adapters); cp x tp is refused. CPU ranks use the HF attention with the hybrids' Ulysses wrapper, the rest is
the production path."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import (  # noqa: E402
    _free_port,
    _import_checkout_unsloth,
    _raised,
    _save_checkpoint,
)

_ENV_KEYS = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_CP_BACKEND",
    "UNSLOTH_PP_MICROBATCHES",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "ACCELERATE_USE_FSDP",
)
_ADAM_EPS = 1e-4  # as test_tp_env_gloo
SEQ = 16
N_LAYERS = 4


def _checkpoint(tmp):
    # 4 heads / 4 kv heads: tp 2 leaves 2 local heads, which Ulysses splits over cp 2.
    path = _save_checkpoint("llama", os.path.join(tmp, "ckpt"), {"num_hidden_layers": N_LAYERS})
    from transformers import AutoTokenizer

    from test_tp_native_oracle import LLAMA

    AutoTokenizer.from_pretrained(LLAMA).save_pretrained(path)
    return path


def _full(t):
    from torch.distributed.tensor import DTensor
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().float().clone()


def _case_train(
    rank,
    world,
    tmp,
    ckpt,
    bs,
    cp = 1,
    steps = 2,
    ga = 2,
    sft_cp = True,
    keep = False,
    rows_dp = None,
):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    import transformers
    import trl
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unsloth import context_parallel as cpm
    from unsloth.models import fsdp2_native, pp_native, tp_env, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix
    from unsloth.utils.ulysses_hybrid import install_hf_attention_wrapper

    patch_gradient_accumulation_fix(transformers.Trainer)
    cpm.patch_sft_trainer()
    cpm.DEVICE_TYPE_TORCH = "cpu"
    # The test seam: CPU ranks run the Hugging Face attention through the Ulysses wrapper.
    cpm._supports_context_parallel = lambda model, backend = "ring": True
    from unsloth.utils import attention_dispatch

    attention_dispatch.HAS_XFORMERS = False  # CUDA only: torch SDPA on the CPU ranks
    attention_dispatch.HAS_FLASH_ATTENTION = False
    layout = tp_env.env_layout()

    def load(
        model_name,
        load_in_4bit = True,
        **kwargs,
    ):
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float32, attn_implementation = "sdpa"
        )

    def peft(model):
        torch.manual_seed(1)
        config = LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = "all-linear",
            init_lora_weights = False,
            lora_dropout = 0.0,
        )
        return get_peft_model(model, config)

    load = tp_env.tensor_parallel_env_load(load)
    peft = tp_env.tensor_parallel_env_peft(peft)
    model = peft(load(ckpt, load_in_4bit = False))
    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            install_hf_attention_wrapper(module, "sdpa", cpm.get_cp_manager)

    dp = 1 if layout is None else layout.dp
    data_dp = getattr(layout, "data_dp", dp) if layout is not None else 1
    if rows_dp is not None:  # DDP / native FSDP2 (no tp layout): the data parallel width
        data_dp = rows_dp
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (steps * ga * bs * data_dp, SEQ), generator = g)
    import datasets

    data = datasets.Dataset.from_dict({"input_ids": ids.tolist()})

    def collate(rows):
        input_ids = torch.tensor([r["input_ids"] for r in rows])
        batch = {"input_ids": input_ids, "labels": input_ids.clone()}
        batch["labels"][
            batch["labels"] % 5 == 0
        ] = -100  # uneven token counts per row and per shard
        return batch

    class SequentialSFT(trl.SFTTrainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    grads = {}

    class Grabber(transformers.TrainerCallback):
        def on_pre_optimizer_step(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            if grads:
                return
            target = trainer.model
            for name, p in target.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grads[name] = _full(p.grad)

    args = trl.SFTConfig(
        output_dir = os.path.join(tmp, "out"),
        per_device_train_batch_size = bs,
        gradient_accumulation_steps = ga,
        max_steps = steps,
        learning_rate = 1e-2,
        optim = "adamw_torch",
        adam_epsilon = _ADAM_EPS,
        lr_scheduler_type = "constant",
        warmup_steps = 0,
        weight_decay = 0.0,
        max_grad_norm = 1.0,
        logging_steps = 1,
        save_strategy = "no",
        report_to = [],
        use_cpu = True,
        seed = 0,
        dataloader_drop_last = True,
        remove_unused_columns = False,
        disable_tqdm = True,
        dataset_kwargs = {"skip_prepare_dataset": True},
        gradient_checkpointing = False,
    )
    if sft_cp:
        args.context_parallel_size = cp
        args.context_parallel_backend = "ulysses"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    trainer = SequentialSFT(
        model = model,
        args = args,
        train_dataset = data,
        data_collator = collate,
        processing_class = tokenizer,
        callbacks = [Grabber()],
    )
    manager = getattr(trainer, "_context_parallel_manager", None)
    seen_lengths, ulysses_calls = set(), []
    embed = trainer.model.get_input_embeddings()
    embed.register_forward_pre_hook(lambda m, a: seen_lengths.add(int(a[0].shape[-1])))
    from unsloth.utils import ulysses_hybrid

    original_ulysses = ulysses_hybrid._ulysses_hf_attention

    def counted(*a, **k):
        ulysses_calls.append(1)
        return original_ulysses(*a, **k)

    ulysses_hybrid._ulysses_hf_attention = counted
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    is_tp = tp_native.is_tensor_parallel(trainer.model)
    if is_tp:
        state = tp_native.tp_full_state_dict(trainer.model, adapters_only = True)
    else:
        state = {k: _full(v) for k, v in trainer.model.state_dict().items() if "lora_" in k}
    stage = pp_native.pipeline_stage_of(trainer.model)
    res = {
        "layout": None if layout is None else repr(layout),
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h.get("grad_norm") for h in logs],
        "grads": {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in grads.items()},
        "state": {
            k.replace("module.", "").replace("_orig_mod.", ""): _full(v)
            for k, v in state.items()
            if "lora_" in k
        },
        "cp_group": None
        if manager is None
        else sorted(int(r) for r in manager.mesh.mesh.flatten().tolist()),
        "is_tp": is_tp,
        "is_fsdp2": fsdp2_native.is_native_fsdp2(trainer.model),
        "seen_lengths": sorted(seen_lengths),
        "ulysses_calls": len(ulysses_calls),
        "stage": None if stage is None else int(stage.stage_index),
    }
    if dist.is_initialized():
        dist.barrier()
    return res


CASES = {"train": _case_train}


def _worker(rank, world, port, case, tmp, env, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.set_num_threads(1)
    import torch.distributed as dist

    try:
        try:
            res = CASES[case](rank, world, tmp, **kwargs)
        except Exception as e:
            import traceback
            res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _spawn(
    case,
    world = 1,
    env = None,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        kwargs["ckpt"] = _checkpoint(tmp)
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
        for p in procs:
            if p.is_alive():
                p.kill()
        results = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            results.append(
                torch.load(path, weights_only = False)
                if os.path.exists(path)
                else {"error": f"rank {r} exit code {procs[r].exitcode}"}
            )
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


_REF = {}


def _ref(bs):
    if bs not in _REF:
        (_REF[bs],) = _spawn("train", 1, bs = bs)
        assert "error" not in _REF[bs], _REF[bs]
    return _REF[bs]


def _close(
    got,
    want,
    rel = 1e-4,
    what = "",
):
    assert "error" not in got, (what, got.get("error"), got.get("trace"))
    assert got["losses"] == pytest.approx(want["losses"], rel = rel), (
        what,
        got["losses"],
        want["losses"],
    )
    assert got["grad_norms"] == pytest.approx(want["grad_norms"], rel = rel), (
        what,
        got["grad_norms"],
        want["grad_norms"],
    )
    for kind in ("grads", "state"):
        assert set(got[kind]) <= set(want[kind]) and got[kind], (
            what,
            kind,
            sorted(set(got[kind]) - set(want[kind])),
        )
        for key, value in got[kind].items():
            torch.testing.assert_close(
                value, want[kind][key], rtol = rel * 10, atol = 1e-6, msg = f"{what} {kind} {key}"
            )


def _engaged(res, cp = 2):
    # Each rank embedded SEQ / cp tokens and every layer's attention went through the Ulysses exchange.
    assert res["seen_lengths"] == [SEQ // cp], res["seen_lengths"]
    assert res["ulysses_calls"] >= N_LAYERS, res["ulysses_calls"]


def test_cp2_alone_matches_one_process():
    """The control: Ulysses cp 2 through SFTTrainer reproduces one process (losses, grads, adapters)."""
    ref = _ref(2)
    assert ref["seen_lengths"] == [SEQ] and ref["ulysses_calls"] == 0
    for rank, res in enumerate(_spawn("train", 2, bs = 2, cp = 2)):
        assert res["cp_group"] == [0, 1]
        _engaged(res)
        _close(res, ref, what = rank)


def test_cp2_with_ddp_dp2_matches_one_process():
    """Plain DDP x cp 2 on 4 ranks: cp peers adjacent, dp replicas averaged, same as one process on the batch."""
    ref = _ref(4)
    for rank, res in enumerate(_spawn("train", 4, bs = 2, cp = 2, rows_dp = 2)):
        assert res["cp_group"] == [rank - rank % 2, rank - rank % 2 + 1], (rank, res.get("error"))
        assert not res["is_fsdp2"] and not res["is_tp"]
        _engaged(res)
        _close(res, ref, what = ("ddp", rank))


@pytest.mark.parametrize("world, ref_bs", [(2, 2), (4, 4)], ids = ["fsdp2_cp2", "fsdp2_dp2_cp2"])
def test_cp2_with_native_fsdp2_matches_one_process(world, ref_bs):
    """UNSLOTH_FSDP2=1 x cp 2: native FSDP2 shards over every rank, the cp peers share a batch."""
    ref = _ref(ref_bs)
    results = _spawn("train", world, {"UNSLOTH_FSDP2": "1"}, bs = 2, cp = 2, rows_dp = world // 2)
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("trace"))
        assert res["is_fsdp2"], "native FSDP2 did not engage"
        assert res["cp_group"] == [rank - rank % 2, rank - rank % 2 + 1], (rank, res["cp_group"])
        _engaged(res)
        _close(res, ref, what = ("fsdp2", world, rank))


def test_cp_with_tensor_parallel_env_is_refused_before_the_split():
    """UNSLOTH_TP_SIZE=2 x cp 2 on 4 ranks trained with the wrong loss (the CP groups were the tp pairs)."""
    for res in _spawn("train", 4, {"UNSLOTH_TP_SIZE": "2"}, bs = 2, cp = 2):
        error = res.get("error") or ""
        assert "does not compose with" in error, error or res["losses"]
        assert "tensor parallel (UNSLOTH_TP_SIZE=2)" in error, error
