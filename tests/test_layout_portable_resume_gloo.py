# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Trainer checkpoints resumed on another parallel layout (gloo CPU, fp32 LoRA): a checkpoint written under
tensor parallel (tp 2), pipeline parallel (pp 2) or expert parallel (ep 2) resumes in one process, a one-process
checkpoint resumes under tp 2 / pp 2 / ep 2, and a tp 2 one under pp 2. The resumed steps (loss, gradient norm,
the LoRA gradients of the first resumed step in one process) and the final adapter match an uninterrupted
one-process run; the resharding path provably ran. On the saved layout the per-rank fast path stays bitwise and
the resharding path does not run. bitsandbytes 8-bit AdamW: tp 2 -> one process requantizes the float32 moments."""

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
from test_tp_native_gloo import (  # noqa: E402
    _Rows,
    _free_port,
    _import_checkout_unsloth,
    _save_checkpoint,
    _trainer_args,
)

_ADAM_EPS = 1e-4  # as test_tp_env_gloo / test_pp_env_gloo: fp32 reduction-order noise stays tiny
_ENV_KEYS = (
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_EP_SIZE",
    "UNSLOTH_PARALLEL",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "ACCELERATE_USE_FSDP",
)
OLMOE = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
_STEPS, _SAVE_AT, _BATCH = 4, 2, 2  # global batch of 2 rows per step, no accumulation
LAYOUTS = {
    "one": (1, {}),
    "tp2": (2, {"UNSLOTH_TP_SIZE": "2"}),
    "pp2": (2, {"UNSLOTH_PP_SIZE": "2"}),
    "ep2": (2, {"UNSLOTH_PARALLEL": "ep=2"}),
    "fsdp2": (2, {"UNSLOTH_FSDP2": "1"}),
}


def _bnb_8bit_everywhere():
    # bitsandbytes keeps 32-bit state below min_8bit_size (4096) elements: every tiny LoRA factor would.
    import bitsandbytes as bnb
    for cls in (bnb.optim.AdamW, bnb.optim.AdamW8bit):
        init = cls.__init__

        def patched(
            self,
            *args,
            __init = init,
            **kwargs,
        ):
            kwargs["min_8bit_size"] = 0
            __init(self, *args, **kwargs)

        cls.__init__ = patched


def _case(
    rank,
    world,
    tmp,
    ckpt,
    family,
    out,
    resume_from = None,
    optim = "adamw_torch",
):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from unsloth.models import portable_resume, tp_env
    from unsloth.models.fsdp2_native import _inner_optimizer as _inner
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)  # Unsloth's Trainer.__init__
    if optim == "adamw_8bit":
        _bnb_8bit_everywhere()
    calls = {"weights": 0, "optimizer": 0, "named": 0}
    for attr, key in (
        ("portable_load_weights", "weights"),
        ("portable_load_optimizer", "optimizer"),
        ("_load_named", "named"),
    ):
        fn = getattr(portable_resume, attr)

        def counted(
            *args,
            __fn = fn,
            __key = key,
            **kwargs,
        ):
            calls[__key] += 1
            return __fn(*args, **kwargs)

        setattr(portable_resume, attr, counted)

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    grads = {}

    class Grads(transformers.TrainerCallback):
        def on_pre_optimizer_step(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            if world == 1:
                grads[state.global_step + 1] = {
                    n: p.grad.detach().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                }

    if family == "olmoe":
        from test_tp_native_oracle import build
        def fresh():
            model = build(OLMOE, lora = False).train()
            config = LoraConfig(
                r = 4,
                lora_alpha = 8,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
                init_lora_weights = False,
            )
            torch.manual_seed(1)
            return tp_env.tensor_parallel_env_peft(lambda m: get_peft_model(m, config))(model)

        vocab = 64
    else:
        load = tp_env.tensor_parallel_env_load(
            lambda name, load_in_4bit = True: AutoModelForCausalLM.from_pretrained(
                name, torch_dtype = torch.float32, attn_implementation = "eager"
            )
        )

        def fresh():
            config = LoraConfig(
                r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False
            )
            torch.manual_seed(1)
            return tp_env.tensor_parallel_env_peft(lambda m: get_peft_model(m, config))(
                load(ckpt, load_in_4bit = False)
            )

        vocab = 1000
    g = torch.Generator().manual_seed(1)
    data = _Rows(list(torch.randint(1, vocab, (_STEPS * _BATCH, 9), generator = g)))
    # Expert parallel and FSDP2 ranks read their own rows (data parallel); tp / pp ranks read the same rows.
    data_parallel = "UNSLOTH_PARALLEL" in os.environ or "UNSLOTH_FSDP2" in os.environ
    bs = _BATCH // world if data_parallel else _BATCH
    args = _trainer_args(
        os.path.join(tmp, out),
        bs,
        optim = optim,
        max_steps = _STEPS,
        gradient_accumulation_steps = 1,
        adam_epsilon = _ADAM_EPS,
        save_strategy = "steps",
        save_steps = _SAVE_AT,
    )
    trainer = SequentialTrainer(model = fresh(), args = args, train_dataset = data, callbacks = [Grads()])
    trainer.train(resume_from_checkpoint = resume_from)
    final = os.path.join(tmp, out, "final")
    trainer.save_model(final)  # the 1-GPU adapter on every layout (collective)
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    return {
        "steps": [h["step"] for h in logs],
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h["grad_norm"] for h in logs],
        "grads": grads,
        "adapter": load_file(os.path.join(final, "adapter_model.safetensors")),
        "calls": calls,
        "optimizer": type(_inner(trainer.optimizer)).__module__
        + "."
        + type(_inner(trainer.optimizer)).__name__,
        "files": sorted(os.listdir(os.path.join(tmp, out, f"checkpoint-{_STEPS}"))),
    }


def _worker(rank, world, port, tmp, env, kwargs):
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
    torch.set_num_threads(2)
    import faulthandler

    faulthandler.dump_traceback_later(570, exit = True)
    import torch.distributed as dist

    try:
        res = _case(rank, world, tmp, **kwargs)
    except Exception as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:2000] + traceback.format_exc()[-6000:]}
    torch.save(res, os.path.join(tmp, f"result_{kwargs['out']}_{rank}.pt"))
    if dist.is_initialized():
        dist.destroy_process_group()


def _run(tmp, layout, **kwargs):
    """One arm in `tmp` (kept between arms: checkpoints are read by later ones); rank 0's result."""
    world, env = LAYOUTS[layout]
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world, port, tmp, env, kwargs)) for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(600)
    codes = [p.exitcode for p in procs]
    for p in procs:
        if p.is_alive():
            p.kill()
    assert codes == [0] * world, f"rank exit codes {codes}"
    results = [
        torch.load(os.path.join(tmp, f"result_{kwargs['out']}_{r}.pt"), weights_only = False)
        for r in range(world)
    ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    for r in results:
        assert "error" not in r, r["error"]
    return results[0]


def _family_of(layouts):
    return "olmoe" if "ep2" in layouts else "llama"


@pytest.fixture(scope = "module")
def refs(tmp_path_factory):
    """{(family, optim): (tmp, uninterrupted one-process run)}; its checkpoint-2 is the one-process checkpoint."""
    cache = {}

    def get(family, optim = "adamw_torch"):
        if (family, optim) not in cache:
            tmp = str(tmp_path_factory.mktemp(f"{family}_{optim}"))
            if family == "olmoe":
                pytest.importorskip("transformers.models.olmoe")
                import transformers

                if int(transformers.__version__.split(".")[0]) < 5:
                    pytest.skip("expert parallel needs transformers 5 (fused expert stacks)")
                ckpt = None
            else:
                ckpt = _save_checkpoint(
                    "llama", os.path.join(tmp, "base"), {"num_hidden_layers": 4}
                )
            ref = _run(tmp, "one", ckpt = ckpt, family = family, out = "ref", optim = optim)
            cache[(family, optim)] = (tmp, ckpt, ref)
        return cache[(family, optim)]

    return get


def _close(
    got,
    want,
    what,
    rel = 1e-5,
    grads = True,
):
    assert got["steps"] == want["steps"] == [1, 2, 3, 4], (what, got["steps"])
    # Steps 1-2 come from the checkpoint's trainer_state.json; 3-4 ran after the resume.
    assert got["losses"][2:] == pytest.approx(want["losses"][2:], rel = rel), (
        what,
        got["losses"],
        want["losses"],
    )
    assert got["grad_norms"][2:] == pytest.approx(want["grad_norms"][2:], rel = 1e-4), (
        what,
        got["grad_norms"],
        want["grad_norms"],
    )
    if grads:
        step = _SAVE_AT + 1  # the first step after the resume
        assert set(got["grads"][step]) == set(want["grads"][step]) and got["grads"][step], what
        for name, g in want["grads"][step].items():
            torch.testing.assert_close(
                got["grads"][step][name], g, rtol = 1e-4, atol = 1e-6, msg = f"{what} {name}"
            )
    assert set(got["adapter"]) == set(want["adapter"]) and want["adapter"], what
    for key, value in want["adapter"].items():
        torch.testing.assert_close(
            got["adapter"][key], value, rtol = 1e-4, atol = 1e-5, msg = f"{what} {key}"
        )


def _needs_pipelining(layouts):
    if any(layout in ("ep2", "fsdp2") for layout in layouts):
        import transformers
        if int(transformers.__version__.split(".")[0]) < 5:
            # transformers 4.x Trainer.train restores the checkpoint's per-device batch size on resume (5.x only
            # with auto_find_batch_size), so a data parallel checkpoint resumes in one process on other rows.
            pytest.skip("transformers 4.x resumes with the checkpoint's per-device batch size")
    if "pp2" not in layouts:
        return
    import inspect

    try:
        from torch.distributed.pipelining import ScheduleGPipe
    except ImportError:
        pytest.skip("torch.distributed.pipelining")
    if "return_outputs" not in inspect.signature(ScheduleGPipe.step).parameters:
        pytest.skip("torch.distributed.pipelining with step(return_outputs) (torch >= 2.10)")


@pytest.mark.parametrize("saved", ["tp2", "pp2", "ep2", "fsdp2"])
def test_parallel_checkpoint_resumes_in_one_process(saved, refs):
    _needs_pipelining([saved])
    tmp, ckpt, ref = refs(_family_of([saved]))
    family = _family_of([saved])
    src = _run(tmp, saved, ckpt = ckpt, family = family, out = f"src_{saved}")
    assert "optimizer.pt" in src["files"], src["files"]
    got = _run(
        tmp,
        "one",
        ckpt = ckpt,
        family = family,
        out = f"to_one_{saved}",
        resume_from = os.path.join(tmp, f"src_{saved}", f"checkpoint-{_SAVE_AT}"),
    )
    assert got["calls"]["named"] == 1, got["calls"]  # the name-keyed optimizer load ran
    _close(got, ref, f"{saved} -> one")


@pytest.mark.parametrize("target", ["tp2", "pp2", "ep2", "fsdp2"])
def test_one_process_checkpoint_resumes_in_parallel(target, refs):
    _needs_pipelining([target])
    family = _family_of([target])
    tmp, ckpt, ref = refs(family)
    got = _run(
        tmp,
        target,
        ckpt = ckpt,
        family = family,
        out = f"from_one_{target}",
        resume_from = os.path.join(tmp, "ref", f"checkpoint-{_SAVE_AT}"),
    )
    if target != "fsdp2":  # native FSDP2 reads a 1-GPU checkpoint with its own loader
        assert got["calls"]["weights"] == 1 and got["calls"]["optimizer"] == 1, got["calls"]
    _close(got, ref, f"one -> {target}", grads = False)


def test_tensor_parallel_checkpoint_resumes_under_pipeline_parallel(refs):
    _needs_pipelining(["pp2"])
    tmp, ckpt, ref = refs("llama")
    if not os.path.isdir(os.path.join(tmp, "src_tp2")):
        _run(tmp, "tp2", ckpt = ckpt, family = "llama", out = "src_tp2")
    got = _run(
        tmp,
        "pp2",
        ckpt = ckpt,
        family = "llama",
        out = "tp2_to_pp2",
        resume_from = os.path.join(tmp, "src_tp2", f"checkpoint-{_SAVE_AT}"),
    )
    assert got["calls"]["weights"] == 1 and got["calls"]["optimizer"] == 1, got["calls"]
    _close(got, ref, "tp2 -> pp2", grads = False)


@pytest.mark.parametrize("layout", ["tp2", "ep2"])
def test_same_layout_resume_keeps_the_rank_files(layout, refs):
    family = _family_of([layout])
    tmp, ckpt, _ = refs(family)
    src_dir = f"src_{layout}"
    if not os.path.isdir(os.path.join(tmp, src_dir)):
        _run(tmp, layout, ckpt = ckpt, family = family, out = src_dir)
    src = _run(
        tmp, layout, ckpt = ckpt, family = family, out = f"again_{layout}"
    )  # uninterrupted, same layout
    got = _run(
        tmp,
        layout,
        ckpt = ckpt,
        family = family,
        out = f"same_{layout}",
        resume_from = os.path.join(tmp, src_dir, f"checkpoint-{_SAVE_AT}"),
    )
    assert got["calls"] == {"weights": 0, "optimizer": 0, "named": 0}, got["calls"]
    assert (
        got["losses"][2:] == src["losses"][2:] and got["grad_norms"][2:] == src["grad_norms"][2:]
    ), (got, src)
    for key, value in src["adapter"].items():
        assert torch.equal(got["adapter"][key], value), key


def _worst(got, want):
    return max(
        float((got[k] - v).abs().max() / v.abs().max().clamp_min(1e-12)) for k, v in want.items()
    )


def _without_optimizer(tmp, src, name):
    """A copy of `src` without optimizer.pt and the per-rank files: a resume of it starts with fresh moments."""
    import shutil

    dst = os.path.join(tmp, name)
    shutil.copytree(src, dst)
    for name in ("optimizer.pt", "unsloth_tp"):
        path = os.path.join(dst, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    return dst


def test_eight_bit_resume_between_tensor_parallel_and_one_process(refs):
    pytest.importorskip("bitsandbytes")
    tmp, ckpt, ref = refs("llama", "adamw_8bit")
    if not ref["optimizer"].startswith("bitsandbytes"):
        pytest.skip(f"bitsandbytes cannot step CPU tensors here ({ref['optimizer']})")
    kw = dict(ckpt = ckpt, family = "llama", optim = "adamw_8bit")
    src = _run(tmp, "tp2", out = "src_tp2_8bit", **kw)
    saved = os.path.join(tmp, "src_tp2_8bit", f"checkpoint-{_SAVE_AT}")
    to_one = _run(tmp, "one", out = "to_one_8bit", resume_from = saved, **kw)
    from_one = _run(
        tmp,
        "tp2",
        out = "from_one_8bit",
        resume_from = os.path.join(tmp, "ref", f"checkpoint-{_SAVE_AT}"),
        **kw,
    )
    fresh = _run(
        tmp,
        "one",
        out = "fresh_8bit",
        resume_from = _without_optimizer(tmp, saved, "no_optimizer"),
        **kw,
    )
    assert to_one["calls"]["named"] == 1 and from_one["calls"]["optimizer"] == 1, (
        to_one["calls"],
        from_one["calls"],
    )
    assert to_one["optimizer"].startswith("bitsandbytes") and from_one["optimizer"].startswith(
        "bitsandbytes"
    )
    # 8-bit blocks follow each rank's shard, so tp 2 and one process already differ uninterrupted (the noise
    # floor); a requantized resume stays within it, a resume with fresh moments (the control) does not.
    floor = _worst(src["adapter"], ref["adapter"])
    worst = {
        arm: _worst(r["adapter"], ref["adapter"])
        for arm, r in (("to_one", to_one), ("from_one", from_one), ("fresh", fresh))
    }
    print(f"8-bit: tp2 vs one uninterrupted {floor:.3e}, resumed vs one {worst}")
    for arm in ("to_one", "from_one"):
        assert worst[arm] <= 2 * floor + 1e-4, (arm, worst, floor)
        assert worst[arm] < worst["fresh"] / 2, (arm, worst)
