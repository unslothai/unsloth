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
    aux = False,
    adapter_name = "default",
    modules_to_save = None,
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
                r = 4,
                lora_alpha = 8,
                target_modules = "all-linear",
                init_lora_weights = False,
                modules_to_save = modules_to_save,
            )

            def peft(m):
                torch.manual_seed(1)
                m = get_peft_model(m, config, adapter_name = adapter_name)
                if aux:  # a frozen, inactive second adapter: PEFT saves it in <checkpoint>/aux
                    m.add_adapter("aux", config)
                    if resume_from is None:
                        with torch.no_grad():
                            for n, p in m.named_parameters():
                                if ".aux." in n:
                                    p.fill_(0.25)
                return m

            return tp_env.tensor_parallel_env_peft(peft)(load(ckpt, load_in_4bit = False))

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
    aux_state = {}
    if aux:
        import re

        from unsloth.models import pp_native

        stage = pp_native.pipeline_stage_of(trainer.model)
        for k, v in trainer.model.state_dict().items():
            layer = re.search(r"\.layers\.(\d+)\.", k)
            if ".aux." not in k or (
                stage is not None and not stage.start <= int(layer.group(1)) < stage.end
            ):
                continue  # another stage's layer (a stand-in here)
            aux_state[k] = v.detach().clone()
    return {
        "aux": aux_state,
        "steps": [h["step"] for h in logs],
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h["grad_norm"] for h in logs],
        "grads": grads,
        "adapter": load_file(
            os.path.join(
                final,
                "" if adapter_name == "default" else adapter_name,
                "adapter_model.safetensors",
            )
        ),
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


def test_frozen_saved_tensors_are_restored_on_another_layout(tmp_path):
    """An inactive second adapter (frozen, saved by PEFT under <checkpoint>/aux) comes back from a one-process
    checkpoint under pp 2 with its saved values, not the fresh ones (tensor parallel takes one adapter)."""
    _needs_pipelining(["pp2"])
    tmp = str(tmp_path)
    ckpt = _save_checkpoint("llama", os.path.join(tmp, "base"), {"num_hidden_layers": 4})
    src = _run(tmp, "one", ckpt = ckpt, family = "llama", out = "aux_src", aux = True)
    got = _run(
        tmp,
        "pp2",
        ckpt = ckpt,
        family = "llama",
        out = "aux_pp2",
        aux = True,
        resume_from = os.path.join(tmp, "aux_src", f"checkpoint-{_SAVE_AT}"),
    )
    assert got["aux"] and set(got["aux"]) <= set(src["aux"]), sorted(
        set(got["aux"]) - set(src["aux"])
    )[:4]
    for key, value in got["aux"].items():
        assert float(src["aux"][key].abs().max()) == 0.25, key  # the saved, perturbed values
        torch.testing.assert_close(value, src["aux"][key], msg = key)


def test_named_adapter_checkpoint_resumes_under_tensor_parallel(tmp_path):
    """The model's only adapter is not "default": PEFT saves it in <checkpoint>/<name>, and a one-process
    checkpoint of it resumes under tp 2 as the uninterrupted run continued."""
    tmp = str(tmp_path)
    ckpt = _save_checkpoint("llama", os.path.join(tmp, "base"), {"num_hidden_layers": 4})
    kw = dict(ckpt = ckpt, family = "llama", adapter_name = "main")
    src = _run(tmp, "one", out = "named_src", **kw)
    got = _run(
        tmp,
        "tp2",
        out = "named_tp2",
        resume_from = os.path.join(tmp, "named_src", f"checkpoint-{_SAVE_AT}"),
        **kw,
    )
    assert got["calls"]["weights"] == 1 and got["calls"]["optimizer"] == 1, got["calls"]
    _close(got, src, "one -> tp2, adapter 'main'", grads = False)


def _portable_module():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
    from unsloth_models_loader import load_models_module
    return load_models_module("portable_resume")


def test_positional_state_keeps_the_optimizer_parameter_order():
    """A 1-GPU (int-keyed) optimizer state maps through the optimizer's own order, not module order: a group
    built as [second, first] keeps each moment on its own parameter."""
    portable = _portable_module()
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias = False), torch.nn.Linear(2, 2, bias = False)
    )
    first, second = model[0].weight, model[1].weight
    optimizer = torch.optim.AdamW([second, first], lr = 1e-2)
    first.grad, second.grad = torch.ones(2, 2), torch.full((2, 2), 3.0)
    optimizer.step()
    named = portable.as_named_state(model, optimizer, optimizer.state_dict())
    assert named["param_groups"][0]["params"] == ["1.weight", "0.weight"], named["param_groups"]
    assert torch.equal(named["state"]["0.weight"]["exp_avg"], optimizer.state[first]["exp_avg"])
    assert torch.equal(named["state"]["1.weight"]["exp_avg"], optimizer.state[second]["exp_avg"])


def test_one_process_checkpoint_with_modules_to_save_resumes_under_tensor_parallel(tmp_path):
    """A trained plain tensor (modules_to_save) next to the LoRA DTensors: tensor parallel splits the optimizer
    group into a DTensor and a plain group, the 1-GPU checkpoint has one; the positions (and the LR scheduler's
    per-group entries) still map."""
    tmp = str(tmp_path)
    ckpt = _save_checkpoint("llama", os.path.join(tmp, "base"), {"num_hidden_layers": 4})
    kw = dict(ckpt = ckpt, family = "llama", modules_to_save = ["lm_head"])
    src = _run(tmp, "one", out = "mts_src", **kw)
    got = _run(
        tmp,
        "tp2",
        out = "mts_tp2",
        resume_from = os.path.join(tmp, "mts_src", f"checkpoint-{_SAVE_AT}"),
        **kw,
    )
    tp_ref = _run(tmp, "tp2", out = "mts_tp2_ref", **kw)

    assert got["calls"]["weights"] == 1 and got["calls"]["optimizer"] == 1, got["calls"]
    # tp 2's own spread from one process on the vocab-sized lm_head gradient (2e-4 relative in the grad norm) is
    # above the shared tolerance: compared with the uninterrupted tp 2 run, and its loss with one process.
    assert got["losses"][2:] == pytest.approx(src["losses"][2:], rel = 1e-5), (
        got["losses"],
        src["losses"],
    )
    _close(got, tp_ref, "one -> tp2, modules_to_save", grads = False)


def _case_save_gate(
    rank,
    world,
    tmp,
    full,
    max_gb,
    rank_r = 4,
    overrides = None,
    repeats = 0,
):
    """apply_tensor_parallel on tp 2 (LoRA, or full finetuning), one AdamW step, then the Trainer checkpoint
    save: how many gathers the portable optimizer.pt ran, whether it was written, and (repeats) ms per save."""
    import time
    import types

    from test_tp_native_oracle import LLAMA, build, tpn
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models import portable_resume

    if max_gb is not None:
        os.environ["UNSLOTH_PORTABLE_RESUME_MAX_GB"] = str(max_gb)
    gathers = [0]
    full_value = portable_resume._full_value

    def counted(*args, **kwargs):
        gathers[0] += 1
        return full_value(*args, **kwargs)

    portable_resume._full_value = counted
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    model = build(LLAMA, overrides, lora = not full, r = rank_r, lora_alpha = 2 * rank_r).train()
    if full:
        for p in model.parameters():
            p.requires_grad_(True)
    tpn.apply_tensor_parallel(model, mesh)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr = 1e-3)
    ids = torch.randint(0, 64, (2, 8), generator = torch.Generator().manual_seed(0))
    model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
    optimizer.step()
    trainer = types.SimpleNamespace(
        model = model,
        optimizer = optimizer,
        lr_scheduler = None,
        args = types.SimpleNamespace(should_save = rank == 0),
    )
    directory = os.path.join(tmp, f"ckpt_{int(full)}_{rank_r}")
    tpn._tp_save_optimizer_and_scheduler(trainer, directory)
    out = {
        "gathers": gathers[0],
        "written": os.path.isfile(os.path.join(directory, "optimizer.pt")),
        "trainable": sum(p.numel() for p in params),
    }
    if repeats:
        import torch.distributed as dist
        for env in ("0", "1"):
            os.environ["UNSLOTH_PORTABLE_RESUME"] = env
            times = []
            for _ in range(repeats):
                dist.barrier()
                start = time.perf_counter()
                tpn._tp_save_optimizer_and_scheduler(trainer, directory)
                times.append(1000 * (time.perf_counter() - start))
            out[f"ms_{env}"] = sorted(times)
        out["portable_mb"] = os.path.getsize(os.path.join(directory, "optimizer.pt")) / 2**20
    return out


def _gate_worker(rank, world, port, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = str(rank), WORLD_SIZE = str(world)
    )
    for key in _ENV_KEYS + ("UNSLOTH_PORTABLE_RESUME", "UNSLOTH_PORTABLE_RESUME_MAX_GB"):
        os.environ.pop(key, None)
    torch.set_num_threads(2)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        res = _case_save_gate(rank, world, tmp, **kwargs)
    except Exception as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:2000] + traceback.format_exc()[-6000:]}
    torch.save(res, os.path.join(tmp, f"gate_{rank}.pt"))
    dist.destroy_process_group()


def _run_gate(tmp, **kwargs):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target = _gate_worker, args = (r, 2, port, tmp, kwargs)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(600)
    assert [p.exitcode for p in procs] == [0, 0]
    results = [torch.load(os.path.join(tmp, f"gate_{r}.pt"), weights_only = False) for r in range(2)]
    for r in results:
        assert "error" not in r, r["error"]
    return results


@pytest.mark.parametrize("full", [False, True], ids = ["lora", "full_finetuning"])
def test_portable_optimizer_is_skipped_for_full_finetuning(full, tmp_path):
    """tp 2 save: LoRA (a small share of the model) gathers and writes the 1-GPU optimizer.pt; full finetuning
    over the byte budget (set to 1 KiB here, a tiny model's moments) gathers nothing and writes none."""
    for rank, r in enumerate(_run_gate(str(tmp_path), full = full, max_gb = 2**-20)):
        if full:
            assert r["gathers"] == 0 and not r["written"], (rank, r)
        else:
            assert r["gathers"] > 0 and r["written"], (rank, r)
