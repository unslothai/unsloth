# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_TP_SIZE (models/tp_env.py): tensor parallel from the environment, on a gloo CPU world, vs one process.

The workers set the torchrun environment and let tp_env create the process group and the mesh. The loader /
get_peft_model hooks wrap AutoModelForCausalLM / PEFT the way they wrap FastLanguageModel / FastModel."""

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
    _Rows,
    _free_port,
    _import_checkout_unsloth,
    _raised,
    _save_checkpoint,
    _trainer_args,
)

# These arms start the adapters at init_lora_weights = False and compare 3 whole Trainer steps. At AdamW's
# default eps 1e-8, m / (sqrt(v) + eps) turns the fp32 reduction-order noise between tp = 2 and one process
# into 1e-5 grad-norm and 1e-4 weight drift by step 3 while the losses still match exactly; at 1e-4 it stays
# below 1e-6, and a real tp error (a missed all-reduce) still lands at 3.5e-4. Only these runs take it: the
# native Trainer tests (B = 0 adapters) pass at the default eps on the CI runners and fail at 1e-4 there.
_ADAM_EPS = 1e-4

_ENV_KEYS = (
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "ACCELERATE_USE_FSDP",
)


def _checkpoint(tmp):
    """A tiny random Llama saved as sharded safetensors (index + shards), like a Hub checkpoint."""
    return _save_checkpoint("llama", os.path.join(tmp, "ckpt"))


def _patch_bnb_8bit_everywhere():
    # bitsandbytes keeps 32-bit state below min_8bit_size (4096) elements: every tiny LoRA factor would.
    # transformers builds optim = "adamw_8bit" as bitsandbytes.optim.AdamW(optim_bits = 8).
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


def _case_units(rank, world, tmp):
    """Layout parsing and refusals (no process group)."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from unsloth.models import tp_env

    out = {}

    def layout(**env):
        for key in _ENV_KEYS + ("WORLD_SIZE",):
            os.environ.pop(key, None)
        os.environ.update({k: str(v) for k, v in env.items()})
        tp_env._reset_env_layout()
        try:
            got = tp_env.env_layout()
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        return None if got is None else (got.tp, got.dp, got.world, got.fsdp2)

    out["unset"] = layout(WORLD_SIZE = 4)
    out["one"] = layout(UNSLOTH_TP_SIZE = 1, WORLD_SIZE = 4)
    out["zero"] = layout(UNSLOTH_TP_SIZE = 0, WORLD_SIZE = 4)
    out["text"] = layout(UNSLOTH_TP_SIZE = "two", WORLD_SIZE = 4)
    out["no_divide"] = layout(UNSLOTH_TP_SIZE = 3, WORLD_SIZE = 4)
    out["no_torchrun"] = layout(UNSLOTH_TP_SIZE = 2)
    out["pp"] = layout(UNSLOTH_PP_SIZE = 2, WORLD_SIZE = 4)
    out["pp_with_tp"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 4)
    out["pp_one"] = layout(UNSLOTH_PP_SIZE = 1, UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 4)
    out["tp_world"] = layout(UNSLOTH_TP_SIZE = 4, WORLD_SIZE = 4)
    out["ddp"] = layout(UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 4)
    out["fsdp2"] = layout(UNSLOTH_TP_SIZE = 2, UNSLOTH_FSDP2 = 1, WORLD_SIZE = 4)
    out["accelerate_fsdp"] = layout(
        UNSLOTH_TP_SIZE = 2, UNSLOTH_FSDP2 = 1, UNSLOTH_FSDP2_BACKEND = "accelerate", WORLD_SIZE = 4
    )

    calls = []

    def loader(
        model_name,
        load_in_4bit = True,
        fast_inference = False,
        full_finetuning = False,
        **kw,
    ):
        calls.append(model_name)
        return "loaded"

    layout(UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 2)
    wrapped = tp_env.tensor_parallel_env_load(loader)
    out["refuse_vllm"] = _raised(lambda: wrapped("m", fast_inference = True))
    out["refuse_full_ft"] = _raised(lambda: wrapped("m", False, full_finetuning = True))
    out["refuse_8bit"] = _raised(lambda: wrapped("m", load_in_8bit = True))
    layout(UNSLOTH_TP_SIZE = 2, UNSLOTH_FSDP2 = 1, UNSLOTH_FSDP2_SHARDED_LOAD = 1, WORLD_SIZE = 2)
    out["refuse_fsdp2_sharded_load"] = _raised(lambda: wrapped("m", False))
    os.environ.pop("UNSLOTH_FSDP2_SHARDED_LOAD", None)
    out["refused_before_load"] = calls == []
    layout(WORLD_SIZE = 2)
    out["no_env_passthrough"] = wrapped("m", fast_inference = True)
    out["four_bit_default"] = tp_env._four_bit({}, loader)
    out["four_bit_16"] = tp_env._four_bit({"load_in_16bit": True}, loader)
    out["four_bit_qconfig"] = tp_env._four_bit(
        {"load_in_4bit": False, "quantization_config": {}}, loader
    )
    # The hooks sit on the real loaders.
    from unsloth import FastLanguageModel, FastModel, FastVisionModel

    out["hooked"] = {}
    for label, cls in (
        ("lm", FastLanguageModel),
        ("model", FastModel),
        ("vision", FastVisionModel),
    ):
        for attr in ("from_pretrained", "get_peft_model"):
            out["hooked"][f"{label}.{attr}"] = getattr(getattr(cls, attr), "_unsloth_tp_env", False)
    return out


def _case_env(
    rank,
    world,
    tmp,
    ckpt,
    bs,
    optim = "adamw_torch",
    four_bit = False,
    resume = False,
    bnb_8bit = False,
    refusals = False,
):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from unsloth.models import tp_env, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    if bnb_8bit:
        _patch_bnb_8bit_everywhere()
    layout = tp_env.env_layout()
    seen = []

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, *args, **kwargs):
            seen.append(inputs["input_ids"].sum().item())
            return super().compute_loss(model, inputs, *args, **kwargs)

    def load(
        model_name,
        load_in_4bit = True,
        **kwargs,
    ):  # FastLanguageModel.from_pretrained's flags
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float32, attn_implementation = "eager"
        )

    def peft(model):
        torch.manual_seed(1)
        config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
        return get_peft_model(model, config)

    load = tp_env.tensor_parallel_env_load(load)
    peft = tp_env.tensor_parallel_env_peft(peft)

    def fresh():
        model = load(ckpt, load_in_4bit = four_bit)
        sharded = "_unsloth_tp_load_stats" in model.__dict__
        return peft(model), sharded

    dp = layout.dp if layout is not None else 1
    steps = 4 if resume else 3
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (steps * 4 * max(dp, bs // 2), 9), generator = g)
    data = _Rows(list(ids))

    def run(out_dir, **overrides):
        model, sharded = fresh()
        args = _trainer_args(
            out_dir, bs, optim = optim, max_steps = steps, adam_epsilon = _ADAM_EPS, **overrides
        )
        trainer = SequentialTrainer(model = model, args = args, train_dataset = data)
        return trainer, sharded

    save = dict(save_strategy = "steps", save_steps = 2) if resume else {}
    out_a = os.path.join(tmp, "a")
    trainer, sharded = run(out_a, **save)
    trainer.train()

    def summary(trainer):
        logs = [h for h in trainer.state.log_history if "loss" in h]
        model = trainer.model
        is_tp = tp_native.is_tensor_parallel(model)
        state = (
            tp_native.tp_full_state_dict(model, adapters_only = True)
            if is_tp
            else {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
        )
        return {
            "steps": [h["step"] for h in logs],
            "losses": [h["loss"] for h in logs],
            "grad_norms": [h.get("grad_norm") for h in logs],
            "state": {k: v for k, v in state.items() if "lora_" in k},
        }

    from unsloth.models.fsdp2_bnb_optim import is_local_shard_bnb_optimizer
    from unsloth.models.fsdp2_native import _inner_optimizer, is_native_fsdp2

    res = {
        "a": summary(trainer),
        "sharded_load": sharded,
        "is_tp": tp_native.is_tensor_parallel(trainer.model),
        "native_fsdp2": is_native_fsdp2(trainer.model),
        "ddp_mesh": tp_native._ddp_mesh(trainer.model) is not None,
        "wrapped": type(trainer.model_wrapped).__name__,
        "optimizer": type(_inner_optimizer(trainer.optimizer)).__module__
        + "."
        + type(_inner_optimizer(trainer.optimizer)).__name__,
        "optim_bits": getattr(
            getattr(_inner_optimizer(trainer.optimizer), "args", None), "optim_bits", None
        ),
        "state_dtypes": sorted(
            {
                str(v.dtype)
                for st in _inner_optimizer(trainer.optimizer).state.values()
                for v in st.values()
                if torch.is_tensor(v) and v.ndim > 0
            }
        ),
        "local_shard_bnb": is_local_shard_bnb_optimizer(_inner_optimizer(trainer.optimizer)),
        "optim_arg": str(getattr(trainer.args.optim, "value", trainer.args.optim)),
        "seen": list(seen),
        "layout": None if layout is None else repr(layout),
    }
    if resume:
        ckpt2 = os.path.join(out_a, "checkpoint-2")
        res["ckpt_files"] = sorted(os.listdir(ckpt2)) + sorted(
            os.listdir(os.path.join(ckpt2, "unsloth_tp")) if layout is not None else []
        )
        trainer_b, _ = run(os.path.join(tmp, "b"), **save)
        trainer_b.train(resume_from_checkpoint = ckpt2)
        res["b"] = summary(trainer_b)
        if layout is not None:
            # A checkpoint written on another layout is refused before anything is loaded.
            bad = os.path.join(tmp, "bad")
            if rank == 0:
                import shutil
                shutil.copytree(ckpt2, bad)
            dist.barrier()
            path = os.path.join(bad, "unsloth_tp", f"rank{rank:05d}.pt")
            payload = torch.load(path, weights_only = True)
            payload["layout"]["tp"] = 99
            torch.save(payload, path)
            trainer_c, _ = run(os.path.join(tmp, "c"), **save)
            res["bad_layout"] = _raised(lambda: trainer_c.train(resume_from_checkpoint = bad))
            dist.barrier()
    if refusals:
        model = trainer.model

        class Stream(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter([data[i] for i in range(len(data))])

        res["iterable"] = _raised(
            lambda: SequentialTrainer(
                model = model, args = _trainer_args(os.path.join(tmp, "it"), bs), train_dataset = Stream()
            ).train()
        )
        res["eval"] = _raised(
            lambda: SequentialTrainer(
                model = model,
                args = _trainer_args(
                    os.path.join(tmp, "ev"), bs, eval_strategy = "steps", eval_steps = 1
                ),
                train_dataset = data,
                eval_dataset = data,
            )
        )
        res["evaluate"] = _raised(lambda: trainer.evaluate(data))
        res["lion"] = _raised(
            lambda: SequentialTrainer(
                model = model,
                args = _trainer_args(os.path.join(tmp, "li"), bs, optim = "lion_8bit"),
                train_dataset = data,
            )
        )
        res["generate"] = _raised(lambda: model.generate(input_ids = ids[:1, :3]))
    if dist.is_initialized():
        dist.barrier()
    return res


def _case_dp_average(rank, world, tmp):
    """DDP-style dp averaging: a gradient on one replica only is averaged with zeros (DDP's behaviour), one on
    none stays None, and both replicas end equal."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models import tp_native

    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("dp",))
    one, both, none, frozen = (torch.nn.Parameter(torch.zeros(3)) for _ in range(4))
    frozen.requires_grad_(False)
    one.grad = torch.full((3,), 4.0) if rank == 0 else None
    both.grad = torch.full((3,), float(rank + 1))
    tp_native._dp_average([one, both, none, frozen], mesh["dp"])
    return {
        "one": one.grad.tolist(),
        "both": both.grad.tolist(),
        "none": none.grad,
        "frozen": frozen.grad,
    }


def _case_accelerate_fsdp(rank, world, tmp, ckpt):
    """UNSLOTH_FSDP2=1 that resolves to accelerate's FSDP: the dp dimension would be left unsynced."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from unsloth.models import tp_env
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    model = tp_env.tensor_parallel_env_load(
        lambda name, load_in_4bit = True: AutoModelForCausalLM.from_pretrained(
            name, torch_dtype = torch.float32
        )
    )(ckpt, load_in_4bit = False)
    model = get_peft_model(model, LoraConfig(r = 4, target_modules = "all-linear"))
    data = _Rows([torch.arange(9)] * 8)
    return {
        "refused": _raised(
            lambda: transformers.Trainer(
                model = model, args = _trainer_args(tmp, 2), train_dataset = data
            )
        )
    }


CASES = {
    "units": _case_units,
    "env": _case_env,
    "dp_average": _case_dp_average,
    "accelerate_fsdp": _case_accelerate_fsdp,
}


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
    torch.set_num_threads(2)
    import torch.distributed as dist

    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(
    case,
    world = 1,
    env = None,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        if case in ("env", "accelerate_fsdp"):
            kwargs["ckpt"] = _checkpoint(tmp)
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
            for r in range(world)
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
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def _close(
    got,
    want,
    rel = 1e-5,
    what = "",
):
    assert got["steps"] == want["steps"], (what, got["steps"], want["steps"])
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
    assert set(got["state"]) == set(want["state"]) and want["state"]
    for key, value in want["state"].items():
        torch.testing.assert_close(got["state"][key], value, rtol = rel * 10, atol = 1e-6, msg = key)


def test_env_layout_parsing_and_refusals():
    (res,) = _spawn("units")
    assert res["unset"] is None and res["one"] is None
    assert "must be >= 1" in res["zero"] and "must be an integer" in res["text"]
    assert "must divide the number of ranks (WORLD_SIZE=4)" in res["no_divide"]
    assert (
        "WORLD_SIZE=1" in res["no_torchrun"] and "torchrun --nproc-per-node 2" in res["no_torchrun"]
    )
    for key in ("pp", "pp_with_tp"):
        assert "UNSLOTH_PP_SIZE=2" in res[key] and "Python API" in res[key], res[key]
    assert res["pp_one"] == (2, 2, 4, False)
    assert res["tp_world"] == (4, 1, 4, False)
    assert res["ddp"] == (2, 2, 4, False)
    assert res["fsdp2"] == (2, 2, 4, True)
    assert "native FSDP2 only" in res["accelerate_fsdp"]
    assert "vLLM" in res["refuse_vllm"] and "UNSLOTH_TP_SIZE=2" in res["refuse_vllm"]
    assert "full finetuning" in res["refuse_full_ft"]
    assert "8-bit weights" in res["refuse_8bit"]
    assert "UNSLOTH_FSDP2_SHARDED_LOAD=1" in (res["refuse_fsdp2_sharded_load"] or ""), res[
        "refuse_fsdp2_sharded_load"
    ]
    assert res["refused_before_load"] and res["no_env_passthrough"] == "loaded"
    assert res["four_bit_default"] and not res["four_bit_16"] and res["four_bit_qconfig"]
    assert all(res["hooked"].values()), res["hooked"]


def _ref(bs, **kwargs):
    return _spawn("env", 1, bs = bs, **kwargs)[0]


@pytest.mark.parametrize("four_bit", [False, True], ids = ["sharded_16bit", "full_load"])
def test_env_tp2_trainer_matches_one_process(four_bit):
    ref = _ref(2, four_bit = four_bit)
    assert ref["layout"] is None and not ref["is_tp"]
    results = _spawn("env", 2, {"UNSLOTH_TP_SIZE": "2"}, bs = 2, four_bit = four_bit, refusals = True)
    for rank, res in enumerate(results):
        assert res["layout"] == "tp 2" and res["is_tp"], res
        assert res["sharded_load"] is (not four_bit), res["sharded_load"]
        assert not res["native_fsdp2"] and not res["ddp_mesh"]
        assert "DistributedDataParallel" not in res["wrapped"]
        _close(res["a"], ref["a"], what = rank)
        hint = "UNSLOTH_TP_SIZE=2: unset it"
        assert "map-style dataset" in res["iterable"] and hint in res["iterable"], res["iterable"]
        assert "evaluation during training" in res["eval"] and hint in res["eval"], res["eval"]
        assert "Trainer.evaluate is not supported" in res["evaluate"] and hint in res["evaluate"]
        assert "lion_8bit" in res["lion"], res["lion"]
        assert "generation is not supported" in res["generate"], res[
            "generate"
        ]  # A3: still refused
    assert results[0]["seen"] == results[1]["seen"]  # both tp ranks read the same batches


def test_env_tp2_with_fsdp2_flag_and_no_dp_dimension():
    ref = _ref(2)
    for res in _spawn("env", 2, {"UNSLOTH_TP_SIZE": "2", "UNSLOTH_FSDP2": "1"}, bs = 2):
        assert res["layout"] == "tp 2" and res["is_tp"] and not res["native_fsdp2"], res
        _close(res["a"], ref["a"])


@pytest.mark.parametrize("fsdp2", [True, False], ids = ["fsdp2_dp2", "ddp_dp2"])
def test_env_tp2_x_dp2_trainer_matches_one_process(fsdp2):
    ref = _ref(4)
    env = {"UNSLOTH_TP_SIZE": "2"}
    if fsdp2:
        env["UNSLOTH_FSDP2"] = "1"
    results = _spawn("env", 4, env, bs = 2)
    for rank, res in enumerate(results):
        assert res["layout"] == (
            "dp 2 (native FSDP2) x tp 2" if fsdp2 else "dp 2 (DDP) x tp 2"
        ), res
        assert res["is_tp"] and res["native_fsdp2"] is fsdp2 and res["ddp_mesh"] is (not fsdp2)
        assert "DistributedDataParallel" not in res["wrapped"]
        _close(res["a"], ref["a"], what = rank)
    for rank, res in enumerate(results):
        assert res["seen"] == results[rank ^ 1]["seen"]  # tp peers: same rows
        assert res["seen"] != results[(rank + 2) % 4]["seen"]  # dp replicas: different rows


def _bnb_cpu():
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _bnb_cpu(), reason = "bitsandbytes not installed")
@pytest.mark.parametrize(
    "world,env", [(2, {}), (4, {"UNSLOTH_FSDP2": "1"})], ids = ["tp2", "tp2_fsdp2_dp2"]
)
def test_env_tp_adamw_8bit_steps_local_shards(world, env):
    ref = _ref(2 * (world // 2), optim = "adamw_8bit", bnb_8bit = True)
    if not ref["optimizer"].startswith("bitsandbytes"):
        pytest.skip(f"bitsandbytes cannot step CPU tensors here ({ref['optimizer']})")
    assert ref["optim_bits"] == 8 and "torch.uint8" in ref["state_dtypes"], ref
    results = _spawn(
        "env", world, {"UNSLOTH_TP_SIZE": "2", **env}, bs = 2, optim = "adamw_8bit", bnb_8bit = True
    )
    for rank, res in enumerate(results):
        assert res["optim_arg"] == "adamw_8bit" and res["local_shard_bnb"], res["optimizer"]
        assert res["optim_bits"] == 8 and "torch.uint8" in res["state_dtypes"], res["state_dtypes"]
        # 8-bit state is quantized per 256-element block of each local shard (one GPU: of the whole
        # tensor), so the run matches one process within the 8-bit band, not bitwise.
        assert res["a"]["losses"] == pytest.approx(ref["a"]["losses"], rel = 2e-3), (
            rank,
            res["a"]["losses"],
            ref["a"]["losses"],
        )
        for key, value in ref["a"]["state"].items():
            torch.testing.assert_close(res["a"]["state"][key], value, rtol = 0.05, atol = 2e-3, msg = key)


@pytest.mark.parametrize(
    "world,env,optim",
    [
        (2, {}, "adamw_torch"),
        (4, {"UNSLOTH_FSDP2": "1"}, "adamw_torch"),
        (4, {}, "adamw_torch"),
        (2, {}, "adamw_8bit"),
    ],
    ids = ["tp2", "tp2_fsdp2_dp2", "tp2_ddp_dp2", "tp2_adamw_8bit"],
)
def test_env_tp_checkpoint_resume_equals_uninterrupted(world, env, optim):
    if optim == "adamw_8bit" and not _bnb_cpu():
        pytest.skip("bitsandbytes not installed")
    results = _spawn(
        "env",
        world,
        {"UNSLOTH_TP_SIZE": "2", **env},
        bs = 2,
        optim = optim,
        resume = True,
        bnb_8bit = optim == "adamw_8bit",
    )
    for rank, res in enumerate(results):
        a, b = res["a"], dict(res["b"])
        # The resumed log history starts with steps 1-2 restored from trainer_state.json.
        assert a["steps"] == [1, 2, 3, 4] and b["steps"] == [1, 2, 3, 4], (a["steps"], b["steps"])
        b["losses"], b["grad_norms"] = b["losses"][2:], b["grad_norms"][2:]
        assert "adapter_model.safetensors" in res["ckpt_files"], res["ckpt_files"]
        assert f"rank{rank:05d}.pt" in res["ckpt_files"], res["ckpt_files"]
        # Bitwise: the same local shards, optimizer state, RNG and data position as the uninterrupted run.
        assert b["losses"] == a["losses"][2:], (rank, b["losses"], a["losses"])
        assert b["grad_norms"] == a["grad_norms"][2:], (rank, b["grad_norms"], a["grad_norms"])
        for key, value in a["state"].items():
            assert torch.equal(b["state"][key], value), key
        assert "written on another layout" in (res["bad_layout"] or ""), res["bad_layout"]


def test_dp_average_matches_ddp_for_gradients_on_some_replicas():
    for res in _spawn("dp_average", 2):
        assert res["one"] == [2.0] * 3 and res["both"] == [1.5] * 3, res
        assert res["none"] is None and res["frozen"] is None, res


def test_env_tp_x_accelerate_fsdp_is_refused():
    env = {"UNSLOTH_TP_SIZE": "2", "UNSLOTH_FSDP2": "1", "ACCELERATE_USE_FSDP": "true"}
    for res in _spawn("accelerate_fsdp", 4, env):
        assert "needs Unsloth's native FSDP2" in (res["refused"] or ""), res
