# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 sharded resume on a gloo CPU world of 2 (fsdp2_native.py loaded on its own, no GPU):

* weights: a full-model safetensors checkpoint whose keys match the sharded state dict is read with
  dcp.load + HuggingFaceStorageReader (each rank its own slices); the result is bitwise equal to the
  full load, for untied and tied embeddings and a multi-file index;
* fallbacks to the full load: a .bin checkpoint, renamed keys, a stray safetensors file, a dtype
  that differs, a LoRA adapter checkpoint, no usable reader (torch < 2.9, 2.8's signature);
* optimizer: the sharded directory (native_save_optimizer_sharded) restores the same state, bitwise,
  as the full optimizer.pt;
* Trainer: save_steps checkpoints and resume_from_checkpoint with UNSLOTH_FSDP2_DCP_OPTIM=1 and =0
  end bitwise equal to the uninterrupted run, and a reused checkpoint directory keeps one format."""

import datetime
import os
import sys

import pytest
import torch

if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _hf_reader_available():
    import inspect

    import torch.distributed.checkpoint as dcp

    reader = getattr(dcp, "HuggingFaceStorageReader", None)
    return reader is not None and "thread_count" in inspect.signature(reader.__init__).parameters


def _model(
    seed,
    tied = False,
    lora = False,
):
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None  # tiny repo's -1 fails save_pretrained's config check
    config.tie_word_embeddings = tied
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None
    if tied:
        model.tie_weights()
    elif model.lm_head.weight is model.model.embed_tokens.weight:
        # Some transformers versions tie this tiny config regardless of the flag.
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied
    g = torch.Generator().manual_seed(seed)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02, generator = g)
    elif seed:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator = g) * 0.01)
    return model


def _batch(step, rank):
    g = torch.Generator().manual_seed(100 * step + rank)
    ids = torch.randint(3, 1000, (2, 16), generator = g)
    return {"input_ids": ids, "labels": ids.clone(), "use_cache": False}


def _step(model, opt, step, rank):
    loss = model(**_batch(step, rank)).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none = True)
    return loss.detach().clone()


def _full_model(model):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options = StateDictOptions(full_state_dict = True))
    return {k: v.detach().clone() for k, v in sd.items()}


def _full_optim(model, opt):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    return get_optimizer_state_dict(model, opt, options = StateDictOptions(full_state_dict = True))


def _diff_models(a, b):
    """Names whose tensors are not bitwise equal (or missing on one side)."""
    out = sorted(set(a) ^ set(b))
    out += [k for k in a if k in b and not torch.equal(a[k], b[k])]
    return out


def _diff_optims(a, b):
    out = sorted(set(a["state"]) ^ set(b["state"]))
    for k, v in a["state"].items():
        w = b["state"].get(k, {})
        for name, t in v.items():
            u = w.get(name)
            if isinstance(t, torch.Tensor):
                if not isinstance(u, torch.Tensor) or not torch.equal(t, u):
                    out.append(f"{k}.{name}")
            elif t != u:
                out.append(f"{k}.{name}")

    def _plain(groups):
        # The full load hands Adam's betas back as a list, the sharded one as the tuple it was.
        return [{k: list(v) if isinstance(v, tuple) else v for k, v in g.items()} for g in groups]

    if _plain(a["param_groups"]) != _plain(b["param_groups"]):
        out.append(("param_groups", a["param_groups"], b["param_groups"]))
    return out


def _init(
    rank,
    world,
    port,
    timeout = 300,
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = timeout)
    )
    return dist


def _spy(native):
    """Records what each _native_load_hf_sharded call returned (None = the full load ran)."""
    calls = []
    original = native._native_load_hf_sharded

    def spy(root, checkpoint):
        n = original(root, checkpoint)
        calls.append(n)
        return n

    native._native_load_hf_sharded = spy
    return calls


def _case_roundtrip(native, dist, rank, tmp, tied, shard_files):
    cfg = native.FSDP2Config(param_dtype = torch.float32)
    calls = _spy(native)

    def _new(seed):
        m = native.shard_model(_model(seed, tied = tied), cfg)
        o = torch.optim.AdamW(m.parameters(), lr = 1e-2, foreach = False)
        return m, o

    model, opt = _new(1)
    for step in (1, 2):
        _step(model, opt, step, rank)
    ckpt = os.path.join(tmp, "ckpt")
    state = native.native_full_state_dict(model)
    if rank == 0:
        kwargs = {"max_shard_size": "200KB"} if shard_files else {}
        model.save_pretrained(ckpt, state_dict = state, safe_serialization = True, **kwargs)
    native.native_save_optimizer(
        model, opt, os.path.join(ckpt, "optimizer.pt"), should_save = rank == 0
    )
    native.native_save_optimizer_sharded(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    dist.barrier()
    want = [_step(model, opt, step, rank).item() for step in (3, 4)]

    res = {"files": sorted(os.listdir(ckpt))}
    arms = {}
    for arm in ("old", "new"):
        if arm == "old":
            os.environ["UNSLOTH_FSDP2_DCP_RESUME"] = "0"
        else:
            os.environ.pop("UNSLOTH_FSDP2_DCP_RESUME", None)
        m, o = _new(2)  # different fresh values: the load must replace them
        native.native_load_model_checkpoint(m, ckpt)
        opt_src = "optimizer.pt" if arm == "old" else "optimizer_dcp"
        native.native_load_optimizer(m, o, os.path.join(ckpt, opt_src))
        arms[arm] = {"model": _full_model(m), "optim": _full_optim(m, o)}
        arms[arm]["losses"] = [_step(m, o, step, rank).item() for step in (3, 4)]
        arms[arm]["after"] = _full_model(m)
        if tied:
            arms[arm]["still_tied"] = m.lm_head.weight is m.model.embed_tokens.weight
    os.environ.pop("UNSLOTH_FSDP2_DCP_RESUME", None)
    res["calls"] = calls
    res["model_diff"] = _diff_models(arms["old"]["model"], arms["new"]["model"])
    res["optim_diff"] = _diff_optims(arms["old"]["optim"], arms["new"]["optim"])
    res["after_diff"] = _diff_models(arms["old"]["after"], arms["new"]["after"])
    res["want"] = want
    res["losses"] = {arm: arms[arm]["losses"] for arm in arms}
    res["n_tensors"] = len(arms["new"]["model"])
    if tied:
        res["still_tied"] = {arm: arms[arm]["still_tied"] for arm in arms}
    return res


def _case_fallbacks(native, dist, rank, tmp):
    import shutil

    from safetensors.torch import load_file, save_file

    cfg = native.FSDP2Config(param_dtype = torch.float32)
    calls = _spy(native)
    src = native.shard_model(_model(1), cfg)
    want = _full_model(src)
    state = native.native_full_state_dict(src)
    base = os.path.join(tmp, "base")
    dirs = {k: os.path.join(tmp, k) for k in ("bin", "renamed", "stray", "bf16", "gate", "sig")}
    if rank == 0:
        src.save_pretrained(base, state_dict = state, safe_serialization = True)
        saved = load_file(os.path.join(base, "model.safetensors"))
        for k in ("bin", "renamed", "stray", "bf16", "gate", "sig"):
            shutil.copytree(base, dirs[k])
        # transformers 5 no longer writes .bin weights: an old-style checkpoint by hand.
        os.remove(os.path.join(dirs["bin"], "model.safetensors"))
        torch.save(saved, os.path.join(dirs["bin"], "pytorch_model.bin"))
        save_file(
            {f"base.{k}": v for k, v in saved.items()},
            os.path.join(dirs["renamed"], "model.safetensors"),
            metadata = {"format": "pt"},
        )
        save_file({"x": torch.zeros(1)}, os.path.join(dirs["stray"], "extra.safetensors"))
        save_file(
            {k: v.to(torch.bfloat16) for k, v in saved.items()},
            os.path.join(dirs["bf16"], "model.safetensors"),
            metadata = {"format": "pt"},
        )
    dist.barrier()
    res = {}
    for name in ("bin", "stray", "renamed", "bf16"):
        dst = native.shard_model(_model(2), cfg)
        before = len(calls)
        try:
            native.native_load_model_checkpoint(dst, dirs[name])
            got = _full_model(dst)
            res[name] = {"diff": _diff_models(want, got) if name != "bf16" else None}
        except Exception as exc:
            res[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        res[name]["calls"] = calls[before:]

    # torch < 2.9 (no reader with thread_count): simulated at the gate.
    original_gate = native._hf_reader_class
    native._hf_reader_class = lambda: None
    dst = native.shard_model(_model(2), cfg)
    before = len(calls)
    native.native_load_model_checkpoint(dst, dirs["gate"])
    res["gate"] = {"calls": calls[before:], "diff": _diff_models(want, _full_model(dst))}
    native._hf_reader_class = original_gate

    # torch 2.8's public reader has another signature: the probe must refuse it.
    import torch.distributed.checkpoint as dcp

    class Reader28:
        def __init__(
            self,
            path,
            token = None,
        ):
            pass

    held = getattr(dcp, "HuggingFaceStorageReader", None)
    dcp.HuggingFaceStorageReader = Reader28
    try:
        res["sig_probe"] = native._hf_reader_class()
        dst = native.shard_model(_model(2), cfg)
        before = len(calls)
        native.native_load_model_checkpoint(dst, dirs["sig"])
        res["sig"] = {"calls": calls[before:], "diff": _diff_models(want, _full_model(dst))}
    finally:
        dcp.HuggingFaceStorageReader = held
    res["sig_probe"] = res["sig_probe"] is not None

    # A LoRA adapter checkpoint stays on the adapter load.
    lsrc = native.shard_model(_model(1, lora = True), cfg)
    lwant = {n: v for n, v in _full_model(lsrc).items() if "lora_" in n}
    lstate = native.native_full_state_dict(lsrc, trainable_only = True)
    lora_dir = os.path.join(tmp, "lora")
    if rank == 0:
        lsrc.save_pretrained(lora_dir, state_dict = lstate, safe_serialization = True)
    dist.barrier()
    ldst = native.shard_model(_model(2, lora = True), cfg)
    before = len(calls)
    native.native_load_model_checkpoint(ldst, lora_dir)
    lgot = {n: v for n, v in _full_model(ldst).items() if "lora_" in n}
    res["lora"] = {
        "calls": calls[before:],
        "diff": _diff_models(lwant, lgot),
        "n": len(lwant),
        "files": sorted(os.listdir(lora_dir)),
    }
    return res


def _case_trainer(native, dist, rank, tmp):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    cfg = native.FSDP2Config(param_dtype = torch.float32)
    rows = [
        {"input_ids": list(range(5 + i, 21 + i)), "labels": list(range(5 + i, 21 + i))}
        for i in range(4)
    ]
    ds = Dataset.from_list(rows * 4)

    def _args(out):
        return TrainingArguments(
            output_dir = out,
            report_to = [],
            use_cpu = True,
            max_steps = 4,
            save_steps = 2,
            save_strategy = "steps",
            learning_rate = 1e-2,
            per_device_train_batch_size = 2,
            lr_scheduler_type = "linear",
            logging_steps = 1,
            seed = 0,
            data_seed = 0,
            dataloader_num_workers = 0,
        )

    res = {}
    for fmt in ("dcp", "full"):
        out = os.path.join(tmp, f"out_{fmt}")
        os.environ["UNSLOTH_FSDP2_DCP_OPTIM"] = "1" if fmt == "dcp" else "0"
        model = native.shard_model(_model(1), cfg)
        Trainer(model = model, args = _args(out), train_dataset = ds).train()
        want = _full_model(model)
        ck2 = os.path.join(out, "checkpoint-2")
        files = sorted(os.listdir(ck2))
        dist.barrier()
        # Resume with the flag flipped: the checkpoint's own format is read either way, and the
        # re-save of checkpoint-4 must replace the other format, not sit beside it.
        os.environ["UNSLOTH_FSDP2_DCP_OPTIM"] = "0" if fmt == "dcp" else "1"
        resumed = native.shard_model(_model(2), cfg)
        Trainer(model = resumed, args = _args(out), train_dataset = ds).train(resume_from_checkpoint = ck2)
        got = _full_model(resumed)
        dist.barrier()
        res[fmt] = {
            "ck2_files": files,
            "ck4_files": sorted(os.listdir(os.path.join(out, "checkpoint-4"))),
            "diff": _diff_models(want, got),
            "n": len(want),
        }
    os.environ.pop("UNSLOTH_FSDP2_DCP_OPTIM", None)
    return res


class _ReaderError(OSError):
    pass


def _case_diverge(native, dist, rank, tmp, part):
    """Two ranks with different environment / torch / file views must still take one path."""
    import shutil
    import types

    import torch.distributed.checkpoint as dcp
    from transformers import Trainer

    cfg = native.FSDP2Config(param_dtype = torch.float32)
    res = {}
    if part in ("env", "gate", "read"):
        calls = _spy(native)
        src = native.shard_model(_model(1), cfg)
        want = _full_model(src)
        state = native.native_full_state_dict(src)
        ckpt = os.path.join(tmp, "ckpt")
        if rank == 0:
            src.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
        dist.barrier()

        def _load(name):
            dst = native.shard_model(_model(2), cfg)
            before = len(calls)
            native.native_load_model_checkpoint(dst, ckpt)
            res[name] = {"calls": calls[before:], "diff": _diff_models(want, _full_model(dst))}

        if part == "env":
            # Rank 1 alone turns the sharded read off.
            if rank == 1:
                os.environ["UNSLOTH_FSDP2_DCP_RESUME"] = "0"
            _load("env")
            os.environ.pop("UNSLOTH_FSDP2_DCP_RESUME", None)
        elif part == "gate":
            # Rank 1 alone has no usable reader (another torch build).
            if rank == 1:
                native._hf_reader_class = lambda: None
            _load("gate")
        else:
            # A read error, on rank 1 alone and on both: dcp.load raises a CheckpointException.
            reader_cls = dcp.HuggingFaceStorageReader
            held_read = reader_cls.read_data
            for name, ranks in (("read_error_r1", (1,)), ("read_error_both", (0, 1))):

                def _fail(self, *args, **kwargs):
                    raise _ReaderError("injected unreadable shard")

                if rank in ranks:
                    reader_cls.read_data = _fail
                _load(name)
                reader_cls.read_data = held_read
        return res

    model = native.shard_model(_model(1), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False)
    _step(model, opt, 1, rank)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    args = types.SimpleNamespace(should_save = rank == 0)
    if part == "save":
        # Optimizer save through the Trainer hook with a different flag per rank.
        trainer = types.SimpleNamespace(
            model = model, optimizer = opt, lr_scheduler = sched, args = args, is_fsdp_enabled = False
        )
        save_dir = os.path.join(tmp, "opt_mixed_flag")
        os.makedirs(save_dir, exist_ok = True)  # save_model made it, in a Trainer checkpoint
        os.environ["UNSLOTH_FSDP2_DCP_OPTIM"] = "1" if rank == 0 else "0"
        Trainer._save_optimizer_and_scheduler(trainer, save_dir)
        os.environ.pop("UNSLOTH_FSDP2_DCP_OPTIM", None)
        dist.barrier()
        res["save_mixed_flag_files"] = sorted(os.listdir(save_dir))
        return res

    # Node-local views: every rank passes its own directory. both = .pt + dcp + scheduler,
    # pt = .pt + scheduler, dcp = dcp + scheduler.
    want_opt = _full_optim(model, opt)
    views = {k: os.path.join(tmp, f"view_{k}") for k in ("both", "pt", "dcp")}
    native.native_save_optimizer_sharded(model, opt, os.path.join(views["both"], "optimizer_dcp"))
    native.native_save_optimizer(
        model, opt, os.path.join(views["both"], "optimizer.pt"), should_save = rank == 0
    )
    if rank == 0:
        torch.save(sched.state_dict(), os.path.join(views["both"], "scheduler.pt"))
        shutil.copytree(views["both"], views["pt"])
        shutil.rmtree(os.path.join(views["pt"], "optimizer_dcp"))
        shutil.copytree(views["both"], views["dcp"])
        os.remove(os.path.join(views["dcp"], "optimizer.pt"))
    dist.barrier()

    def _resume(name, view):
        m = native.shard_model(_model(2), cfg)
        o = torch.optim.AdamW(m.parameters(), lr = 1e-2, foreach = False)
        sc = torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0)
        t = types.SimpleNamespace(
            model = m, optimizer = o, lr_scheduler = sc, args = args, is_fsdp_enabled = False
        )
        try:
            Trainer._load_optimizer_and_scheduler(t, view)
            res[name] = {"diff": _diff_optims(want_opt, _full_optim(m, o))}
        except RuntimeError as exc:
            res[name] = {"error": str(exc)[:200]}

    # Rank 0 sees both formats, rank 1 only optimizer.pt: both read optimizer.pt.
    _resume("load_both_vs_pt", views["both"] if rank == 0 else views["pt"])
    # Rank 0 sees only the sharded dir, rank 1 only optimizer.pt: no common format, both refuse.
    _resume("load_dcp_vs_pt", views["dcp"] if rank == 0 else views["pt"])
    return res


def _case_amsgrad(native, dist, rank, tmp):
    """A replicated (ignored) parameter in an AMSGrad group, resumed into an optimizer built with
    amsgrad = False: the restored hyperparameters ask for max_exp_avg_sq on the next step."""
    cfg_holder = {}

    def _new(seed, amsgrad):
        model = _model(seed)
        first = model.model.layers[0].input_layernorm.weight
        cfg = native.FSDP2Config(param_dtype = torch.float32, ignored_params = {first})
        model = native.shard_model(model, cfg)
        rep = [model.model.layers[0].input_layernorm.weight]
        rest = [p for p in model.parameters() if p is not rep[0]]
        cfg_holder["rep_is_dtensor"] = type(rep[0]).__name__
        opt = torch.optim.AdamW(
            [{"params": rest}, {"params": rep, "amsgrad": amsgrad}], lr = 1e-2, foreach = False
        )
        return model, opt

    model, opt = _new(1, True)
    for step in (1, 2):
        _step(model, opt, step, rank)
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt, exist_ok = True)
    native.native_save_optimizer(
        model, opt, os.path.join(ckpt, "optimizer.pt"), should_save = rank == 0
    )
    native.native_save_optimizer_sharded(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    state = native.native_full_state_dict(model)
    if rank == 0:
        model.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
    dist.barrier()
    res = {"rep_type": cfg_holder["rep_is_dtensor"]}
    arms = {}
    for arm, src in (("old", "optimizer.pt"), ("new", "optimizer_dcp")):
        m, o = _new(2, False)
        native.native_load_model_checkpoint(m, ckpt)
        try:
            native.native_load_optimizer(m, o, os.path.join(ckpt, src))
            loss = _step(m, o, 3, rank).item()
            arms[arm] = {"loss": loss, "after": _full_model(m), "optim": _full_optim(m, o)}
        except Exception as exc:
            res[f"{arm}_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if "old" in arms and "new" in arms:
        res["loss"] = [arms["old"]["loss"], arms["new"]["loss"]]
        res["after_diff"] = _diff_models(arms["old"]["after"], arms["new"]["after"])
        res["optim_diff"] = [
            str(d)[:200] for d in _diff_optims(arms["old"]["optim"], arms["new"]["optim"])
        ]
    return res


def _case_unstepped(native, dist, rank, tmp):
    """One trainable parameter never got a gradient before the save (an unused expert or branch), so
    the saved optimizer holds no state for it; a fresh optimizer's target does."""

    def _new(seed):
        model = native.shard_model(_model(seed), native.FSDP2Config(param_dtype = torch.float32))
        return model, torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False)

    def _skip(model):
        return model.model.layers[0].post_attention_layernorm.weight

    model, opt = _new(1)
    for step in (1, 2):
        model(**_batch(step, rank)).loss.backward()
        _skip(model).grad = None
        opt.step()
        opt.zero_grad(set_to_none = True)
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt, exist_ok = True)
    native.native_save_optimizer(
        model, opt, os.path.join(ckpt, "optimizer.pt"), should_save = rank == 0
    )
    native.native_save_optimizer_sharded(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    state = native.native_full_state_dict(model)
    if rank == 0:
        model.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
    dist.barrier()
    res = {"saved_has_state": _skip(model) in opt.state}
    # The uninterrupted run's next step is what a resume has to reproduce.
    arms = {
        "ref": {
            "has_state": False,
            "loss": _step(model, opt, 3, rank).item(),
            "after": _full_model(model),
            "optim": _full_optim(model, opt),
        }
    }
    for arm, src in (("old", "optimizer.pt"), ("new", "optimizer_dcp")):
        m, o = _new(2)
        native.native_load_model_checkpoint(m, ckpt)
        try:
            native.native_load_optimizer(m, o, os.path.join(ckpt, src))
            arms[arm] = {
                "has_state": _skip(m) in o.state and bool(o.state[_skip(m)]),
                "loss": _step(m, o, 3, rank).item(),
                "after": _full_model(m),
                "optim": _full_optim(m, o),
            }
        except BaseException as exc:  # CheckpointException is a BaseException
            res[f"{arm}_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    if "new" in arms:
        res["has_state"] = arms["new"]["has_state"]
        res["loss"] = [arms["ref"]["loss"], arms["new"]["loss"]]
        res["after_diff"] = _diff_models(arms["ref"]["after"], arms["new"]["after"])
        res["optim_diff"] = [
            str(d)[:200] for d in _diff_optims(arms["ref"]["optim"], arms["new"]["optim"])
        ]
    return res


def _case_momentum(native, dist, rank, tmp):
    """SGD saved with momentum, resumed into SGD built with momentum = 0, whose fresh optimizer state
    is empty: the momentum buffers must still be read (the restored param groups ask for them)."""

    def _new(seed, momentum):
        model = native.shard_model(_model(seed), native.FSDP2Config(param_dtype = torch.float32))
        return model, torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = momentum, foreach = False)

    model, opt = _new(1, 0.9)
    for step in (1, 2):
        _step(model, opt, step, rank)
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt, exist_ok = True)
    native.native_save_optimizer_sharded(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    state = native.native_full_state_dict(model)
    if rank == 0:
        model.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
    dist.barrier()
    ref = {
        "loss": _step(model, opt, 3, rank).item(),
        "after": _full_model(model),
        "optim": _full_optim(model, opt),
    }
    m, o = _new(2, 0.0)
    native.native_load_model_checkpoint(m, ckpt)
    res = {}
    try:
        native.native_load_optimizer(m, o, os.path.join(ckpt, "optimizer_dcp"))
        res["buffers"] = sum("momentum_buffer" in st for st in o.state.values())
        res["momentum"] = o.param_groups[0]["momentum"]
        loss = _step(m, o, 3, rank).item()
        res["loss"] = [ref["loss"], loss]
        res["after_diff"] = _diff_models(ref["after"], _full_model(m))
        res["optim_diff"] = [str(d)[:200] for d in _diff_optims(ref["optim"], _full_optim(m, o))]
    except BaseException as exc:  # CheckpointException is a BaseException
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


def _case_bnb_keys(native, dist, rank, tmp):
    """A sharded checkpoint holding the local-shard bitsandbytes Adam keys (state1 / state2), resumed
    into torch's AdamW: the keys are renamed as the optimizer.pt path renames them."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    def _new(seed):
        model = native.shard_model(_model(seed), native.FSDP2Config(param_dtype = torch.float32))
        return model, torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False)

    model, opt = _new(1)
    for step in (1, 2):
        _step(model, opt, step, rank)
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt, exist_ok = True)
    state = get_optimizer_state_dict(model, opt, options = StateDictOptions(full_state_dict = False))
    names = {
        "exp_avg": "state1",
        "exp_avg_sq": "state2",
    }  # a copy: the live entries stay as they are
    state["state"] = {
        fqn: {names.get(k, k): v for k, v in entry.items()} for fqn, entry in state["state"].items()
    }
    dcp.save(state, checkpoint_id = os.path.join(ckpt, "optimizer_dcp"))
    weights = native.native_full_state_dict(model)
    if rank == 0:
        model.save_pretrained(ckpt, state_dict = weights, safe_serialization = True)
    dist.barrier()
    ref = {"loss": _step(model, opt, 3, rank).item(), "after": _full_model(model)}
    m, o = _new(2)
    native.native_load_model_checkpoint(m, ckpt)
    res = {}
    try:
        native.native_load_optimizer(m, o, os.path.join(ckpt, "optimizer_dcp"))
        res["keys"] = sorted({k for st in o.state.values() for k in st})
        res["loss"] = [ref["loss"], _step(m, o, 3, rank).item()]
        res["after_diff"] = _diff_models(ref["after"], _full_model(m))
    except BaseException as exc:  # CheckpointException is a BaseException
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


def _case_reshard(native, dist, rank, tmp, phase):
    """Save with one world size, resume with another: the weights (sharded read) and the
    optimizer_dcp directory both come back equal to the saver's full state."""
    cfg = native.FSDP2Config(param_dtype = torch.float32)
    calls = _spy(native)
    ckpt = os.path.join(tmp, "ckpt")
    ref = os.path.join(tmp, "ref.pt")
    model = native.shard_model(_model(1 if phase == "save" else 2), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False)
    if phase == "save":
        for step in (1, 2):
            _step(model, opt, step, rank)
        state = native.native_full_state_dict(model)
        if rank == 0:
            model.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
        native.native_save_optimizer_sharded(model, opt, os.path.join(ckpt, "optimizer_dcp"))
        full = {"model": _full_model(model), "optim": _full_optim(model, opt)}
        if rank == 0:
            torch.save(full, ref)
        dist.barrier()
        return {"world": dist.get_world_size()}
    native.native_load_model_checkpoint(model, ckpt)
    native.native_load_optimizer(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    want = torch.load(ref, weights_only = False)
    res = {
        "world": dist.get_world_size(),
        "calls": calls,
        "model_diff": _diff_models(want["model"], _full_model(model)),
        "optim_diff": _diff_optims(want["optim"], _full_optim(model, opt)),
    }
    res["loss"] = _step(model, opt, 3, rank).item()  # the restored state steps
    return res


_SHORT_TIMEOUT_CASES = ("amsgrad", "unstepped", "momentum", "bnb_keys")


def _worker(rank, world, port, out, case, tmp):
    # Divergence cases run on a short collective timeout: a rank waiting on a collective the
    # other never joins fails within a minute instead of hanging the suite.
    short = case in _SHORT_TIMEOUT_CASES or case.startswith("diverge_")
    dist = _init(rank, world, port, timeout = 60 if short else 300)
    try:
        native = _load_native()
        if case == "fallbacks":
            res = _case_fallbacks(native, dist, rank, tmp)
        elif case == "trainer":
            res = _case_trainer(native, dist, rank, tmp)
        elif case.startswith("diverge_"):
            res = _case_diverge(native, dist, rank, tmp, case[len("diverge_") :])
        elif case == "amsgrad":
            res = _case_amsgrad(native, dist, rank, tmp)
        elif case == "unstepped":
            res = _case_unstepped(native, dist, rank, tmp)
        elif case == "momentum":
            res = _case_momentum(native, dist, rank, tmp)
        elif case == "bnb_keys":
            res = _case_bnb_keys(native, dist, rank, tmp)
        elif case.startswith("reshard_"):
            res = _case_reshard(native, dist, rank, tmp, case[len("reshard_") :])
        else:
            res = _case_roundtrip(
                native, dist, rank, tmp, tied = case == "tied", shard_files = case == "index"
            )
        out.put((rank, res))
    except BaseException as exc:  # CheckpointException is a BaseException
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp,
    timeout = 900,
    world = 2,
):
    import queue

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world, port, out, case, tmp)) for r in range(world)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, r = out.get(timeout = timeout)
            results[rank] = r
    except queue.Empty:
        pass
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.kill()
    assert sorted(results) == list(
        range(world)
    ), f"{case}: ranks did not report within {timeout} s: {results}"
    for rank, r in results.items():
        print("DCP_RESUME", case, rank, {k: v for k, v in r.items() if k != "crash"})
        assert "crash" not in r, r["crash"]
    return results


@pytest.mark.parametrize("case", ["untied", "tied", "index"])
def test_sharded_resume_matches_full_load(case, tmp_path):
    pytest.importorskip("transformers")
    if not _hf_reader_available():
        pytest.skip("torch < 2.9: no HuggingFaceStorageReader(path, thread_count)")
    for rank, r in _run(case, str(tmp_path)).items():
        # The sharded read ran once (the "new" arm) and the full load once (the "old" arm).
        assert r["calls"][0] is None and isinstance(r["calls"][1], int) and r["calls"][1] > 0, r
        assert r["model_diff"] == [], r["model_diff"]
        assert r["optim_diff"] == [], r["optim_diff"]
        assert r["after_diff"] == [], r["after_diff"]
        assert r["losses"]["old"] == r["want"] and r["losses"]["new"] == r["want"], r
        if case == "tied":
            assert r["still_tied"] == {"old": True, "new": True}, r
        if case == "index":
            assert "model.safetensors.index.json" in r["files"], r["files"]


def test_sharded_resume_fallbacks(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    reader = _hf_reader_available()
    for rank, r in _run("fallbacks", str(tmp_path)).items():
        for name in ("bin", "stray", "bf16", "gate", "sig", "lora"):
            assert r[name]["calls"] == [None], (name, r[name])
        for name in ("bin", "stray", "gate", "sig", "lora"):
            assert r[name].get("diff") == [], (name, r[name])
        assert r["lora"]["n"] > 0 and "adapter_model.safetensors" in r["lora"]["files"], r["lora"]
        assert "error" not in r["bf16"], r["bf16"]
        # Renamed keys: the full load runs and reports the missing tensors, as it always did.
        assert r["renamed"]["calls"] == [None], r["renamed"]
        assert "found no saved tensor" in r["renamed"].get("error", ""), r["renamed"]
        assert r["sig_probe"] is False
        if not reader:
            pytest.skip("torch < 2.9: every case is a fallback already")


def test_trainer_resume_both_optimizer_formats(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    for rank, r in _run("trainer", str(tmp_path)).items():
        dcp, full = r["dcp"], r["full"]
        assert "optimizer_dcp" in dcp["ck2_files"] and "optimizer.pt" not in dcp["ck2_files"], dcp
        assert (
            "optimizer.pt" in full["ck2_files"] and "optimizer_dcp" not in full["ck2_files"]
        ), full
        # checkpoint-4 re-saved with the flag flipped holds only the newer format.
        assert "optimizer.pt" in dcp["ck4_files"] and "optimizer_dcp" not in dcp["ck4_files"], dcp
        assert (
            "optimizer_dcp" in full["ck4_files"] and "optimizer.pt" not in full["ck4_files"]
        ), full
        assert dcp["diff"] == [] and full["diff"] == [], r


@pytest.mark.parametrize("part", ["env", "gate", "read", "save", "load"])
def test_ranks_with_different_views_agree(part, tmp_path):
    pytest.importorskip("transformers")
    if part == "read" and not _hf_reader_available():
        pytest.skip("torch < 2.9: no sharded read to fail")
    for rank, r in _run(f"diverge_{part}", str(tmp_path), timeout = 600).items():
        names = {"env": ["env"], "gate": ["gate"], "read": ["read_error_r1", "read_error_both"]}
        for name in names.get(part, []):
            # Every rank took the full load (a lone sharded rank would have hung or crashed).
            assert r[name]["calls"] == [None], (name, r[name])
            assert r[name]["diff"] == [], (name, r[name])
        if part == "save":
            # Mixed UNSLOTH_FSDP2_DCP_OPTIM: the vote picks the full file on every rank.
            assert "optimizer.pt" in r["save_mixed_flag_files"], r
            assert "optimizer_dcp" not in r["save_mixed_flag_files"], r
        if part == "load":
            assert r["load_both_vs_pt"].get("diff") == [], r["load_both_vs_pt"]
            assert "different optimizer" in r["load_dcp_vs_pt"].get("error", ""), r


def test_sharded_optimizer_restores_amsgrad_state(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("amsgrad", str(tmp_path), timeout = 600).items():
        assert r["rep_type"] != "DTensor", r  # the AMSGrad group's parameter is replicated
        assert "old_error" not in r, r
        assert "new_error" not in r, r
        assert r["loss"][0] == r["loss"][1], r
        assert r["after_diff"] == [] and r["optim_diff"] == [], r


@pytest.mark.parametrize("save_world, load_world", [(2, 3), (3, 2)])
def test_sharded_resume_on_another_world_size(save_world, load_world, tmp_path):
    pytest.importorskip("transformers")
    if not _hf_reader_available():
        pytest.skip("torch < 2.9: no HuggingFaceStorageReader(path, thread_count)")
    _run("reshard_save", str(tmp_path), world = save_world)
    results = _run("reshard_load", str(tmp_path), world = load_world)
    for rank, r in results.items():
        assert r["world"] == load_world, r
        assert (
            r["calls"] and isinstance(r["calls"][0], int) and r["calls"][0] > 0
        ), r  # sharded read
        assert r["model_diff"] == [], r["model_diff"]
        assert r["optim_diff"] == [], r["optim_diff"]


def test_sharded_optimizer_resumes_a_parameter_never_stepped(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("unstepped", str(tmp_path), timeout = 600).items():
        assert r["saved_has_state"] is False, r
        # The optimizer.pt path refuses this checkpoint in torch's strict set_optimizer_state_dict
        # (a separate, older limitation); the sharded one restores it.
        assert "new_error" not in r, r
        assert r["has_state"] is False, r  # still without state, as when saved
        assert r["loss"][0] == r["loss"][1], r
        assert r["after_diff"] == [] and r["optim_diff"] == [], r


def test_sharded_optimizer_restores_state_the_fresh_optimizer_lacks(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("momentum", str(tmp_path), timeout = 600).items():
        assert "error" not in r, r
        assert r["momentum"] == 0.9 and r["buffers"] > 0, r
        assert r["loss"][0] == r["loss"][1], r
        assert r["after_diff"] == [] and r["optim_diff"] == [], r


def test_sharded_optimizer_renames_bitsandbytes_adam_keys(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("bnb_keys", str(tmp_path), timeout = 600).items():
        assert "error" not in r, r
        assert "exp_avg" in r["keys"] and "state1" not in r["keys"], r
        assert r["loss"][0] == r["loss"][1], r
        assert r["after_diff"] == [], r
