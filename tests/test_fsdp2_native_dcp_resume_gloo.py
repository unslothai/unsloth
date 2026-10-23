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


def _init(rank, world, port):
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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
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


def _worker(rank, world, port, out, case, tmp):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        if case == "fallbacks":
            res = _case_fallbacks(native, dist, rank, tmp)
        elif case == "trainer":
            res = _case_trainer(native, dist, rank, tmp)
        else:
            res = _case_roundtrip(
                native, dist, rank, tmp, tied = case == "tied", shard_files = case == "index"
            )
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def _run(case, tmp):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target = _worker, args = (r, 2, port, out, case, tmp)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict(out.get(timeout = 900) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
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
