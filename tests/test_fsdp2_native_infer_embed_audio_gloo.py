# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2 where a process group is needed):

* `_get_inference_mode_context_manager` picks no_grad (not inference_mode) under native FSDP2, whose
  launch exports no accelerate FSDP variable: a unit gathered under inference mode breaks the next
  training step;
* a resume restores LoRA on an nn.Embedding (`lora_embedding_A.<adapter>`, saved without the name);
* `input_values` audio (PeAudioVideo-style, run by a submodule only some ranks reach) is agreed on a
  host every rank runs, and a mixed batch raises on every rank;
* q_galore_config is refused before sharding;
* native_clip_grad_norm_ with a negative norm type and an empty local shard matches torch."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import importlib  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import (  # noqa: E402
    REPO_ID,
    _free_port,
    _full,
    _stub_unsloth_save,
)
from test_fsdp2_native_trainer_args_gloo import _targs  # noqa: E402
from test_fsdp2_native_guards_gloo import _Stop, _tiny_llama  # noqa: E402

NEGATIVE_NORMS = (-1.0, -2.0, -0.5, -math.inf)


def _inference_helper():
    """`_get_inference_mode_context_manager` from _utils.py (self-contained: GRPO's compiled trainer
    copies its source too)."""
    path = os.path.join(REPO, "unsloth", "models", "_utils.py")
    src = open(path, encoding = "utf-8").read()
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_inference_mode_context_manager"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
    return ns["_get_inference_mode_context_manager"]


def _case_inference(native):
    helper = _inference_helper()
    model = native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))
    inner = model.model  # what generate() / GRPO log-prob code pass: often not the sharded root
    ids = torch.randint(1, 100, (1, 8), generator = torch.Generator().manual_seed(0))
    res = {"kind": type(helper(inner)).__name__}
    try:
        with helper(inner):
            model(input_ids = ids)
        model(input_ids = ids, labels = ids).loss.backward()
        res["train_after"] = "ok"
    except Exception as exc:
        res["train_after"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


def _embedding_lora(seed):
    import transformers
    from peft import LoraConfig, get_peft_model

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None
    cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["embed_tokens", "q_proj"])
    model = get_peft_model(model, cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.02, generator = g)
    return model


def _case_embedding(native, tmp):
    from transformers import Trainer
    import torch.distributed as dist

    cfg = native.FSDP2Config(param_dtype = torch.float32)
    ckpt = os.path.join(tmp, "ckpt")
    src = native.shard_model(_embedding_lora(1), cfg)
    Trainer(model = src, args = _targs(os.path.join(tmp, "out"))).save_model(ckpt)
    dist.barrier()
    want = _full(src, trainable_only = True)
    dst = native.shard_model(_embedding_lora(2), cfg)
    res = {"embedding_keys": sorted(n for n in want if "lora_embedding" in n)}
    try:
        native.native_load_model_checkpoint(dst, ckpt)
        got = _full(dst, trainable_only = True)
        res["diff"] = max((got[n] - want[n]).abs().max().item() for n in want)
    except Exception as exc:
        res["load_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


class _AudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = torch.nn.Linear(3, 8)
        self.embed = torch.nn.Embedding(16, 8)

    def forward(self, input_ids, input_values):
        return self.embed(input_ids) + self.audio_encoder(input_values).mean(1, keepdim = True)


class _AudioText(torch.nn.Module):
    """PeAudioVideoModel-like: `audio_model` (with its own forward taking input_values) runs only
    when audio is given; the text path otherwise."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.audio_model = _AudioModel()
        self.text = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(
        self,
        input_ids,
        input_values = None,
    ):
        h = (
            self.audio_model(input_ids, input_values)
            if input_values is not None
            else self.text(input_ids)
        )
        return self.head(h).square().mean()


def _case_audio(native, rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = init_device_mesh("cpu", (2,))
    model = _AudioText()
    fully_shard(model.audio_model.audio_encoder, mesh = mesh)
    fully_shard(model, mesh = mesh)
    host = vision.install_vision_dummy(model)
    res = {"host": type(host).__name__ if host is not None else None}
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    for step, audio in (("all_text", False), ("all_audio", True), ("mixed", rank == 0)):
        kwargs = {"input_values": torch.randn(1, 6, 3)} if audio else {}
        try:
            model(ids, **kwargs).backward()
            res[step] = "ok"
        except Exception as exc:
            res[step] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


class _Uneven(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(5, 6)
        self.small = torch.nn.Linear(
            6, 1
        )  # weight (1, 6), bias (1,): dim 0 < world, rank 1 holds nothing

    def forward(self, x):
        return self.small(torch.tanh(self.a(x))).square().sum()


def _case_negative_norm(native):
    import copy

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    reference = _Uneven()
    model = copy.deepcopy(reference)
    mesh = init_device_mesh("cpu", (2,))
    fully_shard(model.a, mesh = mesh)
    fully_shard(model.small, mesh = mesh)
    fully_shard(model, mesh = mesh)
    x = torch.randn(3, 5, generator = torch.Generator().manual_seed(2))
    res = {"local_numel": model.small.weight.to_local().numel()}
    for norm_type in NEGATIVE_NORMS:
        for m in (model, reference):
            m.zero_grad(set_to_none = True)
            m(x).backward()
        want = torch.nn.utils.clip_grad_norm_(
            list(reference.parameters()), 1e6, norm_type = norm_type
        )
        got = native.native_clip_grad_norm_(list(model.parameters()), 1e6, norm_type = norm_type)
        res[str(norm_type)] = {"got": float(got), "want": float(want)}
    return res


def _worker(rank, world, port, out, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        if case == "inference":
            res = _case_inference(native)
        elif case == "embedding":
            res = _case_embedding(native, tmp)
        elif case == "audio":
            res = _case_audio(native, rank)
        elif case == "negative_norm":
            res = _case_negative_norm(native)
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp_path,
    timeout = 180,
):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, case, str(tmp_path)),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(
                    f"{case}: the ranks did not finish within {timeout} s (hung collective)"
                )
    except mp.ProcessExitedException as exc:
        pytest.fail(f"{case}: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND6_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_native_fsdp2_inference_context_is_no_grad(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("inference", tmp_path).items():
        assert r["kind"] == "no_grad", (rank, r)
        assert r["train_after"] == "ok", (rank, r)


def test_resume_restores_embedding_lora(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("embedding", tmp_path).items():
        assert r["embedding_keys"], (rank, r)
        assert "load_error" not in r and r["diff"] == 0.0, (rank, r)


def test_input_values_audio_is_agreed_on_a_host_every_rank_runs(tmp_path):
    for rank, r in _run("audio", tmp_path).items():
        assert r["host"] in ("_AudioText", "FSDP_AudioText"), (rank, r)  # not the inner _AudioModel
        assert r["all_text"] == "ok" and r["all_audio"] == "ok", (rank, r)
        assert r["mixed"].startswith("RuntimeError") and "audio on 1 of 2 ranks" in r["mixed"], (
            rank,
            r,
        )


def test_negative_norm_with_an_empty_shard_matches_torch(tmp_path):
    results = _run("negative_norm", tmp_path)
    assert results[1]["local_numel"] == 0, results
    for rank, r in results.items():
        for norm_type in NEGATIVE_NORMS:
            v = r[str(norm_type)]
            assert v["want"] > 0, (rank, norm_type, v)
            assert math.isclose(v["got"], v["want"], rel_tol = 1e-5), (rank, norm_type, v)


# ---- CPU unit checks (no process group) -------------------------------------------------------


def test_inference_helper_env_and_marker(monkeypatch):
    helper = _inference_helper()
    for key in ("ACCELERATE_USE_FSDP", "FSDP_VERSION", "UNSLOTH_FSDP2", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising = False)
    plain = torch.nn.Linear(2, 2)
    assert isinstance(helper(plain), torch.inference_mode)
    marked = torch.nn.Linear(2, 2)
    marked._unsloth_fsdp2_native = True
    assert isinstance(helper(marked), torch.no_grad)
    assert isinstance(helper(types.SimpleNamespace(module = marked)), torch.no_grad)
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert isinstance(helper(plain), torch.no_grad)
    monkeypatch.setenv("WORLD_SIZE", "1")  # single process: no sharding, inference mode is fine
    assert isinstance(helper(plain), torch.inference_mode)


def test_q_galore_is_refused_before_sharding(monkeypatch):
    pytest.importorskip("transformers")
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    for key in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_MIXED_PRECISION"):
        monkeypatch.delenv(key, raising = False)
    native = _load_native()

    def _stop(device, *_):
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _stop)
    base = dict(optim = "adamw_torch", save_on_each_node = False, fsdp = None)
    args = types.SimpleNamespace(q_galore_config = object(), **base)
    model = _tiny_llama()
    with pytest.raises(ValueError, match = "q_galore_config"):
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
    assert not native.is_native_fsdp2(model)
    with pytest.raises(_Stop):  # q_galore_config = None: sharding proceeds
        args = types.SimpleNamespace(q_galore_config = None, **base)
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)


@pytest.mark.parametrize(
    "optim",
    [
        "galore_adamw_layerwise",
        "galore_adamw_8bit_layerwise",
        "galore_adafactor_layerwise",
        "apollo_adamw_layerwise",
        "galore_adamw",
    ],
)
def test_galore_and_apollo_are_refused_before_sharding(monkeypatch, optim):
    pytest.importorskip("transformers")
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    for key in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_MIXED_PRECISION"):
        monkeypatch.delenv(key, raising = False)
    native = _load_native()

    def _stop(device, *_):
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _stop)
    base = dict(save_on_each_node = False, fsdp = None, q_galore_config = None)
    model = _tiny_llama()
    with pytest.raises(ValueError, match = optim):
        args = types.SimpleNamespace(optim = optim, **base)
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
    assert not native.is_native_fsdp2(model)
    with pytest.raises(_Stop):  # a torch optimizer: sharding proceeds
        args = types.SimpleNamespace(optim = "adamw_torch", **base)
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
