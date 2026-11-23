# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2 where a process group is needed):

* Trainer.save_model of a LoRA model the Trainer holds wrapped by torch.compile (`_orig_mod`) still
  writes the adapter checkpoint;
* a model on the CPU is sharded onto the local GPU under a gloo-only process group (CUDA without
  NCCL), unless the CPU is asked for explicitly;
* VideoLlava-style separate image / video towers (`pixel_values_images` / `pixel_values_videos`) get
  the per-modality agreement: image on one rank and video on the other raises on both;
* a prebuilt bitsandbytes optimizer instance is refused (Trainer and accelerate prepare)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _full, _stub_unsloth_save  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import (  # noqa: E402
    _install_wrapped_trainer_init,
    _lora,
    _targs,
)
from test_fsdp2_native_guards_gloo import _Stop, _tiny_llama  # noqa: E402
from test_fsdp2_native_optim_reentry_gloo import AdamW8bitLike  # noqa: E402


def _case_compiled_save(native, tmp):
    from safetensors.torch import load_file
    from transformers import Trainer
    import torch.distributed as dist

    model = native.shard_model(_lora(1), native.FSDP2Config(param_dtype = torch.float32))
    want = _full(model, trainable_only = True)
    trainer = Trainer(model = model, args = _targs(os.path.join(tmp, "out")))
    trainer.model = torch.compile(model)  # what accelerate hands back with torch_compile = True
    ckpt = os.path.join(tmp, "ckpt")
    res = {"full_gathers": 0}
    full = native.native_full_state_dict

    def counting(*args, **kwargs):  # the whole-model gather a LoRA save must not take
        res["full_gathers"] += 1
        return full(*args, **kwargs)

    native.native_full_state_dict = counting
    try:
        trainer.save_model(ckpt)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        native.native_full_state_dict = full
    dist.barrier()
    res["files"] = sorted(os.listdir(ckpt)) if os.path.isdir(ckpt) else []
    path = os.path.join(ckpt, "adapter_model.safetensors")
    if os.path.isfile(path):
        saved = load_file(path)
        worst = 0.0
        for k, v in saved.items():
            name = k.replace(".weight", ".default.weight")
            worst = (
                max(worst, (v - want[name]).abs().max().item()) if name in want else float("inf")
            )
        res["n"] = len(saved)
        res["diff"] = worst
    # Resume / load_best_model through the same wrapper.
    dst = native.shard_model(_lora(2), native.FSDP2Config(param_dtype = torch.float32))
    try:
        native.native_load_model_checkpoint(torch.compile(dst), ckpt)
        got = _full(dst, trainable_only = True)
        res["load_diff"] = max((got[n] - want[n]).abs().max().item() for n in want)
    except Exception as exc:
        res["load_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 8)

    def forward(self, pixels):
        return self.proj(pixels.flatten(1))


class _LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.layer = torch.nn.Linear(8, 8)

    def forward(self, inputs_embeds):
        return torch.tanh(self.layer(inputs_embeds))


class _VideoLlavaLike(torch.nn.Module):
    """transformers VideoLlavaModel's layout: image_tower, video_tower, multi_modal_projector,
    language_model; get_image_features(pixel_values_images), get_video_features(pixel_values_videos)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.config = types.SimpleNamespace(
            vision_config = types.SimpleNamespace(image_size = 4, patch_size = 2, num_channels = 1)
        )
        self.image_tower = _Tower()
        self.video_tower = _Tower()
        self.multi_modal_projector = torch.nn.Linear(8, 8)
        self.language_model = _LM()

    def get_input_embeddings(self):
        return self.language_model.embed

    def get_image_features(self, pixel_values_images):
        return self.multi_modal_projector(self.image_tower(pixel_values_images))

    def get_video_features(self, pixel_values_videos):
        return self.multi_modal_projector(self.video_tower(pixel_values_videos))

    def forward(
        self,
        input_ids,
        pixel_values_images = None,
        pixel_values_videos = None,
    ):
        embeds = self.language_model.embed(input_ids)
        if pixel_values_images is not None:
            embeds = embeds + self.get_image_features(pixel_values_images).mean(0)
        if pixel_values_videos is not None:
            embeds = embeds + self.get_video_features(pixel_values_videos).mean(0)
        return self.language_model(inputs_embeds = embeds).square().mean()


def _case_video(native, rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = init_device_mesh("cpu", (2,))
    model = _VideoLlavaLike()
    for module in (model.image_tower, model.video_tower, model.language_model.layer):
        fully_shard(module, mesh = mesh)
    fully_shard(model, mesh = mesh)
    host = vision.install_vision_dummy(model)
    res = {"host": type(host).__name__ if host is not None else None}
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    image = {"pixel_values_images": torch.randn(1, 1, 4, 4)}
    video = {"pixel_values_videos": torch.randn(2, 1, 4, 4)}
    steps = [
        ("text", {}),
        ("both", {**image, **video}),
        ("image_vs_video", image if rank == 0 else video),
    ]
    for step, kwargs in steps:
        try:
            model(ids, **kwargs).backward()
            res[step] = "ok"
        except Exception as exc:
            res[step] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


def _case_bnb_instance(native, tmp):
    _install_wrapped_trainer_init()
    from accelerate import Accelerator
    from transformers import Trainer

    model = native.shard_model(_lora(1), native.FSDP2Config(param_dtype = torch.float32))
    trainable = [p for p in model.parameters() if p.requires_grad]
    res = {}
    for name, cls in (("bnb_like", AdamW8bitLike), ("torch", torch.optim.AdamW)):
        optimizer = cls(trainable, lr = 1e-3)
        try:
            Trainer(
                model = model, args = _targs(os.path.join(tmp, "out")), optimizers = (optimizer, None)
            )
            res[f"{name}.trainer"] = ""
        except Exception as exc:
            res[f"{name}.trainer"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    # A Trainer's accelerator (marked) with the sharded model registered.
    fake_accelerator = types.SimpleNamespace(_models = [model])
    setattr(fake_accelerator, native._TRAINER_ACCELERATOR, True)
    try:
        Accelerator.prepare_optimizer(fake_accelerator, AdamW8bitLike(trainable, lr = 1e-3))
        res["prepare"] = ""
    except Exception as exc:
        res["prepare"] = f"{type(exc).__name__}: {str(exc)[:200]}"
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
        if case == "compiled_save":
            res = _case_compiled_save(native, tmp)
        elif case == "video":
            res = _case_video(native, rank)
        elif case == "bnb_instance":
            res = _case_bnb_instance(native, tmp)
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
        print("ROUND7_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_compiled_peft_model_saves_an_adapter_checkpoint(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    r = _run("compiled_save", tmp_path)[0]
    assert "error" not in r, r
    assert "adapter_model.safetensors" in r["files"] and "model.safetensors" not in r["files"], r
    assert r["n"] > 0 and r["diff"] == 0.0, r
    assert r["full_gathers"] == 0, r  # adapter-only gather, not the whole model
    assert "load_error" not in r and r["load_diff"] == 0.0, r


def test_separate_image_and_video_towers_are_agreed(tmp_path):
    for rank, r in _run("video", tmp_path).items():
        assert r["host"] in ("_VideoLlavaLike", "FSDP_VideoLlavaLike"), (rank, r)
        assert r["text"] == "ok" and r["both"] == "ok", (rank, r)
        assert r["image_vs_video"].startswith("RuntimeError"), (rank, r)
        assert "on 1 of 2 ranks" in r["image_vs_video"], (rank, r)


def test_prebuilt_bnb_optimizer_instance_is_refused(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")
    for rank, r in _run("bnb_instance", tmp_path).items():
        assert (
            r["bnb_like.trainer"].startswith("ValueError")
            and "bitsandbytes" in r["bnb_like.trainer"]
        ), (rank, r)
        assert r["torch.trainer"] == "", (rank, r)
        assert r["prepare"].startswith("ValueError") and "bitsandbytes" in r["prepare"], (rank, r)


# ---- CPU unit checks (no process group) -------------------------------------------------------


def test_gloo_process_group_on_a_cuda_host_shards_onto_the_gpu(monkeypatch):
    pytest.importorskip("transformers")
    import torch.distributed as dist

    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0", LOCAL_RANK = "1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ACCELERATE_USE_CPU", raising = False)
    native = _load_native()
    model = _tiny_llama()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")  # CUDA build without NCCL
    seen = {}

    def _record(device, *_):
        seen["device"] = device
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _record)
    cfg = native.FSDP2Config(param_dtype = torch.float32)
    with pytest.raises(_Stop):
        native.shard_model(model, cfg)
    assert seen["device"] == torch.device("cuda", 1), seen
    # Explicit CPU requests still keep the CPU: ACCELERATE_USE_CPU, or config.device (use_cpu).
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    with pytest.raises(_Stop):
        native.shard_model(_tiny_llama(), cfg)
    assert seen["device"] == torch.device("cpu"), seen
    monkeypatch.delenv("ACCELERATE_USE_CPU")
    with pytest.raises(_Stop):
        native.shard_model(
            _tiny_llama(), native.FSDP2Config(param_dtype = torch.float32, device = "cpu")
        )
    assert seen["device"] == torch.device("cpu"), seen
