# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 media agreement is per input, not per modality: each input is encoded by its own
call, so a rank with two inputs of one modality runs the encoder twice while a rank with one runs it
once.

* Moshi (transformers/models/moshi/modeling_moshi.py MoshiForConditionalGeneration.forward) takes
  `user_input_values` and `moshi_input_values` and runs `audio_encoder.encode` once per given input:
  both names are recognised as audio, and a rank with both next to a rank with one raises on every
  rank (gloo world of 2) instead of desynchronising the encoder's collectives;
* Qwen-VL runs `visual` once for pixel_values and again for pixel_values_videos
  (transformers/models/qwen2_vl/modeling_qwen2_vl.py Qwen2VLModel.forward): mixed videos raise (no
  video dummy), mixed images still run the image dummy."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import inspect  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402


class _Codec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 8)

    def encode(self, values):
        return (self.proj(values).mean(1, keepdim = True),)


class _MoshiLike(torch.nn.Module):
    """MoshiForConditionalGeneration-like: `audio_encoder.encode` once per given audio input."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.audio_encoder = _Codec()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(
        self,
        input_ids,
        user_input_values = None,
        moshi_input_values = None,
    ):
        h = self.embed(input_ids)
        if user_input_values is not None:
            h = h + self.audio_encoder.encode(user_input_values)[0]
        if moshi_input_values is not None:
            h = h + self.audio_encoder.encode(moshi_input_values)[0]
        return self.head(h).square().mean()


def _case_moshi(rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = init_device_mesh("cpu", (2,))
    model = _MoshiLike()
    fully_shard(
        model.audio_encoder.proj, mesh = mesh
    )  # a unit inside the codec (encode is no forward)
    fully_shard(model, mesh = mesh)
    host = vision.install_vision_dummy(model)
    res = {"host": type(host).__name__ if host is not None else None}
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    audio = lambda: torch.randn(1, 6, 3)  # noqa: E731
    steps = (
        ("both_everywhere", {"user_input_values": audio(), "moshi_input_values": audio()}),
        ("user_everywhere", {"user_input_values": audio()}),
        # rank 0 encodes twice, rank 1 once
        (
            "both_vs_user",
            {"user_input_values": audio(), "moshi_input_values": audio()}
            if rank == 0
            else {"user_input_values": audio()},
        ),
    )
    for step, kwargs in steps:
        try:
            model(ids, **kwargs).backward()
            res[step] = "ok"
        except Exception as exc:
            res[step] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


def _worker(rank, world, port, out):
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
        _load_native()
        out.put((rank, _case_moshi(rank)))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def test_moshi_inputs_are_agreed_per_input():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + 150
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail("moshi: the ranks did not finish (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"moshi: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND11_RESULT moshi", rank, r)
        assert "crash" not in r, r["crash"]
        assert r["host"] in ("_MoshiLike", "FSDP_MoshiLike"), (rank, r)
        assert r["both_everywhere"] == "ok" and r["user_everywhere"] == "ok", (rank, r)
        assert r["both_vs_user"].startswith("RuntimeError"), (rank, r)
        assert "audio on 1 of 2 ranks (input `moshi_input_values`)" in r["both_vs_user"], (rank, r)


def test_real_moshi_signature_inputs_are_audio():
    transformers = pytest.importorskip("transformers")
    _load_native()
    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    moshi = transformers.MoshiForConditionalGeneration
    names = vision._audio_inputs(inspect.signature(moshi.forward))
    assert "user_input_values" in names and "moshi_input_values" in names, names


class _QwenVLHost(torch.nn.Module):
    def forward(
        self,
        input_ids = None,
        pixel_values = None,
        pixel_values_videos = None,
    ):
        return input_ids


def _simulated(vision, monkeypatch, other):
    """One process: the other rank's flags are added to this rank's (world of 2)."""
    monkeypatch.setattr(
        vision,
        "_agree_counts",
        lambda flags, device, *_: ([int(f) + o for f, o in zip(flags, other)], 2),
    )


def test_qwen_vl_videos_are_agreed_apart_from_images(monkeypatch):
    _load_native()
    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    modality = vision._Modality("images", vision._VISION_INPUTS, "", lambda: True)
    host = _QwenVLHost()
    vision._install_presence_agreement(host, [modality], lambda: torch.device("cpu"))
    x = torch.zeros(1)
    # Flags: [image group, pixel_values_videos, unready]. The other rank has images and a video.
    _simulated(vision, monkeypatch, [1, 1, 0])
    with pytest.raises(
        RuntimeError, match = r"images on 1 of 2 ranks \(input `pixel_values_videos`\)"
    ):
        host(pixel_values = x)  # this rank: images only, so `visual` runs once here, twice there
    # The other rank has images only: this rank, with a video only, runs the image dummy.
    _simulated(vision, monkeypatch, [1, 0, 0])
    with pytest.raises(RuntimeError, match = "pixel_values_videos"):
        host(pixel_values_videos = x)
    _simulated(vision, monkeypatch, [1, 0, 0])
    host(input_ids = x)
    assert modality.run_dummy is True
    _simulated(vision, monkeypatch, [1, 1, 0])
    host(pixel_values = x, pixel_values_videos = x)
    assert modality.run_dummy is False
