# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for data-parallel DiT LoRA training (diffusion_ddp) on gloo: the gradient all-reduce, the fallback
rules, and the real loop on a tiny Qwen-Image-2.1 where two ranks at batch 1 train exactly the run one process trains
at batch 2, with identical replicas and rank 1 reading the conditioning warm."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.training import diffusion_ddp

BACKEND = Path(__file__).resolve().parents[1]
FAMILY = "qwen-image-2.1"
TINY_PIPE = "hf-internal-testing/tiny-qwenimage21-pipe"


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(
    mode: str,
    world: int,
    *args: str,
    timeout: int = 600,
) -> None:
    port = str(_port())
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(BACKEND), os.environ.get("PYTHONPATH", "")]),
    }
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, mode, str(rank), str(world), port, *args],
            cwd = str(BACKEND),
            env = env,
        )
        for rank in range(world)
    ]
    codes = [p.wait(timeout = timeout) for p in procs]
    assert codes == [0] * world, codes


def _init(rank: int, world: int, port: int):
    import torch.distributed as dist

    store = dist.TCPStore(
        "127.0.0.1", port, world, is_master = rank == 0, timeout = timedelta(seconds = 300)
    )
    dist.init_process_group("gloo", store = store, rank = rank, world_size = world)
    return diffusion_ddp.activate_for_tests(rank, world, store)


# ---- child entry points -------------------------------------------------------------------------------------


def _child_grads(rank: int, world: int, port: int, out: str) -> None:
    ctx = _init(rank, world, port)
    torch.manual_seed(rank)
    a = torch.nn.Parameter(torch.zeros(3, 4))
    b = torch.nn.Parameter(torch.zeros(5))
    none = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.randn(3, 4)
    b.grad = torch.randn(5)
    local = {"a": a.grad.clone(), "b": b.grad.clone()}
    loss = ctx.all_reduce_grads([a, b, none], torch.tensor(float(rank + 1), dtype = torch.float64))
    flag = ctx.any_flag(rank == 1)
    quiet = ctx.any_flag(False)
    pick = ctx.agree_precision("bf16" if rank == 0 else "nf4")
    # Diverged generators (a resume restores each card's from rank 0's save): sync_rng continues rank 0's stream.
    torch.manual_seed(100)
    ref_draw = torch.rand(6)
    torch.manual_seed(100 if rank == 0 else 7)
    ctx.sync_rng()
    draw = torch.rand(6)
    torch.save(
        {
            "local": local,
            "reduced": {"a": a.grad, "b": b.grad, "none": none.grad},
            "loss": loss,
            "flag": flag,
            "quiet": quiet,
            "pick": pick,
            "ref_draw": ref_draw,
            "draw": draw,
            "stats": ctx.stats(),
        },
        f"{out}/grads_{rank}.pt",
    )


def _cpu_patches(dit, precision = "bf16"):
    from core.training import diffusion_gc_budget as gb

    dit._resolve_base_precision = lambda cfg, spec, device: precision
    dit._assert_trusted_base_model = lambda *a, **k: None
    sys.modules["unsloth_zoo"] = None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.memory_allocated = lambda *a, **k: 0
    dit.enable_budgeted_checkpointing = lambda model, device: None
    dit.enable_diffusion_gradient_checkpointing = lambda model, mode = None: "plain"
    gb._memory_allocated = lambda: 0


def _child_train(
    rank: int,
    world: int,
    port: int,
    out: str,
    pipe: str,
    data: str,
    batch: str,
    steps: str,
    cache: str = "1",
) -> None:
    import core.inference.diffusion_lora as dl
    from core.training import diffusion_dit_trainer as dit
    from core.training.diffusion_train_common import DiffusionLoraConfig

    root = Path(out)
    dl.loras_dir = lambda: root / "catalog"
    (root / "catalog").mkdir(parents = True, exist_ok = True)
    _cpu_patches(dit)
    if world > 1:
        _init(rank, world, port)
    spec = dit._SPECS[FAMILY]
    load, load_cond = spec.load_transformer, spec.load_conditioners
    seen = {"transformer": None, "conditioners": 0}

    def load_transformer(*a, **k):
        seen["transformer"] = load(*a, **k).float()
        return seen["transformer"]

    def load_conditioners(*a, **k):
        seen["conditioners"] += 1
        return load_cond(*a, **k)

    spec.load_transformer = load_transformer
    spec.load_conditioners = load_conditioners
    encode, load_px = spec.encode_latents, dit._load_pixel_tensor
    seen["encoded"], seen["decoded"] = [], 0

    def encode_latents(vae, px):
        seen["encoded"].append(int(px.shape[0]))
        return encode(vae, px)

    def load_pixel_tensor(*a, **k):
        seen["decoded"] += 1
        return load_px(*a, **k)

    spec.encode_latents = encode_latents
    dit._load_pixel_tensor = load_pixel_tensor
    tag = f"w{world}_b{batch}_r{rank}" + ("" if cache == "1" else "_nocache")
    events = []
    cfg = DiffusionLoraConfig(
        base_model = pipe,
        model_family = FAMILY,
        data_dir = data,
        output_dir = str(root / f"out_{tag}"),
        resolution = 64,
        train_steps = int(steps),
        train_batch_size = int(batch),
        learning_rate = 1e-2,
        lora_rank = 4,
        cache_variants = 2,
        cache_latents = cache == "1",
        center_crop = cache == "1",
        cfg_dropout = 0.25,
        cond_cache_dir = str(root / f"cond_w{world}"),
        seed = 0,
    )
    dit.run_dit_lora_training(cfg, on_event = events.append if rank == 0 else None)
    lora = {
        n: p.detach().clone() for n, p in seen["transformer"].named_parameters() if p.requires_grad
    }
    torch.save(
        {
            "lora": lora,
            "events": json.loads(json.dumps(events, default = str)),
            "conditioners": seen["conditioners"],
            "encoded": seen["encoded"],
            "decoded": seen["decoded"],
        },
        root / f"train_{tag}.pt",
    )


# ---- tests ---------------------------------------------------------------------------------------------------


def test_all_reduce_averages_grads_and_loss_and_flags(tmp_path):
    _spawn("grads", 2, str(tmp_path))
    r0, r1 = (torch.load(tmp_path / f"grads_{r}.pt") for r in range(2))
    for k in ("a", "b"):
        want = (r0["local"][k] + r1["local"][k]) / 2
        assert torch.allclose(r0["reduced"][k], want) and torch.equal(
            r0["reduced"][k], r1["reduced"][k]
        )
    # A parameter without a gradient takes part as zeros and gets the (zero) mean.
    assert torch.equal(r0["reduced"]["none"], torch.zeros(2))
    assert r0["loss"] == r1["loss"] == pytest.approx(1.5)
    assert r0["flag"] is True and r1["flag"] is True
    assert r0["quiet"] is False and r1["quiet"] is False
    # Ranks that resolve different bases train the most frugal one.
    assert r0["pick"] == r1["pick"] == "nf4"
    assert r0["stats"]["allreduce_calls"] == 1 and r0["stats"]["world_size"] == 2
    assert torch.equal(r0["draw"], r0["ref_draw"]) and torch.equal(r1["draw"], r0["ref_draw"])


def _cfg(**kw):
    base = dict(num_gpus = 2, block_swap = "")
    base.update(kw)
    return SimpleNamespace(**base)


def test_plan_world_single_device_and_refusals(monkeypatch):
    assert diffusion_ddp.plan_world(_cfg(num_gpus = 1), "cuda") == (1, None)
    world, why = diffusion_ddp.plan_world(_cfg(), "cpu")
    assert world == 1 and "CUDA" in why
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
    world, why = diffusion_ddp.plan_world(_cfg(), "cuda")
    assert world == 1 and "NCCL" in why


def _fake_cards(monkeypatch, cards):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cards))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name = cards[i][0], total_memory = cards[i][1]),
    )


def test_plan_world_counts_identical_cards(monkeypatch):
    _fake_cards(monkeypatch, [("Tesla T4", 15 << 30)] * 2)
    monkeypatch.delenv("UNSLOTH_DIT_BLOCK_SWAP", raising = False)
    assert diffusion_ddp.plan_world(_cfg(), "cuda") == (2, None)
    world, why = diffusion_ddp.plan_world(_cfg(num_gpus = 4), "cuda")
    assert world == 2 and "2 GPUs are visible" in why
    world, why = diffusion_ddp.plan_world(_cfg(block_swap = "auto"), "cuda")
    assert world == 1 and "block swap" in why
    monkeypatch.setenv("UNSLOTH_DIT_BLOCK_SWAP", "8")
    assert diffusion_ddp.plan_world(_cfg(), "cuda")[0] == 1


def test_plan_world_refuses_mismatched_or_single_cards(monkeypatch):
    _fake_cards(monkeypatch, [("Tesla T4", 15 << 30), ("NVIDIA L4", 22 << 30)])
    world, why = diffusion_ddp.plan_world(_cfg(), "cuda")
    assert world == 1 and "identical" in why
    _fake_cards(monkeypatch, [("Tesla T4", 15 << 30)])
    world, why = diffusion_ddp.plan_world(_cfg(), "cuda")
    assert world == 1 and "only 1 GPU" in why


def test_num_gpus_is_validated_and_defaults_to_one():
    from core.training.diffusion_train_common import DiffusionLoraConfig, _config_from_dict

    cfg = DiffusionLoraConfig(base_model = "x", data_dir = "d", output_dir = "o")
    assert cfg.num_gpus == 1
    assert (
        _config_from_dict(
            {"base_model": "x", "data_dir": "d", "output_dir": "o", "num_gpus": "2"}
        ).num_gpus
        == "2"
    )
    with pytest.raises(ValueError, match = "num_gpus"):
        DiffusionLoraConfig(
            base_model = "black-forest-labs/FLUX.2-klein-base-4B",
            data_dir = "d",
            output_dir = "o",
            num_gpus = 0,
        ).normalized()


def test_the_start_request_carries_num_gpus():
    from models.training import DiffusionTrainingStartRequest
    field = DiffusionTrainingStartRequest.model_fields["num_gpus"]
    assert field.default == 1


@pytest.fixture
def tiny_pipe_dir():
    try:
        from diffusers import QwenImage21Pipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"diffusers Qwen-Image-2.1 classes unavailable: {exc}")
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(TINY_PIPE)
    except Exception as exc:  # noqa: BLE001 -- offline CI
        pytest.skip(f"{TINY_PIPE} unavailable: {exc}")


class _LaunchSeen(Exception):
    pass


@pytest.mark.parametrize("world, batch, expected", [(2, 1, 8), (1, 1, 16), (2, 2, 4), (3, 1, 6)])
def test_epoch_steps_count_the_whole_world(monkeypatch, tmp_path, world, batch, expected):
    """num_epochs counts passes over the dataset: with N GPUs a micro-step consumes N x train_batch_size images, so
    16 images for 1 epoch on 2 GPUs at batch 1 is 8 steps (it was 16, consuming 32 images). The resolved count is
    what the started ranks receive."""
    from PIL import Image

    import core.training.diffusion_dit_trainer as dit
    from core.training.diffusion_train_common import DiffusionLoraConfig

    data = tmp_path / "data"
    data.mkdir()
    for i in range(16):
        Image.new("RGB", (64, 64), (10 * i, 20, 30)).save(data / f"img{i:02d}.png")
        (data / f"img{i:02d}.txt").write_text(f"tile {i}")
    seen = {}

    def fake_launch(
        cfg,
        world_size,
        on_failure = None,
    ):
        seen["cfg"], seen["world"] = cfg, world_size
        raise _LaunchSeen

    monkeypatch.setattr(dit, "_assert_trusted_base_model", lambda *a, **k: None)
    monkeypatch.setattr(dit, "_assert_gated_access", lambda *a, **k: None)
    monkeypatch.setattr(dit, "resolve_train_device", lambda: "cuda")
    monkeypatch.setattr(dit, "dit_fp16_fallback", lambda fam: False)
    monkeypatch.setattr(dit, "flow_bf16_trainable", lambda: True)
    monkeypatch.setattr(diffusion_ddp, "plan_world", lambda cfg, device: (world, None))
    monkeypatch.setattr(diffusion_ddp, "launch", fake_launch)
    monkeypatch.setattr(dit, "_train_dit", lambda cfg, *a, **k: seen.setdefault("single", cfg))
    cfg = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.2-klein-base-4B",
        model_family = "flux.2-klein",
        data_dir = str(data),
        output_dir = str(tmp_path / "out"),
        num_epochs = 1,
        train_batch_size = batch,
        num_gpus = max(world, 2),
    )
    try:
        dit.run_dit_lora_training(cfg)
    except _LaunchSeen:
        pass
    got = seen.get("cfg") or seen.get("single")
    assert got.train_steps == expected and got.num_epochs == 0
    if world > 1:
        assert seen["world"] == world


@pytest.mark.allow_network
def test_two_ranks_at_batch_one_train_the_one_process_batch_two_run(tmp_path, tiny_pipe_dir):
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    for i in range(5):
        Image.new("RGB", (64, 64), (200, 30 * i, 40)).save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text(f"a red tile {i}")
    steps = "4"
    _spawn("train", 2, str(tmp_path), tiny_pipe_dir, str(data), "1", steps)
    _spawn("train", 1, str(tmp_path), tiny_pipe_dir, str(data), "2", steps)
    r0 = torch.load(tmp_path / "train_w2_b1_r0.pt")
    r1 = torch.load(tmp_path / "train_w2_b1_r1.pt")
    ref = torch.load(tmp_path / "train_w1_b2_r0.pt")

    # The replicas never drift: identical adapters on both ranks.
    assert r0["lora"].keys() == r1["lora"].keys() == ref["lora"].keys()
    for k in r0["lora"]:
        assert torch.equal(r0["lora"][k], r1["lora"][k]), k
    # Same global batch, same draws: the two-rank run is the batch-2 run (up to the batch-1 vs batch-2 GEMM order).
    for k in r0["lora"]:
        assert torch.allclose(r0["lora"][k], ref["lora"][k], rtol = 1e-4, atol = 1e-6), k
    lw2 = [e["loss"] for e in r0["events"] if e["type"] == "progress"]
    lw1 = [e["loss"] for e in ref["events"] if e["type"] == "progress"]
    assert len(lw2) == len(lw1) == int(steps)
    assert lw2 == pytest.approx(lw1, rel = 1e-4, abs = 1e-6)

    done = [e for e in r0["events"] if e["type"] == "complete"][-1]
    mg = done["multi_gpu"]
    assert (
        mg["world_size"] == 2
        and mg["allreduce_calls"] == int(steps)
        and mg["params_in_sync"] is True
    )
    assert [r["steps"] for r in mg["ranks"]] == [int(steps)] * 2
    assert "multi_gpu" not in [e for e in ref["events"] if e["type"] == "complete"][-1]
    # Throughput counts the whole global batch; rank 1 read rank 0's conditioning warm, so it loaded no encoders.
    assert r1["conditioners"] == 0 and r0["conditioners"] == 1
    assert Path(done["lora_path"]).exists()
    assert not (tmp_path / "out_w2_b1_r1").exists()


@pytest.mark.allow_network
def test_two_ranks_without_a_latent_cache_encode_only_their_slice(tmp_path, tiny_pipe_dir):
    """No latent cache: each rank decodes and VAE-encodes its own image only (it encoded the whole global batch), the
    replicas stay identical, and the noise and crop streams stay shared."""
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    for i in range(5):
        # Non-square: the random crop draws an offset, so a skipped image must still advance the shared stream.
        Image.new("RGB", (96 + 8 * i, 64), (200, 30 * i, 40)).save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text(f"a red tile {i}")
    steps = 3
    _spawn("train", 2, str(tmp_path), tiny_pipe_dir, str(data), "1", str(steps), "0")
    r0 = torch.load(tmp_path / "train_w2_b1_r0_nocache.pt")
    r1 = torch.load(tmp_path / "train_w2_b1_r1_nocache.pt")
    for r in (r0, r1):
        assert r["encoded"] == [1] * steps
        assert r["decoded"] == steps
    for k in r0["lora"]:
        assert torch.equal(r0["lora"][k], r1["lora"][k]), k
    losses = [e["loss"] for e in r0["events"] if e["type"] == "progress"]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)


def test_skipping_an_image_advances_the_crop_stream_like_loading_it(tmp_path):
    """A rank skips the images outside its slice from the header alone; its stream must match a rank that decoded
    them, including an EXIF-rotated image whose crop range follows the rotated size."""
    import random

    from PIL import Image

    import core.training.diffusion_dit_trainer as dit

    paths = []
    for i, (w, h) in enumerate([(96, 64), (64, 160), (300, 200)]):
        img = Image.new("RGB", (w, h), (10, 20 * i, 30))
        path = tmp_path / f"img{i}.jpg"
        if i == 2:
            exif = Image.Exif()
            exif[0x0112] = 6  # rotate 90: the oriented size is 200 x 300
            img.save(path, exif = exif)
        else:
            img.save(path)
        paths.append(str(path))
    for center_crop in (False, True):
        for flip in (False, True):
            for res in (64, (64, 96)):
                a, b = random.Random(3), random.Random(3)
                for p in paths:
                    dit._load_pixel_tensor(p, res, center_crop, flip, a)
                    dit._skip_pixel_draws(p, res, center_crop, flip, b)
                    assert a.getstate() == b.getstate(), (p, center_crop, flip, res)


def test_a_rank_private_generator_leaves_the_shared_stream():
    import core.training.diffusion_dit_trainer as dit

    torch.manual_seed(0)
    ref = torch.randn(4)
    torch.manual_seed(0)
    with dit._rank_private_rng("cpu", 11):
        own = torch.randn(4)
    with dit._rank_private_rng("cpu", 12):
        other = torch.randn(4)
    assert torch.equal(torch.randn(4), ref)
    assert not torch.equal(own, other)


def test_ranks_pin_nccl_to_loopback_unless_the_user_chose(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    env = {}
    diffusion_ddp._pin_loopback(env)
    assert env == {"NCCL_SOCKET_IFNAME": "lo"}
    env = {"NCCL_SOCKET_IFNAME": "eth0"}
    diffusion_ddp._pin_loopback(env)
    assert env == {"NCCL_SOCKET_IFNAME": "eth0"}


def test_a_checkpoint_resumes_only_at_its_gpu_count():
    """Every rank draws train_batch_size images, so a 2-GPU bundle continued on 1 GPU would train at half the global
    batch on a different sampler stream: refused, like a batch-size change."""
    from core.training.diffusion_checkpoint import CheckpointIdentity, with_world_size

    base = CheckpointIdentity(
        family = FAMILY,
        base_model = "m",
        lora_target_modules = ("to_q",),
        lora_rank = 4,
        lora_alpha = 4,
        precision = "bf16",
        base_precision = "bf16",
        resolution = 64,
        train_batch_size = 1,
    )
    two, one = with_world_size(base, 2), with_world_size(base, 1)
    assert "GPU count (2 vs 1)" in two.mismatch_reason(one)
    assert one.mismatch_reason(two) is not None
    assert two.mismatch_reason(with_world_size(base, 2)) is None
    # The route's preflight knows no world yet: cannot tell, not a mismatch.
    assert two.mismatch_reason(base) is None
    # A bundle from before the field trained on one GPU.
    assert base.mismatch_reason(one) is None
    assert base.mismatch_reason(two) is not None
    assert CheckpointIdentity.from_dict(two.as_dict()).world_size == 2


if __name__ == "__main__":
    mode, rank, world, port, *rest = sys.argv[1:]
    fn = {"grads": _child_grads, "train": _child_train}[mode]
    fn(int(rank), int(world), int(port), *rest)
