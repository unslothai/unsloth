# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2, gloo CPU worlds of 2 (and 4 for the 2-D mesh):

* per-modality presence agreement (fsdp2_vision): a vision `get_image_features` with a required
  argument no dummy supplies, and an audio tower, raise on every rank when present on some ranks
  only (instead of one rank raising or skipping collectives); a buildable image dummy still runs
  on the ranks without images; all-present / all-absent batches train;
* native_clip_grad_norm_ on an HSDP (2 replicate x 2 shard) mesh matches torch on an unsharded copy;
* a prebuilt optimizer + LR scheduler over mixed DTensor / replicated groups is refused, and one
  built from split_param_groups is accepted with its groups intact;
* Unsloth block swap is refused before sharding, and the post-init check recognises native FSDP2."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _stub_unsloth_save  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import (  # noqa: E402
    _install_wrapped_trainer_init,
    _lora,
    _targs,
)
from test_fsdp2_native_guards_gloo import _Stop, _tiny_llama  # noqa: E402
from test_fsdp2_native_trainer_fsdp_clip_gloo import NORM_TYPES, _Net  # noqa: E402

H = 8
SIZE = 4


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(SIZE * SIZE, H)

    def forward(self, pixel_values):
        return self.proj(pixel_values.flatten(1))


class _LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, H)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(2)])

    def forward(self, inputs_embeds):
        h = inputs_embeds
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return h


class _MM(torch.nn.Module):
    """Llava-like: vision tower + projector via get_image_features, an audio tower run inline."""

    def __init__(self, high_res):
        super().__init__()
        torch.manual_seed(0)
        self.high_res = high_res
        self.config = types.SimpleNamespace(
            vision_config = types.SimpleNamespace(image_size = SIZE, patch_size = 2, num_channels = 1)
        )
        self.vision_tower = _Tower()
        self.multi_modal_projector = torch.nn.Linear(H, H)
        self.audio_tower = torch.nn.Linear(3, H)
        self.language_model = _LM()

    def get_input_embeddings(self):
        return self.language_model.embed

    def _image(self, pixel_values):
        return self.multi_modal_projector(self.vision_tower(pixel_values))

    def forward(
        self,
        input_ids,
        pixel_values = None,
        high_res_pixel_values = None,
        input_features = None,
    ):
        embeds = self.language_model.embed(input_ids)
        if pixel_values is not None:
            if self.high_res:
                feats = self.get_image_features(pixel_values, high_res_pixel_values)
            else:
                feats = self.get_image_features(pixel_values)
            embeds = embeds + feats.mean(0)
        if input_features is not None:
            embeds = embeds + self.audio_tower(input_features).mean(1, keepdim = True)
        return self.language_model(inputs_embeds = embeds).square().mean()


class _MMHighRes(_MM):
    def get_image_features(self, pixel_values, high_res_pixel_values):
        return self._image(pixel_values) + 0.0 * high_res_pixel_values.sum()


class _MMPlain(_MM):
    def get_image_features(self, pixel_values):
        return self._image(pixel_values)


def _modality_step(model, rank, image, audio):
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    kwargs = {}
    if image:
        kwargs["pixel_values"] = torch.randn(1, 1, SIZE, SIZE)
        kwargs["high_res_pixel_values"] = torch.randn(1, 1, 2 * SIZE, 2 * SIZE)
    if audio:
        kwargs["input_features"] = torch.randn(1, 6, 3)
    model(ids, **kwargs).backward()


def _case_modality(native, rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = init_device_mesh("cpu", (2,))
    res = {}
    for name, cls in (("plain", _MMPlain), ("high_res", _MMHighRes)):
        model = cls(high_res = cls is _MMHighRes)
        for module in (model.vision_tower, model.audio_tower, *model.language_model.layers):
            fully_shard(module, mesh = mesh)
        fully_shard(model, mesh = mesh)
        vision.install_vision_dummy(model)
        steps = [("all_text", False, False), ("all_image", True, False), ("all_audio", False, True)]
        steps += [("mixed_image", rank == 0, False), ("mixed_audio", False, rank == 1)]
        for step, image, audio in steps:
            try:
                _modality_step(model, rank, image, audio)
                res[f"{name}.{step}"] = "ok"
            except Exception as exc:
                res[f"{name}.{step}"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    return res


def _case_clip_hsdp(native):
    import copy

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    reference = _Net()
    model = copy.deepcopy(reference)
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("replicate", "shard"))
    for name in ("a", "b", "dead"):
        fully_shard(getattr(model, name), mesh = mesh)
    fully_shard(model, mesh = mesh, ignored_params = {model.rep})
    x = torch.randn(4, 7, generator = torch.Generator().manual_seed(1))  # same batch everywhere
    res = {"placements": str(model.a.weight.placements)}
    for norm_type in NORM_TYPES:
        for m in (model, reference):
            m.zero_grad(set_to_none = True)
            m(x).backward()
        want = torch.nn.utils.clip_grad_norm_(
            list(reference.parameters()), 0.5, norm_type = norm_type
        )
        got = native.native_clip_grad_norm_(list(model.parameters()), 0.5, norm_type = norm_type)
        diff = 0.0
        for p, q in zip(model.parameters(), reference.parameters()):
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            diff = max(diff, (g - q.grad).abs().max().item())
        res[str(norm_type)] = {"got": float(got), "want": float(want), "grad_diff": diff}
    return res


def _case_prebuilt_optimizer(native, tmp):
    _install_wrapped_trainer_init()
    from transformers import Trainer

    model = _lora(1)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n)
    cfg = native.FSDP2Config(param_dtype = torch.float32, ignored_params = {first})
    model = native.shard_model(model, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    res = {}
    optimizer = torch.optim.AdamW(trainable, lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    try:
        trainer = Trainer(
            model = model, args = _targs(os.path.join(tmp, "out")), optimizers = (optimizer, scheduler)
        )
        trainer.create_optimizer()
        res["mixed"] = {"groups": len(optimizer.param_groups), "base_lrs": len(scheduler.base_lrs)}
    except RuntimeError as exc:
        res["mixed"] = {"error": str(exc)}
    if not hasattr(native, "split_param_groups"):
        return res  # (before the helper existed)
    optimizer = torch.optim.AdamW(native.split_param_groups(trainable), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    trainer = Trainer(
        model = model, args = _targs(os.path.join(tmp, "out")), optimizers = (optimizer, scheduler)
    )
    trainer.create_optimizer()
    res["split"] = {"groups": len(optimizer.param_groups), "base_lrs": len(scheduler.base_lrs)}
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
        if case == "modality":
            res = _case_modality(native, rank)
        elif case == "clip_hsdp":
            res = _case_clip_hsdp(native)
        elif case == "prebuilt_optimizer":
            res = _case_prebuilt_optimizer(native, tmp)
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
    world = 2,
    timeout = 180,
):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (world, _free_port(), out, case, str(tmp_path)),
        nprocs = world,
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
    assert sorted(results) == list(range(world)), results
    for rank, r in results.items():
        print("ROUND4_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_modality_presence_is_agreed_over_ranks(tmp_path):
    for rank, r in _run("modality", tmp_path).items():
        for name in ("plain", "high_res"):
            for step in ("all_text", "all_image", "all_audio"):
                assert r[f"{name}.{step}"] == "ok", (rank, name, step, r)
            # Audio has no dummy: a mixed audio batch raises on every rank.
            assert r[f"{name}.mixed_audio"].startswith("RuntimeError"), (rank, r)
            assert "audio on 1 of 2 ranks" in r[f"{name}.mixed_audio"], (rank, r)
        # A buildable image dummy runs on the rank without images.
        assert r["plain.mixed_image"] == "ok", (rank, r)
        # get_image_features needs high_res_pixel_values, which no dummy supplies: raise everywhere.
        assert r["high_res.mixed_image"].startswith("RuntimeError"), (rank, r)
        assert "high_res_pixel_values" in r["high_res.mixed_image"], (rank, r)


def test_native_clip_on_an_hsdp_mesh_matches_torch(tmp_path):
    for rank, r in _run("clip_hsdp", tmp_path, world = 4, timeout = 240).items():
        assert "Replicate" in r["placements"] and "Shard" in r["placements"], (rank, r)
        for norm_type in NORM_TYPES:
            v = r[str(norm_type)]
            assert math.isclose(v["got"], v["want"], rel_tol = 1e-5, abs_tol = 1e-6), (
                rank,
                norm_type,
                v,
            )
            assert v["grad_diff"] < 1e-6, (rank, norm_type, v)


def test_prebuilt_mixed_optimizer_with_scheduler_is_refused(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("prebuilt_optimizer", tmp_path).items():
        assert "split_param_groups" in r["mixed"].get("error", ""), (rank, r)
        assert r["split"]["groups"] == r["split"]["base_lrs"] == 2, (
            rank,
            r,
        )  # split before the scheduler


# ---- CPU unit checks (no process group) -------------------------------------------------------


@pytest.fixture
def native_env(monkeypatch):
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    return _load_native()


def test_block_swap_is_refused_before_sharding(native_env, monkeypatch):
    pytest.importorskip("transformers")
    native = native_env

    def _stop(device, *_):
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _stop)
    model = _tiny_llama()
    model.model.layers._unsloth_block_swap = types.SimpleNamespace(indices = [1])
    model._unsloth_block_swap = model.model.layers._unsloth_block_swap
    with pytest.raises(ValueError, match = r"offload_layers = 1 .*native FSDP2"):
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
    assert not native.is_native_fsdp2(model)
    # An "auto" rank that swapped nothing refuses too (no rank shards alone).
    model = _tiny_llama()
    model._unsloth_offload_layers_auto = 1
    with pytest.raises(ValueError, match = r"offload_layers = 'auto' \(this rank swapped 0"):
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
    with pytest.raises(_Stop):  # no block swap: sharding proceeds
        native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))


def test_post_init_block_swap_check_recognises_native_fsdp2(monkeypatch):
    pytest.importorskip("unsloth_zoo")
    from test_block_swap_auto_ddp import _ignore_helpers

    check = _ignore_helpers(monkeypatch)["_block_swap_for_trainer_backend"]
    model = types.SimpleNamespace(
        _unsloth_fsdp2_native = True, _unsloth_block_swap = types.SimpleNamespace(indices = [2])
    )
    trainer = types.SimpleNamespace(model = model, is_fsdp_enabled = False, is_deepspeed_enabled = False)
    with pytest.raises(ValueError, match = "native FSDP2"):
        check(trainer)
