# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 generic media agreement (gloo world of 2): a multimodal model with trainable or
FSDP-managed parameters outside its language model gets one agreement on its outermost forward
over every non-text forward input, whatever the encoder is called.

* Fuyu (transformers/models/fuyu/modeling_fuyu.py: FuyuModel.forward takes `image_patches` and runs
  `vision_embed_tokens` only when it is given, :147-181) with a trainable `vision_embed_tokens`;
* MusicFlamingo / VoxtralRealtime (transformers/models/musicflamingo/modeling_musicflamingo.py
  get_audio_features :223-247, voxtral_realtime/modeling_voxtral_realtime.py :897-928): a frozen,
  replicated audio tower and a sharded `multi_modal_projector` reached only inside
  get_audio_features;

a mixed batch raises on every rank, naming the inputs; uniform batches train; a text-only model
installs nothing."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(2))

    def forward(self, h):
        for layer in self.layers:
            h = h + layer(h)
        return h


class _FuyuLike(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens = nn.Embedding(16, 8).requires_grad_(False)
        self.vision_embed_tokens = nn.Linear(12, 8)  # trainable
        self.language_model = _Decoder()
        self.lm_head = nn.Linear(8, 4)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids,
        image_patches = None,
        image_patches_indices = None,
        attention_mask = None,
        labels = None,
    ):
        h = self.embed_tokens(input_ids)
        if image_patches is not None:
            h = h + self.vision_embed_tokens(image_patches).mean(1, keepdim = True)
        return self.lm_head(self.language_model(h)).square().mean()


class _AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(3, 8)

    def forward(
        self,
        input_features,
        input_features_mask = None,
    ):
        return self.conv(input_features)


class _MusicFlamingoLike(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.audio_tower = _AudioTower().requires_grad_(False)  # frozen, replicated
        self.multi_modal_projector = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.embed_tokens = nn.Embedding(16, 8).requires_grad_(False)
        self.language_model = _Decoder()
        self.lm_head = nn.Linear(8, 4)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_audio_features(self, input_features, input_features_mask):
        return self.multi_modal_projector(self.audio_tower(input_features, input_features_mask))

    def forward(
        self,
        input_ids,
        input_features = None,
        input_features_mask = None,
        attention_mask = None,
    ):
        h = self.embed_tokens(input_ids)
        if input_features is not None:
            h = h + self.get_audio_features(input_features, input_features_mask).mean(
                1, keepdim = True
            )
        return self.lm_head(self.language_model(h)).square().mean()


class _TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(2))

    def forward(
        self,
        input_ids,
        attention_mask = None,
        labels = None,
        num_items_in_batch = None,
    ):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h.square().mean()


def _shard(
    model,
    mesh,
    extra = (),
):
    from torch.distributed.fsdp import fully_shard

    for layer in model.language_model.layers:
        fully_shard(layer, mesh = mesh)
    for module in extra:
        fully_shard(module, mesh = mesh)
    ignored = set(model.audio_tower.parameters()) if hasattr(model, "audio_tower") else None
    fully_shard(model, mesh = mesh, ignored_params = ignored)
    return model


def _steps(model, rank, media):
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    res = {}
    for step, given in (("all_text", False), ("all_media", True), ("mixed", rank == 0)):
        kwargs = media() if given else {}
        try:
            model(ids, **kwargs).backward()
            res[step] = "ok"
        except Exception as exc:
            res[step] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return res


def _case(rank, case):
    from torch.distributed.device_mesh import init_device_mesh

    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = init_device_mesh("cpu", (2,))
    if case == "fuyu":
        model = _shard(_FuyuLike(), mesh)
        media = lambda: {  # noqa: E731
            "image_patches": torch.randn(1, 3, 12),
            "image_patches_indices": torch.arange(3).view(1, 3),
        }
    elif case == "flamingo":
        model = _MusicFlamingoLike()
        model = _shard(model, mesh, extra = [model.multi_modal_projector])
        media = lambda: {  # noqa: E731
            "input_features": torch.randn(1, 6, 3),
            "input_features_mask": torch.ones(1, 6, dtype = torch.long),
        }
    else:
        from torch.distributed.fsdp import fully_shard

        model = _TextOnly()
        for layer in model.layers:
            fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)
        host = vision.install_vision_dummy(model)
        return {"host": None if host is None else type(host).__name__}
    host = vision.install_vision_dummy(model)
    res = {"host": None if host is None else type(host).__name__}
    res.update(_steps(model, rank, media))
    return res


def _worker(rank, world, port, out, case):
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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        _load_native()
        out.put((rank, _case(rank, case)))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(case, timeout = 150):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out, case), nprocs = 2, join = False, start_method = "spawn"
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
        print("ROUND14_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


@pytest.mark.parametrize(
    "case, names",
    [
        ("fuyu", ("image_patches", "image_patches_indices")),
        ("flamingo", ("input_features", "input_features_mask")),
    ],
)
def test_mixed_media_raises_on_every_rank(case, names):
    for rank, r in _run(case).items():
        assert r["host"] is not None, (rank, r)
        assert r["all_text"] == "ok" and r["all_media"] == "ok", (rank, r)
        assert r["mixed"].startswith("RuntimeError") and "on 1 of 2 ranks" in r["mixed"], (rank, r)
        for name in names:
            assert name in r["mixed"], (rank, name, r)


def test_text_only_model_installs_nothing():
    for rank, r in _run("text").items():
        assert r["host"] is None, (rank, r)
