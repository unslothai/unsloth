# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2:

* the process group native setup creates uses TrainingArguments.ddp_backend / ddp_timeout (or the
  arguments' own device setup), with the NCCL-or-gloo default only when none is given;
* a model already on CUDA shards onto cuda:LOCAL_RANK, not its first weight's GPU; quantized weights on
  another GPU, or a model split over GPUs, are refused;
* transformers 5 parallelism_config (TP / CP / SP / replicate) and tp_plan models are refused;
* FSDP2 names resolve on torch 2.4 / 2.5's `torch.distributed._composable.fsdp` too;
* Phi-4-multimodal-style `image_embed` / `audio_embed` towers (and an unrecognised encoder, through
  the fallback) get the per-modality agreement (gloo world of 2)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_guards_gloo import _Stop, _tiny_llama  # noqa: E402

BASE_ARGS = dict(optim = "adamw_torch", save_on_each_node = False, fsdp = None)


@pytest.fixture
def native_env(monkeypatch):
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0", LOCAL_RANK = "1").items():
        monkeypatch.setenv(key, value)
    for key in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_CPU"):
        monkeypatch.delenv(key, raising = False)
    return _load_native()


def _fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)


# ---- 1. backend / timeout ---------------------------------------------------------------------


def test_process_group_uses_trainer_backend_and_timeout(native_env, monkeypatch):
    pytest.importorskip("transformers")
    import torch.distributed as dist

    native = native_env
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    calls = []

    def _init(**kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(dist, "init_process_group", _init)
    args = types.SimpleNamespace(ddp_backend = "gloo", ddp_timeout = 123, **BASE_ARGS)
    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(model = _tiny_llama(), phase = "pre", training_args = args)
    assert calls[-1].get("backend") == "gloo", calls
    assert calls[-1].get("timeout") == datetime.timedelta(seconds = 123), calls
    # No ddp_backend: the default (NCCL for CUDA when available, else gloo), still with the timeout.
    args = types.SimpleNamespace(ddp_backend = None, ddp_timeout = 77, **BASE_ARGS)
    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(model = _tiny_llama(), phase = "pre", training_args = args)
    assert calls[-1]["backend"] == ("nccl" if dist.is_nccl_available() else "gloo"), calls
    assert calls[-1].get("timeout") == datetime.timedelta(seconds = 77), calls

    # Real TrainingArguments: their own device setup (PartialState with backend / timeout) runs first.
    class Args(types.SimpleNamespace):
        @functools.cached_property
        def _setup_devices(self):
            calls.append({"setup_devices": True})
            raise _Stop

    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(
            model = _tiny_llama(), phase = "pre", training_args = Args(**BASE_ARGS)
        )
    assert calls[-1] == {"setup_devices": True}, calls


# ---- 2. cuda:LOCAL_RANK -----------------------------------------------------------------------


class _T:
    def __init__(self, device):
        self.device = torch.device(device)


class Params4bit(_T):
    pass


class _FakeModel:
    def __init__(
        self,
        tensors,
        device_map = None,
    ):
        self._tensors = tensors
        if device_map is not None:
            self.hf_device_map = device_map

    def parameters(self):
        return iter(self._tensors)

    def buffers(self):
        return iter(())

    def modules(self):
        return iter((self,))


def test_placed_model_shards_onto_the_local_rank_gpu(native_env, monkeypatch):
    native = native_env
    _fake_cuda(monkeypatch)
    # device_map = {"": 0} on every rank: rank 1 must still use cuda:1.
    model = _FakeModel([_T("cuda:0"), _T("cuda:0")], device_map = {"": 0})
    assert native._shard_device(model) == torch.device("cuda", 1)
    with pytest.raises(RuntimeError, match = "quantized weights are on cuda:0"):
        native._shard_device(_FakeModel([_T("cuda:0"), Params4bit("cuda:0")]))
    with pytest.raises(RuntimeError, match = "split over several GPUs"):
        native._shard_device(_FakeModel([_T("cuda:0"), _T("cuda:1")], device_map = {"a": 0, "b": 1}))
    # Quantized weights already on this rank's GPU are fine.
    assert native._shard_device(_FakeModel([Params4bit("cuda:1"), _T("cuda:0")])) == torch.device(
        "cuda", 1
    )


# ---- 3. parallelism_config / tp_plan ----------------------------------------------------------


def test_parallelism_config_and_tp_models_are_refused(native_env, monkeypatch):
    pytest.importorskip("transformers")
    native = native_env
    monkeypatch.setattr(
        native, "_ensure_process_group", lambda device, *_: (_ for _ in ()).throw(_Stop())
    )

    def pc(**sizes):
        base = dict(tp_size = 1, cp_size = 1, sp_size = 1, dp_replicate_size = 1, dp_shard_size = 1)
        return types.SimpleNamespace(**{**base, **sizes})

    for sizes in (dict(tp_size = 2), dict(cp_size = 2), dict(sp_size = 2), dict(dp_replicate_size = 2)):
        args = types.SimpleNamespace(parallelism_config = pc(**sizes), **BASE_ARGS)
        model = _tiny_llama()
        with pytest.raises(RuntimeError, match = "parallelism_config"):
            native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
        assert not native.is_native_fsdp2(model)
    try:
        from accelerate import ParallelismConfig
        real = ParallelismConfig(tp_size = 2)
    except Exception:
        real = None
    if real is not None:
        args = types.SimpleNamespace(parallelism_config = real, **BASE_ARGS)
        with pytest.raises(RuntimeError, match = "parallelism_config"):
            native.native_fsdp2_trainer_setup(model = _tiny_llama(), phase = "pre", training_args = args)
    # A plain shard over every rank is what native FSDP2 does: sharding proceeds.
    args = types.SimpleNamespace(parallelism_config = pc(dp_shard_size = 2), **BASE_ARGS)
    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(model = _tiny_llama(), phase = "pre", training_args = args)
    # A tensor-parallel model (from_pretrained(tp_plan = ...) sets tp_size).
    model = _tiny_llama()
    model._tp_size = 2
    with pytest.raises(RuntimeError, match = "already distributed"):
        native.native_fsdp2_trainer_setup(
            model = model, phase = "pre", training_args = types.SimpleNamespace(**BASE_ARGS)
        )
    # transformers expert parallelism (DistributedConfig(tp_size = N, enable_expert_parallel = True)):
    # refused with a message that names it.
    for tp_size in (2, 4):
        for dc in (
            types.SimpleNamespace(enable_expert_parallel = True, tp_size = tp_size),
            {"enable_expert_parallel": True, "tp_size": tp_size},
        ):
            model = _tiny_llama()
            model._tp_size = tp_size
            model.config.distributed_config = dc
            with pytest.raises(RuntimeError, match = "expert parallelism") as e:
                native.native_fsdp2_trainer_setup(
                    model = model, phase = "pre", training_args = types.SimpleNamespace(**BASE_ARGS)
                )
            assert "UNSLOTH_FSDP2_BACKEND=accelerate with" not in str(e.value)
            assert not native.is_native_fsdp2(model)
    # Expert parallelism off, or a stale flag on a model that was not distributed: not refused.
    for enable in (False, True):
        model = _tiny_llama()
        model.config.distributed_config = types.SimpleNamespace(
            enable_expert_parallel = enable, tp_size = 1
        )
        native._refuse_tensor_parallel_model(model)


# ---- 4. torch 2.4 / 2.5 layout ----------------------------------------------------------------


def test_fsdp_names_resolve_on_the_composable_layout(native_env, monkeypatch):
    native = native_env
    sentinel = object()
    public = types.ModuleType("torch.distributed.fsdp")  # 2.4 / 2.5: no FSDP2 names promoted here
    composable = types.ModuleType("torch.distributed._composable.fsdp")
    composable.CPUOffloadPolicy = sentinel
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", public)
    monkeypatch.setitem(sys.modules, "torch.distributed._composable.fsdp", composable)
    assert native._fsdp_attr("CPUOffloadPolicy") is sentinel


_FSDP_FILES = (
    "fsdp2_native.py",
    "fsdp2_perf.py",
    "fsdp2_vision.py",
    "fsdp2_native_quant.py",
    "fsdp2_save.py",
    "fsdp2_qlora.py",
)


def test_no_unguarded_public_fsdp2_import():
    """Every `from torch.distributed.fsdp import <FSDP2 name>` sits in a try with an ImportError (or
    Exception) handler, so torch 2.4 / 2.5 falls back to `_composable.fsdp`."""
    names = {
        "fully_shard",
        "FSDPModule",
        "MixedPrecisionPolicy",
        "OffloadPolicy",
        "CPUOffloadPolicy",
        "register_fsdp_forward_method",
    }
    bad = []
    for name in _FSDP_FILES:
        path = os.path.join(REPO, "unsloth", "models", name)
        tree = ast.parse(open(path, encoding = "utf-8").read())
        guarded = set()
        fallback = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                any(
                    isinstance(n, (ast.Import, ast.ImportFrom))
                    and "torch.distributed._composable.fsdp"
                    in (getattr(n, "module", None) or " ".join(a.name for a in n.names))
                    for n in ast.walk(h)
                )
                for h in node.handlers
            ):
                for inner in node.body:
                    fallback.update(id(n) for n in ast.walk(inner))
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                h.type is None or getattr(h.type, "id", "") in ("ImportError", "Exception")
                for h in node.handlers
            ):
                for inner in node.body:
                    guarded.update(id(n) for n in ast.walk(inner))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch.distributed.fsdp"
                and names.intersection(a.name for a in node.names)
                and id(node) not in guarded
            ):
                bad.append(f"{name}:{node.lineno}")
            # FSDP2's private modules move with the layout (`_composable.fsdp.<name>` on 2.4 / 2.5):
            # a direct import must sit in a try whose ImportError handler imports the same module
            # from `torch.distributed._composable.fsdp` (the _fsdp_private loaders).
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("torch.distributed.fsdp._fully_shard")
                and id(node) not in fallback
            ):
                bad.append(f"{name}:{node.lineno} (private)")
    assert not bad, bad


# ---- 5. Phi-4 multimodal-style towers (gloo) --------------------------------------------------


class _ImageEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_processor = torch.nn.Linear(16, 8)

    def forward(self, image_pixel_values):
        return self.img_processor(image_pixel_values.flatten(1))


class _AudioEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 8)

    def forward(self, audio_input_features):
        return self.encoder(audio_input_features).mean(1)


class _FeatureEmbedding(torch.nn.Module):
    """Phi4MultimodalFeatureEmbedding's shape: image_embed / audio_embed run only for their input."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.image_embed = _ImageEmbed()
        self.audio_embed = _AudioEmbed()

    def forward(
        self,
        input_ids,
        image_pixel_values = None,
        audio_input_features = None,
    ):
        h = self.embed(input_ids)
        if image_pixel_values is not None:
            h = h + self.image_embed(image_pixel_values).mean(0)
        if audio_input_features is not None:
            h = h + self.audio_embed(audio_input_features).mean(0)
        return h


class _Phi4Like(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_tokens_extend = _FeatureEmbedding()
        self.head = torch.nn.Linear(8, 4)

    def forward(
        self,
        input_ids,
        image_pixel_values = None,
        audio_input_features = None,
    ):
        h = self.embed_tokens_extend(input_ids, image_pixel_values, audio_input_features)
        return self.head(h).square().mean()


class _Unknown(torch.nn.Module):
    """Takes pixel_values, runs an encoder no name rule recognises, only for images."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(16, 8)
        self.patch_net = torch.nn.Linear(16, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(
        self,
        input_ids,
        pixel_values = None,
    ):
        h = self.embed(input_ids)
        if pixel_values is not None:
            h = h + self.patch_net(pixel_values.flatten(1)).mean(0)
        return self.head(h).square().mean()


def _case_media(rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    _load_native()
    vision = importlib.import_module("unsloth.models.fsdp2_vision")
    mesh = init_device_mesh("cpu", (2,))
    res = {}
    ids = torch.randint(0, 16, (1, 5), generator = torch.Generator().manual_seed(rank))
    image = {"image_pixel_values": torch.randn(1, 1, 4, 4)}
    audio = {"audio_input_features": torch.randn(1, 6, 3)}
    model = _Phi4Like()
    fully_shard(model.embed_tokens_extend.image_embed, mesh = mesh)
    fully_shard(model.embed_tokens_extend.audio_embed.encoder, mesh = mesh)
    fully_shard(model, mesh = mesh)
    host = vision.install_vision_dummy(model)
    res["phi4.host"] = type(host).__name__ if host is not None else None
    steps = [
        ("text", {}),
        ("image", image),
        ("audio", audio),
        ("mixed_image", image if rank == 0 else {}),
        ("mixed_audio", audio if rank == 1 else {}),
    ]
    for step, kwargs in steps:
        try:
            model(ids, **kwargs).backward()
            res[f"phi4.{step}"] = "ok"
        except Exception as exc:
            res[f"phi4.{step}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    model = _Unknown()
    fully_shard(model.patch_net, mesh = mesh)
    fully_shard(model, mesh = mesh)
    vision.install_vision_dummy(model)
    for step, kwargs in (
        ("text", {}),
        ("mixed", {"pixel_values": torch.randn(1, 1, 4, 4)} if rank == 0 else {}),
    ):
        try:
            model(ids, **kwargs).backward()
            res[f"unknown.{step}"] = "ok"
        except Exception as exc:
            res[f"unknown.{step}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


def _worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        out.put((rank, _case_media(rank)))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def test_phi4_style_and_unknown_encoders_are_agreed():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + 180
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail("the ranks did not finish within 180 s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND8_RESULT", rank, r)
        assert "crash" not in r, r["crash"]
        assert r["phi4.host"] in ("_Phi4Like", "FSDP_Phi4Like"), (rank, r)
        for step in ("text", "image", "audio"):
            assert r[f"phi4.{step}"] == "ok", (rank, step, r)
        assert "images on 1 of 2 ranks" in r["phi4.mixed_image"], (rank, r)
        assert "audio on 1 of 2 ranks" in r["phi4.mixed_audio"], (rank, r)
        assert r["unknown.text"] == "ok", (rank, r)
        assert "images on 1 of 2 ranks" in r["unknown.mixed"], (rank, r)
