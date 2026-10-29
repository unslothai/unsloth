# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2, on a gloo CPU world of 2 where a process group is needed:

* TrainingArguments(fsdp = ...) is refused under an explicit UNSLOTH_FSDP2_BACKEND=native and wins
  (nothing sharded natively) without one;
* native_clip_grad_norm_ matches torch.nn.utils.clip_grad_norm_ for norm types 0, 1, 2 and inf;
* Mllama (vision tower run inline, cross-attention layers skipped without images) raises on every
  rank when images are on some ranks only, instead of hanging; all-image and all-text batches train;
* TrainingArguments(use_cpu = True) keeps a CPU mesh on a CUDA host."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

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

NORM_TYPES = (0.0, 1.0, 2.0, math.inf)


def _case_trainer_fsdp(native, tmp):
    _install_wrapped_trainer_init()
    from transformers import Trainer

    res = {}
    for backend in ("", "native"):
        os.environ["UNSLOTH_FSDP2_BACKEND"] = backend
        # Accelerator(fsdp_plugin = ...) exports ACCELERATE_USE_FSDP=true for the process
        # (accelerate/accelerator.py): each case starts from a plain torchrun env.
        os.environ.pop("ACCELERATE_USE_FSDP", None)
        model = _lora(1)
        try:
            targs = _targs(os.path.join(tmp, "out"), fsdp = "full_shard")
            trainer = Trainer(model = model, args = targs)
            # On CPU accelerate builds no FSDP state (MULTI_CPU), so is_fsdp_enabled stays False
            # here; the plugin arguments show the Trainer took the FSDP request.
            res[backend or "none"] = {
                "error": "",
                "fsdp_plugin": getattr(trainer.args, "fsdp_plugin_args", None) is not None
                or bool(getattr(trainer.args, "fsdp", None)),
            }
        except Exception as exc:
            res[backend or "none"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        res[backend or "none"]["sharded"] = native.is_native_fsdp2(model)
    os.environ.pop("UNSLOTH_FSDP2_BACKEND", None)
    return res


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = torch.nn.Linear(7, 9)
        self.b = torch.nn.Linear(9, 5)
        self.dead = torch.nn.Linear(9, 3)  # its gradient is exactly zero
        self.rep = torch.nn.Parameter(torch.randn(5))  # replicated (ignored_params)

    def forward(self, x):
        h = torch.tanh(self.a(x))
        return (self.b(h) * self.rep).square().sum() + 0.0 * self.dead(h).sum()


def _case_clip(native):
    import copy

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    reference = _Net()
    model = copy.deepcopy(reference)
    mesh = init_device_mesh("cpu", (2,))
    for name in ("a", "b", "dead"):
        fully_shard(getattr(model, name), mesh = mesh)
    fully_shard(model, mesh = mesh, ignored_params = {model.rep})
    x = torch.randn(4, 7, generator = torch.Generator().manual_seed(1))  # same batch on both ranks
    res = {}
    for norm_type in NORM_TYPES:
        for m in (model, reference):
            m.zero_grad(set_to_none = True)
            m(x).backward()
        want = torch.nn.utils.clip_grad_norm_(
            list(reference.parameters()), 0.5, norm_type = norm_type
        )
        try:
            got = native.native_clip_grad_norm_(list(model.parameters()), 0.5, norm_type = norm_type)
        except Exception as exc:
            res[str(norm_type)] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        grad_diff = 0.0
        for (n, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            grad_diff = max(grad_diff, (g - q.grad).abs().max().item())
        res[str(norm_type)] = {"got": float(got), "want": float(want), "grad_diff": grad_diff}
    return res


def _case_trainer_clip(native):
    """Trainer._clip_grad_norm with sharded bfloat16 and replicated float32 grads."""
    from accelerate import Accelerator
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor
    from transformers import Trainer

    native.install_native_fsdp2_hooks()
    model = _Net()
    for name in ("a", "b", "dead"):
        getattr(model, name).to(torch.bfloat16)
    mesh = init_device_mesh("cpu", (2,))
    for name in ("a", "b", "dead"):
        fully_shard(getattr(model, name), mesh = mesh)
    fully_shard(model, mesh = mesh, ignored_params = {model.rep})
    setattr(model, native._NATIVE_FLAG, True)
    x = torch.randn(64, 7, generator = torch.Generator().manual_seed(1)).to(torch.bfloat16)

    class _T:
        _has_mixed_mesh_grads = Trainer._has_mixed_mesh_grads
        _clip_grad_norm = Trainer._clip_grad_norm
        _mixed_mesh_grad_norm = getattr(Trainer, "_mixed_mesh_grad_norm", None)

    t = _T()
    t.args = types.SimpleNamespace(max_grad_norm = 1.0)
    t._mixed_mesh_grads = None
    t.model = model
    t.accelerator = Accelerator(cpu = True)
    t.accelerator._models.append(model)
    model.zero_grad(set_to_none = True)
    (model(x).float() * 50.0).backward()
    grads = [
        p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad for p in model.parameters()
    ]
    dtypes = sorted({str(g.dtype) for g in grads})
    fp64 = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
    one_gpu = float(torch.nn.utils.get_total_norm([g.clone() for g in grads], 2.0))
    want = float(native.native_clip_grad_norm_(list(model.parameters()), math.inf))
    mixed = float(t._mixed_mesh_grad_norm(model, math.inf)) if t._mixed_mesh_grad_norm else None
    got = float(t._clip_grad_norm(model))
    return {
        "got": got,
        "want": want,
        "mixed": mixed,
        "one_gpu": one_gpu,
        "fp64": fp64,
        "dtypes": dtypes,
    }


def _mllama():
    from transformers import MllamaConfig, MllamaForConditionalGeneration

    config = MllamaConfig(
        vision_config = dict(
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 2,
            num_global_layers = 1,
            attention_heads = 2,
            image_size = 28,
            patch_size = 14,
            max_num_tiles = 1,
            intermediate_layers_indices = [0],
            vision_output_dim = 32,
            supported_aspect_ratios = [[1, 1]],
        ),
        text_config = dict(
            vocab_size = 64,
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            cross_attention_layers = [1],
            pad_token_id = 0,
        ),
        image_token_index = 63,
    )
    torch.manual_seed(0)
    return MllamaForConditionalGeneration(config)


def _mllama_step(model, image):
    ids = torch.randint(1, 60, (1, 6), generator = torch.Generator().manual_seed(3))
    kwargs = dict(input_ids = ids, labels = ids)
    if image:
        kwargs.update(
            pixel_values = torch.randn(1, 1, 1, 3, 28, 28),
            aspect_ratio_ids = torch.tensor([[1]]),
            aspect_ratio_mask = torch.tensor([[[1]]]),
            cross_attention_mask = torch.ones(1, 6, 1, 1, dtype = torch.long),
        )
    model(**kwargs).loss.backward()


def _case_mllama(native, rank):
    model = native.shard_model(_mllama(), native.FSDP2Config(param_dtype = torch.float32))
    res = {}
    for name, image in (("all_image", True), ("all_text", False), ("mixed", rank == 0)):
        try:
            _mllama_step(model, image)
            res[name] = "ok"
        except Exception as exc:
            res[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
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
        if case == "trainer_fsdp":
            res = _case_trainer_fsdp(native, tmp)
        elif case == "clip":
            res = _case_clip(native)
        elif case == "trainer_clip":
            res = _case_trainer_clip(native)
        elif case == "mllama":
            res = _case_mllama(native, rank)
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
        print("FSDP_CLIP_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_trainer_fsdp_args_refused_with_native_backend_and_win_without(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("trainer_fsdp", tmp_path).items():
        native = r["native"]
        assert native["error"].startswith("RuntimeError"), (rank, r)
        assert "UNSLOTH_FSDP2_BACKEND=accelerate" in native["error"], (rank, r)
        assert native["sharded"] is False, (rank, r)
        # No explicit backend: the Trainer's FSDP request wins, the model is left to accelerate.
        assert r["none"]["sharded"] is False, (rank, r)
        assert r["none"]["error"] == "" and r["none"]["fsdp_plugin"], (rank, r)


def test_native_clip_matches_torch_for_every_norm_type(tmp_path):
    for rank, r in _run("clip", tmp_path).items():
        for norm_type in NORM_TYPES:
            v = r[str(norm_type)]
            assert "error" not in v, (rank, norm_type, v)
            assert math.isclose(v["got"], v["want"], rel_tol = 1e-5, abs_tol = 1e-6), (
                rank,
                norm_type,
                v,
            )
            assert v["grad_diff"] < 1e-6, (rank, norm_type, v)


def test_trainer_clip_with_replicated_and_sharded_grads_uses_native_clip(tmp_path):
    # Trainer._mixed_mesh_grad_norm combined bf16 norms in bf16, off by up to 2e-3.
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    if not hasattr(transformers.Trainer, "_has_mixed_mesh_grads"):
        pytest.skip("this transformers has no mixed DTensor / plain gradient norm path")
    for rank, r in _run("trainer_clip", tmp_path).items():
        assert r["dtypes"] == ["torch.bfloat16", "torch.float32"], (rank, r)
        assert r["got"] == r["want"], (rank, r)
        if r["mixed"] is not None:  # this batch separates the two combines
            assert not math.isclose(r["mixed"], r["want"], rel_tol = 1e-4), (rank, r)
        # One extra 16-bit rounding over single-GPU clip_grad_norm_.
        assert math.isclose(r["got"], r["one_gpu"], rel_tol = 2**-6), (rank, r)


def test_mllama_mixed_image_batch_raises_instead_of_hanging(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.mllama")
    for rank, r in _run("mllama", tmp_path, timeout = 150).items():
        assert r["all_image"] == "ok" and r["all_text"] == "ok", (rank, r)
        assert r["mixed"].startswith("RuntimeError") and "1 of 2 ranks" in r["mixed"], (rank, r)


# ---- CPU unit checks (no process group) -------------------------------------------------------


@pytest.mark.parametrize("flag", ["use_cpu", "no_cuda"])
def test_use_cpu_keeps_a_cpu_mesh_on_a_cuda_host(flag, monkeypatch):
    pytest.importorskip("transformers")
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0", LOCAL_RANK = "1").items():
        monkeypatch.setenv(key, value)
    for key in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_CPU", "UNSLOTH_FSDP2_BACKEND"):
        monkeypatch.delenv(key, raising = False)
    native = _load_native()
    model = _tiny_llama()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    seen = {}

    def _record(device, *_):
        seen["device"] = device
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _record)
    # 4.57's `no_cuda` alone (its __post_init__ copies it to use_cpu; a namespace does not).
    args = types.SimpleNamespace(
        **{flag: True}, optim = "adamw_torch", save_on_each_node = False, fsdp = None
    )
    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
    assert seen["device"] == torch.device("cpu"), seen
    # Without it, the CUDA host shards onto the local GPU.
    args = types.SimpleNamespace(optim = "adamw_torch", save_on_each_node = False, fsdp = None)
    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(model = _tiny_llama(), phase = "pre", training_args = args)
    assert seen["device"] == torch.device("cuda", 1), seen
