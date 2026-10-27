# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 Trainer checkpoints on a gloo CPU world of 2 (fsdp2_native.py loaded on its own):

* `Trainer._load_best_model` (load_best_model_at_end) goes through the native loader, LoRA and full;
* a resume restores persistent buffers (a routing bias), not only parameters;
* `Trainer.save_model` of a LoRA model writes the frozen tensors PEFT saves: the resized embedding
  tables and an inactive adapter, with their full values.
Plus the CPU-only unit checks of the process-group bootstrap and the bitsandbytes optimizer swap."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _stub_unsloth_save():
    """unsloth/save.py needs the whole package (GPU); the hook only needs its PEFT selector, taken
    verbatim from the source."""
    if "unsloth.save" in sys.modules:
        return
    path = os.path.join(REPO, "unsloth", "save.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_fsdp2_peft_saved_frozen"
    )
    module = types.ModuleType("unsloth.save")
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), module.__dict__)
    sys.modules["unsloth.save"] = module


def _model(
    lora,
    seed,
    tmp,
    resize = False,
    other_adapter = False,
    buffers = False,
):
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None  # tiny repo's -1 fails save_pretrained's config check
    base_dir = os.path.join(tmp, "base_config")
    if resize and not os.path.isdir(base_dir):
        config.save_pretrained(base_dir)  # PEFT compares the vocab with the base config's
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None
    g = torch.Generator().manual_seed(seed)
    if resize:
        model.resize_token_embeddings(config.vocab_size + 8)
        model.config._name_or_path = base_dir
        model.name_or_path = base_dir
        with torch.no_grad():
            for module in (model.get_input_embeddings(), model.get_output_embeddings()):
                module.weight.normal_(0, 0.02, generator = g)
    if buffers:
        # Persistent buffers outside and inside a block (DeepSeek-like routing correction bias).
        model.model.register_buffer("running_scale", torch.full((4,), float(seed)))
        model.model.layers[0].mlp.register_buffer("route_bias", torch.arange(4.0) * seed)
    if lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"])
        model = get_peft_model(model, cfg)
        if other_adapter:
            model.add_adapter("other", LoraConfig(r = 2, lora_alpha = 4, target_modules = ["k_proj"]))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.normal_(0, 0.02, generator = g)
    elif not resize:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator = g) * 0.01)
    return model


def _full(model, trainable_only = False):
    from torch.distributed.tensor import DTensor
    return {
        n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad or not trainable_only
    }


def _trainer(model, tmp):
    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(output_dir = os.path.join(tmp, "out"), report_to = [], use_cpu = True)
    return Trainer(model = model, args = args)


def _worker(rank, world, port, out, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        cfg = native.FSDP2Config(param_dtype = torch.float32)
        ckpt = os.path.join(tmp, "ckpt")
        res = {}
        if case in ("best_lora", "best_full"):
            lora = case == "best_lora"
            src = native.shard_model(_model(lora, 1, tmp), cfg)
            _trainer(src, tmp).save_model(ckpt)
            dist.barrier()
            want = _full(src, trainable_only = lora)
            dst = native.shard_model(_model(lora, 2, tmp), cfg)
            trainer = _trainer(dst, tmp)
            trainer.state.best_model_checkpoint = ckpt
            try:
                trainer._load_best_model()
                got = _full(dst, trainable_only = lora)
                res["max_diff"] = max((got[n] - want[n]).abs().max().item() for n in want)
                res["n"] = len(want)
            except Exception as exc:
                res["load_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        elif case == "full_save_wait":
            # Full finetuning: only rank 0 writes the gathered model. Slow its write down; rank 1
            # must not return from save_model before the files exist.
            import time

            src = native.shard_model(_model(False, 1, tmp), cfg)
            if rank == 0:
                save = native._save_unwrapped

                def slow_save(*args, **kwargs):
                    time.sleep(3.0)
                    return save(*args, **kwargs)

                native._save_unwrapped = slow_save
            _trainer(src, tmp).save_model(ckpt)
            res["written_on_return"] = os.path.isfile(os.path.join(ckpt, "model.safetensors"))
        elif case == "buffers":
            src = native.shard_model(_model(False, 1, tmp, buffers = True), cfg)
            state = native.native_full_state_dict(src)
            if rank == 0:
                src.save_pretrained(ckpt, state_dict = state, safe_serialization = True)
            dist.barrier()
            dst = native.shard_model(_model(False, 3, tmp, buffers = True), cfg)
            native.native_load_model_checkpoint(dst, ckpt)
            res["running_scale"] = dst.model.running_scale.tolist()
            res["route_bias"] = dst.model.layers[0].mlp.route_bias.tolist()
        elif case == "save_frozen":
            from safetensors.torch import load_file

            model = native.shard_model(_model(True, 1, tmp, resize = True, other_adapter = True), cfg)
            want = _full(model)
            _trainer(model, tmp).save_model(ckpt)
            dist.barrier()
            if rank == 0:
                saved = load_file(os.path.join(ckpt, "adapter_model.safetensors"))
                other_path = os.path.join(ckpt, "other", "adapter_model.safetensors")
                other = load_file(other_path) if os.path.isfile(other_path) else {}
                res["keys"] = sorted(saved)
                res["other_keys"] = sorted(other)

                def _match(saved_dict, adapter):
                    worst = 0.0
                    for k, v in saved_dict.items():
                        name = k
                        if "lora_" in k:
                            name = k.replace(".weight", f".{adapter}.weight")
                        if name not in want:
                            return f"no model tensor for {k}"
                        worst = max(worst, (v - want[name]).abs().max().item())
                    return worst

                res["diff"] = _match(saved, "default")
                res["other_diff"] = _match(other, "other")
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def _run(case, tmp_path):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker,
        args = (2, _free_port(), out, case, str(tmp_path)),
        nprocs = 2,
        join = True,
        start_method = "spawn",
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("CKPT_RESULT", case, rank, r)
        assert "error" not in r, r["error"]
    return results


@pytest.mark.parametrize("case", ["best_lora", "best_full"])
def test_load_best_model_uses_native_loader(case, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run(case, tmp_path).items():
        assert "load_error" not in r, (rank, r)
        assert r["n"] > 0 and r["max_diff"] == 0.0, (rank, r)


def test_full_save_non_writers_wait_for_the_write(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("full_save_wait", tmp_path).items():
        assert r["written_on_return"], (rank, r)


def test_resume_restores_persistent_buffers(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("buffers", tmp_path).items():
        assert r["running_scale"] == [1.0] * 4, (rank, r)
        assert r["route_bias"] == [0.0, 1.0, 2.0, 3.0], (rank, r)


def test_trainer_save_writes_frozen_peft_tensors(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    r = _run("save_frozen", tmp_path)[0]
    assert any("embed_tokens" in k for k in r["keys"]), r
    assert any("lm_head" in k for k in r["keys"]), r
    assert any("lora_A" in k for k in r["other_keys"]), r
    assert r["diff"] == 0.0 and r["other_diff"] == 0.0, r


# ---- CPU unit checks (no process group) -------------------------------------------------------


def test_bootstrap_uses_gloo_without_nccl(monkeypatch):
    import torch.distributed as dist

    native = _load_native()
    calls = {}
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
    monkeypatch.setattr(
        dist, "init_process_group", lambda backend = None, **kw: calls.update(b = backend)
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    native._ensure_process_group(torch.device("cuda", 0))
    assert calls["b"] == "gloo"
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    native._ensure_process_group(torch.device("cuda", 0))
    assert calls["b"] == "nccl"
    native._ensure_process_group(torch.device("cpu"))
    assert calls["b"] == "gloo"


@pytest.mark.parametrize(
    "optim",
    [
        "lion_8bit",
        "lion_32bit",
        "paged_lion_8bit",
        "paged_lion_32bit",
        "ademamix",
        "ademamix_8bit",
        "paged_ademamix_32bit",
        "paged_ademamix_8bit",
    ],
)
def test_bnb_optimizers_without_torch_equivalent_are_refused(optim):
    native = _load_native()
    args = types.SimpleNamespace(optim = optim)
    with pytest.raises(ValueError, match = "adamw_torch"):
        native._swap_bnb_optimizer(args)
    assert args.optim == optim


@pytest.mark.parametrize(
    "optim,target",
    [
        ("adamw_8bit", "adamw_torch"),
        ("paged_adamw_32bit", "adamw_torch"),
        ("rmsprop_bnb", "rmsprop"),
        ("rmsprop_bnb_8bit", "rmsprop"),
        ("rmsprop_bnb_32bit", "rmsprop"),
        ("adamw_torch", "adamw_torch"),
    ],
)
def test_bnb_optimizers_switch_to_torch(optim, target, monkeypatch):
    # bitsandbytes Adam / AdamW step local shards by default; UNSLOTH_FSDP2_BNB_LOCAL=0 switches them.
    monkeypatch.setenv("UNSLOTH_FSDP2_BNB_LOCAL", "0")
    native = _load_native()
    args = types.SimpleNamespace(optim = optim)
    native._swap_bnb_optimizer(args)
    value = str(getattr(args.optim, "value", args.optim))
    assert value in (target, target + "_fused"), value  # fused only when CUDA is visible
    # Every bitsandbytes optimizer transformers knows is either switched or refused.
    try:
        from transformers.trainer_optimizer import _BITSANDBYTES_OPTIMIZERS
    except ImportError:
        return
    known = set(native._BNB_OPTIMS) | set(native._BNB_UNSUPPORTED)
    assert {str(o.value) for o in _BITSANDBYTES_OPTIMIZERS} <= known


@pytest.mark.parametrize(
    "optim", ["adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "paged_adamw_32bit"]
)
def test_bnb_adam_is_kept_for_local_shards(optim):
    pytest.importorskip("bitsandbytes")
    native = _load_native()
    args = types.SimpleNamespace(optim = optim)
    assert native._swap_bnb_optimizer(args) is False
    assert args.optim == optim
    assert set(native._BNB_LOCAL_SHARD_OPTIMS) <= set(native._BNB_OPTIMS)


def _non_blockwise_adamw8bit():
    """bitsandbytes Adam with per-tensor 8-bit state (block_wise = False, the kwarg older bitsandbytes
    takes): an Optimizer2State 'adam' subclass the class check alone accepts."""
    import bitsandbytes as bnb

    class NonBlockwiseAdamW8bit(bnb.optim.AdamW8bit):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.args.block_wise = False

    return NonBlockwiseAdamW8bit


@pytest.mark.parametrize("which", ["lamb", "non_blockwise", "adamw8bit", "paged_adamw8bit"])
def test_bnb_optimizer_cls_preflight_uses_the_localization_predicate(which):
    bnb = pytest.importorskip("bitsandbytes")
    native = _load_native()
    cls = {
        "lamb": bnb.optim.LAMB8bit,
        "non_blockwise": _non_blockwise_adamw8bit(),
        "adamw8bit": bnb.optim.AdamW8bit,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
    }[which]
    if which in ("lamb", "non_blockwise"):
        with pytest.raises(ValueError, match = "cannot"):
            native._refuse_bnb_optimizer_cls((cls, {"lr": 1e-3}))
    else:
        native._refuse_bnb_optimizer_cls((cls, {"lr": 1e-3}))


def test_cpu_shards_keep_the_torch_adamw_switch_when_bnb_cannot_step_cpu(monkeypatch):
    """CPU-offloaded / CPU-mesh shards are CPU tensors; bitsandbytes releases before 0.50 refuse CPU
    inputs in the 8-bit Adam step, so adamw_8bit falls back to torch's AdamW there."""
    pytest.importorskip("bitsandbytes")
    native = _load_native()
    import unsloth.models.fsdp2_bnb_optim as local_shards

    for cpu_shards, can_step_cpu, kept in (
        (False, False, True),
        (True, True, True),
        (True, False, False),
    ):
        monkeypatch.setattr(local_shards, "bnb_steps_cpu_tensors", lambda: can_step_cpu)
        args = types.SimpleNamespace(optim = "adamw_8bit")
        native._swap_bnb_optimizer(args, cpu_shards)
        value = str(getattr(args.optim, "value", args.optim))
        assert (value == "adamw_8bit") == kept, (cpu_shards, can_step_cpu, value)
        if not kept:
            assert value in ("adamw_torch", "adamw_torch_fused"), value
    # A prebuilt bitsandbytes optimizer over CPU tensors is then not localized (and refused).
    import bitsandbytes as bnb

    opt = bnb.optim.AdamW8bit([torch.nn.Parameter(torch.zeros(4))])
    assert native._localize_bnb_optimizer(opt) is False
    with pytest.raises(ValueError, match = "cannot"):
        native._refuse_bnb_optimizer_instance(opt)
    # Where the shards will live: UNSLOTH_FSDP2_CPU_OFFLOAD, a CPU Trainer, or no CUDA device.
    monkeypatch.setattr(native.torch.cuda, "is_available", lambda: True)
    args = types.SimpleNamespace(use_cpu = False, no_cuda = False)
    monkeypatch.delenv("ACCELERATE_USE_CPU", raising = False)
    monkeypatch.delenv("UNSLOTH_FSDP2_CPU_OFFLOAD", raising = False)
    assert native._native_shards_on_cpu(None, args) is False
    monkeypatch.setenv("UNSLOTH_FSDP2_CPU_OFFLOAD", "1")
    assert native._native_shards_on_cpu(None, args) is True
    monkeypatch.delenv("UNSLOTH_FSDP2_CPU_OFFLOAD")
    assert native._native_shards_on_cpu(None, types.SimpleNamespace(use_cpu = True)) is True


def _q_galore_class():
    """unsloth/optimizers/q_galore_adamw.py loaded on its own (unsloth_zoo's device helper stubbed)."""
    import importlib.util

    zoo = types.ModuleType("unsloth_zoo.device_type")
    zoo.device_synchronize = lambda *a, **k: None
    sys.modules.setdefault("unsloth_zoo", types.ModuleType("unsloth_zoo"))
    sys.modules["unsloth_zoo.device_type"] = zoo
    optimizers = types.ModuleType("unsloth.optimizers")
    optimizers.__path__ = [os.path.join(REPO, "unsloth", "optimizers")]
    sys.modules["unsloth.optimizers"] = optimizers
    name = "unsloth.optimizers.q_galore_adamw"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "unsloth", "optimizers", "q_galore_adamw.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.QGaLoreAdamW8bit


def test_q_galore_is_refused_not_localized():
    bnb = pytest.importorskip("bitsandbytes")
    native = _load_native()
    from unsloth.models.fsdp2_bnb_optim import supports_local_shards

    QGaLoreAdamW8bit = _q_galore_class()
    opt = QGaLoreAdamW8bit([torch.nn.Parameter(torch.zeros(64, 64))], lr = 1e-3)
    assert opt.optimizer_name == "adam"  # what the old predicate keyed on
    assert supports_local_shards(opt) is False and supports_local_shards(QGaLoreAdamW8bit) is False
    assert native._localize_bnb_optimizer(opt) is False
    with pytest.raises(ValueError, match = "Q-GaLore"):
        native._refuse_bnb_optimizer_instance(opt)
    with pytest.raises(ValueError, match = "cannot"):
        native._refuse_bnb_optimizer_cls((QGaLoreAdamW8bit, {"lr": 1e-3}))
    # bitsandbytes' own Adam classes still step local shards.
    for cls in (
        bnb.optim.AdamW8bit,
        bnb.optim.Adam8bit,
        bnb.optim.PagedAdamW8bit,
        bnb.optim.PagedAdamW32bit,
    ):
        assert supports_local_shards(cls([torch.nn.Parameter(torch.zeros(4))])), cls
