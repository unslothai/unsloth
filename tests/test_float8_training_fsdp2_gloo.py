# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float8_training under native FSDP2 (UNSLOTH_FSDP2=1), gloo world of 2 on the CPU with torchao's
emulated fp8: the converted Float8Linear layers survive shard_model, their weights become DTensors,
and four AdamW steps give exactly the losses of the same fp8 model trained unsharded."""

import importlib.util
import json
import multiprocessing
import os
import sys
import types

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("torchao.float8")
pytest.importorskip("torch.distributed.fsdp")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _native():
    for name in ("unsloth", "unsloth.models"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(REPO, *name.split("."))]
        sys.modules.setdefault(name, pkg)
    _load("unsloth.fsdp2_launch", os.path.join(REPO, "unsloth", "fsdp2_launch.py"))
    return _load(
        "unsloth.models.fsdp2_native", os.path.join(REPO, "unsloth", "models", "fsdp2_native.py")
    )


def _build():
    from torchao.float8 import Float8LinearConfig
    from transformers import LlamaConfig, LlamaForCausalLM

    f8t = _load(
        "_unsloth_float8_training_fsdp2",
        os.path.join(REPO, "unsloth", "models", "float8_training.py"),
    )
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    model = LlamaForCausalLM(config).float()
    fp8 = Float8LinearConfig.from_recipe_name("tensorwise")
    object.__setattr__(fp8, "emulate", True)
    names = f8t.convert_model_to_float8_training(model, fp8, size_filter = False)
    return model, names


def _train(model, steps = 4):
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, 256, (2, 32), generator = g)
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


def _worker(rank, port, out):
    import torch.distributed as dist
    from torchao.float8.float8_linear import Float8Linear

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = "2",
        LOCAL_RANK = str(rank),
    )
    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        native = _native()
        model, names = _build()
        model = native.shard_model(
            model, native.FSDP2Config(param_dtype = torch.float32, device = "cpu")
        )
        fp8 = [m for m in model.modules() if isinstance(m, Float8Linear)]
        sharded = sum(type(m.weight).__name__ == "DTensor" for m in fp8)
        losses = _train(model)
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(
                    dict(converted = len(names), fp8 = len(fp8), sharded = sharded, losses = losses), f
                )
    finally:
        dist.destroy_process_group()


def _reference_worker(rank, out):
    with open(out, "w", encoding = "utf-8") as f:
        json.dump(_train(_build()[0]), f)


def test_float8_linear_trains_under_native_fsdp2(tmp_path):
    import torch.multiprocessing as mp
    from test_fsdp2_frozen_dtypes_gloo import _free_port

    # The unsharded reference trains in a fresh process too. Computed in the pytest worker it
    # inherits whatever earlier tests left there, and in the CPU rest shard that moved its step-4
    # loss to 5.603441 while the sharded ranks (and a clean process) give 5.603824.
    ref_out = str(tmp_path / "unsharded_fp8.json")
    mp.spawn(_reference_worker, args = (ref_out,), nprocs = 1, join = True)
    reference = json.load(open(ref_out, encoding = "utf-8"))
    out = str(tmp_path / "fsdp2_fp8.json")
    mp.spawn(_worker, args = (_free_port(), out), nprocs = 2, join = True)
    result = json.load(open(out, encoding = "utf-8"))
    assert result["converted"] == 14
    assert result["fp8"] == 14 and result["sharded"] == 14
    assert result["losses"] == pytest.approx(reference, rel = 1e-5, abs = 1e-6)
