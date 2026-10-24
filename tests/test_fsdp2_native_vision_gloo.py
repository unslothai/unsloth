# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2, mixed image / text micro-batches (gloo CPU world of 2): rank 0 trains on an image +
text batch, rank 1 on text only. Without the dummy vision pass (fsdp2_vision) the vision units
all-gather on rank 0 only and the run fails or times out, and with the vision LoRA held by the root
(frozen base replicated) the reduce-scatter sets differ. With it, two steps complete and the gradients
equal the unsharded model's on both micro-batches, averaged. Loads fsdp2_native.py on its own (no
`import unsloth`), so it runs without a GPU."""

import datetime
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

# Spawned gloo ranks re-import this module: keep them on the CPU (deterministic on GPU hosts; CUDA
# reads the mask at its lazy first use, after this).
if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"


def _load_native():
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    for name, path in (
        ("unsloth.fsdp2_launch", os.path.join(REPO, "unsloth", "fsdp2_launch.py")),
        ("unsloth.models.fsdp2_native", os.path.join(REPO, "unsloth", "models", "fsdp2_native.py")),
        ("unsloth.models.fsdp2_vision", os.path.join(REPO, "unsloth", "models", "fsdp2_vision.py")),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["unsloth.models.fsdp2_native"], sys.modules["unsloth.models.fsdp2_vision"]


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _build(lora):
    import transformers

    config = transformers.AutoConfig.from_pretrained(TINY)
    torch.manual_seed(0)
    model = transformers.AutoModelForImageTextToText.from_config(config, torch_dtype = torch.float32)
    if lora:
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(2)
        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02, generator = torch.Generator().manual_seed(len(n)))
    return model.train()


def _batches(model):
    config = model.config
    vision = config.vision_config
    m = vision.spatial_merge_size
    grid = torch.tensor([[1, 2 * m, 2 * m]])
    n_patches = int(grid.prod())
    n_tokens = n_patches // (m * m)
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    g = torch.Generator().manual_seed(1)
    text = lambda n: torch.randint(3, 1000, (n,), generator = g)  # noqa: E731
    image_ids = torch.cat(
        [
            text(4),
            torch.tensor([config.vision_start_token_id]),
            torch.full((n_tokens,), config.image_token_id),
            torch.tensor([config.vision_end_token_id]),
            text(10 - n_tokens),
        ]
    )[None]
    image = {
        "input_ids": image_ids,
        "labels": image_ids.clone(),
        "pixel_values": torch.randn(n_patches, patch_dim, generator = g),
        "image_grid_thw": grid,
        "use_cache": False,
    }
    text_ids = text(image_ids.shape[1])[None]
    plain = {"input_ids": text_ids, "labels": text_ids.clone(), "use_cache": False}
    return [image, plain]


def _worker(rank, world, port, lora, root_only, dummy, out):
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
    result = {"rank": rank}
    try:
        native, vision = _load_native()
        if not dummy:
            vision.install_vision_dummy = lambda model: None
        ref = _build(lora)
        model = _build(lora)
        batches = _batches(model)
        config = native.FSDP2Config(param_dtype = torch.float32)
        if root_only:
            # The fit layout with LoRA: no block units, frozen base replicated, adapters in the root.
            config = native.FSDP2Config(param_dtype = torch.float32, units = [], shard_frozen = False)
        native.shard_model(model, config)
        result["installed"] = any(getattr(m, vision._INSTALLED, False) for m in model.modules())
        from torch.distributed.tensor import DTensor

        ref_opt = torch.optim.SGD([p for p in ref.parameters() if p.requires_grad], lr = 0.5)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.5)
        worst = 0.0
        compared = 0
        losses = []
        for step in range(2):
            ref_opt.zero_grad(set_to_none = True)
            for b in batches:
                (ref(**b).loss / len(batches)).backward()
            ref_grads = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None}
            ref_opt.step()
            opt.zero_grad(set_to_none = True)
            loss = model(**batches[rank]).loss
            loss.backward()
            losses.append(loss.item())
            for n, p in model.named_parameters():
                if n not in ref_grads:
                    continue
                if p.grad is None:
                    worst = float("inf")
                    result.setdefault("missing", []).append(n)
                    continue
                g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                r = ref_grads[n]
                compared += 1
                worst = max(worst, (g - r).abs().max().item() / (r.abs().max().item() + 1e-8))
            opt.step()
        result.update(ok = True, worst_rel_grad = worst, compared = compared, losses = losses)
    except Exception as exc:
        result.update(ok = False, error = f"{type(exc).__name__}: {str(exc)[:300]}")
    # Written whole or not at all: a surviving rank is killed once its peer dies, possibly mid-write.
    with open(f"{out}.{rank}.tmp", "w", encoding = "utf-8") as f:
        json.dump(result, f)
    os.replace(f"{out}.{rank}.tmp", f"{out}.{rank}")
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _run(lora, root_only, dummy, tmp_path):
    import torch.multiprocessing as mp

    out = str(tmp_path / "result.json")
    # CPU only: the spawned children inherit this environment before they import torch.
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        ctx = mp.start_processes(
            _worker,
            args = (2, _free_port(), lora, root_only, dummy, out),
            nprocs = 2,
            join = False,
            start_method = "spawn",
        )
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    import time

    deadline = time.time() + 300
    while True:
        try:
            if ctx.join(timeout = 5):
                break
        except Exception:  # a rank died (gloo aborted the collective): the expected failure mode
            break
        if time.time() > deadline:
            break
    for p in ctx.processes:
        if p.is_alive():
            p.kill()
    results = []
    for rank in range(2):
        try:
            with open(f"{out}.{rank}", encoding = "utf-8") as f:
                results.append(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            results.append({"rank": rank, "ok": False, "error": "no result (hung or killed)"})
    return results


def _skip_without_tiny():
    try:
        import transformers
        transformers.AutoConfig.from_pretrained(TINY)
    except Exception as exc:
        pytest.skip(f"{TINY}: {type(exc).__name__}: {exc}")


LAYOUTS = [(False, False), (True, False), (True, True)]
LAYOUT_IDS = ["full-blocks", "lora-blocks", "lora-root-only"]


@pytest.mark.parametrize("lora,root_only", LAYOUTS, ids = LAYOUT_IDS)
def test_text_only_rank_runs_dummy_vision(lora, root_only, tmp_path):
    _skip_without_tiny()
    results = _run(lora, root_only, True, tmp_path)
    for r in results:
        assert r["ok"], results
        assert r["installed"], results
        assert r["worst_rel_grad"] < 1e-4, results
        assert r["compared"] > 0, results


@pytest.mark.parametrize("lora,root_only", LAYOUTS, ids = LAYOUT_IDS)
def test_without_dummy_vision_desyncs(lora, root_only, tmp_path):
    """Negative control: the same run without the dummy pass fails, times out or gets wrong grads."""
    _skip_without_tiny()
    results = _run(lora, root_only, False, tmp_path)
    assert not all(r["ok"] and r["worst_rel_grad"] < 1e-4 for r in results), results
