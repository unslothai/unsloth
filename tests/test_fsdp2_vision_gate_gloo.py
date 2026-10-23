# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 dummy vision pass (fsdp2_vision) on a tiny random Llava, gloo CPU world of 2: rank 0
trains on an image + text batch, rank 1 on text only.

- frozen tower, trainable FSDP2-managed `multi_modal_projector`: the projector's reduce-scatter sets
  differ unless the gate counts the projector.
- frozen tower except replicated trainable norms (gradients averaged by the all-reduce hook): the
  all-reduce runs on the image rank only unless the gate counts those hooked parameters.
- `get_image_features(..., image_sizes = None)` (Llava on transformers >= 5.19): a CLIP tower needs
  `vision_config.image_size`, not a patch-sized dummy.
Each case must finish on both ranks with gradients equal to the unsharded model's on both batches.
Loads fsdp2_native.py on its own (no `import unsloth`), so it runs without a GPU."""

import datetime
import importlib.util
import inspect
import json
import os
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_SIZE = 8
PATCH = 4
IMAGE_TOKEN = 999


def _load_native():
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    for name in ("fsdp2_launch", "models.fsdp2_native", "models.fsdp2_vision", "models.fsdp2_save"):
        path = os.path.join(REPO, "unsloth", *name.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(f"unsloth.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"unsloth.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["unsloth.models.fsdp2_native"], sys.modules["unsloth.models.fsdp2_vision"]


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _config():
    import transformers

    vision = transformers.CLIPVisionConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        image_size = IMAGE_SIZE,
        patch_size = PATCH,
        num_channels = 3,
    )
    text = transformers.LlamaConfig(
        vocab_size = 1000,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
    )
    return transformers.LlavaConfig(
        vision_config = vision,
        text_config = text,
        image_token_index = IMAGE_TOKEN,
        vision_feature_layer = -1,
        vision_feature_select_strategy = "default",
    )


def _build(case):
    import transformers

    torch.manual_seed(0)
    model = transformers.LlavaForConditionalGeneration(_config()).float().train()
    if case in ("projector", "norms"):
        for name, p in model.named_parameters():
            if "vision_tower" in name:
                p.requires_grad_(case == "norms" and "layer_norm" in name)
            elif "multi_modal_projector" in name:
                p.requires_grad_(case == "projector")
    return model


def _batches():
    g = torch.Generator().manual_seed(1)
    n_tokens = (IMAGE_SIZE // PATCH) ** 2
    text = lambda n: torch.randint(3, 900, (n,), generator = g)  # noqa: E731
    image_ids = torch.cat([text(4), torch.full((n_tokens,), IMAGE_TOKEN), text(6)])[None]
    image = {
        "input_ids": image_ids,
        "labels": image_ids.clone(),
        "pixel_values": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator = g),
        "use_cache": False,
    }
    text_ids = text(image_ids.shape[1])[None]
    return [image, {"input_ids": text_ids, "labels": text_ids.clone(), "use_cache": False}]


def _explicit_image_sizes():
    """transformers >= 5.19 Llava: `get_image_features(..., image_sizes = None, **kwargs)`, passed on
    to the backbone only when given. Older versions take it through **kwargs only."""
    from transformers.models.llava import modeling_llava

    cls = modeling_llava.LlavaModel
    if "image_sizes" in inspect.signature(cls.get_image_features).parameters:
        return
    original = cls.get_image_features

    def get_image_features(
        self,
        pixel_values,
        vision_feature_layer = None,
        vision_feature_select_strategy = None,
        image_sizes = None,
        **kwargs,
    ):
        return original(
            self,
            pixel_values = pixel_values,
            vision_feature_layer = vision_feature_layer,
            vision_feature_select_strategy = vision_feature_select_strategy,
            **kwargs,
        )

    cls.get_image_features = get_image_features


def _worker(rank, world, port, case, out):
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
        if case == "image_sizes":
            _explicit_image_sizes()
        ref = _build(case)
        model = _build(case)
        ignored = None
        if case == "norms":
            # Replicated trainable vision norms: their gradients are all-reduced by a hook.
            ignored = {
                p for n, p in model.named_parameters() if "vision_tower" in n and p.requires_grad
            }
        config = native.FSDP2Config(
            param_dtype = torch.float32, units = [], shard_frozen = False, ignored_params = ignored
        )
        native.shard_model(model, config)
        result["installed"] = any(getattr(m, vision._INSTALLED, False) for m in model.modules())
        from torch.distributed.tensor import DTensor

        batches = _batches()
        ref_opt = torch.optim.SGD([p for p in ref.parameters() if p.requires_grad], lr = 0.5)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.5)
        worst, compared = 0.0, 0
        for step in range(2):
            ref_opt.zero_grad(set_to_none = True)
            for b in batches:
                (ref(**b).loss / len(batches)).backward()
            ref_grads = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None}
            ref_opt.step()
            opt.zero_grad(set_to_none = True)
            model(**batches[rank]).loss.backward()
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
        result.update(ok = True, worst_rel_grad = worst, compared = compared)
    except Exception as exc:
        result.update(ok = False, error = f"{type(exc).__name__}: {str(exc)[:300]}")
    with open(f"{out}.{rank}", "w", encoding = "utf-8") as f:
        json.dump(result, f)
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _run(case, tmp_path):
    import time

    import torch.multiprocessing as mp

    out = str(tmp_path / "result.json")
    # CPU only: the spawned children inherit this environment before they import torch.
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        ctx = mp.start_processes(
            _worker,
            args = (2, _free_port(), case, out),
            nprocs = 2,
            join = False,
            start_method = "spawn",
        )
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    deadline = time.time() + 300
    while True:
        try:
            if ctx.join(timeout = 5):
                break
        except Exception:  # a rank died (gloo aborted the collective)
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
        except FileNotFoundError:
            results.append({"rank": rank, "ok": False, "error": "no result (hung or killed)"})
    return results


@pytest.mark.parametrize("case", ["projector", "norms", "image_sizes"])
def test_text_only_rank_matches_image_rank_collectives(case, tmp_path):
    pytest.importorskip("transformers.models.llava")
    if sys.platform == "win32" or not torch.distributed.is_gloo_available():
        pytest.skip("needs torch.distributed with gloo")
    results = _run(case, tmp_path)
    for r in results:
        assert r["ok"], results
        assert r["installed"], results
        assert r["compared"] > 0, results
        assert r["worst_rel_grad"] < 1e-4, results
