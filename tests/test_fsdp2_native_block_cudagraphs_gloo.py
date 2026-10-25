# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_BLOCK_CUDAGRAPHS=1 under native FSDP2 (gloo, world 2): gathered weights keep one address across
steps so block CUDA graphs replay, and losses still match the run without graphs bitwise."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_layout_agree_gloo import REPO_ID, _init, _spawn  # noqa: E402
from test_fsdp2_native_units import _build, _load_native  # noqa: E402

STEPS = 3


def _gathered(model):
    from torch.distributed.fsdp import FSDPModule

    out = []
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        group = getattr(module._get_fsdp_state(), "_fsdp_param_group", None)
        for fsdp_param in getattr(group, "fsdp_params", None) or ():
            outputs = getattr(fsdp_param, "all_gather_outputs", None) or []
            for t in outputs:
                out.append((t.untyped_storage().size(), t.untyped_storage().data_ptr()))
    return out


def _run(native, layout, graphs):
    os.environ["UNSLOTH_BLOCK_CUDAGRAPHS"] = "1" if graphs else "0"
    rank = int(os.environ["RANK"])
    torch.manual_seed(0)
    model = _build(REPO_ID, "AutoModelForCausalLM")
    if layout == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator = g) * 0.02)
        cfg = native.FSDP2Config(param_dtype = torch.float32, fit_layout = True)
    else:
        cfg = native.FSDP2Config(param_dtype = torch.float32, fit_layout = False)
    native.shard_model(model, cfg)
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.5)
    losses, gathered = [], []
    for step in range(STEPS):
        ids = torch.randint(
            3, 1000, (2, 16), generator = torch.Generator().manual_seed(100 * rank + step)
        )
        loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss.detach()))
        gathered.append(_gathered(model))
    return {"losses": losses, "gathered": gathered}


def _worker(rank, world, port, out, path):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        res = {
            (layout, graphs): _run(native, layout, graphs)
            for layout in ("lora", "full")
            for graphs in (False, True)
        }
        torch.save(res, f"{path}.rank{rank}")
        out.put("ok")
    except Exception as exc:
        import traceback
        out.put({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        os.environ.pop("UNSLOTH_BLOCK_CUDAGRAPHS", None)
        dist.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    path = str(tmp_path_factory.mktemp("cudagraphs") / "res.pt")
    payloads = _spawn(_worker, path)
    assert payloads == ["ok", "ok"], payloads
    return [torch.load(f"{path}.rank{r}") for r in range(2)]


@pytest.mark.parametrize("layout", ["lora", "full"])
def test_gathered_weights_keep_one_address_under_block_cudagraphs(ranks, layout):
    for res in ranks:
        steps = res[layout, True]["gathered"]
        assert steps[0], "no FSDP-managed parameters found"
        for snap in steps:
            assert all(size > 0 for size, _ in snap), snap
        assert all(snap == steps[0] for snap in steps), steps


@pytest.mark.parametrize("layout", ["lora", "full"])
def test_kept_weights_follow_the_optimizer(ranks, layout):
    for res in ranks:
        assert res[layout, True]["losses"] == res[layout, False]["losses"]
        assert len(set(res[layout, True]["losses"])) == STEPS  # the weights did move


@pytest.mark.parametrize("layout", ["lora", "full"])
def test_without_graphs_the_gathered_weights_are_freed(ranks, layout):
    for res in ranks:
        snap = res[layout, False]["gathered"][-1]
        assert any(size == 0 for size, _ in snap), snap
