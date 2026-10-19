# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (unsloth/models/fsdp2_native.py) on a gloo CPU world of 2: unit discovery for many
tiny architectures, then fully_shard per unit + root, and a forward / backward whose loss and
gradients match the unsharded model. Loads `fsdp2_native.py` on its own (no `import unsloth`), so
it runs without a GPU."""

import importlib.util
import os
import socket
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (architecture, tiny repo, loader class, LoRA target modules or None)
ARCHS = [
    ("Llama", "hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM"),
    ("Qwen3", "trl-internal-testing/tiny-Qwen3ForCausalLM", "AutoModelForCausalLM"),
    ("Mistral", "hf-internal-testing/tiny-random-MistralForCausalLM", "AutoModelForCausalLM"),
    ("Gemma2", "trl-internal-testing/tiny-Gemma2ForCausalLM", "AutoModelForCausalLM"),
    ("Qwen3Moe", "trl-internal-testing/tiny-Qwen3MoeForCausalLM", "AutoModelForCausalLM"),
    ("GptOss", "trl-internal-testing/tiny-GptOssForCausalLM", "AutoModelForCausalLM"),
    ("Qwen3Next", "hf-internal-testing/tiny-random-Qwen3NextForCausalLM", "AutoModelForCausalLM"),
    (
        "Gemma3VL",
        "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
        "AutoModelForImageTextToText",
    ),
    (
        "Qwen2_5_VL",
        "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
    ),
    (
        "Qwen3VL",
        "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
    ),
    (
        "Gemma4",
        "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
        "AutoModelForImageTextToText",
    ),
    (
        "Qwen3_5",
        "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration",
        "AutoModelForImageTextToText",
    ),
]


def _load_native():
    """fsdp2_native.py without importing the unsloth package (which needs a GPU)."""
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    for name, path in (
        ("unsloth.fsdp2_launch", os.path.join(REPO, "unsloth", "fsdp2_launch.py")),
        ("unsloth.models.fsdp2_native", os.path.join(REPO, "unsloth", "models", "fsdp2_native.py")),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["unsloth.models.fsdp2_native"]


def _build(repo, loader):
    import transformers

    config = transformers.AutoConfig.from_pretrained(repo)
    torch.manual_seed(0)
    cls = getattr(transformers, loader)
    model = cls.from_config(config, torch_dtype = torch.float32)
    return model.eval()


def _expected_blocks(model):
    """Every ModuleList child whose class is a decoder / encoder block by name."""
    n = 0
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            kids = list(module.children())
            if kids and all(
                any(k in type(c).__name__ for k in ("Layer", "Block"))
                and "Expert" not in type(c).__name__
                for c in kids
            ):
                n += len(kids)
    return n


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _batch(model, seed = 1):
    config = getattr(model.config, "text_config", None) or model.config
    vocab = config.vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, min(vocab, 1000), (2, 16), generator = g)
    return {"input_ids": ids, "labels": ids.clone()}


def _worker(rank, world, port, repo, loader, lora, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        assert native.native_fsdp2_requested()
        ref = _build(repo, loader)
        model = _build(repo, loader)
        if lora:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear")
            torch.manual_seed(2)
            ref = get_peft_model(ref, cfg)
            torch.manual_seed(2)
            model = get_peft_model(model, cfg)
            # lora_B starts at zero, which would hide a wrong base weight gather in the gradients.
            with torch.no_grad():
                for (n1, p1), (n2, p2) in zip(ref.named_parameters(), model.named_parameters()):
                    if "lora_B" in n1:
                        p1.normal_(0, 0.02, generator = torch.Generator().manual_seed(len(n1)))
                        p2.copy_(p1)
        units = native.find_shard_units(model)
        expected = _expected_blocks(model)
        names = {type(u).__name__ for u in units}
        batch = _batch(model)
        ref_loss = ref(**batch).loss
        ref_loss.backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}

        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
        assert native.is_native_fsdp2(model)
        assert native.shard_model(model) is model  # idempotent
        loss = model(**batch).loss
        loss.backward()
        from torch.distributed.tensor import DTensor

        worst = 0.0
        missing = []
        n_dtensor = 0
        for n, p in model.named_parameters():
            if isinstance(p, DTensor):
                n_dtensor += 1
            if p.grad is None:
                if n in ref_grads:
                    missing.append(n)
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            r = ref_grads[n]
            worst = max(worst, (g - r).abs().max().item() / (r.abs().max().item() + 1e-8))
        n_params = sum(1 for _ in model.named_parameters())
        result = {
            "units": len(units),
            "expected": expected,
            "unit_classes": sorted(names),
            "loss": loss.item(),
            "ref_loss": ref_loss.item(),
            "worst_rel_grad": worst,
            "missing": missing,
            "dtensor_params": n_dtensor,
            "params": n_params,
            "n_grads": len(ref_grads),
        }
        if rank == 0:
            import json
            with open(out, "w") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def _run(repo, loader, lora, tmp_path):
    import torch.multiprocessing as mp

    out = str(tmp_path / "result.json")
    mp.spawn(_worker, args = (2, _free_port(), repo, loader, lora, out), nprocs = 2, join = True)
    import json

    with open(out) as f:
        return json.load(f)


@pytest.mark.parametrize("lora", [False, True], ids = ["full", "lora"])
@pytest.mark.parametrize("arch,repo,loader", ARCHS, ids = [a[0] for a in ARCHS])
def test_native_fsdp2_units_and_grads(arch, repo, loader, lora, tmp_path):
    try:
        import transformers
        transformers.AutoConfig.from_pretrained(repo)
    except Exception as exc:  # unknown architecture in this transformers
        pytest.skip(f"{repo}: {type(exc).__name__}: {exc}")
    r = _run(repo, loader, lora, tmp_path)
    assert r["units"] >= max(1, r["expected"]), r
    assert r["dtensor_params"] == r["params"], r  # root covers every leftover parameter
    assert not r["missing"], r
    assert abs(r["loss"] - r["ref_loss"]) <= 1e-5 * max(1.0, abs(r["ref_loss"])), r
    assert r["worst_rel_grad"] < 1e-4, r


def test_native_selection_env(monkeypatch):
    native = _load_native()
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising = False)
    monkeypatch.delenv("UNSLOTH_FSDP2_BACKEND", raising = False)
    monkeypatch.setenv("UNSLOTH_FSDP2", "0")
    assert not native.native_fsdp2_requested()
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    assert native.native_fsdp2_requested()
    monkeypatch.setenv("UNSLOTH_FSDP2", "native")
    assert native.native_fsdp2_requested()
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "accelerate")
    assert not native.native_fsdp2_requested()
    monkeypatch.delenv("UNSLOTH_FSDP2_BACKEND")
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    assert not native.native_fsdp2_requested()
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "native")
    assert native.native_fsdp2_requested()
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not native.native_fsdp2_requested()
    launch = sys.modules["unsloth.fsdp2_launch"]
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("ACCELERATE_USE_FSDP")
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert "ACCELERATE_USE_FSDP" not in os.environ


def test_find_units_jamba_like_hybrid():
    native = _load_native()

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class Mamba(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class M(torch.nn.Module):
        _no_split_modules = ["Attn", "Mamba"]

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)
            self.layers = torch.nn.ModuleList([Attn(), Mamba(), Attn()])
            self.heads = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])

    m = M()
    units = native.find_shard_units(m)
    assert [type(u).__name__ for u in units] == ["Attn", "Mamba", "Attn"]
