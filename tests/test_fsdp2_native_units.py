# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (unsloth/models/fsdp2_native.py) on a gloo CPU world of 2: unit discovery for many
tiny architectures, then fully_shard per unit + root, and a forward / backward whose loss and
gradients match the unsharded model. Loads `fsdp2_native.py` on its own (no `import unsloth`), so
it runs without a GPU."""

import importlib.util
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
        "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
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
    from tests._rendezvous_port import free_port
    return free_port()


def _batch(model, seed = 1):
    config = getattr(model.config, "text_config", None) or model.config
    vocab = config.vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, min(vocab, 1000), (2, 16), generator = g)
    return {"input_ids": ids, "labels": ids.clone(), "use_cache": False}


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
        zero_grads = 0
        for n, p in model.named_parameters():
            if isinstance(p, DTensor):
                n_dtensor += 1
            if p.grad is None:
                if n in ref_grads:
                    missing.append(n)
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            if n not in ref_grads:
                # Only fsdp2_vision's dummy image (output times 0) gives a text-only batch vision grads.
                assert float(g.abs().max()) == 0.0, n
                zero_grads += 1
                continue
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
            "extra_units": len(getattr(model, "_unsloth_fsdp2_extra_units", [])),
            "zero_grads": zero_grads,
        }
        if loader == "AutoModelForImageTextToText":
            import importlib

            vision = importlib.import_module("unsloth.models.fsdp2_vision")
            plan = vision._vision_plan(model)
            # A dummy image this family can run (every required get_image_features argument given).
            result["dummy_viable"] = plan is not None and plan[2] is not None
        # generate after a backward: root-held weights must be gathered (PeftModel.generate skips root hooks).
        if loader == "AutoModelForCausalLM":
            prompt = batch["input_ids"][:1, :6]
            try:
                with torch.no_grad():
                    ref_tokens = ref.generate(input_ids = prompt, max_new_tokens = 4, do_sample = False)
            except (
                Exception
            ):  # transformers cannot generate this tiny model at all (Qwen3-Next cache)
                ref_tokens = None
            if ref_tokens is not None:
                with torch.no_grad():
                    tokens = model.generate(input_ids = prompt, max_new_tokens = 4, do_sample = False)
                result["generate_equal"] = bool(torch.equal(ref_tokens, tokens))
                result["after_generate_sharded"] = all(
                    isinstance(p, DTensor) for p in model.parameters()
                )
        if rank == 0:
            import json
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def _run(repo, loader, lora, tmp_path):
    import torch.multiprocessing as mp

    out = str(tmp_path / "result.json")
    mp.spawn(_worker, args = (2, _free_port(), repo, loader, lora, out), nprocs = 2, join = True)
    import json

    with open(out, encoding = "utf-8") as f:
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
    if "generate_equal" in r:
        assert r["generate_equal"] and r["after_generate_sharded"], r
    if arch == "Gemma4":
        assert r["extra_units"] >= 1, r  # embed_tokens_per_layer gets its own unit
    if loader == "AutoModelForImageTextToText":
        # Text-only on every rank: the presence agreement sees no mixed batch, so no dummy runs and
        # the vision tower gets no gradient anywhere (matching collectives). The dummy for this family
        # can be built (it runs when another rank has images: test_fsdp2_native_vision_gloo).
        assert r["zero_grads"] == 0, r
        assert r["dummy_viable"], r


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


def _mixed_worker(
    rank,
    world,
    port,
    out,
    lora = False,
    average = True,
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import json

    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        from torch.distributed.tensor import DTensor

        def build():
            m = _build("hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM")
            m = m.to(torch.bfloat16)
            for name, p in m.named_parameters():
                if "norm" in name:  # Unsloth full finetuning: float32 norms, 16-bit linears
                    p.data = p.data.float()
            if lora:  # frozen float32 norms next to frozen bf16 linears and bf16 adapters
                from peft import LoraConfig, get_peft_model

                torch.manual_seed(2)
                m = get_peft_model(m, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
                with torch.no_grad():
                    for n, p in m.named_parameters():
                        if "lora_" in n:
                            p.data = p.data.to(torch.bfloat16)
                        if "lora_B" in n:
                            p.normal_(0, 0.02, generator = torch.Generator().manual_seed(len(n)))
            return m.train()

        ref, model = build(), build()
        batch = _batch(model)
        # float32 norm outputs meet bf16 linears: autocast, as the Trainer runs one GPU.
        amp = lambda: torch.autocast("cpu", dtype = torch.bfloat16)
        with amp():
            ref_loss = ref(**batch).loss
        ref_loss.backward()
        ref_grads = {
            n: p.grad.float().clone() for n, p in ref.named_parameters() if p.grad is not None
        }
        if not average:  # negative control: no gradient averaging for replicated parameters
            import importlib
            importlib.import_module(
                "unsloth.models.fsdp2_save"
            ).reduce_unsharded_trainable_grads = lambda m: []
        native.shard_model(model, native.FSDP2Config())
        norms_plain = all(
            not isinstance(p, DTensor) and p.dtype == torch.float32
            for n, p in model.named_parameters()
            if "norm" in n
        )
        linears_sharded = all(
            isinstance(p, DTensor) and p.dtype == torch.bfloat16
            for n, p in model.named_parameters()
            if "norm" not in n
        )
        with amp():
            loss = model(**batch).loss
        loss.backward()
        worst = {}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            r = ref_grads[n]
            worst[n] = (g.float() - r).abs().max().item() / (r.abs().max().item() + 1e-8)
        # Weights read outside the blocks' forwards (decode loops, generate): gathered while inside.
        inner = model.get_base_model() if lora else model
        ref_inner = ref.get_base_model() if lora else ref
        with native.native_fsdp2_unsharded(model):
            inside_plain = not any(isinstance(p, DTensor) for p in model.parameters())
            with torch.no_grad(), amp():
                logits = inner.model.layers[0].mlp.down_proj.weight.float().sum().item()
            with torch.no_grad(), amp():
                ref_w = ref_inner.model.layers[0].mlp.down_proj.weight.float().sum().item()
        after_sharded = isinstance(inner.model.layers[0].mlp.down_proj.weight, DTensor)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr = 1e-3, foreach = True
        )
        splits = native._split_mixed_param_groups(opt)
        opt.step()
        clip = native.native_clip_grad_norm_(list(model.parameters()), 1e9)
        ref_norm = torch.sqrt(sum((g**2).sum() for g in ref_grads.values())).item()
        # Replicated parameters must not drift across ranks: different data per rank, then a step.
        rep = [p for p in model.parameters() if p.requires_grad and not isinstance(p, DTensor)]
        before = [p.detach().clone() for p in rep]
        opt.zero_grad()
        with amp():
            loss2 = model(**_batch(model, seed = 10 + rank)).loss
        loss2.backward()
        opt.step()
        flat = torch.cat([p.detach().float().flatten() for p in rep]) if rep else torch.zeros(1)
        gathered = [torch.zeros_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        drift = (gathered[0] - gathered[1]).abs().max().item()
        moved = any((p.detach() != b).any().item() for p, b in zip(rep, before))
        result = {
            "norms_plain": norms_plain,
            "linears_sharded": linears_sharded,
            "worst": max(worst.values()),
            "worst_name": max(worst, key = worst.get),
            "splits": splits,
            "clip": float(clip),
            "ref_norm": ref_norm,
            "replicated": getattr(model, "_unsloth_fsdp2_replicated", None),
            "drift": drift,
            "moved": moved,
            "inside_plain": inside_plain,
            "after_sharded": after_sharded,
            "direct_read_diff": abs(logits - ref_w),
        }
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_mixed_dtype_full_ft_keeps_norms_replicated(tmp_path):
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "mixed.json")
    mp.spawn(_mixed_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    assert r["norms_plain"] and r["linears_sharded"], r
    assert r["replicated"] and r["replicated"] > 0, r
    assert r["splits"] == 1, r
    assert r["moved"] and r["drift"] == 0.0, r  # averaged gradients: replicas stay identical
    _check_unsharded(r)


def _check_unsharded(r):
    assert r["inside_plain"] and r["after_sharded"], r
    assert r["direct_read_diff"] < 1e-3, r
    assert r["worst"] < 2e-2, r  # bf16 compute, float32 reduce
    assert abs(r["clip"] - r["ref_norm"]) <= 2e-2 * r["ref_norm"], r


def test_lora_frozen_fp32_norms_stay_replicated(tmp_path):
    """LoRA: frozen float32 norms next to bf16 weights are not cast to bf16 by the gather (1-GPU numerics)."""
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "lora_mixed.json")
    mp.spawn(_mixed_worker, args = (2, _free_port(), out, True), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    assert r["norms_plain"] and r["linears_sharded"], r
    assert r["splits"] == 0, r  # trainable adapters are all sharded
    _check_unsharded(r)


def test_default_path_untouched():
    """No flag: no torch.distributed.fsdp import, no accelerate / Trainer patch, the model is returned as is."""
    import subprocess

    code = f"""
import os, sys
os.environ.pop("UNSLOTH_FSDP2", None)
os.environ["WORLD_SIZE"] = "2"
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
import test_fsdp2_native_units as t
native = t._load_native()
import torch
m = torch.nn.Linear(2, 2)
assert native.native_fsdp2_trainer_setup(model = m, phase = "pre") is m
assert native.native_fsdp2_trainer_setup(trainer = object(), phase = "post") is None
assert "torch.distributed.fsdp" not in sys.modules, "fsdp imported on the default path"
import accelerate, transformers
assert not getattr(accelerate.Accelerator, native._PATCHED, False)
assert not getattr(transformers.Trainer, native._PATCHED, False)
assert not native.is_native_fsdp2(m)
print("DEFAULT_OK")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 300)
    assert "DEFAULT_OK" in out.stdout, out.stdout[-2000:] + out.stderr[-3000:]


def _knobs_worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import json

    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        from peft import LoraConfig, get_peft_model
        from torch.distributed.tensor import DTensor

        cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear")
        models = []
        for _ in range(2):
            m = _build("hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM")
            torch.manual_seed(2)
            m = get_peft_model(m, cfg)
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if "lora_B" in n:
                        p.normal_(0, 0.02, generator = torch.Generator().manual_seed(len(n)))
            models.append(m)
        ref, model = models
        batch = _batch(model)
        ref(**batch).loss.backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}
        calls = []
        config = native.FSDP2Config(
            param_dtype = torch.float32,
            shard_frozen = False,
            forward_prefetch = 1,
            backward_prefetch = 2,
            reshard_last_block = True,
            unit_reshard_after_forward = lambda i, u: calls.append(i) or (False if i == 0 else None),
        )
        native.shard_model(model, config)
        frozen_plain = all(
            not isinstance(p, DTensor) for p in model.parameters() if not p.requires_grad
        )
        lora_sharded = all(isinstance(p, DTensor) for p in model.parameters() if p.requires_grad)
        units = model._unsloth_fsdp2_units
        reshards = [u._unsloth_reshard_after_forward for u in units]
        model(**batch).loss.backward()
        worst = 0.0
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            worst = max(
                worst,
                (g - ref_grads[n]).abs().max().item() / (ref_grads[n].abs().max().item() + 1e-8),
            )
        result = {
            "frozen_plain": frozen_plain,
            "lora_sharded": lora_sharded,
            "calls": calls,
            "reshards": reshards,
            "worst": worst,
            "n_units": len(units),
        }
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_perf_knobs_shard_frozen_prefetch(tmp_path):
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "knobs.json")
    mp.spawn(_knobs_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    assert r["frozen_plain"] and r["lora_sharded"], r
    assert r["calls"] == list(range(r["n_units"])), r
    assert r["reshards"][0] is False and r["reshards"][-1] is True, r
    assert r["worst"] < 1e-4, r


def _skip_worker(rank, world, port, out, guard):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import json

    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        if not guard:
            native._patch_accumulated_grad_guard = lambda: False

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                return self.proj(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block(), Block()])
                self.tower = torch.nn.ModuleList([Block()])  # skipped on "text" batches
                self.head = torch.nn.Linear(8, 1)

            def forward(self, x, use_tower):
                if use_tower:
                    x = self.tower[0](x)
                for layer in self.layers:
                    x = layer(x)
                return self.head(x).sum()

        torch.manual_seed(0)
        model = Model().to(torch.bfloat16)
        native.shard_model(
            model, native.FSDP2Config(param_dtype = torch.bfloat16, reduce_dtype = torch.float32)
        )
        x = torch.randn(4, 8, dtype = torch.bfloat16)
        error = None
        try:
            model.set_requires_gradient_sync(False)
            model(x, use_tower = False).backward()  # micro-step 1: tower never unsharded
            model.set_requires_gradient_sync(True)
            model(x, use_tower = True).backward()
        except AttributeError as exc:
            error = str(exc)
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump({"error": error}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("guard", [False, True], ids = ["unguarded", "guarded"])
def test_accumulation_with_skipped_unit(guard, tmp_path):
    """float32 reduce + no-sync accumulation + a unit skipped in a micro-step: torch raises without the guard."""
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "skip.json")
    mp.spawn(_skip_worker, args = (2, _free_port(), out, guard), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    if guard:
        assert r["error"] is None, r
    else:
        # Documents the torch bug the guard covers (fixed upstream with a getattr; then both pass).
        if r["error"] is None:
            pytest.skip("this torch already guards _unsharded_param")
        assert "_unsharded_param" in r["error"], r


def test_replicated_params_drift_without_averaging(tmp_path):
    """Negative control for the drift check: without the gradient averaging hook the replicas diverge."""
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "noavg.json")
    mp.spawn(_mixed_worker, args = (2, _free_port(), out, False, False), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    assert r["moved"] and r["drift"] > 0.0, r


def test_find_units_real_jamba_and_peft_linear_lists():
    """R5: mixed-class decoder lists (real Jamba) are units without `_no_split_modules` help; R6: a list of
    PEFT-wrapped Linear layers (Gemma3n altup-like) is not."""
    import transformers
    from peft import LoraConfig, get_peft_model

    native = _load_native()
    cfg = transformers.JambaConfig(
        num_hidden_layers = 4,
        hidden_size = 64,
        intermediate_size = 128,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
        attn_layer_period = 2,
        attn_layer_offset = 1,
        expert_layer_period = 2,
        expert_layer_offset = 1,
        num_experts = 2,
        mamba_d_state = 8,
        mamba_dt_rank = 8,
        use_mamba_kernels = False,
    )
    m = transformers.JambaForCausalLM(cfg)
    m._no_split_modules = []
    for module in m.modules():
        module._no_split_modules = []
    units = native.find_shard_units(m)
    assert [type(u).__name__ for u in units] == [
        "JambaMambaDecoderLayer",
        "JambaAttentionDecoderLayer",
    ] * 2

    base = _build("hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM")
    base.extra_proj = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    peft_model = get_peft_model(base, LoraConfig(r = 4, target_modules = "all-linear"))
    names = {id(mod): n for n, mod in peft_model.named_modules()}
    found = [names[id(u)] for u in native.find_shard_units(peft_model)]
    assert found == ["base_model.model.model.layers.0", "base_model.model.model.layers.1"], found


def test_no_unit_inside_a_compiled_block():
    """torch.compile rule: no FSDP unit nested in a (compiled) decoder block. find_shard_units only picks
    whole blocks; an explicit unit inside a block (an MLP, an expert, a LoRA Linear) is refused before any
    fully_shard or process group."""
    native = _load_native()
    model = _build("hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM")
    layers = model.model.layers
    cls = type(layers[0])
    original = cls.forward

    def compiled_forward(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    compiled_forward._unsloth_block_compiled = True  # what _block_compile marks
    cls.forward = compiled_forward
    try:
        units = native.find_shard_units(model)
        assert [id(u) for u in units] == [id(layer) for layer in layers]
        assert native.check_shard_units(model, units)
        for bad in ([layers[0].mlp], [layers[0], layers[0].self_attn.q_proj]):
            with pytest.raises(RuntimeError, match = "inside the"):
                native.check_shard_units(model, bad)
        with pytest.raises(RuntimeError, match = "inside the"):
            native.shard_model(model, native.FSDP2Config(units = [layers[1].mlp]))
        assert not getattr(model, "_unsloth_fsdp2_native", False)
    finally:
        cls.forward = original
    # Without the compile marker a block is still a block (ModuleList of repeated blocks).
    with pytest.raises(RuntimeError, match = "decoder block"):
        native.check_shard_units(model, [layers[0].mlp])


def _fit_worker(rank, world, port, out, units_mode):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_TRAINABLE_UNITS = units_mode,
    )
    import json

    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        from peft import LoraConfig, get_peft_model
        from torch.distributed.tensor import DTensor

        cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear")
        models = []
        for _ in range(2):
            m = _build("hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM")
            torch.manual_seed(2)
            m = get_peft_model(m, cfg)
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if "lora_B" in n:
                        p.normal_(0, 0.02, generator = torch.Generator().manual_seed(len(n)))
            models.append(m)
        ref, model = models
        # Each rank its own micro batch; the reference averages both (what the reduce-scatter gives).
        batches = [_batch(model, seed = 1 + r) for r in range(world)]
        for b in batches:
            (ref(**b).loss / world).backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}
        # Default config, forced fit layout (CPU has no device memory to size it).
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, fit_layout = True))
        layout = model._unsloth_fsdp2_layout
        frozen_plain = all(
            not isinstance(p, DTensor) for p in model.parameters() if not p.requires_grad
        )
        lora_sharded = all(isinstance(p, DTensor) for p in model.parameters() if p.requires_grad)
        model(**batches[rank]).loss.backward()
        worst = 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None, n
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            worst = max(
                worst,
                (g - ref_grads[n]).abs().max().item() / (ref_grads[n].abs().max().item() + 1e-8),
            )
        result = {
            "layout": layout,
            "frozen_plain": frozen_plain,
            "lora_sharded": lora_sharded,
            "n_units": len(model._unsloth_fsdp2_units),
            "root_only": model._unsloth_fsdp2_root_only,
            "reshards": [u._unsloth_reshard_after_forward for u in model._unsloth_fsdp2_units],
            "worst": worst,
        }
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("units_mode", ["root", "blocks"])
def test_fit_layout_through_shard_model(units_mode, tmp_path):
    """shard_model with the fit layout (fsdp2_perf): frozen base weights replicated (plain tensors, no
    grad), LoRA sharded, one root unit or per-block units kept gathered (last block too), gradients equal
    the averaged unsharded reference."""
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "fit.json")
    mp.spawn(_fit_worker, args = (2, _free_port(), out, units_mode), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    assert r["layout"]["fits"] and r["layout"]["replicate_frozen"], r
    assert r["layout"]["reshard_after_forward"] is False, r
    assert r["frozen_plain"] and r["lora_sharded"], r
    if units_mode == "root":
        assert r["root_only"] and r["n_units"] == 0, r
    else:
        assert not r["root_only"] and r["n_units"] > 1, r
        assert all(x is False for x in r["reshards"]), r
    assert r["worst"] < 1e-4, r


def _gather_dtype_worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_TRAINABLE_UNITS = "root",
    )
    import json

    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        from peft import LoraConfig, get_peft_model

        result = {}
        for name, fit in (("fit", True), ("sharded", False)):
            m = _build("hf-internal-testing/tiny-random-LlamaForCausalLM", "AutoModelForCausalLM")
            m = m.to(torch.bfloat16)
            # PEFT keeps the adapters in float32 (autocast_adapter_dtype), as Unsloth's LoRA does.
            m = get_peft_model(m, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear"))
            assert {p.dtype for p in m.parameters() if p.requires_grad} == {torch.float32}
            native.shard_model(m, native.FSDP2Config(fit_layout = fit))
            result[name] = str(m._unsloth_fsdp2_mp_policy.param_dtype)
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_fit_layout_gathers_float32_masters_as_stored(tmp_path):
    """bf16 base + float32 LoRA: the fit layout (FSDP2 manages only trainable params) gathers the float32
    masters as they are (one GPU keeps them float32 and autocast casts per op); the per-block layout keeps
    the contract's bf16 gather."""
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / "gather.json")
    mp.spawn(_gather_dtype_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    assert r == {"fit": "None", "sharded": "torch.bfloat16"}, r


def _recompute_cast_worker(rank, world, init_file, out, patched):
    os.environ.update(
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import json

    import torch.distributed as dist
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    # A file store, not a TCP port: parallel test workers cannot race for the same port.
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        native = _load_native()
        current = FSDPState._pre_forward
        if not native._recompute_skips_input_cast(current):
            # Emulate torch < 2.13 on a newer torch: the recompute branch returns the inputs uncast.
            def old_pre_forward(self, module, args, kwargs):
                if self._training_state == TrainingState.PRE_BACKWARD:
                    current(self, module, args, kwargs)
                    return args, kwargs
                return current(self, module, args, kwargs)

            FSDPState._pre_forward = old_pre_forward
        if not patched:
            native._patch_recompute_input_cast = lambda: False

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 8, bias = False)

            def forward(self, x):
                # Like Unsloth's LoRA / 4-bit matmul: the weight is cast to the input dtype and saved.
                return x + torch.tanh(x @ self.lin.weight.t().to(x.dtype))

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(Block() for _ in range(2))

            def forward(self, x):
                x = (
                    x.float()
                )  # float32 activations inside the model (the root's pre-forward cast them)
                for layer in self.layers:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant = False)
                return x.float().pow(2).mean()

        torch.manual_seed(0)
        model = Toy()
        native.shard_model(
            model,
            native.FSDP2Config(
                param_dtype = torch.bfloat16, units = list(model.layers), fit_layout = False
            ),
        )
        x = torch.randn(4, 8, generator = torch.Generator().manual_seed(rank))  # float32 inputs
        error = None
        try:
            model(x).backward()
        except Exception as exc:  # CheckpointError: bf16 saved, float32 recomputed
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump({"error": error}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("patched", [False, True])
def test_recompute_sees_the_cast_inputs(patched, tmp_path):
    """torch < 2.13 skipped FSDP2's forward-input cast on activation-checkpoint recompute, so a unit fed
    float32 inputs with bf16 gathered weights saved bf16 tensors in forward and float32 ones in the
    recompute: CheckpointError (Kaggle T4, torch 2.10). The backport casts there too."""
    import json

    import torch.multiprocessing as mp

    out = str(tmp_path / f"recompute_{patched}.json")
    init_file = str(tmp_path / f"recompute_{patched}.store")
    mp.spawn(_recompute_cast_worker, args = (2, init_file, out, patched), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    if patched:
        assert r["error"] is None, r
    else:
        assert r["error"] and "Checkpoint" in r["error"], r
