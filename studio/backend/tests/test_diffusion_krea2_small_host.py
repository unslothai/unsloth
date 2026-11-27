# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 on an fp16-only card: components load at stored bf16 via the small-host route, not fp32. CPU only."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.diffusion_krea2 as krea2  # noqa: E402
import core.inference.diffusion_small_host as sh  # noqa: E402

# From unsloth/Krea-2-Turbo safetensors headers: Qwen3-VL encoder (741 MiB token embedding), 12.8B DiT, bf16.
KREA2_TURBO = {
    "text_encoder": sh.StoredComponent(
        "text_encoder", 8464, "bfloat16", 0, "transformers", embed_mib = 741
    ),
    "transformer": sh.StoredComponent("transformer", 25065, "bfloat16", 0, "diffusers"),
    "vae": sh.StoredComponent("vae", 484, "float32", 0, "diffusers"),
}
KAGGLE_T4 = (32098, 29000)
BIG_HOST = (512 * 1024, 480 * 1024)
ROOMY_DISK = 200 * 1024


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.SMALL_HOST_ENV, sh.HOST_RAM_CHECK_ENV, sh.SPILL_ENV):
        monkeypatch.delenv(name, raising = False)


def _decide(
    comps,
    total,
    available,
    dtype = None,
    **kw,
):
    return sh.decide_small_host(
        comps,
        dtype or torch.float32,
        device = "cuda",
        host_total_mib = total,
        host_available_mib = available,
        **kw,
    )


@pytest.mark.parametrize("disk", [None, ROOMY_DISK])
def test_kaggle_t4_takes_the_route_instead_of_widening(disk):
    d = _decide(KREA2_TURBO, *KAGGLE_T4, spill_free_mib = disk)
    assert d.engaged and d.refuse is None
    assert d.dense_host_mib > KAGGLE_T4[1] > d.route_host_mib
    assert d.storage_dtypes == {"text_encoder": "bfloat16", "transformer": "bfloat16"}


def test_kaggle_t4_with_the_int8_encoder_routes_the_dit():
    """A pre-built int8 text encoder in pipe_kwargs is not sized; the DiT still takes the route."""
    comps = {k: v for k, v in KREA2_TURBO.items() if k != "text_encoder"}
    d = _decide(comps, *KAGGLE_T4)
    assert d.engaged and d.refuse is None
    assert d.storage_dtypes == {"transformer": "bfloat16"}
    assert d.route_host_mib == 25065 // 2


def test_big_host_keeps_the_dense_load():
    d = _decide(KREA2_TURBO, *BIG_HOST)
    assert not d.engaged and d.refuse is None and not d.storage_dtypes


def _krea_branch() -> ast.If:
    tree = ast.parse((_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "fam.name == KREA2_FAMILY_NAME"
            and any(
                isinstance(n, ast.Call) and ast.unparse(n.func) == "load_krea2_pipeline"
                for n in ast.walk(node)
            )
            and any(
                isinstance(n, ast.Call) and ast.unparse(n.func) == "self._stop_load_prefetch"
                for n in ast.walk(node)
            )
        ):
            return node
    raise AssertionError("Krea 2 pipeline branch not found")


def test_pipeline_branch_decides_the_route_and_loads_stored_dtypes():
    branch = _krea_branch()
    body = ast.Module(body = branch.body, type_ignores = [])
    calls = {ast.unparse(n.func): n for n in ast.walk(body) if isinstance(n, ast.Call)}
    assert "self._small_host_decision" in calls, "Krea 2 skips the small-host decision"
    decide = calls["self._small_host_decision"]
    assert {k.arg for k in decide.keywords} >= {"lora_active", "dense_denoiser", "fetch"}
    assert ast.unparse(decide.args[0]) == ast.unparse(calls["load_krea2_pipeline"].args[0])
    fetch = next(k.value for k in decide.keywords if k.arg == "fetch")
    assert ast.unparse(fetch.func) == "_per_component_small_host_fetch"
    load = calls["load_krea2_pipeline"]
    assert "small_host" in ast.unparse(
        next(k.value for k in load.keywords if k.arg == "storage_dtypes")
    )
    assert decide.lineno < load.lineno
    assigns = [
        n
        for n in ast.walk(body)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and n.value is decide
    ]
    assert [ast.unparse(t) for a in assigns for t in a.targets] == ["small_host"]


def test_fetch_skips_supplied_components(monkeypatch):
    import huggingface_hub

    import core.inference.diffusion as diffusion

    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: seen.update(kw) or "/snap"
    )
    fetch = diffusion._per_component_small_host_fetch(
        "unsloth/Krea-2-Turbo", None, False, None, skip = ("text_encoder",)
    )
    assert fetch() == "/snap"
    assert seen["ignore_patterns"] == ["text_encoder/*.safetensors"]
    assert "*/*.safetensors" in seen["allow_patterns"]
    seen.clear()
    diffusion._per_component_small_host_fetch("unsloth/Krea-2-Turbo", None, False, None)()
    assert seen["ignore_patterns"] is None
    assert diffusion._per_component_small_host_fetch("repo", None, True, None) is None
    assert diffusion._per_component_small_host_fetch("repo", "/staged", False, None) is None


def test_loader_reads_routed_components_at_their_stored_dtype(monkeypatch):
    seen = {}

    def _te(repo, dtype, **kw):
        seen["text_encoder"] = dtype
        return object()

    def _component(name):
        def _load(
            repo,
            subfolder = None,
            torch_dtype = None,
            **kw,
        ):
            seen[subfolder] = torch_dtype
            return object()

        return types.SimpleNamespace(from_pretrained = _load)

    fake = types.SimpleNamespace(
        Krea2Pipeline = lambda **kw: types.SimpleNamespace(**kw),
        FlowMatchEulerDiscreteScheduler = types.SimpleNamespace(
            from_pretrained = lambda *a, **k: object()
        ),
        AutoencoderKLQwenImage = _component("vae"),
        Krea2Transformer2DModel = _component("transformer"),
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setattr(krea2, "_load_model_index", lambda *a, **k: {})
    monkeypatch.setattr(krea2, "load_krea2_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(krea2, "load_krea2_text_encoder", _te)

    krea2.load_krea2_pipeline(
        "repo",
        torch.float32,
        storage_dtypes = {"text_encoder": "bfloat16", "transformer": "bfloat16"},
    )
    assert seen == {
        "text_encoder": torch.bfloat16,
        "transformer": torch.bfloat16,
        "vae": torch.float32,
    }
    seen.clear()
    krea2.load_krea2_pipeline("repo", torch.float32)
    assert seen == {
        "text_encoder": torch.float32,
        "transformer": torch.float32,
        "vae": torch.float32,
    }


# unsloth/Krea-2-Turbo Hub tree (2026-11); the loader never reads the root checkpoint or README images.
KREA2_TURBO_FILES = [
    ".gitattributes",
    "README.md",
    "images/00.jpg",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "turbo.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
    "text_encoder/model.fp16.safetensors",
    "extra/weights.safetensors",
]


def _cold_decision(
    monkeypatch,
    dtype,
    device = "cuda",
    patterns = False,
    **kw,
):
    """Run the backend's small-host decision on a cold Hub cache; returns the snapshot_download calls."""
    import huggingface_hub

    import core.inference.diffusion as diffusion

    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **k: calls.append((repo, k)) or None
    )
    monkeypatch.setattr(diffusion, "resolve_snapshot_dir", lambda *a, **k: None)
    monkeypatch.setattr(diffusion, "host_ram_mib", lambda: KAGGLE_T4)
    backend = object.__new__(diffusion.DiffusionBackend)
    fetch = diffusion._per_component_small_host_fetch(
        "unsloth/Krea-2-Turbo",
        None,
        False,
        None,
        skip = ("text_encoder",),
        **({"allow_patterns": krea2.krea2_hub_patterns()} if patterns else {}),
    )
    out = backend._small_host_decision(
        "unsloth/Krea-2-Turbo",
        {"text_encoder": object()},
        types.SimpleNamespace(backend = device),
        dtype,
        lora_active = kw.pop("lora_active", False),
        fetch = fetch,
    )
    assert out is None
    return calls


@pytest.mark.parametrize(
    "dtype, device, env",
    [
        ("bfloat16", "cuda", None),  # bf16 card: the dense load already stays memory-mapped
        ("float32", "cuda", "0"),  # kill switch
        ("float32", "cpu", None),
        ("float32", "mps", None),
    ],
)
def test_cold_load_does_not_fetch_when_the_route_cannot_engage(monkeypatch, dtype, device, env):
    if env is not None:
        monkeypatch.setenv(sh.SMALL_HOST_ENV, env)
    assert _cold_decision(monkeypatch, getattr(torch, dtype), device) == []


def test_cold_load_does_not_fetch_with_lora_unless_forced(monkeypatch):
    assert _cold_decision(monkeypatch, torch.float32, lora_active = True) == []
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "1")
    assert len(_cold_decision(monkeypatch, torch.float32, lora_active = True)) == 1


def test_cold_fp16_only_load_fetches_exactly_what_the_loader_reads(monkeypatch):
    from huggingface_hub.utils import filter_repo_objects

    calls = _cold_decision(monkeypatch, torch.float32, patterns = True)
    assert len(calls) == 1
    kw = calls[0][1]
    picked = set(
        filter_repo_objects(
            KREA2_TURBO_FILES,
            allow_patterns = kw["allow_patterns"],
            ignore_patterns = kw["ignore_patterns"],
        )
    )
    assert picked == {
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    }


def test_pipeline_branch_fetches_the_krea_patterns():
    branch = _krea_branch()
    fetch = next(
        n
        for n in ast.walk(branch)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_per_component_small_host_fetch"
    )
    patterns = next(k.value for k in fetch.keywords if k.arg == "allow_patterns")
    assert ast.unparse(patterns) == "krea2_hub_patterns()"
