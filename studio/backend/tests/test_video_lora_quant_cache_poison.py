# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A LoRA-baked video denoiser is never written to the local prequant cache under the base model's key, so a later
plain load can never be served the merged build."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
safetensors_torch = pytest.importorskip("safetensors.torch")
diffusers = pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import video as vid  # noqa: E402

assert Path(vid.__file__).resolve().is_relative_to(BACKEND), vid.__file__

FAMILY = "hunyuanvideo-1.5"
CLS = "HunyuanVideo15Transformer3DModel"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(C.ENV_DIR, str(tmp_path / "cache"))
    for name in (C.ENV_MODE, C.ENV_MAX_GB, C.ENV_ASYNC, vid.VIDEO_RESIDENT_QUANT_CACHE_ENV):
        monkeypatch.delenv(name, raising = False)
    C._HEADER_MEMO.clear()
    getattr(C, "_UNREACHABLE", {}).clear()
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    yield
    C.cancel_background_writes(timeout = 10.0)


def _tiny():
    torch.manual_seed(0)
    m = diffusers.HunyuanVideo15Transformer3DModel(
        in_channels = 9,
        out_channels = 4,
        num_attention_heads = 4,
        attention_head_dim = 128,
        num_layers = 1,
        num_refiner_layers = 1,
        mlp_ratio = 2.0,
        text_embed_dim = 64,
        text_embed_2_dim = 32,
        image_embed_dim = 32,
        rope_axes_dim = (16, 56, 56),
        target_size = 64,
    )
    return m.to(torch.bfloat16).eval()


def _scheme() -> str:
    from torchao.quantization import quantize_

    from core.inference import prequant_safetensors as PS
    from core.inference.diffusion_transformer_quant import _make_quant_config

    for s in ("int8", "fp8"):
        try:
            m = torch.nn.Sequential(torch.nn.Linear(512, 512)).to(torch.bfloat16)
            quantize_(m, _make_quant_config(s))
            PS._torchao_helpers()[0](m.state_dict())
            return s
        except Exception:  # noqa: BLE001
            continue
    pytest.skip("this torchao cannot write a quantized build on the CPU")


def _lora_file(model, path: Path) -> Path:
    name = next(
        n
        for n, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and mod.in_features >= 512 and mod.out_features >= 512
    )
    lin = dict(model.named_modules())[name]
    g = torch.Generator().manual_seed(3)
    safetensors_torch.save_file(
        {
            f"transformer.{name}.lora_A.weight": torch.randn(4, lin.in_features, generator = g),
            f"transformer.{name}.lora_B.weight": torch.randn(lin.out_features, 4, generator = g),
        },
        str(path),
    )
    return path


def test_a_lora_baked_explicit_quant_build_is_not_cached_as_the_base(env, tmp_path, monkeypatch):
    from core.inference import diffusion_transformer_quant as tq

    scheme = _scheme()
    base = tmp_path / "base"
    _tiny().save_pretrained(str(base / "transformer"), safe_serialization = True)
    fam = types.SimpleNamespace(
        name = FAMILY,
        label = "HV1.5",
        transformer_class = CLS,
        base_repo = str(base),
        is_moe = False,
        modular_workflow = None,
        prequant_repos = (),
    )
    model = _tiny()
    lora = _lora_file(model, tmp_path / "style.safetensors")
    pipe = types.SimpleNamespace(transformer = model)
    # What _load_model does for a LoRA load: bake into the dense bf16 denoiser...
    merged = vid.VideoBackend._bake_video_loras(
        pipe, (("style", str(lora), 1.0),), [{"id": "style", "weight": 1.0}], fam = fam
    )
    assert merged == 1
    # ...then the requested explicit precision quantises it (transformer_quant='int8' is not refused for LoRA loads)...
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    monkeypatch.setattr(tq, "native_quant_scheme", lambda *a, **k: None)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16)
    assert tq.quantize_transformer(pipe, target, mode = scheme, family = FAMILY) == scheme
    # ...and, native_cache_spec being None under LoRA, the `elif transformer_quant_pinned is not None` branch
    # calls this synchronously.
    vid._save_runtime_quant_to_cache(fam, [pipe], base = str(base), source = str(base))
    assert (
        C.lookup(fam, scheme, base_repo = str(base)) is None
    ), "the LoRA-merged build was cached under the bare base's key"
    # Consequence if it were: the next PLAIN auto load (transformer_quant unset) seeds the LoRA'd weights.
    monkeypatch.setattr(vid, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    plain = vid._video_auto_denoiser_scheme(
        fam,
        target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16),
        requested = None,
        base_repo = str(base),
        speed_mode = None,
    )
    assert plain is None


def test_plain_quant_build_is_still_cached(env, tmp_path, monkeypatch):
    """Negative control: the guard must not stop a plain (no LoRA) build from being cached."""
    from core.inference import diffusion_transformer_quant as tq

    scheme = _scheme()
    base = tmp_path / "base"
    _tiny().save_pretrained(str(base / "transformer"), safe_serialization = True)
    fam = types.SimpleNamespace(
        name = FAMILY,
        label = "HV1.5",
        transformer_class = CLS,
        base_repo = str(base),
        is_moe = False,
        modular_workflow = None,
        prequant_repos = (),
    )
    model = _tiny()
    pipe = types.SimpleNamespace(transformer = model)
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    monkeypatch.setattr(tq, "native_quant_scheme", lambda *a, **k: None)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16)
    assert tq.quantize_transformer(pipe, target, mode = scheme, family = FAMILY) == scheme
    vid._save_runtime_quant_to_cache(fam, [pipe], base = str(base), source = str(base))
    assert C.lookup(fam, scheme, base_repo = str(base)) is not None
