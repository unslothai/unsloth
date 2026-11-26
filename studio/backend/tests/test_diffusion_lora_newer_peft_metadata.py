# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A LoRA saved by a newer PEFT (lightx2v/Qwen-Image-2.1-Lightning: PEFT 0.20, ``lora.down`` / ``lora.up`` keys)
loads on a PEFT whose ``LoraConfig`` lacks some of the fields in its ``lora_adapter_metadata``."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
diffusers = pytest.importorskip("diffusers")
from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion_lora_files as lf  # noqa: E402

_LAYERS, _HEADS, _HEAD_DIM, _RANK = 2, 2, 8, 4
_MODULES = ("to_q", "to_k", "to_v", "to_out.0")
# The file's own header plus one field no PEFT has: the load must not depend on which PEFT runs the test.
_NEWER_PEFT_FIELDS = {
    "lora_ga_config": None,
    "monteclora_config": None,
    "use_bdlora": None,
    "velora_config": None,
    "future_lora_option": None,
}


def _metadata(alpha: int, **extra) -> str:
    config = {
        "r": _RANK,
        "lora_alpha": alpha,
        "target_modules": ["attn.to_out.0", "attn.to_v", "attn.to_q", "attn.to_k"],
        "init_lora_weights": "gaussian",
        "peft_type": "LORA",
        "peft_version": "0.20.0",
        "use_dora": False,
        "use_rslora": False,
        **_NEWER_PEFT_FIELDS,
        **extra,
    }
    return json.dumps({f"transformer.{k}": v for k, v in config.items()})


def _state() -> dict:
    gen = torch.Generator().manual_seed(0)
    inner = _HEADS * _HEAD_DIM
    state = {}
    for i in range(_LAYERS):
        for m in _MODULES:
            stem = f"transformer.transformer_blocks.{i}.attn.{m}"
            state[f"{stem}.lora.down.weight"] = torch.randn(_RANK, inner, generator = gen)
            state[f"{stem}.lora.up.weight"] = torch.randn(inner, _RANK, generator = gen)
    return state


def _pipe():
    try:
        from diffusers import QwenImage21Transformer2DModel
        from diffusers.loaders import QwenImageLoraLoaderMixin
    except ImportError as exc:
        pytest.skip(f"this diffusers has no Qwen-Image-2.1 ({exc})")

    class Pipe(QwenImageLoraLoaderMixin):
        hf_device_map = None

        @property
        def components(self):
            return {"transformer": self.transformer}

    pipe = Pipe()
    pipe.transformer = QwenImage21Transformer2DModel(
        num_layers = _LAYERS,
        attention_head_dim = _HEAD_DIM,
        num_attention_heads = _HEADS,
        context_in_dim = 16,
        in_channels = 4,
        out_channels = 4,
        axes_dims_rope = (2, 2, 4),
    )
    return pipe


@pytest.mark.parametrize("alpha", [_RANK, 2 * _RANK])
def test_newer_peft_metadata_loads_every_module(tmp_path, alpha):
    state = _state()
    path = tmp_path / "pytorch_lora_weights.safetensors"
    save_file(
        state, str(path), metadata = {"lora_adapter_metadata": _metadata(alpha), "format": "pt"}
    )
    pipe = _pipe()

    lf.load_adapter(pipe, str(path), "lightning", "qwen-image-2.1")

    modules = dict(pipe.transformer.named_modules())
    loaded = sorted(
        name[: -len(".lora_A.lightning")] for name in modules if name.endswith(".lora_A.lightning")
    )
    expected = sorted(f"transformer_blocks.{i}.attn.{m}" for i in range(_LAYERS) for m in _MODULES)
    assert loaded == expected
    for stem in expected:
        down = state[f"transformer.{stem}.lora.down.weight"]
        up = state[f"transformer.{stem}.lora.up.weight"]
        assert torch.equal(modules[f"{stem}.lora_A.lightning"].weight, down)
        assert torch.equal(modules[f"{stem}.lora_B.lightning"].weight, up)
        assert modules[stem].scaling["lightning"] == alpha / _RANK
    assert path.exists() and len(list(tmp_path.iterdir())) == 1


def test_a_set_field_of_a_component_diffusers_does_not_load_still_loads(tmp_path):
    """Qwen-Image loads only ``transformer.*`` metadata: a set unknown field elsewhere never reaches LoraConfig."""
    path = tmp_path / "adapter.safetensors"
    meta = json.dumps(
        {
            "transformer.r": _RANK,
            "transformer.lora_alpha": 2 * _RANK,
            "transformer.target_modules": ["attn.to_out.0", "attn.to_v", "attn.to_q", "attn.to_k"],
            "text_encoder.future_lora_option": {"enabled": True},
        }
    )
    save_file(_state(), str(path), metadata = {"lora_adapter_metadata": meta})
    pipe = _pipe()
    lf.load_adapter(pipe, str(path), "lightning", "qwen-image-2.1")
    modules = dict(pipe.transformer.named_modules())
    assert modules["transformer_blocks.0.attn.to_q"].scaling["lightning"] == 2.0


def test_only_unset_unknown_fields_are_dropped(tmp_path):
    path = tmp_path / "adapter.safetensors"
    meta = {
        "transformer.r": _RANK,
        "transformer.lora_alpha": _RANK,
        "transformer.future_lora_option": None,
        "transformer.other_future_option": {"k": 1},
        "text_encoder.future_lora_option": {"enabled": True},
    }
    save_file(_state(), str(path), metadata = {"lora_adapter_metadata": json.dumps(meta)})
    header = lf.compatible_lora_header(str(path))
    kept = json.loads(header["__metadata__"]["lora_adapter_metadata"])
    assert kept == {k: v for k, v in meta.items() if k != "transformer.future_lora_option"}


def test_a_forged_header_length_is_not_read(tmp_path):
    import struct
    import tracemalloc

    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    past_end = tmp_path / "past_end.safetensors"
    past_end.write_bytes(struct.pack("<Q", 64) + b"{}")
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")
    tracemalloc.start()
    try:
        for path in (huge, past_end, short):
            assert lf.compatible_lora_header(str(path)) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20


def test_a_header_this_peft_takes_is_left_alone(tmp_path):
    path = tmp_path / "adapter.safetensors"
    meta = json.dumps({"transformer.r": _RANK, "transformer.lora_alpha": _RANK})
    save_file(_state(), str(path), metadata = {"lora_adapter_metadata": meta})
    assert lf.compatible_lora_header(str(path)) is None
    plain = tmp_path / "plain.safetensors"
    save_file(_state(), str(plain))
    assert lf.compatible_lora_header(str(plain)) is None
