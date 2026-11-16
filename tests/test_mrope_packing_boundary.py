# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An mRoPE text model that expands 2D position_ids to 3 rope rows builds no packed mask, so
padding-free / packed rows with a tokenizer processing_class must fall back to padded rows."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
from torch import nn  # noqa: E402

try:
    import unsloth.trainer as trainer_module  # noqa: E402
except ImportError:
    import unsloth
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)

MROPE = {"rope_parameters": {"mrope_section": [1, 1, 0]}}


class _Cfg:
    def __init__(self, model_type, **extra):
        self.model_type = model_type
        self.__dict__.update(extra)

    def to_dict(self):
        return dict(self.__dict__)


class _Text3(nn.Module):
    def forward(self, position_ids = None):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
        else:
            text_position_ids = None
        return text_position_ids


class _Text3KeepsRow(nn.Module):
    """Qwen3-VL on transformers 4.x: 3 rows, but the text row is position_ids[0]."""

    def forward(self, position_ids = None):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        text_position_ids = position_ids[0]
        return text_position_ids


class _Text4(nn.Module):
    def forward(self, position_ids = None):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        return position_ids


class _Model(nn.Module):
    def __init__(self, config, text):
        super().__init__()
        self.config = config
        self.language_model = text

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return None


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)


def _args(model, packing):
    class _Stub:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model, self.args, self.data_collator = model, args, None

    module = SimpleNamespace(SFTTrainer = _Stub)
    trainer_module._patch_sft_trainer_auto_packing(module)
    args = SimpleNamespace(packing = packing, padding_free = None, max_length = 512)
    ds = datasets.Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5]]})
    return module.SFTTrainer(
        model = model, args = args, train_dataset = ds, processing_class = object()
    ).args


@pytest.mark.parametrize("packing", [False, True])
def test_three_row_mrope_falls_back_to_padded_rows(packing):
    args = _args(_Model(_Cfg("qwen2_5_vl", text_config = MROPE), _Text3()), packing)
    assert not args.padding_free and not args.packing


@pytest.mark.parametrize("packing", [False, True])
def test_text_row_mrope_and_plain_rope_keep_padding_free(packing):
    for model in (
        _Model(_Cfg("qwen3_vl", text_config = MROPE), _Text4()),
        _Model(_Cfg("llama"), _Text3()),
    ):
        args = _args(model, packing)
        assert args.padding_free is True and bool(args.packing) is packing


@pytest.mark.parametrize(
    "module,cls,leaks",
    [
        ("qwen2_5_vl", "Qwen2_5_VLTextModel", True),
        ("qwen2_vl", "Qwen2VLTextModel", True),
        ("qwen3_vl", "Qwen3VLTextModel", False),
    ],
)
def test_real_transformers_text_models(module, cls, leaks):
    import importlib

    m = importlib.import_module(f"transformers.models.{module}.modeling_{module}")
    cfg = _Cfg(module, text_config = MROPE)
    model = _Model(cfg, nn.Module())
    model.language_model.__class__ = type("T", (getattr(m, cls),), {})
    assert trainer_module._mrope_2d_position_ids_leak(model, cfg) is leaks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
