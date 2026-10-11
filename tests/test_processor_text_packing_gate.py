# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A processor passed as `processing_class` with a text-only dataset may pack.

FastModel returns a processor for Gemma 3 / Gemma 4 / most VLMs, and notebooks pass it straight
to SFTTrainer. On a text-only dataset TRL tokenizes and collates through the processor's
tokenizer, so packing and padding-free are exactly as safe as with the tokenizer passed
directly. Anything that might carry media, or a model whose packed mask needs more than 2D
`position_ids` (multimodal RoPE), keeps the old block.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
from torch import nn  # noqa: E402
from transformers import ProcessorMixin  # noqa: E402

try:
    import unsloth.trainer as trainer_module  # noqa: E402
    from unsloth.trainer import _is_text_only_dataset, _uses_multimodal_rope  # noqa: E402
except ImportError:
    import unsloth

    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_PROCESSOR_TEXT_PACKING", raising = False)
    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)


class _Proc(ProcessorMixin):
    """A processor without running ProcessorMixin.__init__ (no real sub-processors needed)."""

    def __init__(self, tokenizer = object()):
        self.tokenizer = tokenizer


class _Cfg:
    def __init__(self, **extra):
        self.model_type = "gemma4"
        self.vision_config = {}
        self.__dict__.update(extra)

    def to_dict(self):
        return dict(self.__dict__)


class _Model(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        self.config = config if config is not None else _Cfg()

    def forward(self, input_ids = None, **kwargs):
        return None


def _patched(raise_on = None):
    calls = []

    class _StubSFTTrainer:
        def __init__(self, model = None, args = None, **kwargs):
            calls.append((bool(getattr(args, "packing", False)), getattr(args, "padding_free", None)))
            if raise_on is not None and (args.packing or args.padding_free):
                raise ValueError(raise_on)
            self.model = model
            self.args = args

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    return module, calls


def _config(packing = False):
    return SimpleNamespace(packing = packing, padding_free = None, max_length = 512)


TEXT = datasets.Dataset.from_dict({"text": ["a b c", "d e"]})


@pytest.mark.parametrize("packing", [False, True])
def test_text_only_processor_is_unblocked(packing):
    module, _ = _patched()
    t = module.SFTTrainer(model = _Model(), args = _config(packing), train_dataset = TEXT, processing_class = _Proc())
    assert t.args.padding_free is True
    assert bool(t.args.packing) is packing


def test_kill_switch_restores_the_block(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PROCESSOR_TEXT_PACKING", "0")
    module, _ = _patched()
    t = module.SFTTrainer(model = _Model(), args = _config(True), train_dataset = TEXT, processing_class = _Proc())
    assert t.args.padding_free is False and t.args.packing is False


def _blocked(**kwargs):
    module, _ = _patched()
    kw = dict(model = _Model(), args = _config(True), train_dataset = TEXT, processing_class = _Proc())
    kw.update(kwargs)
    t = module.SFTTrainer(**kw)
    return t.args.padding_free is False and t.args.packing is False


@pytest.mark.parametrize(
    "column",
    ["image", "images", "audio", "input_features", "pixel_values", "video", "mm_token_type_ids"],
)
def test_media_columns_stay_blocked(column):
    ds = datasets.Dataset.from_dict({"text": ["a"], column: [[0]]})
    assert _blocked(train_dataset = ds)


def test_an_image_feature_nested_in_messages_stays_blocked():
    feats = datasets.Features(
        {
            "messages": [
                {
                    "role": datasets.Value("string"),
                    "content": [{"type": datasets.Value("string"), "image": datasets.Image()}],
                }
            ]
        }
    )
    ds = datasets.Dataset.from_dict(
        {"messages": [[{"role": "user", "content": [{"type": "text", "image": None}]}]]},
        features = feats,
    )
    assert not _is_text_only_dataset(ds)
    assert _blocked(train_dataset = ds)


def test_unknown_schema_stays_blocked():
    assert _blocked(train_dataset = [{"text": "a"}])


def test_an_eval_split_with_images_stays_blocked():
    img = datasets.Dataset.from_dict({"text": ["a"], "images": [[0]]})
    assert _blocked(eval_dataset = {"a": TEXT, "b": img})
    assert not _blocked(eval_dataset = {"a": TEXT})


def test_custom_collator_and_tokenizerless_processor_stay_blocked():
    assert _blocked(data_collator = lambda x: x)
    assert _blocked(processing_class = _Proc(tokenizer = None))


def test_multimodal_rope_models_stay_blocked():
    """Qwen2-VL family builds the packed mask only from 4-row position_ids, so 2D ids leak."""
    cfg = _Cfg(model_type = "qwen2_5_vl", text_config = {"rope_parameters": {"mrope_section": [1, 1, 0]}})
    assert _uses_multimodal_rope(cfg)
    assert _blocked(model = _Model(cfg))
    assert not _uses_multimodal_rope(_Cfg(text_config = {"rope_parameters": {"rope_type": "default"}}))


@pytest.mark.parametrize(
    "message",
    [
        "Packing is not supported for vision-language models. Please set `packing=False` in the SFTConfig.",
        "Padding-free training is yet not supported for vision-language models. Please set "
        "`padding_free=False` in the `SFTConfig`.",
    ],
)
def test_trl_022_refusal_falls_back_to_the_old_block(message):
    """TRL < 0.24 raises on `_is_vlm` alone; the gate then behaves exactly as before."""
    module, calls = _patched(raise_on = message)
    packing = message.startswith("Packing")
    t = module.SFTTrainer(model = _Model(), args = _config(packing), train_dataset = TEXT, processing_class = _Proc())
    assert len(calls) == 2
    assert calls[-1] == (False, False)
    assert t.args.padding_free is False and t.args.packing is False


def test_an_unrelated_value_error_still_propagates():
    module, _ = _patched(raise_on = "something else entirely")
    with pytest.raises(ValueError, match = "something else"):
        module.SFTTrainer(model = _Model(), args = _config(False), train_dataset = TEXT, processing_class = _Proc())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
