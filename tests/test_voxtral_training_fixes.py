# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Voxtral fine-tuning (#3013): saving its MistralCommonBackend processor, and training Voxtral
Realtime with input gradients on."""

import pytest

transformers = pytest.importorskip("transformers")

import unsloth  # noqa: E402,F401  applies patch_enable_input_require_grads
import torch  # noqa: E402

from unsloth.import_fixes import fix_transformers_voxtral_realtime_inplace_audio_add  # noqa: E402
from unsloth.save import patch_saving_functions, unsloth_tokenizer_save_pretrained  # noqa: E402

TINY_VOXTRAL = "trl-internal-testing/tiny-VoxtralForConditionalGeneration"


class _StrictSave:
    """MistralCommonBackend.save_pretrained's contract: any extra kwarg raises, even None."""

    def __init__(self):
        self.calls = []

    def original_save_pretrained(
        self,
        save_directory,
        push_to_hub = False,
        token = None,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs)} are not supported")
        self.calls.append(save_directory)
        return (save_directory,)


def test_unset_legacy_format_and_prefix_are_not_forwarded(tmp_path):
    tok = _StrictSave()
    unsloth_tokenizer_save_pretrained(tok, str(tmp_path))
    unsloth_tokenizer_save_pretrained(tok, str(tmp_path), legacy_format = None, filename_prefix = None)
    assert tok.calls == [str(tmp_path)] * 2


def test_set_legacy_format_and_prefix_still_reach_the_tokenizer(tmp_path):
    seen = {}

    class _Recording:
        def original_save_pretrained(self, save_directory, **kwargs):
            seen.update(kwargs)

    unsloth_tokenizer_save_pretrained(
        _Recording(), str(tmp_path), legacy_format = False, filename_prefix = "p"
    )
    assert seen["legacy_format"] is False and seen["filename_prefix"] == "p"


def test_patched_voxtral_processor_saves(tmp_path):
    pytest.importorskip("mistral_common")
    try:
        processor = transformers.AutoProcessor.from_pretrained(TINY_VOXTRAL)
    except Exception as e:
        pytest.skip(f"cannot load {TINY_VOXTRAL}: {e}")
    if not isinstance(processor.tokenizer, transformers.PreTrainedTokenizerBase):
        pytest.skip("transformers 4.x MistralCommonTokenizer; FastModel cannot load Voxtral there")
    patch_saving_functions(processor)
    patch_saving_functions(processor.tokenizer)
    processor.save_pretrained(str(tmp_path / "processor"))
    # save_pretrained_merged's call shape.
    processor.tokenizer.save_pretrained(
        save_directory = str(tmp_path / "tokenizer"), legacy_format = None, filename_prefix = None
    )
    assert (tmp_path / "processor" / "tekken.json").is_file()
    assert (tmp_path / "tokenizer" / "tekken.json").is_file()


def _realtime_model():
    cls = getattr(transformers, "VoxtralRealtimeForConditionalGeneration", None)
    if cls is None:
        pytest.skip("this transformers has no VoxtralRealtimeForConditionalGeneration")
    config = transformers.VoxtralRealtimeConfig(
        audio_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            head_dim = 16,
            num_mel_bins = 16,
        ),
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 2,
            head_dim = 16,
            vocab_size = 64,
        ),
    )
    torch.manual_seed(0)
    model = cls(config).float().train()
    model.requires_grad_(False)
    trainable = [p for n, p in model.named_parameters() if "q_proj" in n and "audio" not in n][0]
    trainable.requires_grad_(True)
    return model, trainable


def _step(model, trainable):
    torch.manual_seed(0)
    features = torch.randn(1, 16, 64)  # 64 frames -> 8 audio tokens
    ids = torch.randint(0, 64, (1, 8))
    trainable.grad = None
    loss = model(input_ids = ids, input_features = features, labels = ids).loss
    loss.backward()
    return loss.detach(), trainable.grad.clone()


def test_voxtral_realtime_trains_with_input_require_grads():
    model, trainable = _realtime_model()
    ref_loss, ref_grad = _step(model, trainable)  # no input-grad hook: upstream's own path

    fix_transformers_voxtral_realtime_inplace_audio_add()
    model.enable_input_require_grads()
    loss, grad = _step(model, trainable)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(grad, ref_grad)


def test_voxtral_realtime_fix_is_idempotent():
    _realtime_model()
    fix_transformers_voxtral_realtime_inplace_audio_add()
    cls = transformers.models.voxtral_realtime.modeling_voxtral_realtime.VoxtralRealtimeModel
    patched = cls.forward
    fix_transformers_voxtral_realtime_inplace_audio_add()
    assert cls.forward is patched
    assert "inputs_embeds +=" in __import__("inspect").getsource(patched.__wrapped__)
